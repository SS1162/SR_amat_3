"""SPFI training script — scale ×4, EMPS dataset.

Usage
-----
    python -m amat3_sr.spfi.train \\
        --emps_dir  <path/to/EMPS>          \\
        --ckpt_dir  checkpoints/spfi_x4     \\
        --lr        1e-4                    \\
        --wd        1e-2                    \\
        --patience  20                      \\
        --val_every 250

All arguments have defaults so the script runs with no flags if the EMPS
directory is at the expected relative location.

Architecture decisions (confirmed):
  scale=4, embed_dim=48, LR patch=64×64, HR patch=256×256
  Loss = Charbonnier + 0.5·(1−SSIM)
  Optimizer = AdamW
  Checkpoint every 250 iterations
  Early stopping on dev-set validation loss (Charbonnier + 0.5·(1−SSIM))

Training recipe options (all off by default = the original recipe above):
  batch_size     patches per update (default 1)
  lr_schedule    "constant" (default) or "cosine": linear warm-up for
                 warmup_iters, then cosine decay to lr·min_lr_ratio at total_iters
  ema_decay      keep an exponential moving average of the weights; validation
                 and checkpoints ("model_state") use the averaged weights
  augment        random flips / 90° rotations of each HR crop
  cache_images   keep the decoded images in memory instead of re-reading the PNG
                 for every patch (same patches, faster)

Data decisions (confirmed):
  Random 80/20 split at image level (no stratification)
  Area-weighted image sampling: one crop per iteration
  Degradation: Gaussian blur, sigma_min=0.2, sigma_max=4.0 (=1.0*scale),
               deterministic seed per (filename, scale), bicubic downscale
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torchvision.transforms.functional import to_tensor

from .image_loading import load_image
from .model import build_spfi
from .preprocessing import normalize_image
from .run_logger import RunLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCALE = 4
EMBED_DIM = 48
LR_PATCH = 64
HR_PATCH = LR_PATCH * SCALE          # 256
SIGMA_MIN = 0.2
SIGMA_MAX = 1.0 * SCALE              # 4.0
LAMBDA_SSIM = 0.5
CKPT_EVERY = 250


# ---------------------------------------------------------------------------
# Degradation helpers
# ---------------------------------------------------------------------------

def _seed_from_pair(filename: str, scale: int) -> int:
    """Deterministic seed per (filename, scale) — stable across runs."""
    key = f"{filename}_{scale}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def degrade(hr: torch.Tensor, filename: str, scale: int = SCALE) -> torch.Tensor:
    """Apply Gaussian blur + bicubic downscale to a [1,H,W] float32 tensor.

    sigma is sampled deterministically from (filename, scale).
    Returns LR tensor [1, H//scale, W//scale].
    """
    rng = np.random.default_rng(_seed_from_pair(filename, scale))
    sigma = float(rng.uniform(SIGMA_MIN, SIGMA_MAX))

    # Gaussian blur via separable 1-D convolution
    radius = math.ceil(3 * sigma)
    ksize = 2 * radius + 1
    x = torch.arange(ksize, dtype=torch.float32) - radius
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    blurred = hr.unsqueeze(0)                                   # [1,1,H,W]
    k = kernel_1d.view(1, 1, 1, -1)
    blurred = F.conv2d(blurred, k, padding=(0, radius))
    k = kernel_1d.view(1, 1, -1, 1)
    blurred = F.conv2d(blurred, k, padding=(radius, 0))
    blurred = blurred.squeeze(0)                                # [1,H,W]

    h, w = blurred.shape[-2], blurred.shape[-1]
    lr = F.interpolate(
        blurred.unsqueeze(0),
        size=(h // scale, w // scale),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0)
    return lr


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 − SSIM, averaged over the batch.  Expects [B,1,H,W] in [0,1]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p = F.avg_pool2d(pred,   kernel_size=11, stride=1, padding=5)
    mu_t = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
    sig_p  = F.avg_pool2d(pred   ** 2, 11, 1, 5) - mu_p2
    sig_t  = F.avg_pool2d(target ** 2, 11, 1, 5) - mu_t2
    sig_pt = F.avg_pool2d(pred * target, 11, 1, 5) - mu_pt
    ssim_map = ((2 * mu_pt + C1) * (2 * sig_pt + C2)) / (
        (mu_p2 + mu_t2 + C1) * (sig_p + sig_t + C2)
    )
    return 1.0 - ssim_map.mean()


def total_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return charbonnier(pred, target) + LAMBDA_SSIM * ssim_loss(pred, target)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EMPSDataset:
    """Lazy-loading EMPS image list with area-weighted sampling.

    Each __getitem__ call returns a single (lr_patch, hr_patch) pair by:
      1. Sampling an image proportional to its pixel area.
      2. Taking one random HR crop of size HR_PATCH × HR_PATCH.
      3. Degrading the crop to produce the LR patch.
    """

    def __init__(self, image_dir: Path, filenames: List[str], scale: int = SCALE,
                 cache: bool = False) -> None:
        self.image_dir = image_dir
        self.filenames = filenames
        self.scale = scale
        self._cache: dict[str, np.ndarray] | None = {} if cache else None   # raw arrays, as on disk

        # Pre-compute image sizes for area-weighted sampling
        self._sizes: List[Tuple[int, int]] = []
        for fn in filenames:
            with Image.open(image_dir / f"{fn}.png") as img:
                self._sizes.append(img.size)   # (W, H)

        areas = np.array([w * h for w, h in self._sizes], dtype=np.float64)
        self._weights = areas / areas.sum()

    # ------------------------------------------------------------------
    def _load_gray_tensor(self, filename: str) -> torch.Tensor:
        """Load PNG as [1, H, W] float32 in [0, 1]."""
        if self._cache is None:
            raw = load_image(self.image_dir / f"{filename}.png")
        else:
            if filename not in self._cache:
                self._cache[filename] = load_image(self.image_dir / f"{filename}.png")
            raw = self._cache[filename]
        img = normalize_image(raw)              # same loading + normalization as inference
        return to_tensor(img)                   # [1, H, W], float32

    # ------------------------------------------------------------------
    def sample(self, rng: random.Random, augment: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one (lr_patch [1,64,64], hr_patch [1,256,256]) pair.

        ``augment``: apply one of the 8 flips / 90° rotations to the HR crop before
        degrading it (the rng is only consumed when augment is on).
        """
        idx = rng.choices(range(len(self.filenames)), weights=self._weights)[0]
        fn  = self.filenames[idx]
        hr  = self._load_gray_tensor(fn)        # [1, H, W]

        _, H, W = hr.shape
        # Ensure the image is large enough for one HR crop
        if H < HR_PATCH or W < HR_PATCH:
            # Upscale to minimum required size (rare edge case)
            new_H = max(H, HR_PATCH)
            new_W = max(W, HR_PATCH)
            hr = F.interpolate(
                hr.unsqueeze(0), size=(new_H, new_W),
                mode="bicubic", align_corners=False
            ).squeeze(0).clamp(0.0, 1.0)
            _, H, W = hr.shape

        top  = rng.randint(0, H - HR_PATCH)
        left = rng.randint(0, W - HR_PATCH)
        hr_patch = hr[:, top:top + HR_PATCH, left:left + HR_PATCH]
        if augment:
            hr_patch = dihedral(hr_patch, rng.randrange(8))

        lr_patch = degrade(hr_patch, filename=fn, scale=self.scale)
        return lr_patch, hr_patch


def dihedral(img: torch.Tensor, k: int) -> torch.Tensor:
    """One of the 8 flips / 90° rotations of a [..., H, W] tensor (k = 0 is the identity)."""
    if k >= 4:
        img = img.flip(-1)
    return torch.rot90(img, k % 4, dims=(-2, -1)).contiguous()


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

LR_SCHEDULES = ("constant", "cosine")


def lr_at(iteration: int, base_lr: float, schedule: str = "constant", warmup_iters: int = 0,
          total_iters: int | None = None, min_lr_ratio: float = 0.01) -> float:
    """Learning rate for update number ``iteration`` (1-based).

    A pure function of the iteration, so a resumed run gets the same rates.
    """
    if warmup_iters > 0 and iteration <= warmup_iters:
        return base_lr * iteration / warmup_iters
    if schedule == "constant":
        return base_lr
    if schedule == "cosine":
        if total_iters is None:
            raise ValueError("the cosine schedule needs total_iters")
        span = max(total_iters - warmup_iters, 1)
        progress = min(max(iteration - warmup_iters, 0) / span, 1.0)
        floor = base_lr * min_lr_ratio
        return floor + (base_lr - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    raise ValueError(f"unknown lr schedule '{schedule}', expected one of {LR_SCHEDULES}")


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------

def split_files(stems: Iterable[str], seed: int) -> Tuple[List[str], List[str]]:
    """Random 80/20 image-level split -> (train, val). Same stems + seed -> same split."""
    files = sorted(stems)
    random.Random(seed).shuffle(files)
    n_train = int(0.8 * len(files))
    return files[:n_train], files[n_train:]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Weight initialization registry
# ---------------------------------------------------------------------------
# Each entry prepares a freshly built model before training starts. Add a new
# mode by decorating a function with @register_init("<name>"); the trainer and
# the --init CLI choices pick it up automatically.

INIT_REGISTRY: dict[str, Callable[["SPFITrainer", Path | None], None]] = {}


def register_init(name: str):
    def wrapper(fn):
        INIT_REGISTRY[name] = fn
        return fn
    return wrapper


def resolve_weights(source: str | Path) -> Path:
    """Return a local path for a weights file. Only local files are supported for now."""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"weights file not found: {path}")
    return path


@register_init("scratch")
def _init_scratch(trainer: "SPFITrainer", weights: Path | None) -> None:
    """Keep the random initialization from build_spfi."""
    if weights is not None:
        raise ValueError("weights are only used with init='finetune'")


@register_init("finetune")
def _init_finetune(trainer: "SPFITrainer", weights: Path | None) -> None:
    """Start from an existing checkpoint's model weights; optimizer and counters start fresh."""
    if weights is None:
        raise ValueError("init='finetune' needs a weights file")
    path = resolve_weights(weights)
    ckpt = torch.load(path, map_location=trainer.device, weights_only=True)
    trainer.model.load_state_dict(ckpt["model_state"])
    print(f"[train] fine-tuning from {path} (trained {ckpt.get('iteration', '?')} iterations)")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SPFITrainer:
    def __init__(
        self,
        emps_dir: Path,
        ckpt_dir: Path,
        lr: float = 1e-4,
        wd: float = 1e-2,
        patience: int = 20,
        val_every: int = CKPT_EVERY,
        seed: int = 0,
        device: str | None = None,
        init: str = "scratch",
        weights: Path | None = None,
        run_dir: Path | None = None,
        log_every: int = 10,
        embed_dim: int = EMBED_DIM,
        split_seed: int | None = None,
        batch_size: int = 1,
        lr_schedule: str = "constant",
        warmup_iters: int = 0,
        total_iters: int | None = None,
        min_lr_ratio: float = 0.01,
        ema_decay: float | None = None,
        augment: bool = False,
        cache_images: bool = False,
    ) -> None:
        """``run_dir``: write meta.json / metrics.jsonl / samples/ there for the UI (see run_logger.py).

        ``embed_dim``: feature channels (model width). ``split_seed``: seed of the
        train/val split; defaults to ``seed``. Set it to keep the same val images
        across runs that use different training seeds.

        Recipe options (see the module docstring): ``batch_size``, ``lr_schedule`` /
        ``warmup_iters`` / ``total_iters`` / ``min_lr_ratio``, ``ema_decay``,
        ``augment``, ``cache_images``. The defaults reproduce the original recipe.
        """
        if init not in INIT_REGISTRY:
            raise ValueError(f"unknown init '{init}', expected one of {sorted(INIT_REGISTRY)}")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if ema_decay is not None and not 0.0 < ema_decay < 1.0:
            raise ValueError("ema_decay must be between 0 and 1")
        lr_at(1, lr, lr_schedule, warmup_iters, total_iters, min_lr_ratio)   # validates the schedule
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.val_every = val_every
        self.patience  = patience

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[train] device={self.device}")

        # ---- reproducibility ----
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ---- data split ----
        image_dir = emps_dir / "images"
        train_files, val_files = split_files(
            (p.stem for p in image_dir.glob("*.png")),
            seed if split_seed is None else split_seed,
        )
        print(f"[train] split: {len(train_files)} train / {len(val_files)} val images")

        self.train_ds = EMPSDataset(image_dir, train_files, cache=cache_images)
        self.val_ds   = EMPSDataset(image_dir, val_files, cache=cache_images)
        self._train_rng = random.Random(seed + 1)

        # ---- model ----
        self.embed_dim = embed_dim
        self.model = build_spfi(scale=SCALE, embed_dim=embed_dim).to(self.device)
        INIT_REGISTRY[init](self, weights)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        # ---- recipe ----
        self.base_lr = lr
        self.batch_size = batch_size
        self.lr_schedule, self.warmup_iters = lr_schedule, warmup_iters
        self.total_iters, self.min_lr_ratio = total_iters, min_lr_ratio
        self.augment = augment
        self.ema_decay = ema_decay
        self.ema_model: nn.Module | None = None
        if ema_decay is not None:
            self.ema_model = copy.deepcopy(self.model).requires_grad_(False)

        # ---- state ----
        self.iteration     = 0
        self.best_val_loss = float("inf")
        self.no_improve    = 0
        self.last_loss: torch.Tensor | None = None
        self.last_val: tuple[float, float] | None = None   # (val_loss, val_psnr) of the latest check

        # ---- run logging (optional) ----
        self.n_train_images, self.n_val_images = len(train_files), len(val_files)
        self.logger = RunLogger(run_dir, log_every) if run_dir is not None else None
        self._samples: list[tuple[torch.Tensor, torch.Tensor]] = []   # fixed (lr, hr) val patches

    # ------------------------------------------------------------------
    @property
    def eval_model(self) -> nn.Module:
        """The weights that are validated and exported: the EMA copy when enabled."""
        return self.ema_model if self.ema_model is not None else self.model

    @torch.no_grad()
    def _update_ema(self) -> None:
        # Short warm-up of the decay (as in common EMA implementations) so the average
        # is not dominated by the random initial weights early in training.
        decay = min(self.ema_decay, (1 + self.iteration) / (10 + self.iteration))
        for e, p in zip(self.ema_model.parameters(), self.model.parameters()):
            e.lerp_(p, 1.0 - decay)
        for e, b in zip(self.ema_model.buffers(), self.model.buffers()):
            e.copy_(b)

    # ------------------------------------------------------------------
    def _validate(self, n_samples: int = 64) -> tuple[float, float, float]:
        """Returns (val_loss, val_psnr, val_ssim). val_loss drives early stopping."""
        model = self.eval_model
        model.eval()
        total_loss_val = 0.0
        total_psnr_val = 0.0
        total_ssim_val = 0.0
        rng = random.Random(42)
        with torch.no_grad():
            for _ in range(n_samples):
                lr, hr = self.val_ds.sample(rng)
                lr = lr.unsqueeze(0).to(self.device)
                hr = hr.unsqueeze(0).to(self.device)
                sr = model(lr)
                total_loss_val += total_loss(sr, hr).item()
                total_psnr_val += psnr(sr, hr)
                total_ssim_val += 1.0 - ssim_loss(sr, hr).item()
        model.train()
        return total_loss_val / n_samples, total_psnr_val / n_samples, total_ssim_val / n_samples

    # ------------------------------------------------------------------
    def _bicubic_psnr(self, n_samples: int = 64) -> float:
        """Baseline PSNR of plain bicubic upscaling, on the same val patches as _validate."""
        total = 0.0
        rng = random.Random(42)
        for _ in range(n_samples):
            lr, hr = self.val_ds.sample(rng)
            sr = F.interpolate(lr.unsqueeze(0), scale_factor=SCALE, mode="bicubic",
                               align_corners=False).clamp(0, 1)
            total += psnr(sr, hr.unsqueeze(0))
        return total / n_samples

    def _gflops(self) -> float | None:
        """Forward-pass GFLOPs on a [1,1,64,64] input; None if counting fails."""
        try:
            from torch.utils.flop_counter import FlopCounterMode

            self.model.eval()
            with torch.no_grad(), FlopCounterMode(display=False) as counter:
                self.model(torch.zeros(1, 1, LR_PATCH, LR_PATCH, device=self.device))
            return counter.get_total_flops() / 1e9
        except Exception as exc:  # noqa: BLE001 - any failure here must not stop training
            print(f"[log] warning: could not count FLOPs ({type(exc).__name__}: {exc})")
            return None
        finally:
            self.model.train()

    def _start_logging(self) -> None:
        """Write meta.json and the fixed start samples, then the "start" event."""
        rng = random.Random(1234)   # dedicated RNG: the training sampler is untouched
        self._samples = [self.val_ds.sample(rng) for _ in range(4)]
        self.logger.write_meta({
            "run_name": self.logger.run_dir.name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "scale": SCALE,
            "embed_dim": self.embed_dim,
            "lr": self.base_lr,
            "wd": self.optimizer.param_groups[0]["weight_decay"],
            "batch_size": self.batch_size,
            "lr_schedule": self.lr_schedule,
            "warmup_iters": self.warmup_iters,
            "total_iters": self.total_iters,
            "min_lr_ratio": self.min_lr_ratio,
            "ema_decay": self.ema_decay,
            "augment": self.augment,
            "patience": self.patience,
            "val_every": CKPT_EVERY,   # the loop validates every CKPT_EVERY steps
            "n_train_images": self.n_train_images,
            "n_val_images": self.n_val_images,
            "params": sum(p.numel() for p in self.model.parameters()),
            "gflops": self._gflops(),
            "bicubic_psnr": self._bicubic_psnr(),
            "sample_ids": [1, 2, 3, 4],
        })
        for k, (lr, hr) in enumerate(self._samples, start=1):
            bicubic = F.interpolate(lr.unsqueeze(0), scale_factor=SCALE, mode="bicubic",
                                    align_corners=False).clamp(0, 1)[0]
            self.logger.save_png(f"sample_{k}_lr.png", lr)
            self.logger.save_png(f"sample_{k}_bicubic.png", bicubic)
            self.logger.save_png(f"sample_{k}_hr.png", hr)
        self.logger.log_event(0, "start")

    def _save_samples(self, step: int) -> None:
        """Model output for the fixed samples at this step (eval mode, then back to train)."""
        model = self.eval_model
        model.eval()
        with torch.no_grad():
            for k, (lr, _) in enumerate(self._samples, start=1):
                sr = model(lr.unsqueeze(0).to(self.device))[0]
                self.logger.save_png(f"sample_{k}_iter{step:07d}.png", sr)
        model.train()

    # ------------------------------------------------------------------
    def _save_checkpoint(self, tag: str) -> None:
        path = self.ckpt_dir / f"spfi_x4_{tag}.pt"
        state = {
            "iteration": self.iteration,
            "model_state": self.eval_model.state_dict(),   # what inference loads (EMA weights if enabled)
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "hparams": {"scale": SCALE, "embed_dim": self.embed_dim},
        }
        if self.ema_model is not None:
            state["train_model_state"] = self.model.state_dict()   # raw weights, to continue training
        torch.save(state, path)
        print(f"[ckpt] saved → {path}")

    # ------------------------------------------------------------------
    def train(self, stop_event: threading.Event | None = None, max_iters: int | None = None) -> None:
        """Run until early stopping, until ``stop_event`` is set, or until
        ``max_iters`` total iterations (exact, for fixed-budget comparisons).

        Calling ``train()`` again after a stop continues from the current
        iteration, model and optimizer state.
        """
        self.model.train()
        if self.logger is not None:
            if self.iteration == 0:
                self._start_logging()
            else:
                self.logger.log_event(self.iteration, "resume")
        print("[train] starting — Ctrl-C to stop early")

        try:
            self._train_loop(stop_event, max_iters)
        except KeyboardInterrupt:
            print(f"[train] interrupted at iteration {self.iteration}")
            self._log_event("stop")

    def _log_event(self, event: str) -> None:
        if self.logger is not None:
            self.logger.log_event(self.iteration, event)

    def _train_loop(self, stop_event: threading.Event | None, max_iters: int | None = None) -> None:
        while True:
            if stop_event is not None and stop_event.is_set():
                print(f"[train] stopped at iteration {self.iteration}")
                self._log_event("stop")
                break
            if max_iters is not None and self.iteration >= max_iters:
                print(f"[train] reached max_iters={max_iters}")
                self._log_event("stop")
                break
            self.iteration += 1

            pairs = [self.train_ds.sample(self._train_rng, self.augment) for _ in range(self.batch_size)]
            lr_t = torch.stack([lr for lr, _ in pairs]).to(self.device)
            hr_t = torch.stack([hr for _, hr in pairs]).to(self.device)

            step_lr = lr_at(self.iteration, self.base_lr, self.lr_schedule, self.warmup_iters,
                            self.total_iters, self.min_lr_ratio)
            for group in self.optimizer.param_groups:
                group["lr"] = step_lr

            self.optimizer.zero_grad()
            sr = self.model(lr_t)
            c = charbonnier(sr, hr_t)
            s = ssim_loss(sr, hr_t)
            loss = c + LAMBDA_SSIM * s          # == total_loss(sr, hr_t), split for logging
            loss.backward()
            self.optimizer.step()
            if self.ema_model is not None:
                self._update_ema()
            self.last_loss = loss.detach()
            if self.logger is not None:
                self.logger.log_train(self.iteration, loss, c, s, self.optimizer.param_groups[0]["lr"])

            # ---- periodic checkpoint + validation ----
            if self.iteration % CKPT_EVERY == 0:
                self._save_checkpoint(f"iter{self.iteration:07d}")

                val_l, val_p, val_s = self._validate()
                self.last_val = (val_l, val_p)
                print(
                    f"[iter {self.iteration:7d}]  loss={loss.item():.4f}"
                    f"  val_loss={val_l:.4f}  val_psnr={val_p:.2f} dB"
                    f"  best_val_loss={self.best_val_loss:.4f}"
                )

                is_best = val_l < self.best_val_loss - 1e-6
                if is_best:
                    self.best_val_loss = val_l
                    self.no_improve    = 0
                    self._save_checkpoint("best")
                else:
                    self.no_improve += 1

                if self.logger is not None:
                    self._save_samples(self.iteration)      # images first: the val line signals they exist
                    self.logger.log_val(self.iteration, val_l, val_p, val_s, is_best, self.no_improve)

                if not is_best and self.no_improve >= self.patience:
                    print(
                        f"[train] early stopping after {self.patience} "
                        f"checks without improvement."
                    )
                    self._log_event("early_stop")
                    break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_emps_dir() -> Path | None:
    """The original layout's dataset location, if this checkout is nested deep enough."""
    parents = Path(__file__).resolve().parents
    return parents[5] / "datasets" / "emps" / "EMPS" if len(parents) > 5 else None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SPFI ×4 on EMPS")
    p.add_argument(
        "--emps_dir",
        type=Path,
        default=_default_emps_dir(),
        help="Path to the EMPS root directory (contains images/)",
    )
    p.add_argument(
        "--ckpt_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "spfi_x4",
        help="Directory for checkpoints",
    )
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--wd",       type=float, default=1e-2)
    p.add_argument("--patience", type=int,   default=20,
                   help="Early-stopping: checks without improvement before stopping")
    p.add_argument("--val_every", type=int,  default=CKPT_EVERY,
                   help="Validate (and checkpoint) every N iterations")
    p.add_argument("--seed",     type=int,   default=0)
    p.add_argument("--split_seed", type=int, default=None,
                   help="Seed of the train/val split (default: --seed)")
    p.add_argument("--embed_dim", type=int,  default=EMBED_DIM,
                   help="Feature channels (model width)")
    p.add_argument("--max_iters", type=int,  default=None,
                   help="Stop after exactly N iterations (default: early stopping only)")
    p.add_argument("--batch_size", type=int, default=1, help="Patches per update")
    p.add_argument("--lr_schedule", type=str, default="constant", choices=LR_SCHEDULES,
                   help="'cosine' decays from --lr to --lr x --min_lr_ratio at --max_iters")
    p.add_argument("--warmup_iters", type=int, default=0, help="Linear LR warm-up iterations")
    p.add_argument("--min_lr_ratio", type=float, default=0.01, help="Final LR / --lr for 'cosine'")
    p.add_argument("--ema_decay", type=float, default=None,
                   help="Average the weights (e.g. 0.999); validation and checkpoints use the average")
    p.add_argument("--augment", action="store_true", help="Random flips / 90° rotations of the patches")
    p.add_argument("--cache_images", action="store_true", help="Keep the decoded images in memory")
    p.add_argument("--device",   type=str,   default=None,
                   help="'cuda', 'cpu', or None for auto-detect")
    p.add_argument("--init",     type=str,   default="scratch", choices=sorted(INIT_REGISTRY),
                   help="How to initialize the model before training")
    p.add_argument("--weights",  type=Path,  default=None,
                   help="Checkpoint (.pt) to start from; required with --init finetune")
    p.add_argument("--run_dir",  type=Path,
                   default=Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S"),
                   help="Run folder for the UI (meta.json, metrics.jsonl, samples/)")
    p.add_argument("--log_every", type=int,  default=10,
                   help="Write an averaged train line to metrics.jsonl every N iterations")
    args = p.parse_args()
    if args.emps_dir is None:
        p.error("--emps_dir is required (no default dataset location for this checkout)")
    if args.init == "finetune" and args.weights is None:
        p.error("--init finetune needs --weights <checkpoint.pt>")
    if args.lr_schedule == "cosine" and args.max_iters is None:
        p.error("--lr_schedule cosine needs --max_iters (the decay ends there)")
    return args


def main() -> None:
    args = _parse_args()
    trainer = SPFITrainer(
        emps_dir  = args.emps_dir,
        ckpt_dir  = args.ckpt_dir,
        lr        = args.lr,
        wd        = args.wd,
        patience  = args.patience,
        val_every = args.val_every,
        seed      = args.seed,
        device    = args.device,
        init      = args.init,
        weights   = args.weights,
        run_dir   = args.run_dir,
        log_every = args.log_every,
        embed_dim = args.embed_dim,
        split_seed = args.split_seed,
        batch_size = args.batch_size,
        lr_schedule = args.lr_schedule,
        warmup_iters = args.warmup_iters,
        total_iters = args.max_iters,
        min_lr_ratio = args.min_lr_ratio,
        ema_decay = args.ema_decay,
        augment = args.augment,
        cache_images = args.cache_images,
    )
    trainer.train(max_iters=args.max_iters)


if __name__ == "__main__":
    main()
