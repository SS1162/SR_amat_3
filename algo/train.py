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

Data decisions (confirmed):
  Random 80/20 split at image level (no stratification)
  Area-weighted image sampling: one crop per iteration
  Degradation: Gaussian blur, sigma_min=0.2, sigma_max=4.0 (=1.0*scale),
               deterministic seed per (filename, scale), bicubic downscale
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import threading
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torchvision.transforms.functional import to_tensor

from .model import build_spfi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCALE = 4
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

    def __init__(self, image_dir: Path, filenames: List[str], scale: int = SCALE) -> None:
        self.image_dir = image_dir
        self.filenames = filenames
        self.scale = scale

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
        path = self.image_dir / f"{filename}.png"
        img = Image.open(path).convert("L")
        return to_tensor(img)                   # [1, H, W], float32

    # ------------------------------------------------------------------
    def sample(self, rng: random.Random) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one (lr_patch [1,64,64], hr_patch [1,256,256]) pair."""
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

        lr_patch = degrade(hr_patch, filename=fn, scale=self.scale)
        return lr_patch, hr_patch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


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
    ) -> None:
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
        all_files = sorted(
            p.stem for p in image_dir.glob("*.png")
        )
        rng_split = random.Random(seed)
        rng_split.shuffle(all_files)
        n_train = int(0.8 * len(all_files))
        train_files = all_files[:n_train]
        val_files   = all_files[n_train:]
        print(f"[train] split: {len(train_files)} train / {len(val_files)} val images")

        self.train_ds = EMPSDataset(image_dir, train_files)
        self.val_ds   = EMPSDataset(image_dir, val_files)
        self._train_rng = random.Random(seed + 1)

        # ---- model ----
        self.model = build_spfi(scale=SCALE, embed_dim=48).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        # ---- state ----
        self.iteration     = 0
        self.best_val_loss = float("inf")
        self.no_improve    = 0
        self.last_loss: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def _validate(self, n_samples: int = 64) -> tuple[float, float]:
        """Returns (val_loss, val_psnr). val_loss drives early stopping."""
        self.model.eval()
        total_loss_val = 0.0
        total_psnr_val = 0.0
        rng = random.Random(42)
        with torch.no_grad():
            for _ in range(n_samples):
                lr, hr = self.val_ds.sample(rng)
                lr = lr.unsqueeze(0).to(self.device)
                hr = hr.unsqueeze(0).to(self.device)
                sr = self.model(lr)
                total_loss_val += total_loss(sr, hr).item()
                total_psnr_val += psnr(sr, hr)
        self.model.train()
        return total_loss_val / n_samples, total_psnr_val / n_samples

    # ------------------------------------------------------------------
    def _save_checkpoint(self, tag: str) -> None:
        path = self.ckpt_dir / f"spfi_x4_{tag}.pt"
        torch.save({
            "iteration": self.iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, path)
        print(f"[ckpt] saved → {path}")

    # ------------------------------------------------------------------
    def train(self, stop_event: threading.Event | None = None) -> None:
        """Run until early stopping, or until ``stop_event`` is set.

        Calling ``train()`` again after a stop continues from the current
        iteration, model and optimizer state.
        """
        self.model.train()
        print("[train] starting — Ctrl-C to stop early")

        while True:
            if stop_event is not None and stop_event.is_set():
                print(f"[train] stopped at iteration {self.iteration}")
                break
            self.iteration += 1

            lr_patch, hr_patch = self.train_ds.sample(self._train_rng)
            lr_t = lr_patch.unsqueeze(0).to(self.device)
            hr_t = hr_patch.unsqueeze(0).to(self.device)

            self.optimizer.zero_grad()
            sr = self.model(lr_t)
            loss = total_loss(sr, hr_t)
            loss.backward()
            self.optimizer.step()
            self.last_loss = loss.detach()

            # ---- periodic checkpoint + validation ----
            if self.iteration % CKPT_EVERY == 0:
                self._save_checkpoint(f"iter{self.iteration:07d}")

                val_l, val_p = self._validate()
                print(
                    f"[iter {self.iteration:7d}]  loss={loss.item():.4f}"
                    f"  val_loss={val_l:.4f}  val_psnr={val_p:.2f} dB"
                    f"  best_val_loss={self.best_val_loss:.4f}"
                )

                if val_l < self.best_val_loss - 1e-6:
                    self.best_val_loss = val_l
                    self.no_improve    = 0
                    self._save_checkpoint("best")
                else:
                    self.no_improve += 1
                    if self.no_improve >= self.patience:
                        print(
                            f"[train] early stopping after {self.patience} "
                            f"checks without improvement."
                        )
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
    p.add_argument("--device",   type=str,   default=None,
                   help="'cuda', 'cpu', or None for auto-detect")
    args = p.parse_args()
    if args.emps_dir is None:
        p.error("--emps_dir is required (no default dataset location for this checkout)")
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
    )
    trainer.train()


if __name__ == "__main__":
    main()
