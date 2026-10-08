"""Fake training run for developing the TensorBoard page without a model or GPU.

Writes a run folder in exactly the format the real training script will produce:

    runs/<run_name>/
    ├── meta.json        written once, before the first iteration
    ├── metrics.jsonl    one JSON object per line: train / val / event
    └── samples/         sample_{k}_{lr,bicubic,hr}.png + sample_{k}_iter{step:07d}.png

The "SR" images are a blend of bicubic and HR that sharpens over time, then slowly
overfits after `--plateau` so early stopping triggers. PSNR / SSIM / val_loss are
computed from those images, so the curves and the pictures always agree.

Usage
-----
    python ui/dev/fake_run.py                 # live: streams lines, for testing auto-refresh
    python ui/dev/fake_run.py --instant       # writes the whole run at once
    python ui/dev/fake_run.py --run_name run_b --speed 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spfi_ui.config import RUNS_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (mirror algo/train.py)
# ---------------------------------------------------------------------------
SCALE = 4
HR_PATCH = 256
LR_PATCH = HR_PATCH // SCALE
VAL_EVERY = 250
LOG_EVERY = 10
LAMBDA_SSIM = 0.5
SAMPLE_IDS = [1, 2, 3, 4]
SAMPLE_KINDS = ["dense", "single", "background", "texture"]
SAMPLE_SIGMAS = [1.2, 2.0, 1.5, 2.5]
FAKE_ITERS_PER_SEC = 4.1     # simulated clock used in --instant mode


# ---------------------------------------------------------------------------
# Synthetic microscopy-like images
# ---------------------------------------------------------------------------

def _particle(canvas: np.ndarray, cy: float, cx: float, ry: float, rx: float, value: float) -> None:
    """Draw a soft-edged ellipse onto canvas in place."""
    yy, xx = np.mgrid[0:canvas.shape[0], 0:canvas.shape[1]]
    d = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    mask = np.clip((1.15 - d) / 0.15, 0.0, 1.0)
    canvas[:] = canvas * (1 - mask) + value * mask


def make_hr(kind: str, rng: np.random.Generator) -> np.ndarray:
    """Return a [H,W] float array in [0,1] that looks roughly like an EMPS crop."""
    yy, xx = np.mgrid[0:HR_PATCH, 0:HR_PATCH] / HR_PATCH
    img = 0.22 + 0.08 * yy + 0.05 * np.sin(3 * xx)

    if kind == "dense":
        for _ in range(28):
            r = rng.uniform(8, 20)
            _particle(img, rng.uniform(0, HR_PATCH), rng.uniform(0, HR_PATCH),
                      r, r * rng.uniform(0.7, 1.3), rng.uniform(0.6, 0.9))
    elif kind == "single":
        _particle(img, 128, 128, 70, 55, 0.85)
        _particle(img, 128, 128, 25, 20, 0.55)
    elif kind == "background":
        for _ in range(3):
            r = rng.uniform(6, 12)
            _particle(img, rng.uniform(0, HR_PATCH), rng.uniform(0, HR_PATCH), r, r, 0.5)
    elif kind == "texture":
        img = img + 0.12 * np.sin(40 * xx + 6 * np.sin(9 * yy)) * np.cos(33 * yy)

    img = img + rng.normal(0, 0.02, img.shape)
    return np.clip(img, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8))


def from_pil(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float64) / 255.0


def degrade(hr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur + bicubic downscale, like algo/train.py:degrade."""
    blurred = to_pil(hr).filter(ImageFilter.GaussianBlur(radius=sigma))
    return from_pil(blurred.resize((LR_PATCH, LR_PATCH), Image.BICUBIC))


def upscale_bicubic(lr: np.ndarray) -> np.ndarray:
    return from_pil(to_pil(lr).resize((HR_PATCH, HR_PATCH), Image.BICUBIC))


def save_png(arr: np.ndarray, path: Path) -> None:
    """Write via a temp file so a polling reader never sees a half-written PNG."""
    tmp = path.with_name(path.name + ".tmp")
    to_pil(arr).save(tmp, format="PNG")
    os.replace(tmp, path)


def write_json(obj: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Metrics (same definitions as algo/train.py, numpy versions)
# ---------------------------------------------------------------------------

def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def charbonnier(pred: np.ndarray, target: np.ndarray, eps: float = 1e-3) -> float:
    return float(np.mean(np.sqrt((pred - target) ** 2 + eps ** 2)))


def _box(x: np.ndarray, k: int = 11) -> np.ndarray:
    """Mean filter with edge padding (stand-in for avg_pool2d in train.py)."""
    p = k // 2
    x = np.pad(x, p, mode="edge")
    c = np.cumsum(np.cumsum(x, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_p, mu_t = _box(pred), _box(target)
    sig_p = _box(pred ** 2) - mu_p ** 2
    sig_t = _box(target ** 2) - mu_t ** 2
    sig_pt = _box(pred * target) - mu_p * mu_t
    ssim_map = ((2 * mu_p * mu_t + c1) * (2 * sig_pt + c2)) / (
        (mu_p ** 2 + mu_t ** 2 + c1) * (sig_p + sig_t + c2)
    )
    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# Simulated model behaviour
# ---------------------------------------------------------------------------

PEAK_SHARPNESS = 0.75   # exaggerated on purpose so the sharpening is easy to see in the UI


def sharpness(step: int, plateau: int) -> float:
    """How far SR has moved from a very blurry first guess (0) towards HR (1)."""
    tau = plateau / 4
    peak = PEAK_SHARPNESS * (1 - math.exp(-plateau / tau))
    if step <= plateau:
        return PEAK_SHARPNESS * (1 - math.exp(-step / tau))
    return max(0.0, peak - 0.01 * (step - plateau) / VAL_EVERY)    # slow overfitting


def fake_sr(bicubic: np.ndarray, hr: np.ndarray, step: int, plateau: int,
            rng: np.random.Generator) -> np.ndarray:
    """Early outputs are blurrier and noisier than bicubic, later ones much closer to HR."""
    q = sharpness(step, plateau)
    first_guess = from_pil(to_pil(bicubic).filter(ImageFilter.GaussianBlur(radius=3)))
    noise = rng.normal(0, 0.04 * (1 - q / PEAK_SHARPNESS) + 0.004, hr.shape)
    return np.clip((1 - q) * first_guess + q * hr + noise, 0.0, 1.0)


def fake_train_loss(step: int, rng: np.random.Generator) -> tuple[float, float]:
    """Return (charbonnier, ssim_loss) for one noisy batch-size-1 iteration."""
    base = 0.035 + 0.12 * math.exp(-step / 1500)
    loss = base * float(rng.lognormal(0, 0.35))
    charb = 0.55 * loss
    return charb, (loss - charb) / LAMBDA_SSIM


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class FakeRun:
    def __init__(self, run_dir: Path, instant: bool, seed: int, expected_iters: int = 0) -> None:
        self.run_dir = run_dir
        self.samples_dir = run_dir / "samples"
        self.instant = instant
        self.rng = np.random.default_rng(seed)
        self.t0 = time.time() - (expected_iters / FAKE_ITERS_PER_SEC if instant else 0)

        if run_dir.exists():
            shutil.rmtree(run_dir)
        self.samples_dir.mkdir(parents=True)
        self._metrics = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")

    def now(self, step: int) -> float:
        return self.t0 + step / FAKE_ITERS_PER_SEC if self.instant else time.time()

    def log(self, record: dict) -> None:
        self._metrics.write(json.dumps(record) + "\n")
        self._metrics.flush()

    def event(self, step: int, name: str) -> None:
        self.log({"type": "event", "step": step, "time": self.now(step), "event": name})

    def close(self) -> None:
        self._metrics.close()


def run(args: argparse.Namespace) -> None:
    # in --instant mode the simulated clock is back-dated so the run ends "now"
    expected_iters = min(args.max_iters, args.plateau + args.patience * VAL_EVERY)
    fr = FakeRun(RUNS_DIR / args.run_name, args.instant, args.seed, expected_iters)
    rng = fr.rng
    delay = 0.0 if args.instant else 0.2 / args.speed

    # ---- fixed samples ----
    samples = []
    for k, kind, sigma in zip(SAMPLE_IDS, SAMPLE_KINDS, SAMPLE_SIGMAS):
        hr = make_hr(kind, rng)
        lr = degrade(hr, sigma)
        bic = upscale_bicubic(lr)
        save_png(hr, fr.samples_dir / f"sample_{k}_hr.png")
        save_png(lr, fr.samples_dir / f"sample_{k}_lr.png")
        save_png(bic, fr.samples_dir / f"sample_{k}_bicubic.png")
        samples.append((k, hr, bic))

    write_json({
        "run_name": args.run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "scale": SCALE, "lr": 1e-4, "wd": 1e-2,
        "patience": args.patience, "val_every": VAL_EVERY,
        "n_train_images": 80, "n_val_images": 20,
        "params": 912_345,
        "gflops": 1.83,
        "bicubic_psnr": round(float(np.mean([psnr(b, h) for _, h, b in samples])), 3),
        "sample_ids": SAMPLE_IDS,
    }, fr.run_dir / "meta.json")

    print(f"[fake] writing to {fr.run_dir}")
    fr.event(0, "start")

    best_val_loss = float("inf")
    no_improve = 0
    acc_c, acc_s = [], []
    step = 0
    try:
        while step < args.max_iters:
            step += 1
            c, s = fake_train_loss(step, rng)
            acc_c.append(c)
            acc_s.append(s)

            if step % LOG_EVERY == 0:
                mc, ms = float(np.mean(acc_c)), float(np.mean(acc_s))
                fr.log({"type": "train", "step": step, "time": fr.now(step),
                        "loss": mc + LAMBDA_SSIM * ms, "charbonnier": mc, "ssim_loss": ms,
                        "lr": 1e-4})
                acc_c, acc_s = [], []
                time.sleep(delay)

            if step % VAL_EVERY == 0:
                # images first: the val line signals that this step's images exist
                losses, psnrs, ssims = [], [], []
                for k, hr, bic in samples:
                    sr = fake_sr(bic, hr, step, args.plateau, rng)
                    save_png(sr, fr.samples_dir / f"sample_{k}_iter{step:07d}.png")
                    sv = ssim(sr, hr)
                    losses.append(charbonnier(sr, hr) + LAMBDA_SSIM * (1 - sv))
                    psnrs.append(psnr(sr, hr))
                    ssims.append(sv)

                val_l = float(np.mean(losses))
                is_best = val_l < best_val_loss - 1e-6
                if is_best:
                    best_val_loss, no_improve = val_l, 0
                else:
                    no_improve += 1

                fr.log({"type": "val", "step": step, "time": fr.now(step),
                        "val_loss": val_l, "val_psnr": float(np.mean(psnrs)),
                        "val_ssim": float(np.mean(ssims)),
                        "is_best": is_best, "no_improve": no_improve})
                print(f"[fake] iter {step:6d}  val_loss={val_l:.4f}  "
                      f"val_psnr={np.mean(psnrs):.2f} dB  no_improve={no_improve}")

                if no_improve >= args.patience:
                    fr.event(step, "early_stop")
                    print(f"[fake] early stop at iter {step}")
                    break
        else:
            fr.event(step, "done")
    except KeyboardInterrupt:
        fr.event(step, "stop")
        print(f"\n[fake] stopped at iter {step}")
    finally:
        fr.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a fake SPFI training run for UI development")
    p.add_argument("--run_name", default="fake_run")
    p.add_argument("--instant", action="store_true", help="write the whole run immediately")
    p.add_argument("--speed", type=float, default=1.0, help="live mode speed multiplier")
    p.add_argument("--plateau", type=int, default=6000,
                   help="iteration after which the fake model stops improving")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max_iters", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
