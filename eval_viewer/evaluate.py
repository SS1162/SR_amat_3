"""Run a model over a fixed random sample of dataset images and cache the results.

Inference only: no training happens here. The LR input is produced with the training
degradation (``algo.train.degrade``) and the metrics are the training ones
(``algo.train.psnr`` / ``ssim_loss``), so the numbers match the training logs.

Cache layout (one folder per sample, shared by all models)::

    eval_cache/<sample_key>/
        sample.json                    the sample definition (data_dir, n, seed, crop, stems)
        hr/ lr/ bicubic/<stem>.png     8-bit previews
        hr16/<stem>.png                exact HR (16-bit) that the model runs on
        reference.csv                  stem, sizes, blur sigma, bicubic PSNR / SSIM
        models/<model_key>/
            meta.json                  model path, hparams, iteration
            metrics.csv                stem, psnr, ssim, seconds
            sr/<stem>.png

CLI::

    python -m eval_viewer.evaluate --model eval_viewer/models/base/spfi_x4_base.pt --data <EMPS dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from algo.image_loading import load_image
from algo.preprocessing import normalize_image
from algo.train import SCALE, SIGMA_MAX, SIGMA_MIN, _seed_from_pair, degrade, psnr, ssim_loss
from inference.runner import _load_model, _select_device

from .config import (
    BASE_MODEL_PATH,
    CACHE_DIR,
    IMAGE_EXTENSIONS,
    MODELS_DIR,
    REPO_ROOT,
    RUNS_DIR,
    SIZE_MULTIPLE,
)

Progress = Callable[[int, int, str], None]

REFERENCE_FIELDS = ["stem", "orig_w", "orig_h", "crop_w", "crop_h", "sigma", "bicubic_psnr", "bicubic_ssim"]
METRIC_FIELDS = ["stem", "psnr", "ssim", "seconds"]


# ---------------------------------------------------------------------------
# Sample definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleSpec:
    data_dir: Path
    n: int
    seed: int
    crop: str            # "256", "320", ... or "full"
    subset: str = "all"  # "all" or "test" (stems listed in the dataset's test.csv)

    @property
    def key(self) -> str:
        digest = hashlib.sha1(str(self.data_dir.resolve()).encode("utf-8")).hexdigest()[:6]
        return f"{_slug(self.data_dir.name)}-{digest}_n{self.n}_s{self.seed}_c{self.crop}_{self.subset}"

    @property
    def dir(self) -> Path:
        return CACHE_DIR / self.key


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "x"


def image_dir(data_dir: Path) -> Path:
    """EMPS keeps its images in ``images/``; any other folder is used as is."""
    sub = data_dir / "images"
    return sub if sub.is_dir() else data_dir


def list_images(data_dir: Path, subset: str = "all") -> list[Path]:
    folder = image_dir(data_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"dataset folder not found: {folder}")
    paths = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."))
    if subset == "test":
        split_file = data_dir / "test.csv"
        if not split_file.is_file():
            raise FileNotFoundError(f"subset 'test' needs {split_file}")
        stems = {line.split(",")[0].strip() for line in split_file.read_text().splitlines() if line.strip()}
        paths = [p for p in paths if p.stem in stems]
    return paths


def select_images(spec: SampleSpec) -> list[Path]:
    """Same spec -> same images, in a stable order."""
    paths = list_images(spec.data_dir, spec.subset)
    if not paths:
        raise FileNotFoundError(f"no images in {image_dir(spec.data_dir)}")
    chosen = random.Random(spec.seed).sample(paths, min(spec.n, len(paths)))
    return sorted(chosen, key=lambda p: p.stem)


def center_crop(hr: torch.Tensor, crop: str) -> torch.Tensor:
    """Center crop of a [1,H,W] tensor; sides are rounded down to a multiple of SIZE_MULTIPLE."""
    _, h, w = hr.shape
    max_h, max_w = h - h % SIZE_MULTIPLE, w - w % SIZE_MULTIPLE
    if crop == "full":
        ch, cw = max_h, max_w
    else:
        side = int(crop)
        side -= side % SIZE_MULTIPLE
        ch, cw = min(side, max_h), min(side, max_w)
    if ch <= 0 or cw <= 0:
        raise ValueError(f"image too small to crop: {w}x{h}")
    top, left = (h - ch) // 2, (w - cw) // 2
    return hr[:, top:top + ch, left:left + cw]


def blur_sigma(stem: str, scale: int = SCALE) -> float:
    """The blur sigma ``degrade`` uses for this file (same seed, same draw)."""
    rng = np.random.default_rng(_seed_from_pair(stem, scale))
    return float(rng.uniform(SIGMA_MIN, SIGMA_MAX))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    return 1.0 - ssim_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()


def save_png(t: torch.Tensor, path: Path) -> None:
    """[1,H,W] or [1,1,H,W] float tensor in [0,1] -> 8-bit grayscale PNG."""
    arr = t.detach().cpu().clamp(0, 1).squeeze().numpy()
    Image.fromarray(np.round(arr * 255).astype(np.uint8), mode="L").save(path)


def save_png16(t: torch.Tensor, path: Path) -> None:
    """Lossless store of the HR: 16 bits keep 8-bit sources exact (x/255 == 257x/65535)."""
    arr = t.detach().cpu().clamp(0, 1).squeeze().numpy()
    Image.fromarray(np.round(arr * 65535).astype(np.uint16)).save(path)


def load_png(path: Path) -> np.ndarray:
    """8- or 16-bit grayscale PNG -> float32 [H,W] in [0,1]."""
    with Image.open(path) as im:
        if im.mode.startswith("I"):
            return np.asarray(im, dtype=np.float32) / 65535.0
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict]:
    """Rows with every non-stem value converted to float."""
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: (v if k == "stem" else float(v)) for k, v in row.items()} for row in csv.DictReader(f)]


def prepare_sample(spec: SampleSpec, progress: Progress | None = None,
                   stop: threading.Event | None = None) -> Path:
    """HR / LR / bicubic previews and bicubic metrics, once per sample. Returns the sample folder."""
    out = spec.dir
    if (out / "reference.csv").is_file():
        return out
    paths = select_images(spec)
    for sub in ("hr", "hr16", "lr", "bicubic"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    for i, path in enumerate(paths):
        if stop is not None and stop.is_set():
            raise InterruptedError("stopped")
        stem = path.stem
        full = to_tensor(normalize_image(load_image(path)))           # [1,H,W]
        hr = center_crop(full, spec.crop)
        hr = torch.round(hr * 65535) / 65535                         # exactly what hr16/ stores
        lr = degrade(hr, filename=stem, scale=SCALE)                 # training degradation
        bicubic = F.interpolate(lr.unsqueeze(0), scale_factor=SCALE, mode="bicubic",
                                align_corners=False).clamp(0, 1)
        save_png(hr, out / "hr" / f"{stem}.png")
        save_png16(hr, out / "hr16" / f"{stem}.png")
        save_png(lr, out / "lr" / f"{stem}.png")
        save_png(bicubic, out / "bicubic" / f"{stem}.png")
        rows.append({
            "stem": stem,
            "orig_w": full.shape[2], "orig_h": full.shape[1],
            "crop_w": hr.shape[2], "crop_h": hr.shape[1],
            "sigma": round(blur_sigma(stem), 4),
            "bicubic_psnr": round(psnr(bicubic, hr.unsqueeze(0)), 4),
            "bicubic_ssim": round(ssim(bicubic, hr.unsqueeze(0)), 5),
        })
        if progress:
            progress(i + 1, len(paths), stem)

    (out / "sample.json").write_text(json.dumps({
        "data_dir": str(spec.data_dir), "n": spec.n, "seed": spec.seed, "crop": spec.crop,
        "subset": spec.subset, "scale": SCALE, "stems": [p.stem for p in paths],
        "image_paths": {p.stem: str(p) for p in paths},
        "created": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    _write_csv(out / "reference.csv", REFERENCE_FIELDS, rows)   # written last: marks the sample done
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def model_label(path: Path) -> str:
    path = path.resolve()
    for root in (REPO_ROOT,):
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return str(path)


def model_key(path: Path) -> str:
    """Cache folder name of a checkpoint; changes when the file is replaced."""
    path = path.resolve()
    st = path.stat()
    digest = hashlib.sha1(f"{path}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()[:8]
    return f"{_slug(model_label(path).replace('/', '__'))[-80:]}-{digest}"


def list_models() -> list[Path]:
    """The base model first, then any .pt under eval_viewer/models/ and runs/*/checkpoints/."""
    found: list[Path] = []
    if BASE_MODEL_PATH.is_file():
        found.append(BASE_MODEL_PATH)
    for pattern_root, pattern in ((MODELS_DIR, "**/*.pt"), (RUNS_DIR, "*/checkpoints/*.pt")):
        if pattern_root.is_dir():
            for p in sorted(pattern_root.glob(pattern)):
                if p.resolve() != BASE_MODEL_PATH.resolve():
                    found.append(p)
    return found


def model_dir(spec: SampleSpec, model_path: Path) -> Path:
    return spec.dir / "models" / model_key(model_path)


def is_evaluated(spec: SampleSpec, model_path: Path) -> bool:
    return (spec.dir / "reference.csv").is_file() and (model_dir(spec, model_path) / "metrics.csv").is_file()


def evaluate_model(spec: SampleSpec, model_path: Path, progress: Progress | None = None,
                   stop: threading.Event | None = None) -> Path:
    """Run the model over the sample (prepared on demand). Returns the model's cache folder."""
    sample_dir = prepare_sample(spec, progress, stop)
    out = model_dir(spec, model_path)
    if (out / "metrics.csv").is_file():
        return out
    (out / "sr").mkdir(parents=True, exist_ok=True)

    device = _select_device()
    model = _load_model(model_path, device)
    ckpt_info = torch.load(model_path, map_location="cpu", weights_only=True)
    stems = json.loads((sample_dir / "sample.json").read_text())["stems"]

    rows = []
    for i, stem in enumerate(stems):
        if stop is not None and stop.is_set():
            raise InterruptedError("stopped")
        hr = torch.from_numpy(load_png(sample_dir / "hr16" / f"{stem}.png"))[None, None]
        # Same HR and degradation as prepare_sample, so bicubic and model metrics compare 1:1.
        lr = degrade(hr[0], filename=stem, scale=SCALE).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.inference_mode():
            sr = model(lr.to(device)).float().cpu().clamp(0, 1)
        seconds = time.perf_counter() - t0
        save_png(sr, out / "sr" / f"{stem}.png")
        rows.append({"stem": stem, "psnr": round(psnr(sr, hr), 4), "ssim": round(ssim(sr, hr), 5),
                     "seconds": round(seconds, 3)})
        if progress:
            progress(i + 1, len(stems), stem)

    (out / "meta.json").write_text(json.dumps({
        "model_path": str(model_path.resolve()), "label": model_label(model_path),
        "hparams": ckpt_info.get("hparams"), "iteration": ckpt_info.get("iteration"),
        "best_val_loss": ckpt_info.get("best_val_loss"), "device": str(device),
        "created": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    _write_csv(out / "metrics.csv", METRIC_FIELDS, rows)   # written last: marks the model done
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    from .config import DEFAULT_CROP, DEFAULT_DATA_DIR, DEFAULT_N_IMAGES, DEFAULT_SEED, CROP_OPTIONS
    p = argparse.ArgumentParser(description="Cache SR results of a model on a random dataset sample")
    p.add_argument("--model", type=Path, action="append",
                   help="Checkpoint (.pt); repeat for several. Default: the base model")
    p.add_argument("--data", type=Path, default=Path(DEFAULT_DATA_DIR), help="Dataset root (EMPS)")
    p.add_argument("--n", type=int, default=DEFAULT_N_IMAGES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--crop", default=DEFAULT_CROP, choices=CROP_OPTIONS)
    p.add_argument("--subset", default="all", choices=["all", "test"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    spec = SampleSpec(args.data, args.n, args.seed, args.crop, args.subset)
    for model_path in args.model or [BASE_MODEL_PATH]:
        print(f"[eval] {model_label(model_path)} on {spec.key}")
        out = evaluate_model(spec, model_path,
                             progress=lambda d, t, s: print(f"  {d}/{t} {s}", end="\r"))
        rows = read_csv(out / "metrics.csv")
        print(f"\n[eval] mean PSNR {np.mean([r['psnr'] for r in rows]):.2f} dB -> {out}")


if __name__ == "__main__":
    main()
