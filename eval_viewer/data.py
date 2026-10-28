"""Reading the cache for the UI: per-image rows, summary stats, error maps. No torch here."""

from __future__ import annotations

import base64
import io
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CACHE_DIR
from .evaluate import SampleSpec, load_png, model_dir, read_csv

# Ties within this many dB count as "same as bicubic / base".
TIE_DB = 0.05


def load_rows(spec: SampleSpec, model_path: Path, base_path: Path | None) -> list[dict] | None:
    """One row per image with model, bicubic and (optional) base metrics; None if not evaluated."""
    ref_file = spec.dir / "reference.csv"
    metrics_file = model_dir(spec, model_path) / "metrics.csv"
    if not ref_file.is_file() or not metrics_file.is_file():
        return None
    model = {r["stem"]: r for r in read_csv(metrics_file)}
    base = {}
    if base_path is not None and base_path.resolve() != model_path.resolve():
        base_file = model_dir(spec, base_path) / "metrics.csv"
        if base_file.is_file():
            base = {r["stem"]: r for r in read_csv(base_file)}

    rows = []
    for ref in read_csv(ref_file):
        m = model.get(ref["stem"])
        if m is None:
            continue
        row = {**ref, "psnr": m["psnr"], "ssim": m["ssim"], "seconds": m["seconds"],
               "d_psnr": m["psnr"] - ref["bicubic_psnr"], "d_ssim": m["ssim"] - ref["bicubic_ssim"]}
        b = base.get(ref["stem"])
        if b is not None:
            row.update(base_psnr=b["psnr"], base_ssim=b["ssim"], d_base=m["psnr"] - b["psnr"])
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    def col(key):
        return [r[key] for r in rows if key in r]

    psnr, ssim, d = col("psnr"), col("ssim"), col("d_psnr")
    out = {
        "n": len(rows),
        "psnr_mean": statistics.fmean(psnr), "psnr_median": statistics.median(psnr),
        "psnr_std": statistics.pstdev(psnr),
        "ssim_mean": statistics.fmean(ssim),
        "bicubic_psnr_mean": statistics.fmean(col("bicubic_psnr")),
        "bicubic_ssim_mean": statistics.fmean(col("bicubic_ssim")),
        "d_psnr_mean": statistics.fmean(d),
        "better_pct": 100 * sum(x > TIE_DB for x in d) / len(d),
        "worse_pct": 100 * sum(x < -TIE_DB for x in d) / len(d),
        "seconds_mean": statistics.fmean(col("seconds")),
    }
    d_base = col("d_base")
    if d_base:
        out.update(base_psnr_mean=statistics.fmean(col("base_psnr")),
                   d_base_mean=statistics.fmean(d_base),
                   better_base_pct=100 * sum(x > TIE_DB for x in d_base) / len(d_base))
    return out


def model_meta(spec: SampleSpec, model_path: Path) -> dict:
    f = model_dir(spec, model_path) / "meta.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def image_url(spec: SampleSpec, kind: str, stem: str, model_path: Path | None = None) -> str:
    """URL of a cached preview, served by the app's /cache route."""
    folder = model_dir(spec, model_path) / "sr" if kind == "sr" else spec.dir / kind
    return "/cache/" + (folder / f"{stem}.png").relative_to(CACHE_DIR).as_posix()


def png_uri(arr: np.ndarray) -> str:
    """float [H,W] in [0,1] -> data URI (for plotly go.Image)."""
    buf = io.BytesIO()
    Image.fromarray(np.round(np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def load_views(spec: SampleSpec, stem: str, model_path: Path, base_path: Path | None) -> dict[str, np.ndarray]:
    """All images of one stem at HR size: lr (nearest-upscaled), bicubic, sr, hr, base."""
    hr = load_png(spec.dir / "hr16" / f"{stem}.png")
    lr = load_png(spec.dir / "lr" / f"{stem}.png")
    ry, rx = hr.shape[0] // lr.shape[0], hr.shape[1] // lr.shape[1]
    views = {
        "lr": np.repeat(np.repeat(lr, ry, axis=0), rx, axis=1),
        "bicubic": load_png(spec.dir / "bicubic" / f"{stem}.png"),
        "sr": load_png(model_dir(spec, model_path) / "sr" / f"{stem}.png"),
        "hr": hr,
    }
    if base_path is not None and base_path.resolve() != model_path.resolve():
        f = model_dir(spec, base_path) / "sr" / f"{stem}.png"
        if f.is_file():
            views["base"] = load_png(f)
    return views
