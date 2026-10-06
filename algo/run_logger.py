"""Run folder writer for the UI's TensorBoard page.

Layout (read by ui/spfi_ui/metrics_reader.py; do not rename fields or files)::

    <run_dir>/
    ├── meta.json        written once, before the first iteration
    ├── metrics.jsonl    one JSON object per line, append-only, flushed after every line
    └── samples/         sample_{k}_{lr,bicubic,hr}.png once, sample_{k}_iter{step:07d}.png per validation

metrics.jsonl lines::

    {"type": "train", "step", "time", "loss", "charbonnier", "ssim_loss", "lr"}   every log_every steps
    {"type": "val",   "step", "time", "val_loss", "val_psnr", "val_ssim", "is_best", "no_improve"}
    {"type": "event", "step", "time", "event": "start" | "resume" | "stop" | "early_stop" | "done"}

Logging is a side effect only: nothing here touches the training RNG or the model's training state.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_to_png(t: torch.Tensor, path: Path) -> None:
    """Save a [1,H,W] (or [H,W]) tensor in [0,1] as an 8-bit grayscale PNG."""
    arr = (t.detach().clamp(0, 1).squeeze().cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)   # 2-D uint8 is saved as "L"; the mode= argument is deprecated


class RunLogger:
    def __init__(self, run_dir: Path, log_every: int = 10) -> None:
        self.run_dir = Path(run_dir)
        self.samples_dir = self.run_dir / "samples"
        metrics_path = self.run_dir / "metrics.jsonl"
        if metrics_path.exists():
            raise FileExistsError(f"{self.run_dir} already holds a run; choose another run directory")
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self._metrics = open(metrics_path, "a", encoding="utf-8", newline="\n")
        self._reset_train_sums()

    # ------------------------------------------------------------------
    def write_meta(self, meta: dict) -> None:
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def save_png(self, name: str, t: torch.Tensor) -> None:
        tensor_to_png(t, self.samples_dir / name)

    # ------------------------------------------------------------------
    def _reset_train_sums(self) -> None:
        self._n = 0
        self._sums: dict[str, torch.Tensor | float] = {"loss": 0.0, "charbonnier": 0.0, "ssim_loss": 0.0}

    def log_train(self, step: int, loss: torch.Tensor, charbonnier: torch.Tensor,
                  ssim_loss: torch.Tensor, lr: float) -> None:
        """Accumulate one step; every ``log_every`` steps write the average since the last train line.

        Values are summed as detached tensors so the GPU only syncs when a line is written.
        """
        for key, value in (("loss", loss), ("charbonnier", charbonnier), ("ssim_loss", ssim_loss)):
            self._sums[key] = self._sums[key] + value.detach()
        self._n += 1
        if step % self.log_every == 0:
            self._write({"type": "train", "step": step, "time": time.time(),
                         **{k: float(v) / self._n for k, v in self._sums.items()}, "lr": lr})
            self._reset_train_sums()

    def log_val(self, step: int, val_loss: float, val_psnr: float, val_ssim: float,
                is_best: bool, no_improve: int) -> None:
        self._write({"type": "val", "step": step, "time": time.time(), "val_loss": val_loss,
                     "val_psnr": val_psnr, "val_ssim": val_ssim, "is_best": is_best, "no_improve": no_improve})

    def log_event(self, step: int, event: str) -> None:
        self._write({"type": "event", "step": step, "time": time.time(), "event": event})

    def _write(self, record: dict) -> None:
        self._metrics.write(json.dumps(record) + "\n")
        self._metrics.flush()

    def close(self) -> None:
        self._metrics.close()
