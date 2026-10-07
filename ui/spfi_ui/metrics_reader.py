"""Read training runs written by algo/train.py (or ui/dev/fake_run.py).

Run folder format:

    <RUNS_DIR>/<run_name>/
    ├── meta.json        written once at start
    ├── metrics.jsonl    one JSON object per line: {"type": "train" | "val" | "event", "step": ..., ...}
    └── samples/         sample_{k}_{lr,bicubic,hr}.png and sample_{k}_iter{step:07d}.png

No Dash code here: the TensorBoard page calls these functions and only handles display.

Quick check from the ui/ folder:
    python -m spfi_ui.metrics_reader fake_run
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from .config import RUNS_DIR

_ITER_RE = re.compile(r"^sample_(\d+)_iter(\d{7})\.png$")


def _run_dir(run: str) -> Path:
    return RUNS_DIR / run


# ---------------------------------------------------------------------------
# Runs and meta
# ---------------------------------------------------------------------------

def list_runs() -> list[str]:
    """Run names that have a meta.json or metrics.jsonl, newest first."""
    if not RUNS_DIR.is_dir():
        return []
    runs = []
    for d in RUNS_DIR.iterdir():
        files = [d / "meta.json", d / "metrics.jsonl"]
        existing = [f for f in files if f.is_file()]
        if d.is_dir() and existing:
            runs.append((max(f.stat().st_mtime for f in existing), d.name))
    return [name for _, name in sorted(runs, reverse=True)]


def read_meta(run: str) -> dict | None:
    """Contents of meta.json, or None if it is missing or not valid JSON yet."""
    try:
        return json.loads((_run_dir(run) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Incremental metrics reading
# ---------------------------------------------------------------------------

class MetricsUpdate(NamedTuple):
    records: list[dict]
    offset: int          # byte position to pass to the next call
    reset: bool          # file got shorter (run was restarted): records are from the start


def read_new_metrics(run: str, offset: int = 0) -> MetricsUpdate:
    """Return only the complete lines added to metrics.jsonl since `offset`.

    A trailing line without "\\n" is still being written by the trainer, so it is
    left for the next call. Lines that are not valid JSON are skipped.
    """
    path = _run_dir(run) / "metrics.jsonl"
    try:
        size = path.stat().st_size
    except OSError:
        return MetricsUpdate([], 0, offset > 0)

    reset = size < offset
    if reset:
        offset = 0

    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()

    end = chunk.rfind(b"\n") + 1          # 0 when there is no complete line yet
    records = []
    for line in chunk[:end].decode("utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and "type" in rec and "step" in rec:
            records.append(rec)
    return MetricsUpdate(records, offset + end, reset)


@dataclass
class RunData:
    name: str
    meta: dict | None = None
    train: list[dict] = field(default_factory=list)
    val: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    offset: int = 0

    def add(self, records: list[dict]) -> None:
        for rec in records:
            {"train": self.train, "val": self.val, "event": self.events}.get(rec["type"], []).append(rec)


_cache: dict[str, RunData] = {}


def load_run(run: str) -> RunData:
    """All data of a run, reading only new lines since the previous call.

    Kept in a server-side cache so the browser never has to send the full history back.
    """
    meta = read_meta(run)
    data = _cache.get(run)
    if data is None or (meta or {}).get("started_at") != (data.meta or {}).get("started_at"):
        data = _cache[run] = RunData(run)     # new run, or the same name was restarted
    data.meta = meta

    update = read_new_metrics(run, data.offset)
    if update.reset:
        data = _cache[run] = RunData(run, meta)
    data.add(update.records)
    data.offset = update.offset
    return data


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------

def sample_steps(run: str) -> list[int]:
    """Iterations for which every sample has an SR image, ascending.

    Only steps that already have a "val" line count: the trainer writes the PNGs first and
    the val line after them, so this never returns a step whose images are half-written.
    """
    sample_dir = _run_dir(run) / "samples"
    if not sample_dir.is_dir():
        return []
    found: dict[int, set[int]] = {}
    for p in sample_dir.iterdir():
        m = _ITER_RE.match(p.name)
        if m:
            found.setdefault(int(m.group(2)), set()).add(int(m.group(1)))
    ids = set((read_meta(run) or {}).get("sample_ids") or [k for ks in found.values() for k in ks])
    validated = {r["step"] for r in load_run(run).val}
    return sorted(step for step, ks in found.items() if ids <= ks and step in validated)


def _sample_path(run: str, k: int, which: str | int) -> Path:
    name = f"sample_{k}_iter{which:07d}.png" if isinstance(which, int) else f"sample_{k}_{which}.png"
    return _run_dir(run) / "samples" / name


def load_sample(run: str, k: int, which: str | int) -> str | None:
    """Image as a data URI for html.Img(src=...).

    `which` is "lr", "bicubic", "hr" or an iteration number. None if the file is missing.
    """
    try:
        raw = _sample_path(run, k, which).read_bytes()
    except OSError:
        return None
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def load_error_map(run: str, k: int, step: int) -> np.ndarray | None:
    """|SR - HR| in [0, 1] as a 2-D array (for a heatmap), or None if an image is missing."""
    try:
        sr = np.asarray(Image.open(_sample_path(run, k, step)).convert("L"), dtype=np.float32)
        hr = np.asarray(Image.open(_sample_path(run, k, "hr")).convert("L"), dtype=np.float32)
    except OSError:
        return None
    if sr.shape != hr.shape:
        return None
    return np.abs(sr - hr) / 255.0


# ---------------------------------------------------------------------------
# Summary for the status cards
# ---------------------------------------------------------------------------

_STATUS_BY_EVENT = {
    "start": "running",
    "resume": "running",
    "stop": "stopped",
    "early_stop": "early_stopped",
    "done": "done",
}


def summarize(data: RunData) -> dict:
    """Numbers shown in the status cards. Missing values are None."""
    meta = data.meta or {}
    all_records = data.train + data.val + data.events
    last_val = data.val[-1] if data.val else None
    prev_val = data.val[-2] if len(data.val) > 1 else None
    best_psnr = max(data.val, key=lambda r: r.get("val_psnr", float("-inf")), default=None)

    its_per_sec = None
    recent = data.train[-10:]
    if len(recent) > 1 and recent[-1]["time"] > recent[0]["time"]:
        its_per_sec = (recent[-1]["step"] - recent[0]["step"]) / (recent[-1]["time"] - recent[0]["time"])

    val_change_pct = None
    if last_val and prev_val and prev_val["val_loss"]:
        val_change_pct = 100 * (last_val["val_loss"] - prev_val["val_loss"]) / prev_val["val_loss"]

    last_time = max((r["time"] for r in all_records if "time" in r), default=None)

    if data.events:
        status = _STATUS_BY_EVENT.get(data.events[-1].get("event"), "running")
    else:
        status = "running" if all_records else "waiting"

    return {
        "status": status,
        "step": max((r["step"] for r in all_records), default=0),
        "its_per_sec": its_per_sec,
        "best_psnr": best_psnr["val_psnr"] if best_psnr else None,
        "best_psnr_step": best_psnr["step"] if best_psnr else None,
        "val_loss": last_val["val_loss"] if last_val else None,
        "val_loss_change_pct": val_change_pct,
        "no_improve": last_val["no_improve"] if last_val else 0,
        "patience": meta.get("patience"),
        "params": meta.get("params"),
        "gflops": meta.get("gflops"),
        "bicubic_psnr": meta.get("bicubic_psnr"),
        "seconds_since_update": max(0.0, time.time() - last_time) if last_time else None,
    }


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else (list_runs() or [None])[0]
    if name is None:
        print(f"No runs in {RUNS_DIR}")
        sys.exit(1)
    run_data = load_run(name)
    print(f"run: {name}   train={len(run_data.train)} val={len(run_data.val)} events={len(run_data.events)}")
    print(f"sample steps: {len(sample_steps(name))}")
    for key, value in summarize(run_data).items():
        print(f"  {key:22s} {value}")
