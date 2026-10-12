"""One inference job of the Inference tab, run inside Dash's background-callback process.

Kept free of Dash and torch: ``inference.runner`` (and so torch / the model) is imported
only inside ``run_job``, so the main app process never loads them.
"""

from __future__ import annotations

import base64
import io
import logging
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np
from diskcache import Cache
from PIL import Image

from inference.events import EventPublisher, ProgressEvent, RunStatus

from ..config import DASH_CACHE_DIR

logger = logging.getLogger(__name__)

__all__ = [
    "RUN_RETENTION_HOURS",
    "JobState",
    "open_cache",
    "new_job_id",
    "create_run_dir",
    "input_dir",
    "output_dir",
    "cleanup_old_runs",
    "save_uploads",
    "array_to_data_url",
    "request_stop",
    "zip_results",
    "run_job",
]

RUN_RETENTION_HOURS = 24
SECONDS_PER_HOUR = 3600
INPUT_DIR_NAME = "input"
OUTPUT_DIR_NAME = "output"
RESULT_GLOB = "*_sr.tif"
STOP_FLAG_PREFIX = "inference-stop-"
DATA_URL_SEPARATOR = ","
PNG_DATA_URL_PREFIX = "data:image/png;base64,"
PNG_FORMAT = "PNG"
UINT8_MAX = np.iinfo(np.uint8).max
# Used when an upload has no usable base name (e.g. "" or "..").
FALLBACK_UPLOAD_NAME = "upload_{index:04d}.png"
INVALID_BASE_NAMES = ("", ".", "..")


class JobState(TypedDict):
    """Outcome of one Run / Continue, kept in the browser's job store."""

    job_id: str
    run_dir: str
    status: str
    next_index: int
    processed: int
    total: int
    error: str | None


def open_cache() -> Cache:
    """Open the diskcache shared by the DiskcacheManager, the Stop button and the job."""
    return Cache(str(DASH_CACHE_DIR))


def new_job_id() -> str:
    """Return a fresh random job id (uuid4 hex)."""
    return uuid.uuid4().hex


def create_run_dir(runs_root: Path, job_id: str) -> Path:
    """Create ``<runs_root>/<job_id>/{input,output}`` and return the job folder."""
    run_dir = runs_root / job_id
    (run_dir / INPUT_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (run_dir / OUTPUT_DIR_NAME).mkdir(parents=True, exist_ok=True)
    return run_dir


def input_dir(run_dir: Path) -> Path:
    """Return the folder holding the job's saved uploads."""
    return run_dir / INPUT_DIR_NAME


def output_dir(run_dir: Path) -> Path:
    """Return the folder holding the job's super-resolved images."""
    return run_dir / OUTPUT_DIR_NAME


def cleanup_old_runs(runs_root: Path, keep: Path, now: float) -> None:
    """Delete job folders under ``runs_root`` last modified over RUN_RETENTION_HOURS ago, never ``keep``."""
    if not runs_root.is_dir():
        return
    max_age_seconds = RUN_RETENTION_HOURS * SECONDS_PER_HOUR
    keep_resolved = keep.resolve()
    for folder in runs_root.iterdir():
        if not folder.is_dir() or folder.resolve() == keep_resolved:
            continue
        if now - folder.stat().st_mtime <= max_age_seconds:
            continue
        try:
            shutil.rmtree(folder)
        except OSError:
            logger.warning("Could not delete old inference run folder %s", folder, exc_info=True)


def _safe_upload_name(filename: str | None, index: int) -> str:
    """Return only the base name of ``filename`` (no folders, so no path traversal)."""
    name = Path(filename or "").name
    if name in INVALID_BASE_NAMES:
        return FALLBACK_UPLOAD_NAME.format(index=index)
    return name


def save_uploads(contents: list[str], filenames: list[str | None], target_dir: Path) -> list[Path]:
    """Decode dcc.Upload data URLs and write each file into ``target_dir`` under its base name."""
    saved: list[Path] = []
    for index, (content, filename) in enumerate(zip(contents, filenames)):
        _, encoded = content.split(DATA_URL_SEPARATOR, 1)
        path = target_dir / _safe_upload_name(filename, index)
        path.write_bytes(base64.b64decode(encoded))
        saved.append(path)
    return saved


def array_to_data_url(image: np.ndarray) -> str:
    """Return a PNG data URL of a float32 [0, 1] image, as 8-bit (display only)."""
    scaled = np.rint(np.clip(image, 0.0, 1.0) * UINT8_MAX).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(scaled).save(buffer, format=PNG_FORMAT)
    return PNG_DATA_URL_PREFIX + base64.b64encode(buffer.getvalue()).decode("ascii")


def _stop_flag_key(job_id: str) -> str:
    """Return the cache key of ``job_id``'s stop flag."""
    return f"{STOP_FLAG_PREFIX}{job_id}"


def request_stop(cache: Cache, job_id: str) -> None:
    """Ask the running job ``job_id`` to stop after its current image."""
    cache.set(_stop_flag_key(job_id), True)


def _clear_stop(cache: Cache, job_id: str) -> None:
    """Remove ``job_id``'s stop flag."""
    cache.delete(_stop_flag_key(job_id))


def _is_stop_requested(cache: Cache, job_id: str) -> bool:
    """Return True when Stop was pressed for ``job_id``."""
    return bool(cache.get(_stop_flag_key(job_id), False))


def zip_results(run_dir: Path) -> bytes:
    """Return a zip archive of every ``*_sr.tif`` in the job's output folder."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir(run_dir).glob(RESULT_GLOB)):
            archive.write(path, arcname=path.name)
    return buffer.getvalue()


def _job_state(
    job_id: str, run_dir: Path, status: RunStatus, next_index: int, processed: int, total: int, error: str | None
) -> JobState:
    """Build the JobState stored in the browser."""
    return JobState(
        job_id=job_id,
        run_dir=str(run_dir),
        status=status.value,
        next_index=next_index,
        processed=processed,
        total=total,
        error=error,
    )


def run_job(
    model_path: Path,
    run_dir: Path,
    job_id: str,
    start_from: int,
    cache: Cache,
    listener: Callable[[ProgressEvent], None],
) -> JobState:
    """Run ``start_inference`` on the job folder and return its final state.

    Any unexpected exception is logged and returned as a FAILED state, so the UI
    always gets a terminal state (defined failures are already returned by the runner).
    """
    # Imported here so torch and the model load only in the background-callback process.
    from inference import runner

    publisher = EventPublisher()
    publisher.subscribe(listener)
    _clear_stop(cache, job_id)
    try:
        result = runner.start_inference(
            model_path=model_path,
            input_path=input_dir(run_dir),
            output_dir=output_dir(run_dir),
            publisher=publisher,
            should_stop=lambda: _is_stop_requested(cache, job_id),
            start_from=start_from,
        )
    except Exception as exc:
        logger.error("Unexpected error in inference job %s", job_id, exc_info=True)
        message = f"Unexpected error: {type(exc).__name__}: {exc}"
        return _job_state(job_id, run_dir, RunStatus.FAILED, start_from, start_from, 0, message)
    error = str(result.error) if result.error is not None else None
    return _job_state(job_id, run_dir, result.status, result.next_index, result.processed, result.total, error)
