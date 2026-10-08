"""Inference core behind the Run / Continue actions.

``start_inference`` processes images one at a time, publishes progress events and
returns an ``InferenceResult``. Defined failures (``InferenceError`` subclasses) are
logged, published as a single FAILED event and returned, never raised. Anything
else is a programming error and propagates to the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from algo.model import build_spfi

from . import image_io
from .dataset import InferenceImageDataset
from .errors import (
    ImageLoadError,
    ImageSaveError,
    InferenceError,
    InputResolveError,
    ModelLoadError,
    ModelRunError,
    NoInputImagesError,
)
from .events import EventPublisher, ProgressEvent, RunStatus
from .postprocessing import postprocess_output

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_NUM_WORKERS", "LOADER_TIMEOUT_SECONDS", "InferenceResult", "start_inference"]

DEFAULT_NUM_WORKERS = 2
LOADER_TIMEOUT_SECONDS = 60
BATCH_SIZE = 1
PERCENT_SCALE = 100
EMPTY_RUN_PERCENT = 0
SPAWN_CONTEXT = "spawn"
CUDA_DEVICE = "cuda"
CPU_DEVICE = "cpu"

# Checkpoint contract: {"model_state": <state_dict>, "hparams": {"scale": int, "embed_dim": int}}
MODEL_STATE_KEY = "model_state"
HPARAMS_KEY = "hparams"
SCALE_KEY = "scale"
EMBED_DIM_KEY = "embed_dim"

Batch = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class InferenceResult:
    """Outcome of one ``start_inference`` call.

    ``next_index`` is where a later Continue should resume (``start_from``).
    """

    status: RunStatus
    next_index: int
    processed: int
    total: int
    error: InferenceError | None


def _compute_percent(processed: int, total: int) -> int:
    """Return floor(processed * 100 / total), or 0 when there is nothing to process."""
    if total == 0:
        return EMPTY_RUN_PERCENT
    return processed * PERCENT_SCALE // total


def _select_device() -> torch.device:
    """Return CUDA when available, otherwise CPU (logging a WARNING on fallback)."""
    if torch.cuda.is_available():
        return torch.device(CUDA_DEVICE)
    logger.warning("No GPU available; running inference on the CPU")
    return torch.device(CPU_DEVICE)


def _resolve_paths(input_path: Path) -> list[Path]:
    """Resolve the input to image paths; raise a defined error on failure or when empty."""
    try:
        paths = image_io.resolve_inputs(input_path)
    except Exception as exc:
        raise InputResolveError("Could not resolve input images", file_name=input_path.name) from exc
    if not paths:
        raise NoInputImagesError("No supported images found", file_name=input_path.name)
    return paths


def _require_key(container: object, key: str, file_name: str) -> object:
    """Return ``container[key]``, raising ``ModelLoadError`` if the checkpoint lacks it."""
    if not isinstance(container, dict) or key not in container:
        raise ModelLoadError(
            f"Checkpoint is missing required key '{key}'. The checkpoint must be saved in the "
            f"new format {{'{MODEL_STATE_KEY}': ..., '{HPARAMS_KEY}': "
            f"{{'{SCALE_KEY}': ..., '{EMBED_DIM_KEY}': ...}}}}",
            file_name=file_name,
        )
    return container[key]


def _load_model(model_path: Path, device: torch.device) -> nn.Module:
    """Build the SPFI model from the checkpoint's hparams and weights, on ``device``, in eval mode."""
    file_name = model_path.name
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise ModelLoadError("Could not read checkpoint", file_name=file_name) from exc

    model_state = _require_key(checkpoint, MODEL_STATE_KEY, file_name)
    hparams = _require_key(checkpoint, HPARAMS_KEY, file_name)
    scale = _require_key(hparams, SCALE_KEY, file_name)
    embed_dim = _require_key(hparams, EMBED_DIM_KEY, file_name)

    try:
        model = build_spfi(scale=scale, embed_dim=embed_dim)
        model.load_state_dict(model_state)
        model.to(device)
    except Exception as exc:
        raise ModelLoadError("Could not build model from checkpoint", file_name=file_name) from exc
    model.eval()
    return model


def _build_loader(paths: list[Path], start_from: int, num_workers: int) -> Iterable[Batch]:
    """Return a DataLoader over images ``start_from..`` (spawn + timeout only with workers)."""
    dataset = Subset(InferenceImageDataset(paths), range(start_from, len(paths)))
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
            multiprocessing_context=SPAWN_CONTEXT,
            timeout=LOADER_TIMEOUT_SECONDS,
        )
    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=num_workers)


def _fetch_next(iterator: Iterator[Batch], file_name: str) -> torch.Tensor:
    """Return the next LR batch; any fetch failure (incl. loader timeout) becomes ``ImageLoadError``."""
    try:
        lr_batch, _ = next(iterator)
    except ImageLoadError:
        raise
    except Exception as exc:
        raise ImageLoadError("Failed to fetch image from loader", file_name=file_name) from exc
    return lr_batch


def _run_model(model: nn.Module, lr_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run ``model`` on ``lr_batch`` on ``device`` without autograd; failures become ``ModelRunError``."""
    try:
        with torch.inference_mode():
            return model(lr_batch.to(device))
    except Exception as exc:
        raise ModelRunError("Model failed on image") from exc


def _save(sr_image: np.ndarray, input_path: Path, output_dir: Path) -> None:
    """Save ``sr_image`` next to its siblings in ``output_dir``; failures become ``ImageSaveError``."""
    try:
        image_io.save_image(sr_image, image_io.output_path_for(input_path, output_dir))
    except Exception as exc:
        raise ImageSaveError("Failed to save image", file_name=input_path.name) from exc


def _finish(
    publisher: EventPublisher,
    status: RunStatus,
    next_index: int,
    total: int,
    current_file: str | None = None,
    lr_image: np.ndarray | None = None,
    sr_image: np.ndarray | None = None,
    error: InferenceError | None = None,
) -> InferenceResult:
    """Publish the run's single terminal event and return the matching result."""
    publisher.publish(
        ProgressEvent(
            status=status,
            processed=next_index,
            total=total,
            percent=_compute_percent(next_index, total),
            current_file=current_file,
            lr_image=lr_image,
            sr_image=sr_image,
            error=error,
        )
    )
    return InferenceResult(status=status, next_index=next_index, processed=next_index, total=total, error=error)


def _fail(publisher: EventPublisher, error: InferenceError, next_index: int, total: int) -> InferenceResult:
    """Log ``error`` at ERROR with traceback, publish FAILED and return the result."""
    logger.error("Inference failed: %s", error, exc_info=error)
    return _finish(publisher, RunStatus.FAILED, next_index, total, error=error)


def _process_images(
    model: nn.Module,
    device: torch.device,
    iterator: Iterator[Batch],
    paths: list[Path],
    output_dir: Path,
    publisher: EventPublisher,
    should_stop: Callable[[], bool],
    start_from: int,
) -> InferenceResult:
    """Run the per-image loop from ``start_from`` and return the terminal result."""
    total = len(paths)
    last_percent = _compute_percent(start_from, total)
    publisher.publish(
        ProgressEvent(
            status=RunStatus.STARTED,
            processed=start_from,
            total=total,
            percent=last_percent,
            current_file=None,
            lr_image=None,
            sr_image=None,
            error=None,
        )
    )

    current_file: str | None = None
    lr_image: np.ndarray | None = None
    sr_image: np.ndarray | None = None
    for index in range(start_from, total):
        if should_stop():
            return _finish(publisher, RunStatus.CANCELLED, index, total, current_file, lr_image, sr_image)

        path = paths[index]
        try:
            lr_batch = _fetch_next(iterator, path.name)
            sr = postprocess_output(_run_model(model, lr_batch, device))
            _save(sr, path, output_dir)
        except InferenceError as error:
            return _fail(publisher, error, index, total)

        current_file, lr_image, sr_image = path.name, lr_batch[0, 0].numpy(), sr
        percent = _compute_percent(index + 1, total)
        if percent > last_percent:
            last_percent = percent
            publisher.publish(
                ProgressEvent(
                    status=RunStatus.RUNNING,
                    processed=index + 1,
                    total=total,
                    percent=percent,
                    current_file=current_file,
                    lr_image=lr_image,
                    sr_image=sr_image,
                    error=None,
                )
            )

    return _finish(publisher, RunStatus.COMPLETED, total, total, current_file, lr_image, sr_image)


def start_inference(
    model_path: Path,
    input_path: Path,
    output_dir: Path,
    publisher: EventPublisher,
    should_stop: Callable[[], bool],
    start_from: int = 0,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> InferenceResult:
    """Super-resolve the images at ``input_path`` into ``output_dir``, starting at ``start_from``.

    Publishes STARTED, RUNNING (once per percent increase) and exactly one terminal
    event (COMPLETED / CANCELLED / FAILED). Inputs are not validated (the UI does
    that); existing output files are overwritten.
    """
    try:
        paths = _resolve_paths(input_path)
    except InferenceError as error:
        return _fail(publisher, error, start_from, total=0)
    total = len(paths)

    device = _select_device()
    try:
        model = _load_model(model_path, device)
    except ModelLoadError as error:
        return _fail(publisher, error, start_from, total)

    iterator = iter(_build_loader(paths, start_from, num_workers))
    try:
        return _process_images(model, device, iterator, paths, output_dir, publisher, should_stop, start_from)
    finally:
        # Dropping the last reference shuts down DataLoader workers on every exit path.
        del iterator
