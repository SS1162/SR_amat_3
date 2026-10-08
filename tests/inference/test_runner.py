"""Unit tests for inference.runner.start_inference (CPU, num_workers=0, fake model and I/O)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pytest
import torch
from torch import nn

from algo import preprocessing
from inference import image_io, runner
from inference.errors import (
    ImageLoadError,
    InputResolveError,
    ModelLoadError,
    NoInputImagesError,
)
from inference.events import EventPublisher, ProgressEvent, RunStatus
from inference.runner import InferenceResult, start_inference

LR_SIZE = 4
SR_FACTOR = 2
PIXEL_MAX = 255.0
FAKE_SCALE = 2
FAKE_EMBED_DIM = 8
TERMINAL_STATUSES = {RunStatus.COMPLETED, RunStatus.CANCELLED, RunStatus.FAILED}


class _FakeSR(nn.Module):
    """Parameter-free stand-in for SPFI: nearest-neighbour x2 upsampling."""

    def __init__(self, scale: int, embed_dim: int) -> None:
        """Accept the build_spfi keyword arguments and build the upsampler."""
        super().__init__()
        self.up = nn.Upsample(scale_factor=SR_FACTOR, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample ``x`` of shape [B, 1, H, W] to [B, 1, 2H, 2W]."""
        return self.up(x)


@dataclass
class _FakeIO:
    """State of the monkeypatched image I/O: input paths and recorded saves."""

    paths: list[Path]
    saved: list[Path] = field(default_factory=list)
    failing_index: int | None = None


@dataclass
class _Run:
    """Outcome of one test run: the result and every published event."""

    result: InferenceResult
    events: list[ProgressEvent]


def _pixel_value(index: int) -> int:
    """Return the constant uint8 pixel value of fake image ``index``."""
    return index % int(PIXEL_MAX + 1)


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_count: int, failing_index: int | None = None
) -> _FakeIO:
    """Replace image I/O, normalization and the model factory with in-memory fakes."""
    paths = [tmp_path / "in" / f"img_{i:03d}.png" for i in range(image_count)]
    fake = _FakeIO(paths=paths, failing_index=failing_index)
    index_of = {path: i for i, path in enumerate(paths)}

    def fake_resolve_inputs(input_path: Path) -> list[Path]:
        """Return the fake input paths."""
        return list(fake.paths)

    def fake_load_image(path: Path) -> np.ndarray:
        """Return a constant 4x4 uint8 image, or raise for the configured failing index."""
        index = index_of[path]
        if index == fake.failing_index:
            raise OSError("corrupt file")
        return np.full((LR_SIZE, LR_SIZE), _pixel_value(index), dtype=np.uint8)

    def fake_save_image(image: np.ndarray, path: Path) -> None:
        """Record the output path instead of writing a file."""
        fake.saved.append(path)

    def fake_normalize_image(image: np.ndarray) -> np.ndarray:
        """Scale uint8 to float32 in [0, 1]."""
        return image.astype(np.float32) / PIXEL_MAX

    monkeypatch.setattr(image_io, "resolve_inputs", fake_resolve_inputs)
    monkeypatch.setattr(image_io, "load_image", fake_load_image)
    monkeypatch.setattr(image_io, "save_image", fake_save_image)
    monkeypatch.setattr(preprocessing, "normalize_image", fake_normalize_image)
    monkeypatch.setattr(runner, "build_spfi", _FakeSR)
    return fake


def _write_checkpoint(tmp_path: Path, with_hparams: bool = True) -> Path:
    """Save a checkpoint for the fake model, optionally without the required hparams."""
    checkpoint: dict[str, object] = {"model_state": {}}
    if with_hparams:
        checkpoint["hparams"] = {"scale": FAKE_SCALE, "embed_dim": FAKE_EMBED_DIM}
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    return path


def _run(
    tmp_path: Path,
    model_path: Path | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    start_from: int = 0,
) -> _Run:
    """Run start_inference on CPU with num_workers=0 and collect all events."""
    publisher = EventPublisher()
    events: list[ProgressEvent] = []
    publisher.subscribe(events.append)
    result = start_inference(
        model_path=model_path if model_path is not None else _write_checkpoint(tmp_path),
        input_path=tmp_path / "in",
        output_dir=tmp_path / "out",
        publisher=publisher,
        should_stop=should_stop,
        start_from=start_from,
        num_workers=0,
    )
    return _Run(result=result, events=events)


def _terminal_events(events: list[ProgressEvent]) -> list[ProgressEvent]:
    """Return the terminal (COMPLETED / CANCELLED / FAILED) events."""
    return [event for event in events if event.status in TERMINAL_STATUSES]


def test_happy_path_completes_and_saves_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """All images are saved, the run COMPLETES with next_index == total and one terminal event."""
    image_count = 3
    fake = _install_fakes(monkeypatch, tmp_path, image_count)

    run = _run(tmp_path)

    assert run.result.status is RunStatus.COMPLETED
    assert run.result.next_index == image_count
    assert run.result.total == image_count
    assert fake.saved == [image_io.output_path_for(path, tmp_path / "out") for path in fake.paths]
    terminal = _terminal_events(run.events)
    assert len(terminal) == 1
    assert terminal[0].status is RunStatus.COMPLETED
    assert terminal[0].sr_image is not None


def test_running_events_once_per_percent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """200 images produce exactly 100 RUNNING events, each with the last processed pair."""
    image_count = 200
    expected_running = 100
    images_per_percent = image_count // expected_running
    _install_fakes(monkeypatch, tmp_path, image_count)

    run = _run(tmp_path)

    running = [event for event in run.events if event.status is RunStatus.RUNNING]
    assert len(running) == expected_running
    for step, event in enumerate(running, start=1):
        last_index = step * images_per_percent - 1
        expected_lr = np.full((LR_SIZE, LR_SIZE), _pixel_value(last_index) / PIXEL_MAX, dtype=np.float32)
        assert event.percent == step
        assert event.current_file == f"img_{last_index:03d}.png"
        assert event.lr_image is not None and event.sr_image is not None
        np.testing.assert_allclose(event.lr_image, expected_lr)
        assert event.sr_image.shape == (LR_SIZE * SR_FACTOR, LR_SIZE * SR_FACTOR)
        np.testing.assert_allclose(event.sr_image, expected_lr.repeat(SR_FACTOR, 0).repeat(SR_FACTOR, 1))


def test_start_from_processes_only_remaining(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Only images from start_from on are saved; STARTED reports start_from and its percent."""
    image_count = 10
    start_from = 4
    fake = _install_fakes(monkeypatch, tmp_path, image_count)

    run = _run(tmp_path, start_from=start_from)

    started = run.events[0]
    assert started.status is RunStatus.STARTED
    assert started.processed == start_from
    assert started.percent == start_from * runner.PERCENT_SCALE // image_count
    assert fake.saved == [image_io.output_path_for(path, tmp_path / "out") for path in fake.paths[start_from:]]
    assert run.result.status is RunStatus.COMPLETED
    assert run.result.next_index == image_count


def test_stop_after_two_images_cancels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Stopping after 2 saves CANCELS with next_index == 2 and no further saves."""
    image_count = 5
    stop_after = 2
    fake = _install_fakes(monkeypatch, tmp_path, image_count)

    run = _run(tmp_path, should_stop=lambda: len(fake.saved) >= stop_after)

    assert run.result.status is RunStatus.CANCELLED
    assert run.result.next_index == stop_after
    assert len(fake.saved) == stop_after
    terminal = _terminal_events(run.events)
    assert len(terminal) == 1
    assert terminal[0].status is RunStatus.CANCELLED


def test_image_load_failure_fails_at_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A load failure at index k FAILS with ImageLoadError and next_index == k."""
    image_count = 5
    failing_index = 3
    fake = _install_fakes(monkeypatch, tmp_path, image_count, failing_index=failing_index)

    run = _run(tmp_path)

    assert run.result.status is RunStatus.FAILED
    assert isinstance(run.result.error, ImageLoadError)
    assert run.result.next_index == failing_index
    assert len(fake.saved) == failing_index
    assert len(_terminal_events(run.events)) == 1


def test_model_load_failure_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A checkpoint without "hparams" FAILS with ModelLoadError."""
    _install_fakes(monkeypatch, tmp_path, image_count=2)

    run = _run(tmp_path, model_path=_write_checkpoint(tmp_path, with_hparams=False))

    assert run.result.status is RunStatus.FAILED
    assert isinstance(run.result.error, ModelLoadError)
    assert len(_terminal_events(run.events)) == 1


def test_empty_input_folder_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An input with no images FAILS with NoInputImagesError."""
    _install_fakes(monkeypatch, tmp_path, image_count=0)

    run = _run(tmp_path)

    assert run.result.status is RunStatus.FAILED
    assert isinstance(run.result.error, NoInputImagesError)
    assert len(_terminal_events(run.events)) == 1


def test_loader_fetch_failure_fails_at_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A loader error (e.g. timeout) at index k FAILS with ImageLoadError and next_index == k."""
    image_count = 5
    failing_index = 2
    _install_fakes(monkeypatch, tmp_path, image_count)

    def failing_loader(paths: list[Path], start_from: int, num_workers: int) -> Iterator[runner.Batch]:
        """Yield valid batches, then raise RuntimeError at the failing index like a timeout."""
        for index in range(start_from, len(paths)):
            if index == failing_index:
                raise RuntimeError("DataLoader timed out")
            yield torch.zeros(1, 1, LR_SIZE, LR_SIZE), torch.tensor([index])

    monkeypatch.setattr(runner, "_build_loader", failing_loader)

    run = _run(tmp_path)

    assert run.result.status is RunStatus.FAILED
    assert isinstance(run.result.error, ImageLoadError)
    assert run.result.next_index == failing_index
    assert len(_terminal_events(run.events)) == 1


def test_resolve_inputs_failure_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A resolve_inputs error FAILS with InputResolveError, total 0, next_index == start_from."""
    start_from = 3
    _install_fakes(monkeypatch, tmp_path, image_count=5)

    def denied_resolve_inputs(input_path: Path) -> list[Path]:
        """Raise as if the input folder were not readable."""
        raise PermissionError("access denied")

    monkeypatch.setattr(image_io, "resolve_inputs", denied_resolve_inputs)

    run = _run(tmp_path, start_from=start_from)

    assert run.result.status is RunStatus.FAILED
    assert isinstance(run.result.error, InputResolveError)
    assert run.result.total == 0
    assert run.result.next_index == start_from
    assert len(_terminal_events(run.events)) == 1
