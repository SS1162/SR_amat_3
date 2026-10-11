"""Unit tests for the Inference tab job helpers (spfi_ui.inference.job)."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Iterator

import pytest
from diskcache import Cache

from inference import runner
from inference.events import ProgressEvent, RunStatus
from spfi_ui.inference import job

FILE_BYTES = b"fake image bytes"
SECONDS_PER_HOUR = 3600
STALE_AGE_SECONDS = (job.RUN_RETENTION_HOURS + 1) * SECONDS_PER_HOUR


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[Cache]:
    """A throwaway diskcache, closed after the test."""
    with Cache(str(tmp_path / "cache")) as test_cache:
        yield test_cache


def _make_old(folder: Path, now: float) -> None:
    """Set ``folder``'s modification time to longer ago than the retention period."""
    old = now - STALE_AGE_SECONDS
    os.utime(folder, (old, old))


def test_upload_with_path_component_is_saved_by_base_name(tmp_path: Path) -> None:
    """A filename containing folders is written as its base name inside the input folder only."""
    input_dir = tmp_path / "run" / "input"
    input_dir.mkdir(parents=True)
    data_url = "data:image/png;base64," + base64.b64encode(FILE_BYTES).decode("ascii")

    saved = job.save_uploads([data_url], ["../../escape/evil.png"], input_dir)

    assert saved == [input_dir / "evil.png"]
    assert (input_dir / "evil.png").read_bytes() == FILE_BYTES
    assert not (tmp_path / "escape").exists()


def test_cleanup_deletes_old_runs_but_keeps_current(tmp_path: Path) -> None:
    """A folder older than the retention period is deleted; the current job's folder is kept even if old."""
    now = time.time()
    old_run = tmp_path / "old"
    current_run = tmp_path / "current"
    old_run.mkdir()
    current_run.mkdir()
    _make_old(old_run, now)
    _make_old(current_run, now)

    job.cleanup_old_runs(tmp_path, keep=current_run, now=now)

    assert not old_run.exists()
    assert current_run.exists()


def test_unexpected_exception_becomes_failed_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cache: Cache
) -> None:
    """An exception that start_inference does not handle is returned as a FAILED job state."""
    start_from = 2

    def crashing_start_inference(**kwargs: object) -> runner.InferenceResult:
        """Raise like a programming error inside the runner."""
        raise RuntimeError("boom")

    def ignore_event(event: ProgressEvent) -> None:
        """Discard progress events."""

    monkeypatch.setattr(runner, "start_inference", crashing_start_inference)

    state = job.run_job(tmp_path / "model.pt", tmp_path / "run", "job123", start_from, cache, ignore_event)

    assert state["status"] == RunStatus.FAILED.value
    assert state["job_id"] == "job123"
    assert state["next_index"] == start_from
    assert state["error"] is not None and "RuntimeError" in state["error"] and "boom" in state["error"]
