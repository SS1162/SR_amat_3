"""One evaluation job at a time, in a thread of the app process (a local dev tool, single user)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from .evaluate import SampleSpec, evaluate_model, is_evaluated, model_label

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_stop = threading.Event()
_thread: threading.Thread | None = None
_state: dict = {"status": "idle", "message": "", "done": 0, "total": 0, "finished": 0}


def state() -> dict:
    with _lock:
        return dict(_state)


def _set(**kw) -> None:
    with _lock:
        _state.update(kw)


def is_running() -> bool:
    return _thread is not None and _thread.is_alive()


def start(spec: SampleSpec, model_paths: list[Path]) -> bool:
    """Evaluate every model that has no cache yet. False if a job is already running."""
    global _thread
    if is_running():
        return False
    todo = list(dict.fromkeys(p for p in model_paths if not is_evaluated(spec, p)))
    _stop.clear()
    _set(status="running", message="Starting…", done=0, total=0)
    _thread = threading.Thread(target=_run, args=(spec, todo), daemon=True)
    _thread.start()
    return True


def stop() -> None:
    _stop.set()


def _run(spec: SampleSpec, todo: list[Path]) -> None:
    try:
        for i, path in enumerate(todo, start=1):
            name = model_label(path)

            def progress(done: int, total: int, stem: str, name=name, i=i) -> None:
                _set(done=done, total=total, message=f"[{i}/{len(todo)}] {name}: {stem}")

            evaluate_model(spec, path, progress=progress, stop=_stop)
        _set(status="done", message="Done")
    except InterruptedError:
        _set(status="stopped", message="Stopped (finished images are not kept; run again to resume)")
    except Exception as exc:   # surface any failure in the UI instead of a dead thread
        logger.exception("evaluation failed")
        _set(status="failed", message=f"{type(exc).__name__}: {exc}")
    finally:
        with _lock:
            _state["finished"] += 1
