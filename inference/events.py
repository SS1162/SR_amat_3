"""Progress events published during an inference run."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .errors import InferenceError

logger = logging.getLogger(__name__)

__all__ = ["RunStatus", "ProgressEvent", "Listener", "EventPublisher"]


class RunStatus(Enum):
    """Lifecycle state of an inference run."""

    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class ProgressEvent:
    """Snapshot of run progress delivered to listeners.

    ``processed`` counts images before ``start_from`` as already processed.
    ``lr_image`` / ``sr_image`` are the input and output of the last processed image.
    """

    status: RunStatus
    processed: int
    total: int
    percent: int
    current_file: str | None
    lr_image: np.ndarray | None
    sr_image: np.ndarray | None
    error: InferenceError | None


Listener = Callable[[ProgressEvent], None]


class EventPublisher:
    """Delivers progress events to subscribed listeners; a failing listener never stops the run."""

    def __init__(self) -> None:
        """Create a publisher with no listeners."""
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Register ``listener`` and return a function that unsubscribes it."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            """Remove the listener; calling it more than once is harmless."""
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def publish(self, event: ProgressEvent) -> None:
        """Send ``event`` to every listener, logging (not raising) listener failures."""
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                logger.exception("Progress listener %r failed on %s event", listener, event.status.name)
