"""Unit tests for inference.events.EventPublisher."""

from __future__ import annotations

from inference.events import EventPublisher, ProgressEvent, RunStatus

TOTAL = 4


def _make_event() -> ProgressEvent:
    """Return a minimal STARTED event."""
    return ProgressEvent(
        status=RunStatus.STARTED,
        processed=0,
        total=TOTAL,
        percent=0,
        current_file=None,
        lr_image=None,
        sr_image=None,
        error=None,
    )


def test_publish_delivers_event_to_all_subscribers() -> None:
    """Every subscriber receives the published event."""
    publisher = EventPublisher()
    first: list[ProgressEvent] = []
    second: list[ProgressEvent] = []
    publisher.subscribe(first.append)
    publisher.subscribe(second.append)
    event = _make_event()

    publisher.publish(event)

    assert first == [event]
    assert second == [event]


def test_failing_subscriber_does_not_block_others() -> None:
    """A subscriber that raises is skipped; later subscribers still receive the event."""
    publisher = EventPublisher()
    received: list[ProgressEvent] = []

    def failing_listener(event: ProgressEvent) -> None:
        """Always raise to simulate a broken listener."""
        raise RuntimeError("listener broke")

    publisher.subscribe(failing_listener)
    publisher.subscribe(received.append)
    event = _make_event()

    publisher.publish(event)

    assert received == [event]
