"""Tests for bounded Engine event queues."""
from __future__ import annotations

from qitos.engine.events import EventStream, EngineEvent, EngineEventType


def test_eventstream_main_queue_has_maxsize():
    """EventStream._queue has maxsize=4096."""
    es = EventStream()
    assert es._queue.maxsize == 4096


def test_eventstream_subscriber_queue_has_maxsize():
    """Subscriber queues have maxsize=1024."""
    es = EventStream()
    sub = es.subscribe()
    assert sub.maxsize == 1024


def test_eventstream_emit_does_not_raise_when_queue_full():
    """Emitting to a full queue records the bounded drop."""
    es = EventStream()
    # Fill the queue to capacity
    event = EngineEvent(event_type=EngineEventType.RUN_START, payload={})
    for _ in range(4096):
        es._queue.put_nowait(event)
    # This should not raise
    es.emit(event)
    assert es.dropped_event_count == 1


def test_eventstream_close_does_not_raise_when_queue_full():
    """Closing a full queue keeps a terminal sentinel by dropping one event."""
    es = EventStream()
    event = EngineEvent(event_type=EngineEventType.RUN_START, payload={})
    for _ in range(4096):
        es._queue.put_nowait(event)

    es.close()

    queued = [es._queue.get_nowait() for _ in range(es._queue.qsize())]
    assert len(queued) == 4096
    assert queued[-1] is None
    assert es.dropped_event_count == 1


def test_eventstream_full_queue_preserves_one_run_end_before_close():
    es = EventStream()
    ordinary = EngineEvent(event_type=EngineEventType.STEP_START)
    for _ in range(4096):
        es._queue.put_nowait(ordinary)
    run_end = EngineEvent(event_type=EngineEventType.RUN_END)

    es.emit(run_end)
    es.emit(EngineEvent(event_type=EngineEventType.RUN_END))
    es.emit(EngineEvent(event_type=EngineEventType.STEP_END))
    es.close()

    queued = [es._queue.get_nowait() for _ in range(es._queue.qsize())]
    assert [event.event_type for event in queued if event is not None].count(
        EngineEventType.RUN_END
    ) == 1
    assert queued[-2:] == [run_end, None]
    assert run_end.payload["dropped_events"] == 2
    assert es.dropped_event_count == 2


def test_eventstream_subscribe_after_close_is_already_terminal():
    es = EventStream()

    es.close()
    subscriber = es.subscribe()

    assert subscriber.get_nowait() is None
