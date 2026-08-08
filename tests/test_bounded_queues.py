"""Tests for bounded queue capacity in EventStream and DurabilityManager."""
from __future__ import annotations

import logging

from qitos.engine.events import EventStream, EngineEvent, EngineEventType
from qitos.checkpoint.durability import DurabilityManager, DurabilityMode
from qitos.checkpoint.memory_store import InMemoryCheckpointStore
from qitos.checkpoint.store import Checkpoint, CheckpointConfig, CheckpointId


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


def test_durability_manager_async_queue_is_bounded():
    """DurabilityManager._queue has maxsize=4096 in ASYNC mode."""
    store = InMemoryCheckpointStore()
    dm = DurabilityManager(store, mode=DurabilityMode.ASYNC)
    try:
        assert dm._queue is not None
        assert dm._queue.maxsize == 4096
    finally:
        dm.shutdown()


def test_durability_manager_sync_has_no_queue():
    """DurabilityManager in SYNC mode has no queue (it is None)."""
    store = InMemoryCheckpointStore()
    dm = DurabilityManager(store, mode=DurabilityMode.SYNC)
    assert dm._queue is None


def test_durability_manager_full_queue_logs_warning(caplog):
    """Putting to a full DurabilityManager queue logs a warning instead of blocking."""
    store = InMemoryCheckpointStore()
    dm = DurabilityManager(store, mode=DurabilityMode.ASYNC)
    try:
        assert dm._queue is not None
        # Fill the queue to capacity with dummy items
        for _ in range(4096):
            dm._queue.put_nowait("dummy")
        # Now try to put a real checkpoint — should log a warning, not block
        cp = Checkpoint(
            id=CheckpointId("cp1"), thread_id="t1", step=0, state_data={"x": 1}
        )
        with caplog.at_level(logging.WARNING, logger="qitos.checkpoint.durability"):
            dm.put(CheckpointConfig(thread_id="t1"), cp, {}, {})
        assert any("queue full" in rec.message.lower() for rec in caplog.records)
    finally:
        dm.shutdown()


def test_durability_manager_flush_full_queue_logs_warning(caplog):
    """Flushing when the queue is full logs a warning for the sentinel."""
    store = InMemoryCheckpointStore()
    dm = DurabilityManager(store, mode=DurabilityMode.ASYNC)
    try:
        assert dm._queue is not None
        # Fill the queue to capacity with dummy items
        for _ in range(4096):
            dm._queue.put_nowait("dummy")
        # Flush should not block; should log a warning about the sentinel
        with caplog.at_level(logging.WARNING, logger="qitos.checkpoint.durability"):
            dm.flush()
        assert any("queue full" in rec.message.lower() for rec in caplog.records)
    finally:
        # Force shutdown even if flush couldn't send sentinel
        dm._shutdown.set()
        if dm._worker is not None:
            dm._worker.join(timeout=2.0)
