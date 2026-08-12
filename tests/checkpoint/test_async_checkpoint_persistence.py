"""Behavior tests for the asynchronous checkpoint persistence owner."""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import threading
from typing import Callable

import pytest

from qitos.checkpoint import (
    Checkpoint,
    CheckpointConfig,
    CheckpointId,
    CheckpointStore,
    InMemoryCheckpointStore,
    SqliteCheckpointStore,
    fork_checkpoint,
    list_fork_history,
)
from qitos.core.history import HistoryMessage, HistorySnapshot
from qitos.trace.events import TraceEvent, TraceStep
from qitos.trace.writer import TraceWriter


def _checkpoint(checkpoint_id: str, *, step: int = 1) -> Checkpoint:
    history = HistorySnapshot.from_messages(
        [
            HistoryMessage(role="user", step_id=step, content="inspect target"),
            HistoryMessage(
                role="assistant",
                step_id=step,
                content=None,
                reasoning_content="check the exposed service",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "{}"},
                    }
                ],
                native_items=[
                    {
                        "type": "reasoning",
                        "id": "reasoning-1",
                        "encrypted_content": "opaque-state",
                    }
                ],
            ),
            HistoryMessage(
                role="tool",
                step_id=step,
                content="port 443 is open",
                tool_call_id="call-1",
                name="shell",
            ),
        ],
        source_revision=7,
    )
    return Checkpoint(
        id=CheckpointId(checkpoint_id),
        thread_id="run-1",
        step=step,
        state_data={"task": "inspect", "current_step": step},
        task_text="inspect",
        task_data={"id": "task-1", "objective": "inspect"},
        history=history,
    )


@pytest.fixture(params=["memory", "sqlite"])
def store_factory(
    request: pytest.FixtureRequest, tmp_path
) -> Callable[[], CheckpointStore]:
    if request.param == "memory":
        return InMemoryCheckpointStore
    return lambda: SqliteCheckpointStore(str(tmp_path / "checkpoints.db"))


@pytest.mark.asyncio
async def test_store_round_trips_recoverable_model_history(store_factory) -> None:
    store = store_factory()
    checkpoint = _checkpoint("cp-1")
    try:
        persisted = await store.put(
            CheckpointConfig(thread_id="run-1"),
            checkpoint,
            {"source": "loop", "step": 1, "run_id": "run-1"},
        )
        loaded = await store.get_tuple(persisted)

        assert loaded is not None
        assert loaded.checkpoint == checkpoint
        assert loaded.checkpoint.history is not None
        native = loaded.checkpoint.history.messages[1].native_items[0]
        assert native["encrypted_content"] == "opaque-state"
        assert loaded.checkpoint.history.messages[2].tool_call_id == "call-1"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_store_lists_and_forks_lineage(store_factory) -> None:
    store = store_factory()
    first = _checkpoint("cp-1", step=1)
    second = _checkpoint("cp-2", step=2)
    second.parent_id = first.id
    try:
        await store.put(CheckpointConfig(thread_id="run-1"), first, {"step": 1})
        await store.put(CheckpointConfig(thread_id="run-1"), second, {"step": 2})

        listed = await store.list(CheckpointConfig(thread_id="run-1"))
        assert [item.checkpoint.id for item in listed] == ["cp-2", "cp-1"]

        forked = await fork_checkpoint(
            store,
            CheckpointConfig(thread_id="run-1", checkpoint_id=second.id),
            new_thread_id="run-1-child",
        )
        branch = await store.get(forked)
        assert branch is not None
        assert branch.parent_id == second.id
        assert branch.parent_thread_id == "run-1"
        assert branch.history == second.history

        lineage = await list_fork_history(store, forked)
        assert [item.checkpoint.id for item in lineage][1:] == ["cp-2", "cp-1"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_cancellation_waits_for_active_write(tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingStore(SqliteCheckpointStore):
        def _run_sync(self, operation):
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release SQLite operation")
            return super()._run_sync(operation)

    store = BlockingStore(str(tmp_path / "cancel.db"))
    write = asyncio.create_task(
        store.put(
            CheckpointConfig(thread_id="run-1"),
            _checkpoint("cp-1"),
            {"source": "loop"},
        )
    )
    assert await asyncio.to_thread(started.wait, 5)
    write.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await write

    loaded = await store.get(CheckpointConfig(thread_id="run-1"))
    assert loaded is not None
    await store.close()


@pytest.mark.asyncio
async def test_store_rejects_mismatched_thread_identity() -> None:
    store = InMemoryCheckpointStore()
    with pytest.raises(ValueError, match="thread_id"):
        await store.put(
            CheckpointConfig(thread_id="other-run"),
            _checkpoint("cp-1"),
            {},
        )


def test_history_snapshot_rejects_orphans_and_detaches_nested_values() -> None:
    call = HistoryMessage(
        role="assistant",
        step_id=1,
        tool_calls=[{"id": "call-1", "function": {"arguments": "{}"}}],
    )
    result = HistoryMessage(
        role="tool",
        step_id=1,
        content={"nested": ["complete"]},
        tool_call_id="call-1",
    )
    orphan = HistoryMessage(
        role="tool", step_id=2, content="orphan", tool_call_id="missing"
    )

    snapshot = HistorySnapshot.from_messages([call, result])
    assert snapshot.messages == (call, result)
    result.content["nested"].append("live mutation")
    assert snapshot.messages[1].content == {"nested": ["complete"]}

    assert HistorySnapshot.from_messages([orphan, call, result]).messages == ()
    assert HistorySnapshot.from_messages([call, orphan, result]).messages == ()


@pytest.mark.asyncio
async def test_store_returns_independent_values(store_factory) -> None:
    store = store_factory()
    try:
        config = await store.put(
            CheckpointConfig(thread_id="run-1"),
            _checkpoint("cp-owned"),
            {"source": "loop", "step": 1},
        )
        first = await store.get_tuple(config)
        assert first is not None
        assert first.checkpoint.history is not None
        first.checkpoint.state_data["task"] = "mutated"
        first.checkpoint.history.messages[1].native_items[0][
            "encrypted_content"
        ] = "mutated"
        first.metadata["source"] = "mutated"

        second = await store.get_tuple(config)
        assert second is not None
        assert second.checkpoint.state_data["task"] == "inspect"
        assert second.checkpoint.history is not None
        assert (
            second.checkpoint.history.messages[1].native_items[0][
                "encrypted_content"
            ]
            == "opaque-state"
        )
        assert second.metadata["source"] == "loop"
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_state",
    [
        {"value": math.nan},
        {"value": {"set"}},
    ],
)
async def test_stores_reject_non_json_checkpoint_values(
    store_factory, invalid_state
) -> None:
    store = store_factory()
    checkpoint = _checkpoint("cp-invalid")
    checkpoint.state_data = invalid_state
    try:
        with pytest.raises(ValueError):
            await store.put(
                CheckpointConfig(thread_id="run-1"), checkpoint, {"step": 1}
            )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_stores_reject_invalid_metadata(store_factory) -> None:
    store = store_factory()
    try:
        with pytest.raises(ValueError, match="unknown fields"):
            await store.put(
                CheckpointConfig(thread_id="run-1"),
                _checkpoint("cp-unknown-meta"),
                {"unknown": "field"},  # type: ignore[typeddict-unknown-key]
            )
        with pytest.raises(ValueError, match="JSON serializable"):
            await store.put(
                CheckpointConfig(thread_id="run-1"),
                _checkpoint("cp-invalid-meta"),
                {"parents": {"run-1": {"not-json"}}},  # type: ignore[typeddict-item]
            )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_store_rejects_cross_event_loop_use(tmp_path) -> None:
    store = SqliteCheckpointStore(str(tmp_path / "loop-owned.db"))
    try:
        await store.get(CheckpointConfig(thread_id="run-1"))

        def use_from_another_loop() -> None:
            asyncio.run(store.get(CheckpointConfig(thread_id="run-1")))

        with pytest.raises(RuntimeError, match="across event loops"):
            await asyncio.to_thread(use_from_another_loop)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_store_migrates_v2_rows(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE checkpoints (
            checkpoint_id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            state_data TEXT NOT NULL,
            state_versions TEXT NOT NULL DEFAULT '{}',
            versions_seen TEXT NOT NULL DEFAULT '{}',
            parent_id TEXT,
            created_at TEXT NOT NULL,
            schema_version TEXT NOT NULL DEFAULT 'v2'
        );
        CREATE TABLE checkpoint_metadata (
            checkpoint_id TEXT PRIMARY KEY REFERENCES checkpoints(checkpoint_id),
            source TEXT,
            step_int INTEGER,
            parents TEXT DEFAULT '{}',
            run_id TEXT
        );
        INSERT INTO checkpoints (
            checkpoint_id, thread_id, step, state_data, created_at
        ) VALUES ('cp-legacy', 'run-1', 1, '{"task": "legacy"}', 'now');
        """
    )
    conn.commit()
    conn.close()

    store = SqliteCheckpointStore(str(db_path))
    try:
        config = CheckpointConfig(
            thread_id="run-1", checkpoint_id=CheckpointId("cp-legacy")
        )
        legacy = await store.get(config)
        assert legacy is not None
        assert legacy.state_data == {"task": "legacy"}
        assert legacy.task_text == ""
        assert legacy.history is None

        replacement = _checkpoint("cp-legacy", step=2)
        await store.put(
            CheckpointConfig(thread_id="run-1"),
            replacement,
            {"source": "update", "step": 2},
        )
        updated = await store.get(config)
        assert updated is not None
        assert updated.step == 2
        assert updated.history == replacement.history
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_rolls_back_checkpoint_when_metadata_write_fails(tmp_path) -> None:
    db_path = tmp_path / "atomic.db"
    store = SqliteCheckpointStore(str(db_path))
    config = CheckpointConfig(thread_id="run-1")
    try:
        assert await store.get(config) is None
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TRIGGER reject_checkpoint_metadata
            BEFORE INSERT ON checkpoint_metadata
            BEGIN
                SELECT RAISE(ABORT, 'metadata rejected');
            END
            """
        )
        conn.commit()
        conn.close()

        with pytest.raises(sqlite3.IntegrityError, match="metadata rejected"):
            await store.put(config, _checkpoint("cp-atomic"), {"step": 1})
        assert await store.get(config) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_store_replaces_checkpoint_without_duplicate_history(store_factory) -> None:
    store = store_factory()
    try:
        first = _checkpoint("cp-1", step=1)
        await store.put(
            CheckpointConfig(thread_id="run-1"),
            first,
            {"source": "loop", "step": 1},
        )
        replacement = _checkpoint("cp-1", step=2)
        replacement.state_data["current_step"] = 2
        await store.put(
            CheckpointConfig(thread_id="run-1"),
            replacement,
            {"source": "update", "step": 2},
        )

        listed = await store.list(CheckpointConfig(thread_id="run-1"))
        assert [item.checkpoint.id for item in listed] == ["cp-1"]
        loaded = await store.get(CheckpointConfig(thread_id="run-1"))
        assert loaded is not None
        assert loaded.step == 2
        assert loaded.state_data["current_step"] == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_store_latest_is_step_ordered_not_insertion_order(store_factory) -> None:
    store = store_factory()
    try:
        await store.put(
            CheckpointConfig(thread_id="run-1"),
            _checkpoint("cp-2", step=2),
            {"step": 2},
        )
        await store.put(
            CheckpointConfig(thread_id="run-1"),
            _checkpoint("cp-1", step=1),
            {"step": 1},
        )

        latest = await store.get(CheckpointConfig(thread_id="run-1"))
        assert latest is not None
        assert latest.id == "cp-2"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_store_before_keeps_same_step_order(store_factory) -> None:
    store = store_factory()
    try:
        for checkpoint_id in ("cp-1", "cp-2", "cp-3"):
            await store.put(
                CheckpointConfig(thread_id="run-1"),
                _checkpoint(checkpoint_id, step=1),
                {"step": 1},
            )

        listed = await store.list(
            CheckpointConfig(thread_id="run-1"),
            before=CheckpointConfig(
                thread_id="run-1", checkpoint_id=CheckpointId("cp-3")
            ),
        )
        assert [item.checkpoint.id for item in listed] == ["cp-2", "cp-1"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_delete_cannot_cross_thread_boundary(store_factory) -> None:
    store = store_factory()
    try:
        checkpoint = _checkpoint("cp-1")
        await store.put(
            CheckpointConfig(thread_id="run-1"), checkpoint, {"step": 1}
        )
        await store.delete(
            CheckpointConfig(thread_id="other-run", checkpoint_id=checkpoint.id)
        )
        assert await store.get(
            CheckpointConfig(thread_id="run-1", checkpoint_id=checkpoint.id)
        ) is not None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_store_rejects_incomplete_history_transaction(store_factory) -> None:
    store = store_factory()
    checkpoint = _checkpoint("cp-incomplete")
    checkpoint.history = HistorySnapshot(
        messages=(
            HistoryMessage(
                role="assistant",
                step_id=1,
                tool_calls=[{"id": "call-missing", "function": {}}],
            ),
        )
    )
    try:
        with pytest.raises(ValueError, match="incomplete"):
            await store.put(
                CheckpointConfig(thread_id="run-1"), checkpoint, {"step": 1}
            )
    finally:
        await store.close()


def test_trace_writer_commits_events_before_step_marker(tmp_path) -> None:
    writer = TraceWriter(str(tmp_path), "run-1", strict_validate=False)
    event = TraceEvent(
        run_id="run-1",
        step_id=1,
        phase="DECIDE",
        payload={"stage": "model_output"},
    )
    writer.write_transaction(
        [event],
        TraceStep(step_id=1, event_start_idx=0, event_end_idx=0),
    )
    writer.finalize(status="completed", summary={})

    events = (tmp_path / "run-1" / "events.jsonl").read_text().splitlines()
    steps = (tmp_path / "run-1" / "steps.jsonl").read_text().splitlines()
    assert json.loads(events[0])["phase"] == "DECIDE"
    assert json.loads(steps[0])["step_id"] == 1
    manifest = json.loads((tmp_path / "run-1" / "manifest.json").read_text())
    assert manifest["event_count"] == 1
    assert manifest["step_count"] == 1


def test_trace_writer_flushes_lifecycle_events_immediately(tmp_path) -> None:
    writer = TraceWriter(str(tmp_path), "run-1", strict_validate=False)
    writer.write_event(
        TraceEvent(
            run_id="run-1",
            step_id=1,
            phase="END",
            ok=False,
            payload={"stop_reason": "cancelled"},
        )
    )

    events = (tmp_path / "run-1" / "events.jsonl").read_text().splitlines()
    assert json.loads(events[0])["phase"] == "END"
    writer.finalize(status="cancelled", summary={})
