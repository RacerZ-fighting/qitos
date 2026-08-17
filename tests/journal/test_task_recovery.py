"""Task projection over the journal: ordering, folding, fail-closed rules."""

from __future__ import annotations

import pytest

from qitos.core.journal import (
    JournalCorruptionError,
    JournalRecordType,
    SessionJournal,
)
from qitos.core.message import AssistantMessage, UserMessage
from qitos.core.model_request import ModelRequest
from qitos.core.model_response import ModelUsage
from qitos.core.task import Task, TaskBlocker, TaskStatus
from qitos.kit.journal import (
    InMemoryJournalStore,
    InMemorySessionJournal,
    JournalTurnTransaction,
    recover_session,
)
from qitos.kit.journal.turn_recorder import (
    encode_task_created,
    encode_task_transition,
)


def _task(task_id: str = "root-task", **kwargs: object) -> Task:
    return Task(task_id=task_id, objective=f"{task_id} objective", **kwargs)  # type: ignore[arg-type]


def _request(run_id: str, turn: int) -> ModelRequest:
    return ModelRequest(
        run_id=run_id,
        transaction_id=f"{run_id}:turn:{turn}:tx",
        provider="scripted",
        model="scripted-model",
        protocol="legacy",
        messages=({"role": "user", "content": "go"},),
    )


async def _create(
    store: InMemoryJournalStore, run_id: str
) -> InMemorySessionJournal:
    journal = InMemorySessionJournal(store)
    await journal.create(run_id, {"purpose": "task-projection-test"})
    return journal


async def _append_turn(journal: SessionJournal, run_id: str, turn: int = 0) -> None:
    """Commit one minimal turn: input, model terminal and the turn commit."""

    recorder = JournalTurnTransaction(journal)
    prompt = UserMessage(content="go")
    await recorder.input_accepted((prompt,))
    assistant = AssistantMessage(text="done")
    await recorder.model_terminal(turn, _request(run_id, turn), assistant)
    await recorder.turn_committed(turn, (prompt, assistant))


async def _recover(store: InMemoryJournalStore, run_id: str):
    journal = InMemorySessionJournal(store)
    await journal.open(run_id)
    try:
        records = await journal.replay()
    finally:
        await journal.close()
    return recover_session(records)


async def _created(journal: SessionJournal, run_id: str, task: Task) -> None:
    await journal.append(
        JournalRecordType.TASK_CREATED,
        encode_task_created(task),
        record_id=f"{run_id}:task:{task.task_id}:created",
    )


async def _transition(
    journal: SessionJournal,
    run_id: str,
    *,
    sequence: int = 0,
    **kwargs: object,
) -> None:
    task_id = str(kwargs["task_id"])
    await journal.append(
        JournalRecordType.TASK_TRANSITION,
        encode_task_transition(**kwargs),  # type: ignore[arg-type]
        record_id=f"{run_id}:task:{task_id}:transition:{sequence}",
    )


# ── projection and ordering ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_root_task_commits_before_input_and_folds_active() -> None:
    store = InMemoryJournalStore()
    journal = await _create(store, "run-a")
    task = _task()
    await _created(journal, "run-a", task)
    await _append_turn(journal, "run-a")
    records = await journal.replay()
    ordering = [record.type for record in records]
    assert ordering.index(JournalRecordType.TASK_CREATED) < ordering.index(
        JournalRecordType.INPUT_ACCEPTED
    )
    recovered = recover_session(records)
    root = recovered.unfinished_root
    assert root is not None
    assert root.definition == task
    assert root.lifecycle.status is TaskStatus.ACTIVE
    assert root.lifecycle.blocker is None
    await journal.close()


@pytest.mark.asyncio
async def test_transitions_fold_through_fork_lineage() -> None:
    store = InMemoryJournalStore()
    journal = await _create(store, "run-a")
    await _created(journal, "run-a", _task())
    await _append_turn(journal, "run-a")
    records = await journal.replay()
    boundary = records[-1].position
    child = await journal.fork(boundary, "run-b")
    await journal.close()
    await _transition(
        child,
        "run-b",
        task_id="root-task",
        from_status=TaskStatus.ACTIVE,
        to_status=TaskStatus.BLOCKED,
        blocker=TaskBlocker(awaiting="input", detail="needs scope approval"),
    )
    await _transition(
        child,
        "run-b",
        sequence=1,
        task_id="root-task",
        from_status=TaskStatus.BLOCKED,
        to_status=TaskStatus.COMPLETED,
        reason="criteria met",
    )
    await child.close()
    recovered = await _recover(store, "run-b")
    root = recovered.tasks["root-task"]
    assert root.lifecycle.status is TaskStatus.COMPLETED
    assert root.lifecycle.terminal_reason == "criteria met"
    assert recovered.unfinished_root is None


@pytest.mark.asyncio
async def test_terminal_root_then_new_root_in_later_leg() -> None:
    store = InMemoryJournalStore()
    journal = await _create(store, "run-a")
    await _created(journal, "run-a", _task("task-1"))
    await _append_turn(journal, "run-a")
    records = await journal.replay()
    child = await journal.fork(records[-1].position, "run-b")
    await journal.close()
    await _transition(
        child,
        "run-b",
        task_id="task-1",
        from_status=TaskStatus.ACTIVE,
        to_status=TaskStatus.FAILED,
        reason="target unreachable",
    )
    await _created(child, "run-b", _task("task-2"))
    await child.close()
    recovered = await _recover(store, "run-b")
    assert recovered.tasks["task-1"].lifecycle.status is TaskStatus.FAILED
    follow_up = recovered.unfinished_root
    assert follow_up is not None
    assert follow_up.definition.task_id == "task-2"


@pytest.mark.asyncio
async def test_usage_snapshot_folds_into_lifecycle() -> None:
    store = InMemoryJournalStore()
    journal = await _create(store, "run-a")
    await _created(journal, "run-a", _task())
    await _transition(
        journal,
        "run-a",
        task_id="root-task",
        from_status=TaskStatus.ACTIVE,
        to_status=TaskStatus.CANCELLED,
        reason="operator stop",
        usage=ModelUsage(total_tokens=77),
    )
    records = await journal.replay()
    lifecycle = recover_session(records).tasks["root-task"].lifecycle
    assert lifecycle.usage is not None
    assert lifecycle.usage.total_tokens == 77
    await journal.close()


# ── fail-closed rules ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_late_root_creation_after_side_effects_fails() -> None:
    store = InMemoryJournalStore()
    journal = await _create(store, "run-a")
    await _append_turn(journal, "run-a")
    await _created(journal, "run-a", _task())
    records = await journal.replay()
    with pytest.raises(JournalCorruptionError):
        recover_session(records)
    await journal.close()


@pytest.mark.asyncio
async def test_second_unfinished_root_task_fails() -> None:
    store = InMemoryJournalStore()
    journal = await _create(store, "run-a")
    await _created(journal, "run-a", _task("task-1"))
    await _created(journal, "run-a", _task("task-2"))
    records = await journal.replay()
    with pytest.raises(JournalCorruptionError):
        recover_session(records)
    await journal.close()


@pytest.mark.asyncio
async def test_transition_for_unknown_task_fails() -> None:
    store = InMemoryJournalStore()
    journal = await _create(store, "run-a")
    await _transition(
        journal,
        "run-a",
        task_id="ghost",
        from_status=TaskStatus.ACTIVE,
        to_status=TaskStatus.COMPLETED,
        reason="done",
    )
    records = await journal.replay()
    with pytest.raises(JournalCorruptionError):
        recover_session(records)
    await journal.close()


@pytest.mark.asyncio
async def test_terminal_once_folds_closed() -> None:
    store = InMemoryJournalStore()
    journal = await _create(store, "run-a")
    await _created(journal, "run-a", _task())
    for sequence in range(2):
        await _transition(
            journal,
            "run-a",
            sequence=sequence,
            task_id="root-task",
            from_status=TaskStatus.ACTIVE,
            to_status=TaskStatus.COMPLETED,
            reason="done",
        )
    records = await journal.replay()
    with pytest.raises(JournalCorruptionError):
        recover_session(records)
    await journal.close()


@pytest.mark.asyncio
async def test_from_status_mismatch_fails() -> None:
    store = InMemoryJournalStore()
    journal = await _create(store, "run-a")
    await _created(journal, "run-a", _task())
    await _transition(
        journal,
        "run-a",
        task_id="root-task",
        from_status=TaskStatus.BLOCKED,
        to_status=TaskStatus.ACTIVE,
    )
    records = await journal.replay()
    with pytest.raises(JournalCorruptionError):
        recover_session(records)
    await journal.close()


@pytest.mark.asyncio
async def test_duplicate_created_idempotent_only_when_identical() -> None:
    store = InMemoryJournalStore()
    journal = await _create(store, "run-a")
    task = _task()
    await _created(journal, "run-a", task)
    await _created(journal, "run-a", task)
    records = await journal.replay()
    assert recover_session(records).tasks["root-task"].definition == task
    await journal.close()

    store_b = InMemoryJournalStore()
    journal_b = await _create(store_b, "run-a")
    await _created(journal_b, "run-a", task)
    changed = Task(
        task_id=task.task_id,
        objective="a different objective",
        created_at=task.created_at,
    )
    await journal_b.append(
        JournalRecordType.TASK_CREATED,
        changed.to_dict(),
        record_id="run-a:task:root-task:created:duplicate",
    )
    records_b = await journal_b.replay()
    with pytest.raises(JournalCorruptionError):
        recover_session(records_b)
    await journal_b.close()


@pytest.mark.asyncio
async def test_transition_payload_shape_fails_closed() -> None:
    store = InMemoryJournalStore()
    journal = await _create(store, "run-a")
    await _created(journal, "run-a", _task())
    await journal.append(
        JournalRecordType.TASK_TRANSITION,
        {"task_id": "root-task", "from_status": "active", "to_status": "done"},
        record_id="run-a:task:root-task:transition:0",
    )
    records = await journal.replay()
    with pytest.raises(JournalCorruptionError):
        recover_session(records)
    await journal.close()
