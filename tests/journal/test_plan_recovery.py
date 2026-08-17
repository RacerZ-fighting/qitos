"""Plan projection over journal lineage and corruption boundaries."""

from __future__ import annotations

import pytest

from qitos.core.journal import JournalCorruptionError, JournalRecordType
from qitos.core.message import AssistantMessage, UserMessage
from qitos.core.model_request import ModelRequest
from qitos.core.plan import Plan, PlanNode, PlanStatus
from qitos.core.task import Task
from qitos.kit.journal import (
    InMemoryJournalStore,
    InMemorySessionJournal,
    JournalTurnTransaction,
    recover_session,
)
from qitos.kit.journal.turn_recorder import (
    encode_plan_updated,
    encode_task_created,
)


def _request(run_id: str) -> ModelRequest:
    return ModelRequest(
        run_id=run_id,
        transaction_id=f"{run_id}:turn:0:tx",
        provider="scripted",
        model="scripted-model",
        protocol="legacy",
        messages=({"role": "user", "content": "go"},),
    )


async def _create_with_committed_plan(
    store: InMemoryJournalStore,
) -> tuple[InMemorySessionJournal, Plan]:
    journal = InMemorySessionJournal(store)
    await journal.create("run-parent", {"purpose": "plan-recovery"})
    task = Task(task_id="task-parent", objective="Do the work")
    await journal.append(
        JournalRecordType.TASK_CREATED,
        encode_task_created(task),
        record_id="run-parent:task:task-parent:created",
    )
    recorder = JournalTurnTransaction(journal)
    prompt = UserMessage(content="go")
    assistant = AssistantMessage(text="planning")
    await recorder.input_accepted((prompt,))
    await recorder.model_terminal(0, _request("run-parent"), assistant)
    plan = Plan((PlanNode("inspect", "Inspect target"),))
    await journal.append(
        JournalRecordType.PLAN_UPDATED,
        encode_plan_updated("task-parent", plan),
        record_id="run-parent:plan:first",
    )
    await recorder.turn_committed(0, (prompt, assistant))
    return journal, plan


@pytest.mark.asyncio
async def test_plan_folds_through_fork_lineage() -> None:
    store = InMemoryJournalStore()
    parent, initial = await _create_with_committed_plan(store)
    records = await parent.replay()
    child = await parent.fork(records[-1].position, "run-child")
    await parent.close()
    completed = Plan(
        (PlanNode("inspect", "Inspect target", PlanStatus.COMPLETED),)
    )
    await child.append(
        JournalRecordType.PLAN_UPDATED,
        encode_plan_updated("task-parent", completed),
        record_id="run-child:plan:completed",
    )

    recovered = recover_session(await child.replay())

    assert initial.node("inspect").status is PlanStatus.PENDING
    assert recovered.plan == completed


@pytest.mark.asyncio
async def test_plan_recovery_fails_closed_on_history_rewrite() -> None:
    store = InMemoryJournalStore()
    journal, _ = await _create_with_committed_plan(store)
    await journal.append(
        JournalRecordType.PLAN_UPDATED,
        encode_plan_updated("task-parent", Plan()),
        record_id="run-parent:plan:invalid-removal",
    )

    with pytest.raises(JournalCorruptionError):
        recover_session(await journal.replay())
