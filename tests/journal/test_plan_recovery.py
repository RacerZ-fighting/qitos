"""Plan projection over journal lineage and corruption boundaries."""

from __future__ import annotations

import pytest

from qitos.core.journal import JournalRecordType
from qitos.core.message import AssistantMessage, UserMessage
from qitos.core.model_request import ModelRequest
from qitos.core.plan import (
    MAX_PLAN_EXPLANATION_CHARS,
    Plan,
    PlanItem,
    PlanStatus,
    PlanUpdate,
    plan_to_dict,
)
from qitos.core.task import Task
from qitos.kit.journal import (
    InMemoryJournalStore,
    InMemorySessionJournal,
    JournalTurnTransaction,
    recover_session,
)
from qitos.kit.journal.turn_recorder import (
    decode_plan_updated,
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
    plan = Plan((PlanItem("Inspect target"),))
    await journal.append(
        JournalRecordType.PLAN_UPDATED,
        encode_plan_updated("task-parent", PlanUpdate(plan)),
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
    completed = Plan((PlanItem("Inspect target", PlanStatus.COMPLETED),))
    await child.append(
        JournalRecordType.PLAN_UPDATED,
        encode_plan_updated("task-parent", PlanUpdate(completed)),
        record_id="run-child:plan:completed",
    )

    recovered = recover_session(await child.replay())

    assert initial.items[0].status is PlanStatus.PENDING
    assert recovered.plan == completed


@pytest.mark.asyncio
async def test_plan_recovery_preserves_replacement_history_and_current_projection() -> (
    None
):
    store = InMemoryJournalStore()
    journal, _ = await _create_with_committed_plan(store)
    await journal.append(
        JournalRecordType.PLAN_UPDATED,
        encode_plan_updated("task-parent", PlanUpdate(Plan())),
        record_id="run-parent:plan:invalid-removal",
    )

    recovered = recover_session(await journal.replay())

    assert recovered.plan == Plan()
    plan_updates = [
        record
        for record in await journal.replay()
        if record.type is JournalRecordType.PLAN_UPDATED
    ]
    assert len(plan_updates) == 2


def test_plan_update_codec_round_trips_the_model_rationale() -> None:
    update = PlanUpdate(
        Plan((PlanItem("Read the advisory", PlanStatus.IN_PROGRESS),)),
        explanation="The version banner ruled the first candidate out.",
    )

    task_id, decoded = decode_plan_updated(
        encode_plan_updated("task-parent", update)
    )

    assert task_id == "task-parent"
    assert decoded == update


def test_plan_update_codec_omits_an_absent_rationale() -> None:
    update = PlanUpdate(Plan((PlanItem("Read the advisory"),)))

    payload = encode_plan_updated("task-parent", update)

    assert "explanation" not in payload
    assert decode_plan_updated(payload)[1] == update


def test_plan_update_codec_reads_a_record_written_before_rationales() -> None:
    plan = Plan((PlanItem("Read the advisory"),))

    _, decoded = decode_plan_updated(
        {"task_id": "task-parent", "plan": plan_to_dict(plan)}
    )

    assert decoded.plan == plan
    assert decoded.explanation is None


@pytest.mark.parametrize(
    "payload",
    [
        {"task_id": "task-parent", "plan": {"items": []}, "explanation": 7},
        {"task_id": "task-parent", "plan": {"items": []}, "explanation": "  "},
        {
            "task_id": "task-parent",
            "plan": {"items": []},
            "explanation": "x" * (MAX_PLAN_EXPLANATION_CHARS + 1),
        },
        {"task_id": "task-parent", "plan": {"items": []}, "rationale": "why"},
    ],
)
def test_plan_update_codec_rejects_an_unusable_rationale(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        decode_plan_updated(payload)


@pytest.mark.asyncio
async def test_plan_rationale_does_not_change_the_folded_checklist() -> None:
    store = InMemoryJournalStore()
    journal, initial = await _create_with_committed_plan(store)
    explained = Plan((PlanItem("Inspect target", PlanStatus.COMPLETED),))
    await journal.append(
        JournalRecordType.PLAN_UPDATED,
        encode_plan_updated(
            "task-parent",
            PlanUpdate(explained, explanation="The banner confirmed the version."),
        ),
        record_id="run-parent:plan:explained",
    )

    recovered = recover_session(await journal.replay())

    assert initial != explained
    assert recovered.plan == explained
