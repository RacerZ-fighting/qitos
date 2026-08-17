"""Child Task binding: launch commits the narrowed Child Task durably."""

from __future__ import annotations

import pytest

from qitos.core.child import (
    ChildLaunchContext,
    ChildLaunchRequest,
    ChildStatus,
)
from qitos.core.journal import JournalRecordType
from qitos.core.task import TaskBudget
from qitos.kit.child import (
    ChildSupervisor,
    build_agent_child_invocation_factory,
)
from qitos.kit.journal import JsonlSessionJournal, recover_session
from qitos.kit.journal.turn_recorder import decode_task_created

from tests.core.agent_fakes import ScriptedModel, text_events


def _children_root(tmp_path):
    return tmp_path / "children"


async def _read_records(root, run_id: str):
    journal = JsonlSessionJournal(root)
    await journal.open(run_id)
    try:
        return await journal.replay()
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_child_journal_commits_narrowed_task_before_input(tmp_path) -> None:
    model = ScriptedModel([text_events("child answer", usage={"total_tokens": 7})])
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            journal_directory=_children_root(tmp_path),
        ),
        child_journal_factory=lambda: JsonlSessionJournal(
            _children_root(tmp_path)
        ),
    )
    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})

    request = ChildLaunchRequest(
        task="enumerate the target",
        description="enumeration child",
        budget=TaskBudget(max_steps=11),
        parent_task_id="root-task",
        plan_assignment="plan-node-1",
    )
    result = await supervisor.launch(
        request,
        ChildLaunchContext(parent_run_id="parent-run", journal=parent_journal),
        background=False,
    )
    assert result.status is ChildStatus.COMPLETED

    started = [
        record
        for record in await parent_journal.replay()
        if record.type is JournalRecordType.CHILD_STARTED
    ]
    assert len(started) == 1
    embedded = started[0].payload["request"]
    assert embedded["parent_task_id"] == "root-task"
    assert embedded["plan_assignment"] == "plan-node-1"

    records = await _read_records(_children_root(tmp_path), result.child_run_id)
    ordering = [record.type for record in records]
    assert ordering.index(JournalRecordType.TASK_CREATED) < ordering.index(
        JournalRecordType.INPUT_ACCEPTED
    )
    created = next(
        record
        for record in records
        if record.type is JournalRecordType.TASK_CREATED
    )
    definition = decode_task_created(created.payload)
    assert definition.task_id == result.handle.child_id
    assert definition.parent_task_id == "root-task"
    assert definition.objective == "enumerate the target"
    assert definition.budget == TaskBudget(max_steps=11)
    assert definition.created_by_run_id == "parent-run"
    assert definition.plan_assignment == "plan-node-1"

    recovered = recover_session(records)
    projected = recovered.tasks[result.handle.child_id]
    assert projected.definition == definition
    assert projected.lifecycle.status.value == "active"
    # A Child Task is not a Root Task of its own lineage.
    assert recovered.unfinished_root is None

    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_child_launch_defaults_keep_task_binding_empty(tmp_path) -> None:
    model = ScriptedModel([text_events("child answer")])
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            journal_directory=_children_root(tmp_path),
        ),
    )
    result = await supervisor.launch(
        ChildLaunchRequest(task="inspect", description="inspect child"),
        ChildLaunchContext(parent_run_id="parent-run"),
        background=False,
    )
    assert result.status is ChildStatus.COMPLETED
    records = await _read_records(_children_root(tmp_path), result.child_run_id)
    created = next(
        record
        for record in records
        if record.type is JournalRecordType.TASK_CREATED
    )
    definition = decode_task_created(created.payload)
    assert definition.task_id == result.handle.child_id
    assert definition.parent_task_id is None
    assert definition.plan_assignment is None
    await supervisor.aclose()
