"""Subagent Task binding: launch commits the narrowed Subagent Task durably."""

from __future__ import annotations

import pytest

from qitos.core.subagent import (
    SubagentLaunchContext,
    SubagentLaunchRequest,
    SubagentStatus,
)
from qitos.core.journal import JournalRecordType
from qitos.core.task import Task, TaskBudget, TaskReference
from qitos.kit.subagent import (
    SubagentSupervisor,
    build_agent_subagent_invocation_factory,
)
from qitos.kit.journal import JsonlSessionJournal, recover_session
from qitos.kit.journal.turn_recorder import (
    decode_task_created,
    encode_task_created,
)

from tests.core.agent_fakes import ScriptedModel, text_events


def _subagents_root(tmp_path):
    return tmp_path / "subagents"


async def _read_records(root, run_id: str):
    journal = JsonlSessionJournal(root)
    await journal.open(run_id)
    try:
        return await journal.replay()
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_subagent_journal_commits_narrowed_task_before_input(tmp_path) -> None:
    model = ScriptedModel([text_events("subagent answer", usage={"total_tokens": 7})])
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            journal_directory=_subagents_root(tmp_path),
        ),
        subagent_journal_factory=lambda: JsonlSessionJournal(
            _subagents_root(tmp_path)
        ),
    )
    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})
    await parent_journal.append(
        JournalRecordType.TASK_CREATED,
        encode_task_created(Task(task_id="root-task", objective="Root work")),
        record_id="parent-run:task:root-task:created",
    )
    request = SubagentLaunchRequest(
        task="enumerate the target",
        description="enumeration subagent",
        success_criteria=("Return verified service evidence",),
        constraints={"scope": "primary"},
        references=(
            TaskReference(kind="artifact", uri="scope://engagement/primary"),
        ),
        budget=TaskBudget(max_steps=11),
        parent_task_id="root-task",
    )
    result = await supervisor.launch(
        request,
        SubagentLaunchContext(parent_run_id="parent-run", journal=parent_journal),
        background=False,
    )
    assert result.status is SubagentStatus.COMPLETED

    started = [
        record
        for record in await parent_journal.replay()
        if record.type is JournalRecordType.SUBAGENT_STARTED
    ]
    assert len(started) == 1
    embedded = started[0].payload["request"]
    assert embedded["parent_task_id"] == "root-task"
    assert "plan_assignment" not in embedded

    records = await _read_records(_subagents_root(tmp_path), result.subagent_run_id)
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
    assert definition.task_id == result.handle.subagent_id
    assert definition.parent_task_id == "root-task"
    assert definition.objective == "enumerate the target"
    assert definition.success_criteria == ("Return verified service evidence",)
    assert definition.constraints == {"scope": "primary"}
    assert definition.references == (
        TaskReference(kind="artifact", uri="scope://engagement/primary"),
    )
    assert definition.budget == TaskBudget(max_steps=11)
    assert definition.created_by_run_id == "parent-run"

    recovered = recover_session(records)
    projected = recovered.tasks[result.handle.subagent_id]
    assert projected.definition == definition
    assert projected.lifecycle.status.value == "completed"
    # A Subagent Task is not a Root Task of its own lineage.
    assert recovered.unfinished_root is None

    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_subagent_launch_allows_no_parent_task_binding(tmp_path) -> None:
    model = ScriptedModel([text_events("subagent answer")])
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            journal_directory=_subagents_root(tmp_path),
        ),
    )
    result = await supervisor.launch(
        SubagentLaunchRequest(task="inspect", description="inspect subagent"),
        SubagentLaunchContext(parent_run_id="parent-run"),
        background=False,
    )
    assert result.status is SubagentStatus.COMPLETED
    records = await _read_records(_subagents_root(tmp_path), result.subagent_run_id)
    created = next(
        record
        for record in records
        if record.type is JournalRecordType.TASK_CREATED
    )
    definition = decode_task_created(created.payload)
    assert definition.task_id == result.handle.subagent_id
    assert definition.parent_task_id is None
    await supervisor.aclose()
