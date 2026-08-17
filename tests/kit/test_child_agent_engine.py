"""Façade-driven child agents: launch, interrupt, mailbox and journal recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from qitos.core.child import (
    ChildHandle,
    ChildLaunchContext,
    ChildLaunchRequest,
    ChildPersistenceError,
    ChildStatus,
)
from qitos.core.journal import JournalRecordType
from qitos.core.model_stream import ModelStreamEvent, ModelStreamEventType
from qitos.core.task import TaskBudget
from qitos.core.tool import tool
from qitos.core.tool_registry import ToolRegistry
from qitos.kit.child import (
    AgentChildEngine,
    ChildSupervisor,
    build_agent_child_invocation_factory,
)
from qitos.kit.journal import JsonlSessionJournal

from tests.core.agent_fakes import (
    ScriptedModel,
    make_hanging_model,
    text_events,
    tool_call_wire,
    tool_events,
)


@tool(name="echo")
def _echo(text: str) -> str:
    return f"echo:{text}"


def _request(task: str, **kwargs: object) -> ChildLaunchRequest:
    return ChildLaunchRequest(task=task, description=f"{task} task", **kwargs)


def _context(**kwargs: object) -> ChildLaunchContext:
    return ChildLaunchContext(parent_run_id="parent-run", **kwargs)


def _children_root(tmp_path):
    return tmp_path / "children"


def _child_journal_factory(tmp_path):
    return lambda: JsonlSessionJournal(_children_root(tmp_path))


async def _read_child_records(tmp_path, run_id: str):
    journal = JsonlSessionJournal(_children_root(tmp_path))
    await journal.open(run_id)
    try:
        return await journal.replay()
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_foreground_child_completes_and_journals_turns(tmp_path) -> None:
    model = ScriptedModel(
        [text_events("child answer", usage={"total_tokens": 7})]
    )
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            journal_directory=_children_root(tmp_path),
        ),
        child_journal_factory=_child_journal_factory(tmp_path),
    )
    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})

    result = await supervisor.launch(
        _request("inspect"),
        _context(journal=parent_journal),
        background=False,
    )

    assert result.status is ChildStatus.COMPLETED
    assert result.conclusion.summary == "child answer"
    assert result.steps == 1
    assert result.total_tokens == 7
    assert result.usage_complete is True
    assert result.cost_complete is False
    assert result.child_run_id

    parent_records = await parent_journal.replay()
    child_started = [
        record
        for record in parent_records
        if record.type is JournalRecordType.CHILD_STARTED
    ]
    child_terminal = [
        record
        for record in parent_records
        if record.type is JournalRecordType.CHILD_TERMINAL
    ]
    assert len(child_started) == len(child_terminal) == 1
    assert child_terminal[0].payload["status"] == ChildStatus.COMPLETED.value

    child_records = await _read_child_records(tmp_path, result.child_run_id)
    child_types = [record.type for record in child_records]
    assert child_types[0] is JournalRecordType.RUN_STARTED
    assert JournalRecordType.MODEL_COMPLETED in child_types
    assert JournalRecordType.STEP_COMMITTED in child_types
    assert child_types[-1] is JournalRecordType.RUN_COMPLETED
    terminal = child_records[-1]
    assert terminal.record_id == f"{result.child_run_id}:run:terminal"
    assert terminal.payload["status"] == "completed"
    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_child_tool_evidence_commits_in_order(tmp_path) -> None:
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "ping"})]),
            text_events("done with echo:ping"),
        ]
    )
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            tool_registry=ToolRegistry().register(_echo),
            journal_directory=_children_root(tmp_path),
        ),
        child_journal_factory=_child_journal_factory(tmp_path),
    )

    result = await supervisor.launch(
        _request("use the tool"),
        _context(),
        background=False,
    )

    assert result.status is ChildStatus.COMPLETED
    assert result.steps == 2
    assert result.conclusion.summary == "done with echo:ping"
    child_records = await _read_child_records(tmp_path, result.child_run_id)
    tool_terminal = [
        record
        for record in child_records
        if record.type is JournalRecordType.TOOL_TERMINAL
    ]
    assert len(tool_terminal) == 1
    assert (
        tool_terminal[0].record_id
        == f"{result.child_run_id}:turn:0:tool:c1:terminal"
    )
    assert tool_terminal[0].payload["result"]["output"] == "echo:ping"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_interrupt_running_child_terminalizes_journal(tmp_path) -> None:
    started = asyncio.Event()
    never = asyncio.Event()

    async def hanging(_request):
        started.set()
        await never.wait()
        yield  # unreachable; keeps the factory an async generator

    model = ScriptedModel([hanging])
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            journal_directory=_children_root(tmp_path),
        ),
        child_journal_factory=_child_journal_factory(tmp_path),
    )

    launched = await supervisor.launch(_request("hang"), _context(), background=True)
    await asyncio.wait_for(started.wait(), timeout=5)
    result = await supervisor.interrupt(launched.handle, wait_seconds=5)

    assert result is not None
    assert result.status is ChildStatus.CANCELLED
    child_records = await _read_child_records(tmp_path, result.child_run_id)
    assert child_records[-1].type is JournalRecordType.RUN_INTERRUPTED
    assert child_records[-1].payload["status"] == "aborted"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_child_budget_exhaustion_maps_from_max_turns(tmp_path) -> None:
    model = ScriptedModel(
        [tool_events([tool_call_wire(f"c{i}", "echo", {"text": "x"})]) for i in range(4)]
    )
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            tool_registry=ToolRegistry().register(_echo),
            journal_directory=_children_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("loop", budget=TaskBudget(max_steps=1)),
        _context(),
        background=False,
    )

    assert result.status is ChildStatus.BUDGET_EXHAUSTED
    assert result.steps == 1
    assert len(model.requests) == 1
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_parent_message_steers_active_child(tmp_path) -> None:
    async def first_response(_request):
        yield ModelStreamEvent(
            type=ModelStreamEventType.COMPLETED,
            finish_reason="tool_calls",
            tool_calls=[tool_call_wire("c1", "echo", {"text": "wait"})],
        )

    model = ScriptedModel(
        [first_response, text_events("acknowledged")],
    )
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            tool_registry=ToolRegistry().register(_echo),
            journal_directory=_children_root(tmp_path),
        ),
        child_journal_factory=_child_journal_factory(tmp_path),
    )
    launched = await supervisor.launch(_request("start"), _context(), background=True)

    accepted, current = await supervisor.message(
        launched.handle,
        "steer note",
        timeout_seconds=5,
    )

    assert accepted is True
    assert current is not None and not current.ready
    final = await supervisor.wait(launched.handle, timeout_seconds=5)
    assert final is not None and final.status is ChildStatus.COMPLETED
    second_request = model.requests[1]
    steered = [
        message
        for message in second_request.messages
        if isinstance(message, Mapping)
        and message.get("role") == "user"
        and "steer note" in str(message.get("content"))
    ]
    assert len(steered) == 1
    child_records = await _read_child_records(tmp_path, final.child_run_id)
    assert any(
        record.type is JournalRecordType.RUNTIME_INPUT_POSTED
        and record.payload["payload"]["content"] == "steer note"
        for record in child_records
    )
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_recovery_rebuilds_completed_child_from_journal(tmp_path) -> None:
    engine = AgentChildEngine(
        model=ScriptedModel([text_events("recovered answer", usage={"total_tokens": 5})]),
        journal_factory=_child_journal_factory(tmp_path),
    )
    completed = await engine.arun("inspect", run_id="run_childcompleted")
    await engine.aclose()
    assert completed.state.stop_reason == "completed"

    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})
    request = _request("inspect")
    handle = ChildHandle(child_id="child-done", parent_run_id="parent-run")
    await parent_journal.append(
        JournalRecordType.CHILD_STARTED,
        {
            "handle": handle.to_dict(),
            "request": request.to_dict(),
            "child_run_id": "run_childcompleted",
        },
        record_id="parent-run:child:child-done:started",
    )

    supervisor = ChildSupervisor(
        invocation_factory=_unused_factory,
        child_journal_factory=_child_journal_factory(tmp_path),
    )
    recovered = await supervisor.recover(
        parent_run_id="parent-run",
        journal=parent_journal,
    )

    assert len(recovered) == 1
    result = recovered[0]
    assert result.status is ChildStatus.COMPLETED
    assert result.conclusion.summary == "recovered answer"
    assert result.steps == 1
    assert result.total_tokens == 5
    assert result.usage_complete is True
    terminal_records = [
        record
        for record in await parent_journal.replay()
        if record.type is JournalRecordType.CHILD_TERMINAL
    ]
    assert len(terminal_records) == 1
    assert terminal_records[0].payload["status"] == ChildStatus.COMPLETED.value
    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_recovery_rebuilds_interrupted_child_from_journal(tmp_path) -> None:
    started = asyncio.Event()
    never = asyncio.Event()

    async def hanging(_request):
        started.set()
        await never.wait()
        yield  # unreachable; keeps the factory an async generator

    engine = AgentChildEngine(
        model=ScriptedModel([hanging]),
        journal_factory=_child_journal_factory(tmp_path),
    )
    run = asyncio.create_task(engine.arun("hang", run_id="run_childaborted"))
    await asyncio.wait_for(started.wait(), timeout=5)
    engine.cancel("immediate")
    aborted = await run
    await engine.aclose()
    assert aborted.state.stop_reason == "cancelled"

    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})
    request = _request("hang")
    handle = ChildHandle(child_id="child-aborted", parent_run_id="parent-run")
    await parent_journal.append(
        JournalRecordType.CHILD_STARTED,
        {
            "handle": handle.to_dict(),
            "request": request.to_dict(),
            "child_run_id": "run_childaborted",
        },
        record_id="parent-run:child:child-aborted:started",
    )

    supervisor = ChildSupervisor(
        invocation_factory=_unused_factory,
        child_journal_factory=_child_journal_factory(tmp_path),
    )
    recovered = await supervisor.recover(
        parent_run_id="parent-run",
        journal=parent_journal,
    )

    assert len(recovered) == 1
    assert recovered[0].status is ChildStatus.CANCELLED
    assert recovered[0].child_run_id == "run_childaborted"
    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_recovery_without_child_journal_marks_interrupted(tmp_path) -> None:
    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})
    request = _request("lost")
    handle = ChildHandle(child_id="child-lost", parent_run_id="parent-run")
    await parent_journal.append(
        JournalRecordType.CHILD_STARTED,
        {
            "handle": handle.to_dict(),
            "request": request.to_dict(),
            "child_run_id": "run_missing",
        },
        record_id="parent-run:child:child-lost:started",
    )

    supervisor = ChildSupervisor(
        invocation_factory=_unused_factory,
        child_journal_factory=_child_journal_factory(tmp_path),
    )
    recovered = await supervisor.recover(
        parent_run_id="parent-run",
        journal=parent_journal,
    )

    assert len(recovered) == 1
    assert recovered[0].status is ChildStatus.INTERRUPTED
    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_recovery_rejects_non_loop_terminal_payload(tmp_path) -> None:
    child_journal = JsonlSessionJournal(_children_root(tmp_path))
    await child_journal.create("run_legacy", {})
    await child_journal.append(
        JournalRecordType.RUN_COMPLETED,
        {"status": "completed", "state": {"final_result": "legacy"}},
        record_id="run_legacy:run:terminal",
    )
    await child_journal.close()

    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})
    request = _request("legacy")
    handle = ChildHandle(child_id="child-legacy", parent_run_id="parent-run")
    await parent_journal.append(
        JournalRecordType.CHILD_STARTED,
        {
            "handle": handle.to_dict(),
            "request": request.to_dict(),
            "child_run_id": "run_legacy",
        },
        record_id="parent-run:child:child-legacy:started",
    )

    supervisor = ChildSupervisor(
        invocation_factory=_unused_factory,
        child_journal_factory=_child_journal_factory(tmp_path),
    )
    with pytest.raises(ChildPersistenceError):
        await supervisor.recover(
            parent_run_id="parent-run",
            journal=parent_journal,
        )
    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_child_tool_groups_narrow_exposure(tmp_path) -> None:
    registry = ToolRegistry().register(_echo)

    @tool(name="other", group="network")
    def _other() -> str:
        return "other"

    registry.register(_other)

    class RecordingModel(ScriptedModel):
        def build_tool_schema_request_options(
            self, payload, *, protocol=None, delivery="prompt_injection"
        ):
            self.seen_tool_names = {
                item.get("function", {}).get("name") for item in payload
            }
            return {"tools": list(payload)}

    model = RecordingModel([text_events("done")])
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_children_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("inspect", allowed_tool_groups=("default",)),
        _context(),
        background=False,
    )

    assert result.status is ChildStatus.COMPLETED
    assert "echo" in model.seen_tool_names
    assert "other" not in model.seen_tool_names
    await supervisor.aclose()


async def _unused_factory(_request, _context):
    raise AssertionError("recovery must not construct a child")
