"""Façade-driven child agents: launch, interrupt, mailbox and journal recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from qitos.core.budget import BudgetLedger
from qitos.core.child import (
    ChildHandle,
    ChildLaunchContext,
    ChildLaunchRequest,
    ChildPersistenceError,
    ChildStatus,
)
from qitos.core.journal import (
    JournalAppendCancelled,
    JournalCommitError,
    JournalCommitState,
    JournalError,
    JournalRecordType,
)
from qitos.core.model_response import ModelPricing
from qitos.core.model_stream import ModelStreamEvent, ModelStreamEventType
from qitos.core.plan import Plan, PlanNode
from qitos.core.task import Task, TaskBudget
from qitos.core.tool import ToolPermissionContext, ToolPermissionRule, tool
from qitos.core.tool_registry import ToolRegistry
from qitos.core.runtime_input import RuntimeInput
from qitos.kit.child import (
    AgentChildEngine,
    ChildSupervisor,
    build_agent_child_invocation_factory,
)
from qitos.kit.journal import JsonlSessionJournal
from qitos.kit.journal.turn_recorder import encode_plan_updated, encode_task_created
from qitos.kit.tool.agent import AgentTool

from tests.core.agent_fakes import (
    ScriptedModel,
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


def _terminal_result_payload(records, terminal):
    """Join one tool.terminal with the transcript entry it references."""

    referenced = terminal.payload["message_record_id"]
    entry = next(
        record
        for record in records
        if record.type is JournalRecordType.TRANSCRIPT_MESSAGE
        and record.record_id == referenced
    )
    return entry.payload["message"]["result"]


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
async def test_agent_tool_threads_explicit_plan_assignment(tmp_path) -> None:
    parent_journal = JsonlSessionJournal(tmp_path / "parent-plan")
    await parent_journal.create("parent-run", {})
    await parent_journal.append(
        JournalRecordType.TASK_CREATED,
        encode_task_created(Task(task_id="parent-task", objective="Parent work")),
        record_id="parent-run:task:parent-task:created",
    )
    await parent_journal.append(
        JournalRecordType.PLAN_UPDATED,
        encode_plan_updated(
            "parent-task",
            Plan((PlanNode("delegate", "Delegate work"),)),
        ),
        record_id="parent-run:plan:initial",
    )
    agent_tool = AgentTool(
        invocation_factory=build_agent_child_invocation_factory(
            model=ScriptedModel([text_events("child answer")]),
            journal_directory=_children_root(tmp_path),
        )
    )

    result = await agent_tool.execute(
        {
            "description": "inspect",
            "prompt": "inspect",
            "plan_assignment": "delegate",
        },
        runtime_context={
            "run_id": "parent-run",
            "task_id": "parent-task",
            "journal": parent_journal,
        },
    )
    started = next(
        record
        for record in await parent_journal.replay()
        if record.type is JournalRecordType.CHILD_STARTED
    )

    assert result.output["child_status"] == ChildStatus.COMPLETED.value
    assert started.payload["request"]["parent_task_id"] == "parent-task"
    assert started.payload["request"]["plan_assignment"] == "delegate"
    await agent_tool.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_child_tool_evidence_commits_in_order(tmp_path) -> None:
    registry = ToolRegistry().register(_echo)
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "ping"})]),
            text_events("done with echo:ping"),
        ]
    )
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_children_root(tmp_path),
        ),
        child_journal_factory=_child_journal_factory(tmp_path),
    )

    result = await supervisor.launch(
        _request("use the tool"),
        _context(parent_tool_authority=registry.freeze()),
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
    assert _terminal_result_payload(child_records, tool_terminal[0])["output"] == (
        "echo:ping"
    )
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
    registry = ToolRegistry().register(_echo)
    model = ScriptedModel(
        [tool_events([tool_call_wire(f"c{i}", "echo", {"text": "x"})]) for i in range(4)]
    )
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_children_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("loop", budget=TaskBudget(max_steps=1)),
        _context(parent_tool_authority=registry.freeze()),
        background=False,
    )

    assert result.status is ChildStatus.BUDGET_EXHAUSTED
    assert result.steps == 1
    assert len(model.requests) == 1
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_parent_message_steers_active_child(tmp_path) -> None:
    registry = ToolRegistry().register(_echo)

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
            tool_registry=registry,
            journal_directory=_children_root(tmp_path),
        ),
        child_journal_factory=_child_journal_factory(tmp_path),
    )
    launched = await supervisor.launch(
        _request("start"),
        _context(parent_tool_authority=registry.freeze()),
        background=True,
    )

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
async def test_parent_message_rejects_pre_run_and_terminal_settlement_windows(
    tmp_path,
) -> None:
    terminal_append_started = asyncio.Event()
    release_terminal_append = asyncio.Event()

    class BlockingTerminalJournal(JsonlSessionJournal):
        async def append(self, record_type, payload, *, record_id):
            if record_type is JournalRecordType.RUN_COMPLETED:
                terminal_append_started.set()
                await release_terminal_append.wait()
            return await super().append(record_type, payload, record_id=record_id)

    engine = AgentChildEngine(
        model=ScriptedModel([text_events("done")]),
        journal_factory=lambda: BlockingTerminalJournal(_children_root(tmp_path)),
    )
    event = RuntimeInput(
        event_id="parent-message",
        kind="agent.parent.message",
        correlation_id="child",
        source="qitos.parent",
        payload={"content": "too late"},
    )

    assert await engine.apost_runtime_event(event, run_id="run_childrace") is False
    running = asyncio.create_task(
        engine.arun("inspect", run_id="run_childrace")
    )
    await asyncio.wait_for(terminal_append_started.wait(), timeout=1)

    assert await engine.apost_runtime_event(event, run_id="run_childrace") is False

    release_terminal_append.set()
    await running
    await engine.aclose()
    records = await _read_child_records(tmp_path, "run_childrace")
    assert JournalRecordType.RUNTIME_INPUT_POSTED not in {
        record.type for record in records
    }


@pytest.mark.asyncio
async def test_parent_message_reserved_before_turn_end_settles_before_terminal(
    tmp_path,
) -> None:
    model_started = asyncio.Event()
    release_model = asyncio.Event()
    runtime_append_started = asyncio.Event()
    release_runtime_append = asyncio.Event()
    turn_boundary_started = asyncio.Event()

    async def first_response(_request):
        model_started.set()
        await release_model.wait()
        for event in text_events("first answer"):
            yield event

    class BlockingRuntimeJournal(JsonlSessionJournal):
        async def append(self, record_type, payload, *, record_id):
            if record_type is JournalRecordType.RUNTIME_INPUT_POSTED:
                runtime_append_started.set()
                await release_runtime_append.wait()
            return await super().append(record_type, payload, record_id=record_id)

    class BoundaryObservedEngine(AgentChildEngine):
        def _close_runtime_event_admission(self):
            turn_boundary_started.set()
            super()._close_runtime_event_admission()

    model = ScriptedModel(
        [first_response, text_events("answer after parent message")]
    )
    engine = BoundaryObservedEngine(
        model=model,
        journal_factory=lambda: BlockingRuntimeJournal(_children_root(tmp_path)),
    )
    event = RuntimeInput(
        event_id="parent-message-race",
        kind="agent.parent.message",
        correlation_id="child",
        source="qitos.parent",
        payload={"content": "reserved note"},
    )

    running = asyncio.create_task(
        engine.arun("inspect", run_id="run_childreserved")
    )
    await asyncio.wait_for(model_started.wait(), timeout=1)
    posting = asyncio.create_task(
        engine.apost_runtime_event(event, run_id="run_childreserved")
    )
    release_model.set()
    await asyncio.wait_for(turn_boundary_started.wait(), timeout=1)
    await asyncio.wait_for(runtime_append_started.wait(), timeout=1)

    assert running.done() is False
    release_runtime_append.set()
    assert await asyncio.wait_for(posting, timeout=1) is True
    result = await asyncio.wait_for(running, timeout=1)
    assert result.state.stop_reason == "completed"
    assert model.requests[1].messages[-1] == {
        "role": "user",
        "content": "reserved note",
    }

    await engine.aclose()
    records = await _read_child_records(tmp_path, "run_childreserved")
    record_types = [record.type for record in records]
    assert record_types.index(JournalRecordType.RUNTIME_INPUT_POSTED) < (
        record_types.index(JournalRecordType.RUN_COMPLETED)
    )


@pytest.mark.asyncio
async def test_accepted_parent_message_is_marked_consumed_after_its_turn_commits(
    tmp_path,
) -> None:
    registry = ToolRegistry().register(_echo)

    async def first_response(_request):
        yield ModelStreamEvent(
            type=ModelStreamEventType.COMPLETED,
            finish_reason="tool_calls",
            tool_calls=[tool_call_wire("c1", "echo", {"text": "wait"})],
        )

    model = ScriptedModel(
        [first_response, text_events("acknowledged")],
    )
    engine = AgentChildEngine(
        model=model,
        tool_registry=registry,
        journal_factory=_child_journal_factory(tmp_path),
    )
    event = RuntimeInput(
        event_id="parent-message-consume",
        kind="agent.parent.message",
        correlation_id="child",
        source="qitos.parent",
        payload={"content": "reserved note"},
    )

    async def _post_when_started() -> None:
        while not engine._accepting_runtime_events:
            await asyncio.sleep(0)
        assert await engine.apost_runtime_event(event, run_id="run_childconsume")

    posting = asyncio.create_task(_post_when_started())
    result = await engine.arun("inspect", run_id="run_childconsume")
    await posting
    assert result.state.stop_reason == "completed"
    await engine.aclose()

    records = await _read_child_records(tmp_path, "run_childconsume")
    record_types = [record.type for record in records]
    posted = record_types.index(JournalRecordType.RUNTIME_INPUT_POSTED)
    consumed = record_types.index(JournalRecordType.RUNTIME_INPUT_CONSUMED)
    committed = record_types.index(JournalRecordType.RUN_COMPLETED)
    # Consumption commits after the steered message's turn committed, before
    # the run terminal, and exactly once.
    assert posted < consumed < committed
    assert record_types.count(JournalRecordType.RUNTIME_INPUT_CONSUMED) == 1
    consumed_record = records[consumed]
    assert consumed_record.payload == {"event_id": "parent-message-consume"}


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawal", ["cancel", "timeout"])
async def test_pending_parent_message_withdrawal_leaves_no_reservation_or_record(
    tmp_path,
    withdrawal,
) -> None:
    model_started = asyncio.Event()
    release_model = asyncio.Event()

    async def response(_request):
        model_started.set()
        await release_model.wait()
        for event in text_events("done"):
            yield event

    engine = AgentChildEngine(
        model=ScriptedModel([response]),
        journal_factory=_child_journal_factory(tmp_path),
    )
    event = RuntimeInput(
        event_id=f"parent-message-{withdrawal}",
        kind="agent.parent.message",
        correlation_id="child",
        source="qitos.parent",
        payload={"content": "withdraw this note"},
    )
    running = asyncio.create_task(
        engine.arun("inspect", run_id=f"run_child_{withdrawal}")
    )
    await asyncio.wait_for(model_started.wait(), timeout=1)

    if withdrawal == "timeout":
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                engine.apost_runtime_event(
                    event,
                    run_id=f"run_child_{withdrawal}",
                ),
                timeout=0.01,
            )
    else:
        posting = asyncio.create_task(
            engine.apost_runtime_event(
                event,
                run_id=f"run_child_{withdrawal}",
            )
        )
        while not engine._runtime_event_reservations:
            await asyncio.sleep(0)
        posting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await posting

    release_model.set()
    result = await asyncio.wait_for(running, timeout=1)
    assert result.state.stop_reason == "completed"
    await engine.aclose()

    records = await _read_child_records(tmp_path, f"run_child_{withdrawal}")
    assert JournalRecordType.RUNTIME_INPUT_POSTED not in {
        record.type for record in records
    }


@pytest.mark.asyncio
async def test_parent_message_append_commit_wins_caller_cancellation(tmp_path) -> None:
    model_started = asyncio.Event()
    release_model = asyncio.Event()
    runtime_append_started = asyncio.Event()

    async def first_response(_request):
        model_started.set()
        await release_model.wait()
        for event in text_events("first answer"):
            yield event

    class BlockingRuntimeJournal(JsonlSessionJournal):
        async def append(self, record_type, payload, *, record_id):
            if record_type is JournalRecordType.RUNTIME_INPUT_POSTED:
                position = await super().append(
                    record_type,
                    payload,
                    record_id=record_id,
                )
                runtime_append_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as cancellation:
                    raise JournalAppendCancelled(position) from cancellation
                return position
            return await super().append(record_type, payload, record_id=record_id)

    model = ScriptedModel([first_response, text_events("continued")])
    engine = AgentChildEngine(
        model=model,
        journal_factory=lambda: BlockingRuntimeJournal(_children_root(tmp_path)),
    )
    event = RuntimeInput(
        event_id="parent-message-commit-wins",
        kind="agent.parent.message",
        correlation_id="child",
        source="qitos.parent",
        payload={"content": "committed note"},
    )
    running = asyncio.create_task(
        engine.arun("inspect", run_id="run_child_commit_wins")
    )
    await asyncio.wait_for(model_started.wait(), timeout=1)
    posting = asyncio.create_task(
        engine.apost_runtime_event(event, run_id="run_child_commit_wins")
    )
    release_model.set()
    await asyncio.wait_for(runtime_append_started.wait(), timeout=1)

    posting.cancel()
    assert posting.done() is False
    assert await asyncio.wait_for(posting, timeout=1) is True
    result = await asyncio.wait_for(running, timeout=1)
    assert result.state.stop_reason == "completed"
    assert model.requests[1].messages[-1] == {
        "role": "user",
        "content": "committed note",
    }
    await engine.aclose()

    records = await _read_child_records(tmp_path, "run_child_commit_wins")
    record_types = [record.type for record in records]
    assert record_types.index(JournalRecordType.RUNTIME_INPUT_POSTED) < (
        record_types.index(JournalRecordType.RUN_COMPLETED)
    )


@pytest.mark.asyncio
async def test_parent_message_accepts_committed_journal_error(tmp_path) -> None:
    model_started = asyncio.Event()
    release_model = asyncio.Event()

    async def first_response(_request):
        model_started.set()
        await release_model.wait()
        for event in text_events("first answer"):
            yield event

    class CommittedErrorJournal(JsonlSessionJournal):
        async def append(self, record_type, payload, *, record_id):
            position = await super().append(
                record_type,
                payload,
                record_id=record_id,
            )
            if record_type is JournalRecordType.RUNTIME_INPUT_POSTED:
                raise JournalCommitError(
                    position,
                    JournalCommitState.COMMITTED,
                    cause=OSError("local projection failed after commit"),
                )
            return position

    model = ScriptedModel([first_response, text_events("continued")])
    engine = AgentChildEngine(
        model=model,
        journal_factory=lambda: CommittedErrorJournal(_children_root(tmp_path)),
    )
    event = RuntimeInput(
        event_id="parent-message-commit-error",
        kind="agent.parent.message",
        correlation_id="child",
        source="qitos.parent",
        payload={"content": "committed error note"},
    )
    running = asyncio.create_task(
        engine.arun("inspect", run_id="run_child_commit_error")
    )
    await asyncio.wait_for(model_started.wait(), timeout=1)
    posting = asyncio.create_task(
        engine.apost_runtime_event(event, run_id="run_child_commit_error")
    )
    release_model.set()

    assert await asyncio.wait_for(posting, timeout=1) is True
    result = await asyncio.wait_for(running, timeout=1)
    assert result.state.stop_reason == "completed"
    assert model.requests[1].messages[-1] == {
        "role": "user",
        "content": "committed error note",
    }
    await engine.aclose()

    records = await _read_child_records(tmp_path, "run_child_commit_error")
    assert sum(
        record.type is JournalRecordType.RUNTIME_INPUT_POSTED for record in records
    ) == 1


@pytest.mark.asyncio
async def test_parent_message_append_rollback_preserves_caller_timeout(
    tmp_path,
) -> None:
    model_started = asyncio.Event()
    release_model = asyncio.Event()
    runtime_append_started = asyncio.Event()

    async def response(_request):
        model_started.set()
        await release_model.wait()
        for event in text_events("done"):
            yield event

    class RollingBackRuntimeJournal(JsonlSessionJournal):
        async def append(self, record_type, payload, *, record_id):
            if record_type is JournalRecordType.RUNTIME_INPUT_POSTED:
                runtime_append_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as cancellation:
                    raise JournalAppendCancelled(None) from cancellation
            return await super().append(record_type, payload, record_id=record_id)

    engine = AgentChildEngine(
        model=ScriptedModel([response]),
        journal_factory=lambda: RollingBackRuntimeJournal(
            _children_root(tmp_path)
        ),
    )
    event = RuntimeInput(
        event_id="parent-message-rollback",
        kind="agent.parent.message",
        correlation_id="child",
        source="qitos.parent",
        payload={"content": "rolled back note"},
    )
    running = asyncio.create_task(
        engine.arun("inspect", run_id="run_child_rollback")
    )
    await asyncio.wait_for(model_started.wait(), timeout=1)
    posting = asyncio.create_task(
        engine.apost_runtime_event(event, run_id="run_child_rollback")
    )
    release_model.set()
    await asyncio.wait_for(runtime_append_started.wait(), timeout=1)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(posting, timeout=0.01)
    result = await asyncio.wait_for(running, timeout=1)
    assert result.state.stop_reason == "completed"
    await engine.aclose()

    records = await _read_child_records(tmp_path, "run_child_rollback")
    assert JournalRecordType.RUNTIME_INPUT_POSTED not in {
        record.type for record in records
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_case", "expected_reason"),
    [
        ("model_failure", "failed:model failed"),
        ("max_turns", "completed"),
        ("cancel", "cancelled"),
        ("budget", "budget_tokens"),
    ],
)
async def test_reserved_parent_message_is_rejected_before_terminal_outcomes(
    tmp_path,
    terminal_case,
    expected_reason,
) -> None:
    model_started = asyncio.Event()
    release_model = asyncio.Event()
    runtime_append_started = asyncio.Event()
    release_runtime_append = asyncio.Event()
    turn_boundary_started = asyncio.Event()

    async def first_response(_request):
        model_started.set()
        await release_model.wait()
        if terminal_case == "model_failure":
            yield ModelStreamEvent(
                type=ModelStreamEventType.FAILED,
                error="model failed",
            )
            return
        for event in text_events(
            "terminal answer",
            usage={"total_tokens": 1} if terminal_case == "budget" else None,
        ):
            yield event

    class BlockingRuntimeJournal(JsonlSessionJournal):
        async def append(self, record_type, payload, *, record_id):
            if record_type is JournalRecordType.RUNTIME_INPUT_POSTED:
                runtime_append_started.set()
                await release_runtime_append.wait()
            return await super().append(record_type, payload, record_id=record_id)

    class BoundaryObservedEngine(AgentChildEngine):
        def _close_runtime_event_admission(self):
            turn_boundary_started.set()
            super()._close_runtime_event_admission()

    engine = BoundaryObservedEngine(
        model=ScriptedModel([first_response]),
        max_turns=1 if terminal_case == "max_turns" else None,
        budget=(
            TaskBudget(max_tokens=1)
            if terminal_case == "budget"
            else TaskBudget()
        ),
        journal_factory=lambda: BlockingRuntimeJournal(_children_root(tmp_path)),
    )
    event = RuntimeInput(
        event_id=f"parent-message-{terminal_case}",
        kind="agent.parent.message",
        correlation_id="child",
        source="qitos.parent",
        payload={"content": "terminal note"},
    )

    running = asyncio.create_task(
        engine.arun("inspect", run_id=f"run_child_{terminal_case}")
    )
    await asyncio.wait_for(model_started.wait(), timeout=1)
    posting = asyncio.create_task(
        engine.apost_runtime_event(
            event,
            run_id=f"run_child_{terminal_case}",
        )
    )
    if terminal_case == "cancel":
        engine.cancel("immediate")
    release_model.set()
    await asyncio.wait_for(turn_boundary_started.wait(), timeout=1)

    assert await asyncio.wait_for(posting, timeout=1) is False
    assert runtime_append_started.is_set() is False
    result = await asyncio.wait_for(running, timeout=1)
    assert result.state.stop_reason == expected_reason
    await engine.aclose()

    records = await _read_child_records(tmp_path, f"run_child_{terminal_case}")
    record_types = [record.type for record in records]
    assert JournalRecordType.RUNTIME_INPUT_POSTED not in record_types


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
async def test_recovery_preserves_root_budget_stop_before_parent_terminal(
    tmp_path,
) -> None:
    engine = AgentChildEngine(
        model=ScriptedModel([text_events("answer", usage={"total_tokens": 10})]),
        journal_factory=_child_journal_factory(tmp_path),
    )
    completed = await engine.arun("inspect", run_id="run_childbudget")
    await engine.aclose()
    assert completed.state.stop_reason == "completed"

    ledger = BudgetLedger(max_tokens=1)
    await ledger.commit(
        origin_run_id="run_childbudget",
        transaction_id="run_childbudget:turn:0:model",
        tokens=10,
        cost_usd=0.0,
        usage_complete=True,
        cost_complete=False,
    )
    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})
    request = _request("inspect")
    handle = ChildHandle(child_id="child-budget", parent_run_id="parent-run")
    await parent_journal.append(
        JournalRecordType.CHILD_STARTED,
        {
            "handle": handle.to_dict(),
            "request": request.to_dict(),
            "child_run_id": "run_childbudget",
        },
        record_id="parent-run:child:child-budget:started",
    )
    supervisor = ChildSupervisor(
        invocation_factory=_unused_factory,
        child_journal_factory=_child_journal_factory(tmp_path),
    )

    recovered = await supervisor.recover(
        parent_run_id="parent-run",
        journal=parent_journal,
        budget_ledger=ledger,
    )

    assert len(recovered) == 1
    assert recovered[0].status is ChildStatus.BUDGET_EXHAUSTED
    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_recovery_attributes_root_budget_to_crossing_child_only(
    tmp_path,
) -> None:
    for run_id, answer, tokens in (
        ("run_childa", "answer-a", 4),
        ("run_childb", "answer-b", 7),
    ):
        engine = AgentChildEngine(
            model=ScriptedModel(
                [text_events(answer, usage={"total_tokens": tokens})]
            ),
            journal_factory=_child_journal_factory(tmp_path),
        )
        completed = await engine.arun("inspect", run_id=run_id)
        await engine.aclose()
        assert completed.state.stop_reason == "completed"

    ledger = BudgetLedger(max_tokens=10)
    for run_id, tokens in (("run_childa", 4), ("run_childb", 7)):
        await ledger.commit(
            origin_run_id=run_id,
            transaction_id=f"{run_id}:turn:0:model",
            tokens=tokens,
            cost_usd=0.0,
            usage_complete=True,
            cost_complete=False,
        )
    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})
    for suffix in ("a", "b"):
        request = _request(f"inspect-{suffix}")
        handle = ChildHandle(
            child_id=f"child-{suffix}",
            parent_run_id="parent-run",
        )
        await parent_journal.append(
            JournalRecordType.CHILD_STARTED,
            {
                "handle": handle.to_dict(),
                "request": request.to_dict(),
                "child_run_id": f"run_child{suffix}",
            },
            record_id=f"parent-run:child:child-{suffix}:started",
        )
    supervisor = ChildSupervisor(
        invocation_factory=_unused_factory,
        child_journal_factory=_child_journal_factory(tmp_path),
    )

    recovered = await supervisor.recover(
        parent_run_id="parent-run",
        journal=parent_journal,
        budget_ledger=ledger,
    )

    by_child = {result.handle.child_id: result for result in recovered}
    assert by_child["child-a"].status is ChildStatus.COMPLETED
    assert by_child["child-b"].status is ChildStatus.BUDGET_EXHAUSTED
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
        _context(parent_tool_authority=registry.freeze()),
        background=False,
    )

    assert result.status is ChildStatus.COMPLETED
    assert "echo" in model.seen_tool_names
    assert "other" not in model.seen_tool_names
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_child_tool_exposure_fails_closed_without_parent_authority(
    tmp_path,
) -> None:
    registry = ToolRegistry().register(_echo)

    class RecordingModel(ScriptedModel):
        seen_tool_names: set[str] = set()

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

    result = await supervisor.launch(_request("inspect"), _context(), background=False)

    assert result.status is ChildStatus.COMPLETED
    assert model.seen_tool_names == set()
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_child_tool_exposure_cannot_exceed_parent_frozen_authority(
    tmp_path,
) -> None:
    registry = ToolRegistry().register(_echo)

    @tool(name="privileged", group="network")
    def _privileged() -> str:
        return "privileged"

    registry.register(_privileged)
    parent_authority = ToolRegistry().register(_echo).freeze()

    class RecordingModel(ScriptedModel):
        seen_tool_names: set[str] = set()

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
        _request("inspect", allowed_tool_groups=("network",)),
        _context(parent_tool_authority=parent_authority),
        background=False,
    )

    assert result.status is ChildStatus.COMPLETED
    assert model.seen_tool_names == set()
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_agent_tool_preserves_parent_parameter_scope_denial(tmp_path) -> None:
    executions: list[str] = []

    @tool(
        name="read_scoped",
        rule_scope_builder=lambda args: str(args.get("path") or ""),
    )
    def _read_scoped(path: str) -> str:
        executions.append(path)
        return path

    registry = ToolRegistry().register(_read_scoped)
    permission_context = ToolPermissionContext(
        deny_rules=[
            ToolPermissionRule(
                effect="deny",
                tool_name="read_scoped",
                scope="/secret",
                message="outside parent scope",
            )
        ]
    )
    model = ScriptedModel(
        [
            tool_events(
                [tool_call_wire("c1", "read_scoped", {"path": "/secret"})]
            ),
            text_events("done"),
        ]
    )
    agent_tool = AgentTool(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_children_root(tmp_path),
        )
    )

    result = await agent_tool.execute(
        {"description": "inspect", "prompt": "inspect"},
        runtime_context={
            "run_id": "parent-run",
            "tool_registry": registry.freeze(),
            "permission_context": permission_context,
        },
    )

    assert result.output["child_status"] == ChildStatus.COMPLETED.value
    assert executions == []
    records = await _read_child_records(tmp_path, result.output["run_id"])
    denied = next(
        record
        for record in records
        if record.type is JournalRecordType.TOOL_TERMINAL
    )
    denied_result = _terminal_result_payload(records, denied)
    assert denied_result["status"] == "denied"
    assert denied_result["metadata"]["permission_scope"] == "/secret"
    await agent_tool.aclose()


@pytest.mark.asyncio
async def test_builtin_child_factory_rejects_different_env_permission_policy(
    tmp_path,
) -> None:
    model_factory_calls = 0

    def model_factory() -> ScriptedModel:
        nonlocal model_factory_calls
        model_factory_calls += 1
        return ScriptedModel([text_events("must not run")])

    class WiderEnv:
        tool_permission_context = ToolPermissionContext(default_decision="allow")

    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model_factory,
            env=WiderEnv(),
            journal_directory=_children_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("inspect"),
        _context(
            parent_permission_context=ToolPermissionContext(
                default_decision="deny"
            )
        ),
        background=False,
    )

    assert result.status is ChildStatus.FAILED
    assert "cannot prove" in result.error
    assert model_factory_calls == 0
    await supervisor.aclose()


@pytest.mark.parametrize(
    ("base_result", "parent_result"),
    [("configured-safe", "parent-privileged"), ("configured-privileged", "parent-safe")],
)
@pytest.mark.asyncio
async def test_child_tool_name_collision_exposes_and_executes_neither_definition(
    tmp_path,
    base_result: str,
    parent_result: str,
) -> None:
    executions: list[str] = []

    @tool(name="shared")
    def _configured() -> str:
        executions.append(base_result)
        return base_result

    @tool(name="shared")
    def _parent() -> str:
        executions.append(parent_result)
        return parent_result

    configured_registry = ToolRegistry().register(_configured)
    parent_authority = ToolRegistry().register(_parent).freeze()

    class RecordingModel(ScriptedModel):
        seen_tool_names: set[str] = set()

        def build_tool_schema_request_options(
            self, payload, *, protocol=None, delivery="prompt_injection"
        ):
            self.seen_tool_names.update(
                item.get("function", {}).get("name") for item in payload
            )
            return {"tools": list(payload)}

    model = RecordingModel(
        [
            tool_events([tool_call_wire("c1", "shared", {})]),
            text_events("done"),
        ]
    )
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            tool_registry=configured_registry,
            journal_directory=_children_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("collision"),
        _context(parent_tool_authority=parent_authority),
        background=False,
    )

    assert result.status is ChildStatus.COMPLETED
    assert model.seen_tool_names == set()
    assert executions == []
    await supervisor.aclose()


@pytest.mark.parametrize(
    "launch_request",
    [
        _request("inspect", profile="restricted"),
        _request("inspect", working_directory="workspace"),
    ],
)
@pytest.mark.asyncio
async def test_builtin_child_factory_rejects_unresolved_runtime_policy(
    tmp_path,
    launch_request: ChildLaunchRequest,
) -> None:
    model_factory_calls = 0

    def model_factory() -> ScriptedModel:
        nonlocal model_factory_calls
        model_factory_calls += 1
        return ScriptedModel([text_events("must not run")])

    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model_factory,
            journal_directory=_children_root(tmp_path),
        )
    )

    result = await supervisor.launch(launch_request, _context(), background=False)

    assert result.status is ChildStatus.FAILED
    assert model_factory_calls == 0
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_child_model_usage_consumes_root_budget_and_closes_admission(
    tmp_path,
) -> None:
    ledger = BudgetLedger(max_tokens=1)
    model = ScriptedModel(
        [
            text_events(
                "over budget",
                usage={
                    "prompt_tokens": 6,
                    "completion_tokens": 4,
                    "total_tokens": 10,
                },
            )
        ]
    )
    base_factory = build_agent_child_invocation_factory(
        model=model,
        journal_directory=_children_root(tmp_path),
    )
    factory_calls = 0

    async def counting_factory(request, context):
        nonlocal factory_calls
        factory_calls += 1
        return await base_factory(request, context)

    supervisor = ChildSupervisor(invocation_factory=counting_factory)
    context = _context(budget_ledger=ledger)

    first = await supervisor.launch(_request("first"), context, background=False)
    second = await supervisor.launch(_request("second"), context, background=False)

    assert first.status is ChildStatus.BUDGET_EXHAUSTED
    assert second.status is ChildStatus.BUDGET_EXHAUSTED
    assert factory_calls == 1
    assert len(model.requests) == 1
    snapshot = ledger.snapshot()
    assert snapshot.total_tokens == 10
    assert snapshot.tokens_exhausted is True
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_immediate_interrupt_cannot_cancel_model_budget_commit(tmp_path) -> None:
    commit_started = asyncio.Event()
    release_commit = asyncio.Event()

    class BlockingBudgetLedger(BudgetLedger):
        async def commit(self, **kwargs):
            commit_started.set()
            await release_commit.wait()
            return await super().commit(**kwargs)

    ledger = BlockingBudgetLedger()
    model = ScriptedModel(
        [text_events("answer", usage={"total_tokens": 10})]
    )
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            journal_directory=_children_root(tmp_path),
        ),
        child_journal_factory=_child_journal_factory(tmp_path),
    )
    launched = await supervisor.launch(
        _request("interrupt"),
        _context(budget_ledger=ledger),
        background=True,
    )
    await asyncio.wait_for(commit_started.wait(), timeout=1)

    assert supervisor.request_interrupt(launched.handle) is True
    release_commit.set()
    terminal = await supervisor.wait(launched.handle, timeout_seconds=1)

    assert terminal is not None
    assert terminal.status is ChildStatus.CANCELLED
    assert ledger.snapshot().total_tokens == 10
    records = await _read_child_records(tmp_path, terminal.child_run_id)
    assert JournalRecordType.MODEL_COMPLETED in {record.type for record in records}
    assert records[-1].type is JournalRecordType.RUN_INTERRUPTED
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_root_budget_is_durable_before_child_model_terminal(tmp_path) -> None:
    class FailingModelJournal(JsonlSessionJournal):
        async def append(self, record_type, payload, *, record_id):
            if record_type is JournalRecordType.MODEL_COMPLETED:
                raise JournalError("model terminal failed")
            return await super().append(record_type, payload, record_id=record_id)

    root_journal = JsonlSessionJournal(tmp_path / "root")
    await root_journal.create("root-run", {})
    ledger = BudgetLedger()
    ledger.attach(root_journal, root_run_id="root-run")
    engine = AgentChildEngine(
        model=ScriptedModel(
            [text_events("answer", usage={"total_tokens": 10})]
        ),
        budget_ledger=ledger,
        journal_factory=lambda: FailingModelJournal(_children_root(tmp_path)),
    )

    with pytest.raises(JournalError, match="model terminal failed"):
        await engine.arun("inspect", run_id="run_childcrash")
    await engine.aclose()

    assert ledger.snapshot().total_tokens == 10
    root_records = await root_journal.replay()
    assert JournalRecordType.BUDGET_COMMITTED in {
        record.type for record in root_records
    }
    child_records = await _read_child_records(tmp_path, "run_childcrash")
    assert JournalRecordType.MODEL_COMPLETED not in {
        record.type for record in child_records
    }
    await root_journal.close()


@pytest.mark.asyncio
async def test_child_token_budget_blocks_tool_side_effects_in_exhausting_turn(
    tmp_path,
) -> None:
    executions = 0

    @tool(name="mutate")
    def _mutate() -> str:
        nonlocal executions
        executions += 1
        return "mutated"

    registry = ToolRegistry().register(_mutate)
    model = ScriptedModel(
        [
            [
                ModelStreamEvent(
                    type=ModelStreamEventType.COMPLETED,
                    finish_reason="tool_calls",
                    tool_calls=[tool_call_wire("c1", "mutate", {})],
                    usage={"total_tokens": 10},
                )
            ]
        ]
    )
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_children_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("mutate", budget=TaskBudget(max_tokens=1)),
        _context(parent_tool_authority=registry.freeze()),
        background=False,
    )

    assert result.status is ChildStatus.BUDGET_EXHAUSTED
    assert executions == 0
    child_records = await _read_child_records(tmp_path, result.child_run_id)
    terminal = next(
        record
        for record in child_records
        if record.type is JournalRecordType.TOOL_TERMINAL
    )
    assert _terminal_result_payload(child_records, terminal)["status"] == "denied"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_child_cost_budget_uses_explicit_frozen_model_pricing(tmp_path) -> None:
    pricing = ModelPricing(
        input_usd_per_million=1.0,
        output_usd_per_million=2.0,
    )
    ledger = BudgetLedger(max_cost_usd=0.000005)
    model = ScriptedModel(
        [
            text_events(
                "over cost",
                usage={
                    "input_tokens": 2,
                    "output_tokens": 2,
                    "total_tokens": 4,
                },
            )
        ]
    )
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            model_pricing=pricing,
            journal_directory=_children_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("cost", budget=TaskBudget(max_cost_usd=0.000005)),
        _context(budget_ledger=ledger),
        background=False,
    )

    assert result.status is ChildStatus.BUDGET_EXHAUSTED
    assert result.total_cost_usd == pytest.approx(0.000006)
    snapshot = ledger.snapshot()
    assert snapshot.total_cost_usd == pytest.approx(0.000006)
    assert snapshot.cost_complete is True
    assert snapshot.cost_exhausted is True
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_child_cost_budget_without_pricing_fails_before_model_construction(
    tmp_path,
) -> None:
    model_factory_calls = 0

    def model_factory() -> ScriptedModel:
        nonlocal model_factory_calls
        model_factory_calls += 1
        return ScriptedModel([text_events("must not run")])

    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model_factory,
            journal_directory=_children_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("cost", budget=TaskBudget(max_cost_usd=1.0)),
        _context(),
        background=False,
    )

    assert result.status is ChildStatus.FAILED
    assert model_factory_calls == 0
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_child_budget_narrows_nested_max_children_runtime_context(
    tmp_path,
) -> None:
    observed: list[int] = []

    @tool(name="capture_budget")
    def _capture_budget(runtime_context=None) -> str:
        observed.append(runtime_context["max_children"])
        return "captured"

    registry = ToolRegistry().register(_capture_budget)
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "capture_budget", {})]),
            text_events("done"),
        ]
    )
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_children_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("nested", budget=TaskBudget(max_children=2)),
        _context(max_children=3, parent_tool_authority=registry.freeze()),
        background=False,
    )

    assert result.status is ChildStatus.COMPLETED
    assert observed == [2]
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_child_runtime_exposes_its_mailbox_for_descendant_completion(
    tmp_path,
) -> None:
    observed: list[bool] = []

    @tool(name="capture_parent_mailbox")
    def _capture_parent_mailbox(runtime_context=None) -> str:
        observed.append(callable(runtime_context.get("post_runtime_event")))
        return "captured"

    registry = ToolRegistry().register(_capture_parent_mailbox)
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "capture_parent_mailbox", {})]),
            text_events("done"),
        ]
    )
    supervisor = ChildSupervisor(
        invocation_factory=build_agent_child_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_children_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("nested completion"),
        _context(parent_tool_authority=registry.freeze()),
        background=False,
    )

    assert result.status is ChildStatus.COMPLETED
    assert observed == [True]
    await supervisor.aclose()


async def _unused_factory(_request, _context):
    raise AssertionError("recovery must not construct a child")
