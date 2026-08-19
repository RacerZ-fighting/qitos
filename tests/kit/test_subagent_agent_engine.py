"""Façade-driven Subagents: launch, interrupt, mailbox and journal recovery."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

import pytest

from qitos.core.budget import BudgetLedger
from qitos.core.agent_loop import NextTurnUpdate
from qitos.core.subagent import (
    SubagentHandle,
    SubagentLaunchContext,
    SubagentLaunchRequest,
    SubagentPersistenceError,
    SubagentStatus,
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
from qitos.core.message import ContextMessage, UserMessage
from qitos.core.task import Task, TaskBudget, TaskReference, TaskStatus
from qitos.core.tool import ToolPermissionContext, ToolPermissionRule, tool
from qitos.core.tool_executor import AfterToolCallOverride
from qitos.core.tool_registry import ToolRegistry
from qitos.core.runtime_input import RuntimeInput
from qitos.kit.subagent import (
    AgentSubagentEngine,
    SubagentSupervisor,
    build_agent_subagent_invocation_factory,
)
from qitos.kit.subagent.agent_engine import subagent_final_text
from qitos.kit.journal import (
    JsonlSessionJournal,
    recover_run_outcome,
    recover_session,
)
from qitos.kit.journal.turn_recorder import encode_task_created
from qitos.kit.tool.subagent import SubagentTool

from tests.core.agent_fakes import (
    ScriptedModel,
    failed_events,
    text_events,
    tool_call_wire,
    tool_events,
)


@tool(name="echo")
def _echo(text: str) -> str:
    return f"echo:{text}"


def _request(task: str, **kwargs: object) -> SubagentLaunchRequest:
    return SubagentLaunchRequest(task=task, description=f"{task} task", **kwargs)


def _context(**kwargs: object) -> SubagentLaunchContext:
    return SubagentLaunchContext(parent_run_id="parent-run", **kwargs)


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


def _subagents_root(tmp_path):
    return tmp_path / "subagents"


def _subagent_journal_factory(tmp_path):
    return lambda: JsonlSessionJournal(_subagents_root(tmp_path))


async def _read_subagent_records(tmp_path, run_id: str):
    journal = JsonlSessionJournal(_subagents_root(tmp_path))
    await journal.open(run_id)
    try:
        return await journal.replay()
    finally:
        await journal.close()


def _model_requests_contain_note(model, note: str) -> bool:
    """Whether any request after the first carries the note as user input."""

    return any(
        isinstance(message, Mapping)
        and message.get("role") == "user"
        and note in str(message.get("content"))
        for request in model.requests[1:]
        for message in request.messages
    )


@pytest.mark.asyncio
async def test_foreground_subagent_completes_and_journals_turns(tmp_path) -> None:
    model = ScriptedModel(
        [text_events("subagent answer", usage={"total_tokens": 7})]
    )
    finalized_run_ids: list[str] = []

    async def _finalize(run_id: str) -> None:
        finalized_run_ids.append(run_id)

    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            journal_directory=_subagents_root(tmp_path),
            trace_directory=tmp_path / "traces",
            run_finalizer=_finalize,
        ),
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )
    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})

    result = await supervisor.launch(
        _request("inspect"),
        _context(journal=parent_journal),
        background=False,
    )

    assert result.status is SubagentStatus.COMPLETED
    assert result.conclusion.summary == "subagent answer"
    assert result.steps == 1
    assert result.total_tokens == 7
    assert result.usage_complete is True
    assert result.cost_complete is False
    assert result.subagent_run_id
    assert finalized_run_ids == [result.subagent_run_id]

    parent_records = await parent_journal.replay()
    subagent_started = [
        record
        for record in parent_records
        if record.type is JournalRecordType.SUBAGENT_STARTED
    ]
    subagent_terminal = [
        record
        for record in parent_records
        if record.type is JournalRecordType.SUBAGENT_TERMINAL
    ]
    assert len(subagent_started) == len(subagent_terminal) == 1
    assert subagent_terminal[0].payload["status"] == SubagentStatus.COMPLETED.value

    subagent_records = await _read_subagent_records(tmp_path, result.subagent_run_id)
    subagent_types = [record.type for record in subagent_records]
    assert subagent_types[0] is JournalRecordType.RUN_STARTED
    assert subagent_types.index(JournalRecordType.TASK_CREATED) < subagent_types.index(
        JournalRecordType.INPUT_ACCEPTED
    )
    assert JournalRecordType.MODEL_COMPLETED in subagent_types
    assert JournalRecordType.STEP_COMMITTED in subagent_types
    assert subagent_types.index(JournalRecordType.TASK_TRANSITION) < subagent_types.index(
        JournalRecordType.RUN_COMPLETED
    )
    assert subagent_types[-1] is JournalRecordType.RUN_COMPLETED
    terminal = subagent_records[-1]
    assert terminal.record_id == f"{result.subagent_run_id}:run:terminal"
    assert terminal.payload["status"] == "completed"
    recovered = recover_session(subagent_records)
    task = next(iter(recovered.tasks.values()))
    assert task.lifecycle.status is TaskStatus.COMPLETED
    assert task.lifecycle.usage is not None
    assert task.lifecycle.usage.total_tokens == result.total_tokens
    manifest = json.loads(
        (tmp_path / "traces" / result.subagent_run_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "completed"
    assert manifest["parent_run_id"] == "parent-run"
    assert manifest["step_count"] == result.steps
    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_subagent_traces_preserve_recursive_run_lineage(tmp_path) -> None:
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=ScriptedModel(
                [text_events("first subagent"), text_events("grandchild")]
            ),
            journal_directory=_subagents_root(tmp_path),
            trace_directory=tmp_path / "traces",
        ),
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )

    child = await supervisor.launch(
        _request("first"),
        _context(),
        background=False,
    )
    grandchild = await supervisor.launch(
        _request("nested"),
        SubagentLaunchContext(parent_run_id=child.subagent_run_id),
        background=False,
    )

    child_manifest = json.loads(
        (tmp_path / "traces" / child.subagent_run_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    grandchild_manifest = json.loads(
        (
            tmp_path
            / "traces"
            / grandchild.subagent_run_id
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert child_manifest["parent_run_id"] == "parent-run"
    assert grandchild_manifest["parent_run_id"] == child.subagent_run_id
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_subagent_trace_lineage_does_not_require_a_run_journal(tmp_path) -> None:
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=ScriptedModel([text_events("subagent result")]),
            trace_directory=tmp_path / "traces",
        ),
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )

    result = await supervisor.launch(
        _request("trace only"),
        _context(),
        background=False,
    )

    manifest = json.loads(
        (tmp_path / "traces" / result.subagent_run_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["parent_run_id"] == "parent-run"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_failed_subagent_finalizes_its_own_trace(tmp_path) -> None:
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=ScriptedModel([failed_events("provider failed")]),
            journal_directory=_subagents_root(tmp_path),
            trace_directory=tmp_path / "traces",
        ),
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )

    result = await supervisor.launch(
        _request("fail"),
        _context(),
        background=False,
    )

    manifest = json.loads(
        (tmp_path / "traces" / result.subagent_run_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.status is SubagentStatus.FAILED
    assert manifest["status"] == "failed"
    assert manifest["parent_run_id"] == "parent-run"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_empty_subagent_answer_gets_one_same_context_conclusion_follow_up(
    tmp_path,
) -> None:
    model = ScriptedModel([text_events(""), text_events("final conclusion")])
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            journal_directory=_subagents_root(tmp_path),
            max_turns=3,
        )
    )

    result = await supervisor.launch(
        _request("inspect"),
        _context(),
        background=False,
    )

    assert result.status is SubagentStatus.COMPLETED
    assert result.conclusion.summary == "final conclusion"
    assert result.steps == 2
    assert len(model.requests) == 2
    follow_up_messages = [
        message
        for message in model.requests[1].message_dicts()
        if message.get("role") == "user"
    ]
    assert len(follow_up_messages) == 2
    assert str(follow_up_messages[-1]["content"]).strip()
    assert follow_up_messages[-1]["content"] != follow_up_messages[0]["content"]
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_empty_conclusion_retry_fails_without_fabricating_tool_output(
    tmp_path,
) -> None:
    model = ScriptedModel(
        [
            tool_events(
                [tool_call_wire("c1", "echo", {"text": "evidence"})],
                text="incidental progress text",
            ),
            text_events(""),
            text_events(""),
        ]
    )
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=ToolRegistry().register(_echo),
            journal_directory=_subagents_root(tmp_path),
            max_turns=4,
        )
    )

    result = await supervisor.launch(
        _request("inspect"),
        _context(),
        background=False,
    )

    assert result.status is SubagentStatus.FAILED
    assert result.conclusion.summary == ""
    assert result.error is not None
    assert "final answer" in result.error
    assert len(model.requests) == 3
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_subagent_tool_threads_parent_task_domain_constraints(tmp_path) -> None:
    parent_journal = JsonlSessionJournal(tmp_path / "parent-plan")
    await parent_journal.create("parent-run", {})
    parent_task = Task(
        task_id="parent-task",
        objective="Parent work",
        constraints={"scope": "primary"},
        references=(
            TaskReference(kind="artifact", uri="scope://engagement/primary"),
        ),
    )
    await parent_journal.append(
        JournalRecordType.TASK_CREATED,
        encode_task_created(parent_task),
        record_id="parent-run:task:parent-task:created",
    )
    subagent_tool = SubagentTool(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=ScriptedModel([text_events("subagent answer")]),
            journal_directory=_subagents_root(tmp_path),
        )
    )
    permission = ToolPermissionContext(
        default_decision="deny",
        allow_rules=(
            ToolPermissionRule(effect="allow", tool_family="filesystem.read"),
        ),
    )

    result = await subagent_tool.execute(
        {
            "description": "inspect",
            "prompt": "inspect",
            "success_criteria": ["Return the inspection result"],
        },
        runtime_context={
            "run_id": "parent-run",
            "task_id": "parent-task",
            "task": parent_task,
            "permission_context": permission,
            "journal": parent_journal,
        },
    )
    started = next(
        record
        for record in await parent_journal.replay()
        if record.type is JournalRecordType.SUBAGENT_STARTED
    )

    assert result.output["subagent_status"] == SubagentStatus.COMPLETED.value
    assert started.payload["request"]["parent_task_id"] == "parent-task"
    assert "plan_assignment" not in started.payload["request"]
    subagent_records = await _read_subagent_records(
        tmp_path,
        result.output["subagent_run_id"],
    )
    subagent_task = next(iter(recover_session(subagent_records).tasks.values())).definition
    assert subagent_task.success_criteria == ("Return the inspection result",)
    assert subagent_task.constraints == parent_task.constraints
    assert subagent_task.references == parent_task.references
    assert started.payload["request"]["permission_context"] == permission.to_dict()
    await subagent_tool.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_subagent_tool_evidence_commits_in_order(tmp_path) -> None:
    registry = ToolRegistry().register(_echo)
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "ping"})]),
            text_events("done with echo:ping"),
        ]
    )
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_subagents_root(tmp_path),
        ),
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )

    result = await supervisor.launch(
        _request("use the tool"),
        _context(parent_tool_authority=registry.freeze()),
        background=False,
    )

    assert result.status is SubagentStatus.COMPLETED
    assert result.steps == 2
    assert result.conclusion.summary == "done with echo:ping"
    subagent_records = await _read_subagent_records(tmp_path, result.subagent_run_id)
    tool_terminal = [
        record
        for record in subagent_records
        if record.type is JournalRecordType.TOOL_TERMINAL
    ]
    assert len(tool_terminal) == 1
    assert (
        tool_terminal[0].record_id
        == f"{result.subagent_run_id}:turn:0:tool:c1:terminal"
    )
    assert _terminal_result_payload(subagent_records, tool_terminal[0])["output"] == (
        "echo:ping"
    )
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_subagent_engine_composes_product_turn_hooks(tmp_path) -> None:
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "ping"})]),
            text_events("done"),
        ]
    )
    observed: list[str] = []
    prepared_context_turns: list[int] = []

    def transform(messages):
        return [*messages, UserMessage(content="projected subagent state")]

    def prepare_context(context):
        prepared_context_turns.append(context.turn)
        return (ContextMessage(content=f"durable subagent state {context.turn}"),)

    def before(context):
        observed.append(f"before:{context.tool_call.name}")
        return None

    def after(context):
        observed.append(f"after:{context.tool_call.name}")
        return AfterToolCallOverride(output="rewritten subagent result")

    def prepare(context):
        observed.append(f"prepare:{context.turn}")
        return NextTurnUpdate(system_prompt="updated subagent instructions")

    def should_stop(context):
        observed.append(f"stop:{context.turn}")
        return context.turn == 1

    engine = AgentSubagentEngine(
        model=model,
        tool_registry=ToolRegistry().register(_echo),
        system_prompt="initial subagent instructions",
        transform_context=transform,
        prepare_turn_context=prepare_context,
        before_tool_call=before,
        after_tool_call=after,
        prepare_next_turn=prepare,
        should_stop_after_turn=should_stop,
        journal_factory=_subagent_journal_factory(tmp_path),
    )

    result = await engine.arun("inspect", run_id="run_subagent_hooks")

    assert result.state.stop_reason == "completed"
    assert "projected subagent state" in {
        message.get("content") for message in model.requests[0].messages
    }
    assert "durable subagent state 0" in {
        message.get("content") for message in model.requests[0].messages
    }
    assert prepared_context_turns == [0, 1]
    assert model.requests[1].messages[0] == {
        "role": "system",
        "content": "updated subagent instructions",
    }
    assert any(
        message.get("role") == "tool"
        and "rewritten subagent result" in str(message.get("content"))
        for message in model.requests[1].messages
    )
    assert observed == [
        "before:echo",
        "after:echo",
        "prepare:0",
        "stop:0",
        "prepare:1",
        "stop:1",
    ]
    await engine.aclose()


@pytest.mark.asyncio
async def test_subagent_policy_can_follow_up_and_terminalize_its_task(tmp_path) -> None:
    model = ScriptedModel(
        [text_events("premature answer"), text_events("finished answer")]
    )
    engine: AgentSubagentEngine

    def prepare(context):
        if context.turn == 0:
            engine.follow_up(UserMessage(content="finish the committed work"))

    async def should_stop(context):
        if context.turn != 1:
            return False
        await engine.complete_task("subagent completion policy passed")
        return True

    engine = AgentSubagentEngine(
        model=model,
        prepare_next_turn=prepare,
        should_stop_after_turn=should_stop,
        journal_factory=_subagent_journal_factory(tmp_path),
        subagent_task=Task(
            task_id="subagent-policy",
            parent_task_id="parent-task",
            objective="finish committed work",
        ),
    )

    result = await engine.arun("inspect", run_id="run_subagent_policy")

    assert result.step_count == 2
    assert any(
        message.get("role") == "user"
        and "finish the committed work" in str(message.get("content"))
        for message in model.requests[1].messages
    )
    await engine.aclose()
    records = await _read_subagent_records(tmp_path, "run_subagent_policy")
    types = [record.type for record in records]
    assert types.count(JournalRecordType.TASK_TRANSITION) == 1
    assert types.index(JournalRecordType.TASK_TRANSITION) < types.index(
        JournalRecordType.RUN_COMPLETED
    )
    recovered = recover_session(records)
    assert recovered.tasks["subagent-policy"].lifecycle.status is TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_interrupt_running_subagent_terminalizes_journal(tmp_path) -> None:
    started = asyncio.Event()
    never = asyncio.Event()

    async def hanging(_request):
        started.set()
        await never.wait()
        yield  # unreachable; keeps the factory an async generator

    model = ScriptedModel([hanging])
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            journal_directory=_subagents_root(tmp_path),
            trace_directory=tmp_path / "traces",
        ),
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )

    launched = await supervisor.launch(_request("hang"), _context(), background=True)
    await asyncio.wait_for(started.wait(), timeout=5)
    result = await supervisor.interrupt(launched.handle, wait_seconds=5)

    assert result is not None
    assert result.status is SubagentStatus.CANCELLED
    subagent_records = await _read_subagent_records(tmp_path, result.subagent_run_id)
    assert subagent_records[-1].type is JournalRecordType.RUN_INTERRUPTED
    assert subagent_records[-1].payload["status"] == "aborted"
    manifest = json.loads(
        (tmp_path / "traces" / result.subagent_run_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "stopped"
    assert manifest["parent_run_id"] == "parent-run"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_interrupted_subagent_terminal_keeps_committed_usage(
    tmp_path,
) -> None:
    registry = ToolRegistry().register(_echo)
    second_started = asyncio.Event()
    never = asyncio.Event()

    async def hanging(_request):
        second_started.set()
        await never.wait()
        yield  # unreachable; keeps the factory an async generator

    model = ScriptedModel(
        [tool_events([tool_call_wire("c1", "echo", {"text": "x"})]), hanging]
    )
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_subagents_root(tmp_path),
        ),
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )

    launched = await supervisor.launch(
        _request("hang after one turn"),
        _context(parent_tool_authority=registry.freeze()),
        background=True,
    )
    await asyncio.wait_for(second_started.wait(), timeout=5)
    result = await supervisor.interrupt(launched.handle, wait_seconds=5)

    assert result is not None
    assert result.status is SubagentStatus.CANCELLED
    assert result.steps >= 1
    assert result.elapsed_seconds > 0
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_subagent_budget_exhaustion_maps_from_max_turns(tmp_path) -> None:
    registry = ToolRegistry().register(_echo)
    model = ScriptedModel(
        [tool_events([tool_call_wire(f"c{i}", "echo", {"text": "x"})]) for i in range(4)]
    )
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_subagents_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("loop", budget=TaskBudget(max_steps=1)),
        _context(parent_tool_authority=registry.freeze()),
        background=False,
    )

    assert result.status is SubagentStatus.BUDGET_EXHAUSTED
    assert result.steps == 1
    assert len(model.requests) == 1
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_subagent_reserves_last_step_for_plain_text_conclusion(tmp_path) -> None:
    executions = 0

    @tool(name="observe")
    def _observe(text: str) -> str:
        nonlocal executions
        executions += 1
        return f"observed:{text}"

    registry = ToolRegistry().register(_observe)
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "observe", {"text": "evidence"})]),
            text_events("conclusion from committed evidence"),
        ]
    )
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_subagents_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("inspect", budget=TaskBudget(max_steps=2)),
        _context(parent_tool_authority=registry.freeze()),
        background=False,
    )

    assert result.status is SubagentStatus.BUDGET_EXHAUSTED
    assert result.conclusion.summary == "conclusion from committed evidence"
    assert result.steps == 2
    assert len(model.requests) == 2
    assert executions == 1
    assert model.requests[1].option_dict().get("tools", []) == []
    final_users = [
        message
        for message in model.requests[1].message_dicts()
        if message.get("role") == "user"
    ]
    assert final_users[-1]["content"] != final_users[0]["content"]
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_parent_message_steers_active_subagent(tmp_path) -> None:
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
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_subagents_root(tmp_path),
        ),
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
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
    assert final is not None and final.status is SubagentStatus.COMPLETED
    second_request = model.requests[1]
    steered = [
        message
        for message in second_request.messages
        if isinstance(message, Mapping)
        and message.get("role") == "user"
        and "steer note" in str(message.get("content"))
    ]
    assert len(steered) == 1
    subagent_records = await _read_subagent_records(tmp_path, final.subagent_run_id)
    assert any(
        record.type is JournalRecordType.RUNTIME_INPUT_POSTED
        and record.payload["payload"]["content"] == "steer note"
        for record in subagent_records
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

    engine = AgentSubagentEngine(
        model=ScriptedModel([text_events("done")]),
        journal_factory=lambda: BlockingTerminalJournal(_subagents_root(tmp_path)),
    )
    event = RuntimeInput(
        event_id="parent-message",
        kind="agent.parent.message",
        correlation_id="subagent",
        source="qitos.parent",
        payload={"content": "too late"},
    )

    assert await engine.apost_runtime_event(event, run_id="run_subagentrace") is False
    running = asyncio.create_task(
        engine.arun("inspect", run_id="run_subagentrace")
    )
    await asyncio.wait_for(terminal_append_started.wait(), timeout=1)

    # Acceptance means queued: a post racing terminal settlement still enters
    # the mailbox, but no turn safe point remains, so the terminal boundary
    # rejects it without a posted record and without delivery.
    assert await engine.apost_runtime_event(event, run_id="run_subagentrace") is True

    release_terminal_append.set()
    await running
    await engine.aclose()
    records = await _read_subagent_records(tmp_path, "run_subagentrace")
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

    class BoundaryObservedEngine(AgentSubagentEngine):
        async def _settle_runtime_event_admissions(self, *, accept, agent=None):
            turn_boundary_started.set()
            await super()._settle_runtime_event_admissions(
                accept=accept,
                agent=agent,
            )

    model = ScriptedModel(
        [first_response, text_events("answer after parent message")]
    )
    engine = BoundaryObservedEngine(
        model=model,
        journal_factory=lambda: BlockingRuntimeJournal(_subagents_root(tmp_path)),
    )
    event = RuntimeInput(
        event_id="parent-message-race",
        kind="agent.parent.message",
        correlation_id="subagent",
        source="qitos.parent",
        payload={"content": "reserved note"},
    )

    running = asyncio.create_task(
        engine.arun("inspect", run_id="run_subagentreserved")
    )
    await asyncio.wait_for(model_started.wait(), timeout=1)
    posting = asyncio.create_task(
        engine.apost_runtime_event(event, run_id="run_subagentreserved")
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
    records = await _read_subagent_records(tmp_path, "run_subagentreserved")
    record_types = [record.type for record in records]
    assert record_types.index(JournalRecordType.RUNTIME_INPUT_POSTED) < (
        record_types.index(JournalRecordType.RUN_COMPLETED)
    )


@pytest.mark.asyncio
async def test_accepted_parent_message_is_marked_consumed_after_its_turn_commits(
    tmp_path,
) -> None:
    registry = ToolRegistry().register(_echo)
    model_started = asyncio.Event()

    async def first_response(_request):
        model_started.set()
        yield ModelStreamEvent(
            type=ModelStreamEventType.COMPLETED,
            finish_reason="tool_calls",
            tool_calls=[tool_call_wire("c1", "echo", {"text": "wait"})],
        )

    model = ScriptedModel(
        [first_response, text_events("acknowledged")],
    )
    engine = AgentSubagentEngine(
        model=model,
        tool_registry=registry,
        journal_factory=_subagent_journal_factory(tmp_path),
    )
    event = RuntimeInput(
        event_id="parent-message-consume",
        kind="agent.parent.message",
        correlation_id="subagent",
        source="qitos.parent",
        payload={"content": "reserved note"},
    )

    running = asyncio.create_task(
        engine.arun("inspect", run_id="run_subagentconsume")
    )
    await asyncio.wait_for(model_started.wait(), timeout=1)
    assert await engine.apost_runtime_event(event, run_id="run_subagentconsume")
    result = await asyncio.wait_for(running, timeout=1)
    assert result.state.stop_reason == "completed"
    await engine.aclose()

    records = await _read_subagent_records(tmp_path, "run_subagentconsume")
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
async def test_parent_message_queues_while_model_turn_is_in_flight(
    tmp_path,
) -> None:
    model_started = asyncio.Event()
    release_model = asyncio.Event()

    async def first_response(_request):
        model_started.set()
        await release_model.wait()
        for event in text_events("first answer"):
            yield event

    model = ScriptedModel([first_response, text_events("acknowledged")])
    engine = AgentSubagentEngine(
        model=model,
        journal_factory=_subagent_journal_factory(tmp_path),
    )
    event = RuntimeInput(
        event_id="parent-message-busy",
        kind="agent.parent.message",
        correlation_id="subagent",
        source="qitos.parent",
        payload={"content": "busy note"},
    )
    running = asyncio.create_task(
        engine.arun("inspect", run_id="run_subagent_busy")
    )
    await asyncio.wait_for(model_started.wait(), timeout=1)

    # The model turn is still in flight; mailbox acceptance must not wait
    # for the turn boundary.
    accepted = await asyncio.wait_for(
        engine.apost_runtime_event(event, run_id="run_subagent_busy"),
        timeout=1,
    )
    assert accepted is True
    assert running.done() is False

    release_model.set()
    result = await asyncio.wait_for(running, timeout=1)
    assert result.state.stop_reason == "completed"
    assert _model_requests_contain_note(model, "busy note")
    await engine.aclose()

    records = await _read_subagent_records(tmp_path, "run_subagent_busy")
    assert any(
        record.type is JournalRecordType.RUNTIME_INPUT_POSTED
        and record.payload["payload"]["content"] == "busy note"
        for record in records
    )


@pytest.mark.asyncio
async def test_parent_message_posted_between_turns_is_queued_and_delivered(
    tmp_path,
) -> None:
    registry = ToolRegistry().register(_echo)
    posted_between_turns = asyncio.Event()
    engine: AgentSubagentEngine

    async def first_response(_request):
        yield ModelStreamEvent(
            type=ModelStreamEventType.COMPLETED,
            finish_reason="tool_calls",
            tool_calls=[tool_call_wire("c1", "echo", {"text": "wait"})],
        )

    async def _post_after_first_turn(context) -> None:
        if context.turn != 0 or posted_between_turns.is_set():
            return None
        # Turn 0 has ended and its admission window is closed; the mailbox
        # still queues the message for the next turn safe point.
        accepted = await engine.apost_runtime_event(
            event,
            run_id="run_subagent_between",
        )
        assert accepted is True
        posted_between_turns.set()
        return None

    event = RuntimeInput(
        event_id="parent-message-between",
        kind="agent.parent.message",
        correlation_id="subagent",
        source="qitos.parent",
        payload={"content": "between turns note"},
    )
    model = ScriptedModel([first_response, text_events("acknowledged")])
    engine = AgentSubagentEngine(
        model=model,
        tool_registry=registry,
        journal_factory=_subagent_journal_factory(tmp_path),
        prepare_next_turn=_post_after_first_turn,
    )

    result = await engine.arun("inspect", run_id="run_subagent_between")
    assert posted_between_turns.is_set()
    assert result.state.stop_reason == "completed"
    assert _model_requests_contain_note(model, "between turns note")
    await engine.aclose()

    records = await _read_subagent_records(tmp_path, "run_subagent_between")
    assert any(
        record.type is JournalRecordType.RUNTIME_INPUT_POSTED
        and record.payload["payload"]["content"] == "between turns note"
        for record in records
    )


@pytest.mark.asyncio
async def test_parent_message_append_commit_wins_journal_cancellation(
    tmp_path,
) -> None:
    model_started = asyncio.Event()
    release_model = asyncio.Event()

    async def first_response(_request):
        model_started.set()
        await release_model.wait()
        for event in text_events("first answer"):
            yield event

    class CommitCancelledJournal(JsonlSessionJournal):
        async def append(self, record_type, payload, *, record_id):
            position = await super().append(
                record_type,
                payload,
                record_id=record_id,
            )
            if record_type is JournalRecordType.RUNTIME_INPUT_POSTED:
                # The record committed before the append observed its
                # cancellation; the durable outcome wins the settlement.
                raise JournalAppendCancelled(position)
            return position

    model = ScriptedModel([first_response, text_events("continued")])
    engine = AgentSubagentEngine(
        model=model,
        journal_factory=lambda: CommitCancelledJournal(_subagents_root(tmp_path)),
    )
    event = RuntimeInput(
        event_id="parent-message-commit-wins",
        kind="agent.parent.message",
        correlation_id="subagent",
        source="qitos.parent",
        payload={"content": "committed note"},
    )
    running = asyncio.create_task(
        engine.arun("inspect", run_id="run_subagent_commit_wins")
    )
    await asyncio.wait_for(model_started.wait(), timeout=1)
    assert (
        await engine.apost_runtime_event(event, run_id="run_subagent_commit_wins")
        is True
    )
    release_model.set()

    result = await asyncio.wait_for(running, timeout=1)
    assert result.state.stop_reason == "completed"
    assert model.requests[1].messages[-1] == {
        "role": "user",
        "content": "committed note",
    }
    await engine.aclose()

    records = await _read_subagent_records(tmp_path, "run_subagent_commit_wins")
    record_types = [record.type for record in records]
    assert record_types.count(JournalRecordType.RUNTIME_INPUT_POSTED) == 1
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
    engine = AgentSubagentEngine(
        model=model,
        journal_factory=lambda: CommittedErrorJournal(_subagents_root(tmp_path)),
    )
    event = RuntimeInput(
        event_id="parent-message-commit-error",
        kind="agent.parent.message",
        correlation_id="subagent",
        source="qitos.parent",
        payload={"content": "committed error note"},
    )
    running = asyncio.create_task(
        engine.arun("inspect", run_id="run_subagent_commit_error")
    )
    await asyncio.wait_for(model_started.wait(), timeout=1)
    posting = asyncio.create_task(
        engine.apost_runtime_event(event, run_id="run_subagent_commit_error")
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

    records = await _read_subagent_records(tmp_path, "run_subagent_commit_error")
    assert sum(
        record.type is JournalRecordType.RUNTIME_INPUT_POSTED for record in records
    ) == 1


@pytest.mark.asyncio
async def test_parent_message_append_rollback_is_not_delivered(tmp_path) -> None:
    model_started = asyncio.Event()
    release_model = asyncio.Event()

    async def first_response(_request):
        model_started.set()
        await release_model.wait()
        for event in text_events("first answer"):
            yield event

    class RollingBackRuntimeJournal(JsonlSessionJournal):
        async def append(self, record_type, payload, *, record_id):
            if record_type is JournalRecordType.RUNTIME_INPUT_POSTED:
                # The append observed cancellation before anything committed;
                # the rolled-back message must not be delivered.
                raise JournalAppendCancelled(None)
            return await super().append(record_type, payload, record_id=record_id)

    model = ScriptedModel([first_response, text_events("continued")])
    engine = AgentSubagentEngine(
        model=model,
        journal_factory=lambda: RollingBackRuntimeJournal(
            _subagents_root(tmp_path)
        ),
    )
    event = RuntimeInput(
        event_id="parent-message-rollback",
        kind="agent.parent.message",
        correlation_id="subagent",
        source="qitos.parent",
        payload={"content": "rolled back note"},
    )
    running = asyncio.create_task(
        engine.arun("inspect", run_id="run_subagent_rollback")
    )
    await asyncio.wait_for(model_started.wait(), timeout=1)
    assert (
        await engine.apost_runtime_event(event, run_id="run_subagent_rollback")
        is True
    )
    release_model.set()

    result = await asyncio.wait_for(running, timeout=1)
    assert result.state.stop_reason == "completed"
    assert not _model_requests_contain_note(model, "rolled back note")
    await engine.aclose()

    records = await _read_subagent_records(tmp_path, "run_subagent_rollback")
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
    runtime_append_attempted = asyncio.Event()

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
                runtime_append_attempted.set()
            return await super().append(record_type, payload, record_id=record_id)

    engine = AgentSubagentEngine(
        model=ScriptedModel([first_response]),
        max_turns=1 if terminal_case == "max_turns" else None,
        budget=(
            TaskBudget(max_tokens=1)
            if terminal_case == "budget"
            else TaskBudget()
        ),
        journal_factory=lambda: BlockingRuntimeJournal(_subagents_root(tmp_path)),
    )
    event = RuntimeInput(
        event_id=f"parent-message-{terminal_case}",
        kind="agent.parent.message",
        correlation_id="subagent",
        source="qitos.parent",
        payload={"content": "terminal note"},
    )

    running = asyncio.create_task(
        engine.arun("inspect", run_id=f"run_subagent_{terminal_case}")
    )
    await asyncio.wait_for(model_started.wait(), timeout=1)
    # Queue semantics: the active run accepts the message into its mailbox;
    # a run that terminalizes before the next turn safe point rejects the
    # reservation at the terminal boundary without a posted record.
    accepted = await engine.apost_runtime_event(
        event,
        run_id=f"run_subagent_{terminal_case}",
    )
    assert accepted is True
    if terminal_case == "cancel":
        engine.cancel("immediate")
    release_model.set()

    result = await asyncio.wait_for(running, timeout=1)
    assert result.state.stop_reason == expected_reason
    assert runtime_append_attempted.is_set() is False
    await engine.aclose()

    records = await _read_subagent_records(tmp_path, f"run_subagent_{terminal_case}")
    record_types = [record.type for record in records]
    assert JournalRecordType.RUNTIME_INPUT_POSTED not in record_types


@pytest.mark.asyncio
async def test_recovery_rebuilds_completed_subagent_from_journal(tmp_path) -> None:
    engine = AgentSubagentEngine(
        model=ScriptedModel([text_events("recovered answer", usage={"total_tokens": 5})]),
        journal_factory=_subagent_journal_factory(tmp_path),
    )
    completed = await engine.arun("inspect", run_id="run_subagentcompleted")
    await engine.aclose()
    assert completed.state.stop_reason == "completed"

    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})
    request = _request("inspect")
    handle = SubagentHandle(subagent_id="subagent-done", parent_run_id="parent-run")
    await parent_journal.append(
        JournalRecordType.SUBAGENT_STARTED,
        {
            "handle": handle.to_dict(),
            "request": request.to_dict(),
            "subagent_run_id": "run_subagentcompleted",
        },
        record_id="parent-run:subagent:subagent-done:started",
    )

    supervisor = SubagentSupervisor(
        invocation_factory=_unused_factory,
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )
    recovered = await supervisor.recover(
        parent_run_id="parent-run",
        journal=parent_journal,
    )

    assert len(recovered) == 1
    result = recovered[0]
    assert result.status is SubagentStatus.COMPLETED
    assert result.conclusion.summary == "recovered answer"
    assert result.steps == 1
    assert result.total_tokens == 5
    assert result.usage_complete is True
    terminal_records = [
        record
        for record in await parent_journal.replay()
        if record.type is JournalRecordType.SUBAGENT_TERMINAL
    ]
    assert len(terminal_records) == 1
    assert terminal_records[0].payload["status"] == SubagentStatus.COMPLETED.value
    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_recovery_preserves_root_budget_stop_before_parent_terminal(
    tmp_path,
) -> None:
    engine = AgentSubagentEngine(
        model=ScriptedModel([text_events("answer", usage={"total_tokens": 10})]),
        journal_factory=_subagent_journal_factory(tmp_path),
    )
    completed = await engine.arun("inspect", run_id="run_subagentbudget")
    await engine.aclose()
    assert completed.state.stop_reason == "completed"

    ledger = BudgetLedger(max_tokens=1)
    await ledger.commit(
        origin_run_id="run_subagentbudget",
        transaction_id="run_subagentbudget:turn:0:model",
        tokens=10,
        cost_usd=0.0,
        usage_complete=True,
        cost_complete=False,
    )
    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})
    request = _request("inspect")
    handle = SubagentHandle(subagent_id="subagent-budget", parent_run_id="parent-run")
    await parent_journal.append(
        JournalRecordType.SUBAGENT_STARTED,
        {
            "handle": handle.to_dict(),
            "request": request.to_dict(),
            "subagent_run_id": "run_subagentbudget",
        },
        record_id="parent-run:subagent:subagent-budget:started",
    )
    supervisor = SubagentSupervisor(
        invocation_factory=_unused_factory,
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )

    recovered = await supervisor.recover(
        parent_run_id="parent-run",
        journal=parent_journal,
        budget_ledger=ledger,
    )

    assert len(recovered) == 1
    assert recovered[0].status is SubagentStatus.BUDGET_EXHAUSTED
    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_recovery_attributes_root_budget_to_crossing_subagent_only(
    tmp_path,
) -> None:
    for run_id, answer, tokens in (
        ("run_subagenta", "answer-a", 4),
        ("run_subagentb", "answer-b", 7),
    ):
        engine = AgentSubagentEngine(
            model=ScriptedModel(
                [text_events(answer, usage={"total_tokens": tokens})]
            ),
            journal_factory=_subagent_journal_factory(tmp_path),
        )
        completed = await engine.arun("inspect", run_id=run_id)
        await engine.aclose()
        assert completed.state.stop_reason == "completed"

    ledger = BudgetLedger(max_tokens=10)
    for run_id, tokens in (("run_subagenta", 4), ("run_subagentb", 7)):
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
        handle = SubagentHandle(
            subagent_id=f"subagent-{suffix}",
            parent_run_id="parent-run",
        )
        await parent_journal.append(
            JournalRecordType.SUBAGENT_STARTED,
            {
                "handle": handle.to_dict(),
                "request": request.to_dict(),
                "subagent_run_id": f"run_subagent{suffix}",
            },
            record_id=f"parent-run:subagent:subagent-{suffix}:started",
        )
    supervisor = SubagentSupervisor(
        invocation_factory=_unused_factory,
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )

    recovered = await supervisor.recover(
        parent_run_id="parent-run",
        journal=parent_journal,
        budget_ledger=ledger,
    )

    by_subagent = {result.handle.subagent_id: result for result in recovered}
    assert by_subagent["subagent-a"].status is SubagentStatus.COMPLETED
    assert by_subagent["subagent-b"].status is SubagentStatus.BUDGET_EXHAUSTED
    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_recovery_rebuilds_interrupted_subagent_from_journal(tmp_path) -> None:
    started = asyncio.Event()
    never = asyncio.Event()

    async def hanging(_request):
        started.set()
        await never.wait()
        yield  # unreachable; keeps the factory an async generator

    engine = AgentSubagentEngine(
        model=ScriptedModel([hanging]),
        journal_factory=_subagent_journal_factory(tmp_path),
    )
    run = asyncio.create_task(engine.arun("hang", run_id="run_subagentaborted"))
    await asyncio.wait_for(started.wait(), timeout=5)
    engine.cancel("immediate")
    aborted = await run
    await engine.aclose()
    assert aborted.state.stop_reason == "cancelled"

    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})
    request = _request("hang")
    handle = SubagentHandle(subagent_id="subagent-aborted", parent_run_id="parent-run")
    await parent_journal.append(
        JournalRecordType.SUBAGENT_STARTED,
        {
            "handle": handle.to_dict(),
            "request": request.to_dict(),
            "subagent_run_id": "run_subagentaborted",
        },
        record_id="parent-run:subagent:subagent-aborted:started",
    )

    supervisor = SubagentSupervisor(
        invocation_factory=_unused_factory,
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )
    recovered = await supervisor.recover(
        parent_run_id="parent-run",
        journal=parent_journal,
    )

    assert len(recovered) == 1
    assert recovered[0].status is SubagentStatus.CANCELLED
    assert recovered[0].subagent_run_id == "run_subagentaborted"
    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_recovery_without_subagent_journal_marks_interrupted(tmp_path) -> None:
    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})
    request = _request("lost")
    handle = SubagentHandle(subagent_id="subagent-lost", parent_run_id="parent-run")
    await parent_journal.append(
        JournalRecordType.SUBAGENT_STARTED,
        {
            "handle": handle.to_dict(),
            "request": request.to_dict(),
            "subagent_run_id": "run_missing",
        },
        record_id="parent-run:subagent:subagent-lost:started",
    )

    supervisor = SubagentSupervisor(
        invocation_factory=_unused_factory,
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )
    recovered = await supervisor.recover(
        parent_run_id="parent-run",
        journal=parent_journal,
    )

    assert len(recovered) == 1
    assert recovered[0].status is SubagentStatus.INTERRUPTED
    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_recovery_rejects_non_loop_terminal_payload(tmp_path) -> None:
    subagent_journal = JsonlSessionJournal(_subagents_root(tmp_path))
    await subagent_journal.create("run_legacy", {})
    await subagent_journal.append(
        JournalRecordType.RUN_COMPLETED,
        {"status": "completed", "state": {"final_result": "legacy"}},
        record_id="run_legacy:run:terminal",
    )
    await subagent_journal.close()

    parent_journal = JsonlSessionJournal(tmp_path / "parent")
    await parent_journal.create("parent-run", {})
    request = _request("legacy")
    handle = SubagentHandle(subagent_id="subagent-legacy", parent_run_id="parent-run")
    await parent_journal.append(
        JournalRecordType.SUBAGENT_STARTED,
        {
            "handle": handle.to_dict(),
            "request": request.to_dict(),
            "subagent_run_id": "run_legacy",
        },
        record_id="parent-run:subagent:subagent-legacy:started",
    )

    supervisor = SubagentSupervisor(
        invocation_factory=_unused_factory,
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )
    with pytest.raises(SubagentPersistenceError):
        await supervisor.recover(
            parent_run_id="parent-run",
            journal=parent_journal,
        )
    await supervisor.aclose()
    await parent_journal.close()


@pytest.mark.asyncio
async def test_subagent_tool_groups_narrow_exposure(tmp_path) -> None:
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
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_subagents_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("inspect", allowed_tool_groups=("default",)),
        _context(parent_tool_authority=registry.freeze()),
        background=False,
    )

    assert result.status is SubagentStatus.COMPLETED
    assert "echo" in model.seen_tool_names
    assert "other" not in model.seen_tool_names
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_subagent_tool_exposure_fails_closed_without_parent_authority(
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
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_subagents_root(tmp_path),
        )
    )

    result = await supervisor.launch(_request("inspect"), _context(), background=False)

    assert result.status is SubagentStatus.COMPLETED
    assert model.seen_tool_names == set()
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_subagent_tool_exposure_cannot_exceed_parent_frozen_authority(
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
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_subagents_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("inspect", allowed_tool_groups=("network",)),
        _context(parent_tool_authority=parent_authority),
        background=False,
    )

    assert result.status is SubagentStatus.COMPLETED
    assert model.seen_tool_names == set()
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_subagent_tool_preserves_parent_parameter_scope_denial(tmp_path) -> None:
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
    subagent_tool = SubagentTool(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_subagents_root(tmp_path),
        )
    )

    result = await subagent_tool.execute(
        {
            "description": "inspect",
            "prompt": "inspect",
            "success_criteria": ["Report whether the Tool was admitted"],
        },
        runtime_context={
            "run_id": "parent-run",
            "tool_registry": registry.freeze(),
            "permission_context": permission_context,
        },
    )

    assert result.output["subagent_status"] == SubagentStatus.COMPLETED.value
    assert executions == []
    records = await _read_subagent_records(
        tmp_path,
        result.output["subagent_run_id"],
    )
    denied = next(
        record
        for record in records
        if record.type is JournalRecordType.TOOL_TERMINAL
    )
    denied_result = _terminal_result_payload(records, denied)
    assert denied_result["status"] == "denied"
    assert denied_result["metadata"]["permission_scope"] == "/secret"
    await subagent_tool.aclose()


@pytest.mark.asyncio
async def test_builtin_subagent_factory_rejects_different_env_permission_policy(
    tmp_path,
) -> None:
    model_factory_calls = 0

    def model_factory() -> ScriptedModel:
        nonlocal model_factory_calls
        model_factory_calls += 1
        return ScriptedModel([text_events("must not run")])

    class WiderEnv:
        tool_permission_context = ToolPermissionContext(default_decision="allow")

    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model_factory,
            env=WiderEnv(),
            journal_directory=_subagents_root(tmp_path),
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

    assert result.status is SubagentStatus.FAILED
    assert "cannot prove" in result.error
    assert model_factory_calls == 0
    await supervisor.aclose()


@pytest.mark.parametrize(
    ("base_result", "parent_result"),
    [("configured-safe", "parent-privileged"), ("configured-privileged", "parent-safe")],
)
@pytest.mark.asyncio
async def test_subagent_tool_name_collision_exposes_and_executes_neither_definition(
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
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=configured_registry,
            journal_directory=_subagents_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("collision"),
        _context(parent_tool_authority=parent_authority),
        background=False,
    )

    assert result.status is SubagentStatus.COMPLETED
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
async def test_builtin_subagent_factory_rejects_unresolved_runtime_policy(
    tmp_path,
    launch_request: SubagentLaunchRequest,
) -> None:
    model_factory_calls = 0

    def model_factory() -> ScriptedModel:
        nonlocal model_factory_calls
        model_factory_calls += 1
        return ScriptedModel([text_events("must not run")])

    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model_factory,
            journal_directory=_subagents_root(tmp_path),
        )
    )

    result = await supervisor.launch(launch_request, _context(), background=False)

    assert result.status is SubagentStatus.FAILED
    assert model_factory_calls == 0
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_subagent_model_usage_consumes_root_budget_and_closes_admission(
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
    base_factory = build_agent_subagent_invocation_factory(
        model=model,
        journal_directory=_subagents_root(tmp_path),
    )
    factory_calls = 0

    async def counting_factory(request, context):
        nonlocal factory_calls
        factory_calls += 1
        return await base_factory(request, context)

    supervisor = SubagentSupervisor(invocation_factory=counting_factory)
    context = _context(budget_ledger=ledger)

    first = await supervisor.launch(_request("first"), context, background=False)
    second = await supervisor.launch(_request("second"), context, background=False)

    assert first.status is SubagentStatus.BUDGET_EXHAUSTED
    assert second.status is SubagentStatus.BUDGET_EXHAUSTED
    assert factory_calls == 1
    assert len(model.requests) == 1
    snapshot = ledger.snapshot()
    assert snapshot.total_tokens == 10
    assert snapshot.tokens_exhausted is True
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_concurrent_subagents_share_one_lineage_step_budget() -> None:
    ledger = BudgetLedger(max_steps=1)
    models = [
        ScriptedModel([text_events(f"answer-{index}")])
        for index in range(2)
    ]
    engines = [
        AgentSubagentEngine(model=model, budget_ledger=ledger)
        for model in models
    ]

    results = await asyncio.gather(
        *(
            engine.arun("inspect", run_id=f"subagent-{index}")
            for index, engine in enumerate(engines)
        )
    )
    await asyncio.gather(*(engine.aclose() for engine in engines))

    assert sorted(result.state.stop_reason for result in results) == [
        "completed",
        "max_steps",
    ]
    assert sum(len(model.requests) for model in models) == 1
    snapshot = ledger.snapshot()
    assert snapshot.total_steps == 1
    assert snapshot.remaining_steps == 0


@pytest.mark.asyncio
async def test_failed_initial_step_lease_releases_conclusion_capacity() -> None:
    class RejectingLeaseLedger(BudgetLedger):
        async def lease_step(self, **kwargs):
            _ = kwargs
            raise RuntimeError("injected lease failure")

    ledger = RejectingLeaseLedger(max_steps=1)
    engine = AgentSubagentEngine(
        model=ScriptedModel([text_events("unused")]),
        budget_ledger=ledger,
    )

    with pytest.raises(RuntimeError, match="injected lease failure"):
        await engine.arun("inspect", run_id="subagent-lease-failure")

    await ledger.reserve_step(
        origin_run_id="root",
        transaction_id="root:turn:0:step",
    )
    assert ledger.snapshot().total_steps == 1
    await engine.aclose()


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
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            journal_directory=_subagents_root(tmp_path),
        ),
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
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
    assert terminal.status is SubagentStatus.CANCELLED
    assert ledger.snapshot().total_tokens == 10
    records = await _read_subagent_records(tmp_path, terminal.subagent_run_id)
    assert JournalRecordType.MODEL_COMPLETED in {record.type for record in records}
    assert records[-1].type is JournalRecordType.RUN_INTERRUPTED
    recovered = recover_run_outcome(records)
    assert recovered is not None
    assert terminal.conclusion.summary == subagent_final_text(recovered.messages)
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_deadline_uses_committed_conclusion_without_retrying_model(tmp_path) -> None:
    second_request_started = asyncio.Event()
    never = asyncio.Event()

    async def hang(_request):
        second_request_started.set()
        await never.wait()
        yield

    model = ScriptedModel(
        [
            tool_events(
                [tool_call_wire("c1", "echo", {"text": "evidence"})],
                text="progress before the deadline",
            ),
            hang,
        ]
    )
    registry = ToolRegistry().register(_echo)
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_subagents_root(tmp_path),
            run_timeout_s=0.1,
        ),
        subagent_journal_factory=_subagent_journal_factory(tmp_path),
    )

    result = await supervisor.launch(
        _request("inspect"),
        _context(parent_tool_authority=registry.freeze()),
        background=False,
    )

    assert second_request_started.is_set()
    assert result.status is SubagentStatus.BUDGET_EXHAUSTED
    assert result.conclusion.summary == ""
    assert len(model.requests) == 2
    records = await _read_subagent_records(tmp_path, result.subagent_run_id)
    recovered = recover_run_outcome(records)
    assert recovered is not None
    assert result.conclusion.summary == subagent_final_text(recovered.messages)
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_root_budget_is_durable_before_subagent_model_terminal(tmp_path) -> None:
    class FailingModelJournal(JsonlSessionJournal):
        async def append(self, record_type, payload, *, record_id):
            if record_type is JournalRecordType.MODEL_COMPLETED:
                raise JournalError("model terminal failed")
            return await super().append(record_type, payload, record_id=record_id)

    root_journal = JsonlSessionJournal(tmp_path / "root")
    await root_journal.create("root-run", {})
    ledger = BudgetLedger()
    ledger.attach(root_journal, root_run_id="root-run")
    engine = AgentSubagentEngine(
        model=ScriptedModel(
            [text_events("answer", usage={"total_tokens": 10})]
        ),
        budget_ledger=ledger,
        journal_factory=lambda: FailingModelJournal(_subagents_root(tmp_path)),
    )

    with pytest.raises(JournalError, match="model terminal failed"):
        await engine.arun("inspect", run_id="run_subagentcrash")
    await engine.aclose()

    assert ledger.snapshot().total_tokens == 10
    root_records = await root_journal.replay()
    assert JournalRecordType.BUDGET_COMMITTED in {
        record.type for record in root_records
    }
    subagent_records = await _read_subagent_records(tmp_path, "run_subagentcrash")
    assert JournalRecordType.MODEL_COMPLETED not in {
        record.type for record in subagent_records
    }
    await root_journal.close()


@pytest.mark.asyncio
async def test_subagent_token_budget_blocks_tool_side_effects_in_exhausting_turn(
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
                    text="attempting a mutation",
                    tool_calls=[tool_call_wire("c1", "mutate", {})],
                    usage={"total_tokens": 10},
                )
            ],
            text_events("budget-limited conclusion", usage={"total_tokens": 1}),
        ]
    )
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_subagents_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("mutate", budget=TaskBudget(max_tokens=1)),
        _context(parent_tool_authority=registry.freeze()),
        background=False,
    )

    assert result.status is SubagentStatus.BUDGET_EXHAUSTED
    assert result.conclusion.summary == "budget-limited conclusion"
    assert len(model.requests) == 2
    assert executions == 0
    subagent_records = await _read_subagent_records(tmp_path, result.subagent_run_id)
    terminal = next(
        record
        for record in subagent_records
        if record.type is JournalRecordType.TOOL_TERMINAL
    )
    assert _terminal_result_payload(subagent_records, terminal)["status"] == "denied"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_subagent_cost_budget_uses_explicit_frozen_model_pricing(tmp_path) -> None:
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
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            model_pricing=pricing,
            journal_directory=_subagents_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("cost", budget=TaskBudget(max_cost_usd=0.000005)),
        _context(budget_ledger=ledger),
        background=False,
    )

    assert result.status is SubagentStatus.BUDGET_EXHAUSTED
    assert result.total_cost_usd == pytest.approx(0.000006)
    snapshot = ledger.snapshot()
    assert snapshot.total_cost_usd == pytest.approx(0.000006)
    assert snapshot.cost_complete is True
    assert snapshot.cost_exhausted is True
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_subagent_cost_budget_without_pricing_fails_before_model_construction(
    tmp_path,
) -> None:
    model_factory_calls = 0

    def model_factory() -> ScriptedModel:
        nonlocal model_factory_calls
        model_factory_calls += 1
        return ScriptedModel([text_events("must not run")])

    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model_factory,
            journal_directory=_subagents_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("cost", budget=TaskBudget(max_cost_usd=1.0)),
        _context(),
        background=False,
    )

    assert result.status is SubagentStatus.FAILED
    assert model_factory_calls == 0
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_subagent_budget_narrows_nested_max_subagents_runtime_context(
    tmp_path,
) -> None:
    observed: list[int] = []

    @tool(name="capture_budget")
    def _capture_budget(runtime_context=None) -> str:
        observed.append(runtime_context["max_subagents"])
        return "captured"

    registry = ToolRegistry().register(_capture_budget)
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "capture_budget", {})]),
            text_events("done"),
        ]
    )
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_subagents_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("nested", budget=TaskBudget(max_subagents=2)),
        _context(max_subagents=3, parent_tool_authority=registry.freeze()),
        background=False,
    )

    assert result.status is SubagentStatus.COMPLETED
    assert observed == [2]
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_subagent_runtime_exposes_its_mailbox_for_descendant_completion(
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
    supervisor = SubagentSupervisor(
        invocation_factory=build_agent_subagent_invocation_factory(
            model=model,
            tool_registry=registry,
            journal_directory=_subagents_root(tmp_path),
        )
    )

    result = await supervisor.launch(
        _request("nested completion"),
        _context(parent_tool_authority=registry.freeze()),
        background=False,
    )

    assert result.status is SubagentStatus.COMPLETED
    assert observed == [True]
    await supervisor.aclose()


async def _unused_factory(_request, _context):
    raise AssertionError("recovery must not construct a subagent")
