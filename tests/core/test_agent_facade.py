"""Agent façade behavior: runs, queues, abort, rejection and subscription."""

from __future__ import annotations

import asyncio

import pytest

from qitos.core.agent import (
    Agent,
    AgentBusyError,
    AgentListenerTimeoutError,
    AgentRunRejected,
    QueueMode,
)
from qitos.core.agent_events import AgentEnd, AgentStart, MessageEnd, ToolExecutionEnd
from qitos.core.agent_loop import AgentLoopResult, AgentRunStatus
from qitos.core.cancellation import CancelSignalView
from qitos.core.message import AssistantMessage, UserMessage
from qitos.core.tool import tool
from qitos.core.tool_registry import ToolRegistry

from .agent_fakes import (
    RecordingTransaction,
    ScriptedModel,
    make_hanging_model,
    text_events,
    tool_call_wire,
    tool_events,
)


@tool(name="echo")
def _echo(text: str) -> str:
    return f"echo:{text}"


def _agent(model, **kwargs) -> Agent:
    registry = ToolRegistry().register(_echo)
    return Agent(model=model, tool_registry=registry, **kwargs)


def test_queue_mode_uses_pi_wire_value_and_rejects_raw_strings() -> None:
    assert QueueMode.ONE_AT_A_TIME.value == "one-at-a-time"
    with pytest.raises(TypeError, match="QueueMode"):
        _agent(
            ScriptedModel([text_events("unused")]),
            steering_mode="all",  # type: ignore[arg-type]
        )

    agent = _agent(ScriptedModel([text_events("unused")]))
    with pytest.raises(TypeError, match="QueueMode"):
        agent.follow_up_mode = "all"  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_prompt_completes_and_updates_state() -> None:
    agent = _agent(ScriptedModel([text_events("answer")]), system_prompt="sys")
    result = await agent.prompt("question")

    assert isinstance(result, AgentLoopResult)
    assert result.status is AgentRunStatus.COMPLETED
    assert [m.role for m in agent.messages] == ["user", "assistant"]
    assert agent.messages[-1].text == "answer"
    assert agent.is_streaming is False
    assert agent.error_message is None


@pytest.mark.asyncio
async def test_prompt_during_run_is_a_typed_rejection() -> None:
    gate = asyncio.Event()
    agent = _agent(ScriptedModel([make_hanging_model(gate, first_text="w")]))

    first = asyncio.create_task(agent.prompt("one"))
    while not agent.is_streaming:
        await asyncio.sleep(0.005)
    second = await agent.prompt("two")
    assert isinstance(second, AgentRunRejected)
    assert second.reason == "busy"

    agent.abort()
    outcome = await first
    assert outcome.status is AgentRunStatus.ABORTED
    await agent.wait_for_idle()
    gate.set()


@pytest.mark.asyncio
async def test_abort_an_idle_agent_is_a_noop() -> None:
    agent = _agent(ScriptedModel([]))
    agent.abort()
    await agent.wait_for_idle()


@pytest.mark.asyncio
async def test_steer_during_run_lands_before_the_next_model_call() -> None:
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "x"})]),
            text_events("final"),
        ]
    )
    agent = _agent(model)

    steered = False

    def _steer_on_turn_end(event) -> None:
        nonlocal steered
        if event.type == "turn_end" and not steered:
            steered = True
            agent.steer(UserMessage(content="steered"))

    agent.subscribe(_steer_on_turn_end)
    result = await agent.prompt("go")

    assert result.status is AgentRunStatus.COMPLETED
    # Listeners settle inside the run, so the steering drain that follows the
    # first turn_end deterministically picks up the message.
    assert model.requests[1].messages[-1] == {
        "role": "user",
        "content": "steered",
    }
    roles = [m.role for m in agent.messages]
    assert roles == ["user", "assistant", "tool", "user", "assistant"]


@pytest.mark.asyncio
async def test_max_turns_restores_steering_drained_before_rejection() -> None:
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "x"})]),
            text_events("continued"),
        ]
    )
    agent = _agent(model, max_turns=1)
    queued = UserMessage(content="keep this")

    def _steer_after_turn(event) -> None:
        if event.type == "turn_end":
            agent.steer(queued)

    unsubscribe = agent.subscribe(_steer_after_turn)
    first = await agent.prompt("go")
    unsubscribe()

    assert first.status is AgentRunStatus.MAX_TURNS
    assert agent.has_queued_messages() is True
    assert not any(message is queued for message in agent.messages)

    second = await agent.continue_run()

    assert isinstance(second, AgentLoopResult)
    assert second.status is AgentRunStatus.COMPLETED
    assert model.requests[1].messages[-1] == {
        "role": "user",
        "content": "keep this",
    }
    assert sum(message is queued for message in agent.messages) == 1


@pytest.mark.asyncio
async def test_follow_up_runs_when_agent_would_stop() -> None:
    agent = _agent(ScriptedModel([text_events("first"), text_events("second")]))
    agent.follow_up(UserMessage(content="more"))
    result = await agent.prompt("go")

    assert result.status is AgentRunStatus.COMPLETED
    assistants = [m.text for m in agent.messages if isinstance(m, AssistantMessage)]
    assert assistants == ["first", "second"]


@pytest.mark.asyncio
async def test_continue_run_rejections_and_queue_fallback() -> None:
    agent = _agent(ScriptedModel([text_events("ok"), text_events("continued")]))
    empty = await agent.continue_run()
    assert isinstance(empty, AgentRunRejected)
    assert empty.reason == "empty_history"

    await agent.prompt("go")
    rejected = await agent.continue_run()
    assert isinstance(rejected, AgentRunRejected)
    assert rejected.reason == "assistant_tail"

    agent.steer(UserMessage(content="queued"))
    continued = await agent.continue_run()
    assert isinstance(continued, AgentLoopResult)
    assert continued.status is AgentRunStatus.COMPLETED
    assert agent.messages[-1].text == "continued"


@pytest.mark.asyncio
async def test_reset_requires_idle() -> None:
    gate = asyncio.Event()
    agent = _agent(ScriptedModel([make_hanging_model(gate)]))
    run = asyncio.create_task(agent.prompt("go"))
    while not agent.is_streaming:
        await asyncio.sleep(0.005)
    with pytest.raises(AgentBusyError):
        agent.reset()
    agent.abort()
    await run
    agent.reset()
    assert agent.messages == ()


@pytest.mark.asyncio
async def test_listeners_are_awaited_in_subscription_order() -> None:
    order = []
    agent = _agent(ScriptedModel([text_events("ok")]))

    async def _first(event) -> None:
        await asyncio.sleep(0.01)
        order.append(("first", event.type))

    async def _second(event) -> None:
        order.append(("second", event.type))

    agent.subscribe(_first)
    agent.subscribe(_second)
    await agent.prompt("go")

    agent_ends = [entry for entry in order if entry[1] == "agent_end"]
    assert agent_ends == [("first", "agent_end"), ("second", "agent_end")]


@pytest.mark.asyncio
async def test_wait_for_idle_includes_listener_settlement() -> None:
    settled = asyncio.Event()
    agent = _agent(ScriptedModel([text_events("ok")]))

    async def _slow(event) -> None:
        if event.type == "agent_end":
            await asyncio.sleep(0.05)
            settled.set()

    agent.subscribe(_slow)
    result_task = asyncio.create_task(agent.prompt("go"))
    result = await result_task
    assert result.status is AgentRunStatus.COMPLETED
    # prompt() resolves only after the run settled, listeners included.
    assert settled.is_set()
    await agent.wait_for_idle()


@pytest.mark.asyncio
async def test_run_timeout_produces_deadline_outcome() -> None:
    gate = asyncio.Event()
    agent = _agent(
        ScriptedModel([make_hanging_model(gate)]),
        run_timeout_s=0.1,
    )
    result = await agent.prompt("go")
    assert result.status is AgentRunStatus.DEADLINE_EXCEEDED
    gate.set()


@pytest.mark.asyncio
async def test_transaction_factory_receives_run_id_and_records() -> None:
    transactions = []

    def _factory(run_id: str):
        transaction = RecordingTransaction()
        transactions.append((run_id, transaction))
        return transaction

    agent = _agent(
        ScriptedModel([text_events("ok")]), transaction_factory=_factory
    )
    result = await agent.prompt("go")

    assert result.status is AgentRunStatus.COMPLETED
    assert len(transactions) == 1
    run_id, transaction = transactions[0]
    assert run_id
    kinds = [record[0] for record in transaction.records]
    assert kinds == [
        "input_accepted",
        "turn_frozen",
        "model_terminal",
        "turn_committed",
        "run_terminal",
    ]


@pytest.mark.asyncio
async def test_caller_cancellation_aborts_and_settles_the_run() -> None:
    gate = asyncio.Event()
    agent = _agent(ScriptedModel([make_hanging_model(gate, first_text="w")]))
    task = asyncio.create_task(agent.prompt("go"))
    while not agent.is_streaming:
        await asyncio.sleep(0.005)
    while agent.streaming_message is None:
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await agent.wait_for_idle()
    assert agent.messages
    assert agent.messages[-1].failed
    gate.set()


@pytest.mark.asyncio
async def test_queue_mode_one_at_a_time_then_all() -> None:
    model = ScriptedModel(
        [
            text_events("one"),
            text_events("two"),
            text_events("three"),
            text_events("four"),
        ]
    )
    agent = _agent(model)
    agent.steering_mode = QueueMode.ONE_AT_A_TIME
    agent.follow_up(UserMessage(content="f1"))
    agent.follow_up(UserMessage(content="f2"))
    await agent.prompt("go")
    # one-at-a-time: only the first follow-up ran before the agent stopped
    # again, then the second was drained on the next stop.
    texts = [m.text for m in agent.messages if isinstance(m, AssistantMessage)]
    assert texts == ["one", "two", "three"]


@pytest.mark.asyncio
async def test_queue_mode_all_drains_everything() -> None:
    model = ScriptedModel([text_events("one"), text_events("two")])
    agent = _agent(model)
    agent.follow_up_mode = QueueMode.ALL
    agent.follow_up(UserMessage(content="f1"))
    agent.follow_up(UserMessage(content="f2"))
    await agent.prompt("go")
    users = [m.content for m in agent.messages if isinstance(m, UserMessage)]
    assert users == ["go", "f1", "f2"]


@pytest.mark.asyncio
async def test_listener_fault_propagates_and_run_is_terminalized() -> None:
    from qitos.core.agent_events import TurnEnd

    transaction = RecordingTransaction()
    agent = _agent(
        ScriptedModel([text_events("ok")]),
        transaction_factory=lambda run_id: transaction,
    )

    def _broken(event) -> None:
        if isinstance(event, TurnEnd):
            raise RuntimeError("listener bug")

    agent.subscribe(_broken)
    # A listener bug is an implementation fault, not a typed run failure.
    with pytest.raises(RuntimeError, match="listener bug"):
        await agent.prompt("go")
    # The run still reached its durable terminal state first.
    kinds = [record[0] for record in transaction.records]
    assert kinds == [
        "input_accepted",
        "turn_frozen",
        "model_terminal",
        "turn_committed",
        "run_terminal",
    ]
    assert transaction.records[-1] == ("run_terminal", "failed")
    assert agent.error_message == "listener bug"
    await agent.wait_for_idle()


@pytest.mark.asyncio
async def test_agent_start_listener_fault_still_records_run_terminal() -> None:
    transaction = RecordingTransaction()
    agent = _agent(
        ScriptedModel([text_events("never reached")]),
        transaction_factory=lambda run_id: transaction,
    )

    def _broken(event) -> None:
        if isinstance(event, AgentStart):
            raise RuntimeError("start listener bug")

    agent.subscribe(_broken)
    with pytest.raises(RuntimeError, match="start listener bug"):
        await agent.prompt("go")

    assert transaction.records == [("run_terminal", "failed")]


@pytest.mark.asyncio
async def test_message_listener_fault_follows_model_terminal_record() -> None:
    transaction = RecordingTransaction()
    agent = _agent(
        ScriptedModel([text_events("answer")]),
        transaction_factory=lambda run_id: transaction,
    )

    def _broken(event) -> None:
        if isinstance(event, MessageEnd) and isinstance(
            event.message, AssistantMessage
        ):
            raise RuntimeError("message listener bug")

    agent.subscribe(_broken)
    with pytest.raises(RuntimeError, match="message listener bug"):
        await agent.prompt("go")

    assert [record[0] for record in transaction.records] == [
        "input_accepted",
        "turn_frozen",
        "model_terminal",
        "run_terminal",
    ]


@pytest.mark.asyncio
async def test_tool_listener_fault_preserves_call_result_pair() -> None:
    transaction = RecordingTransaction()
    agent = _agent(
        ScriptedModel(
            [tool_events([tool_call_wire("c1", "echo", {"text": "x"})])]
        ),
        transaction_factory=lambda run_id: transaction,
    )

    def _broken(event) -> None:
        if isinstance(event, ToolExecutionEnd):
            raise RuntimeError("tool listener bug")

    agent.subscribe(_broken)
    with pytest.raises(RuntimeError, match="tool listener bug"):
        await agent.prompt("go")

    assert [message.role for message in agent.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert agent.messages[-1].tool_call_id == "c1"
    # The façade listener failed before it could project the ToolResult event,
    # but both the in-memory transcript and durable transaction stay paired.
    assert [record[0] for record in transaction.records] == [
        "input_accepted",
        "turn_frozen",
        "model_terminal",
        "tool_started",
        "tool_terminal",
        "run_terminal",
    ]


@pytest.mark.asyncio
async def test_listener_receives_read_only_cancel_signal() -> None:
    seen: list[CancelSignalView] = []
    agent = _agent(ScriptedModel([text_events("never reached")]))

    def _abort_on_start(event, signal: CancelSignalView) -> None:
        if isinstance(event, AgentStart):
            seen.append(signal)
            assert not hasattr(signal, "request_cancel")
            assert signal is agent.signal
            agent.abort()

    agent.subscribe(_abort_on_start)
    result = await agent.prompt("go")

    assert isinstance(result, AgentLoopResult)
    assert result.status is AgentRunStatus.ABORTED
    assert len(seen) == 1
    assert seen[0].immediate_requested is True
    assert agent.signal is None


@pytest.mark.asyncio
async def test_single_argument_listener_remains_supported() -> None:
    seen = []
    agent = _agent(ScriptedModel([text_events("ok")]))
    agent.subscribe(lambda event: seen.append(event.type))

    await agent.prompt("go")

    assert seen[0] == "agent_start"
    assert seen[-1] == "agent_end"


def test_listener_signature_is_validated_at_subscription() -> None:
    agent = _agent(ScriptedModel([text_events("unused")]))

    def _invalid(_event, _signal, _extra) -> None:
        return None

    with pytest.raises(TypeError, match="must accept"):
        agent.subscribe(_invalid)


@pytest.mark.asyncio
async def test_prompt_cancellation_settles_the_run_before_raising() -> None:
    from qitos.core.agent_events import MessageUpdate

    gate = asyncio.Event()  # never set: the model stream hangs mid-run
    transaction = RecordingTransaction()
    agent = _agent(
        ScriptedModel([make_hanging_model(gate, first_text="w")]),
        transaction_factory=lambda run_id: transaction,
    )
    streaming = asyncio.Event()

    def _on_event(event) -> None:
        if isinstance(event, MessageUpdate):
            streaming.set()

    agent.subscribe(_on_event)
    prompt_task = asyncio.create_task(agent.prompt("go"))
    await streaming.wait()
    prompt_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prompt_task
    # The run reached its durable terminal state before the caller observed
    # the cancellation.
    assert transaction.records[-1] == ("run_terminal", "aborted")
    assert agent.is_streaming is False
    await agent.wait_for_idle()


@pytest.mark.asyncio
async def test_prompt_cancellation_awaits_listener_cleanup() -> None:
    listener_started = asyncio.Event()
    listener_settled = asyncio.Event()
    transaction = RecordingTransaction()
    agent = _agent(
        ScriptedModel([text_events("never reached")]),
        transaction_factory=lambda run_id: transaction,
    )

    async def _listener(event, signal: CancelSignalView) -> None:
        if not isinstance(event, AgentStart):
            return
        try:
            listener_started.set()
            await signal.wait_immediate()
        finally:
            await asyncio.sleep(0)
            listener_settled.set()

    agent.subscribe(_listener)
    prompt_task = asyncio.create_task(agent.prompt("go"))
    await listener_started.wait()
    prompt_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await prompt_task

    assert listener_settled.is_set()
    assert transaction.records[-1] == ("run_terminal", "aborted")


@pytest.mark.asyncio
async def test_aborted_run_awaits_agent_end_listener_normally() -> None:
    gate = asyncio.Event()
    listener_completed = asyncio.Event()
    agent = _agent(ScriptedModel([make_hanging_model(gate, first_text="w")]))

    async def _listener(event, _signal: CancelSignalView) -> None:
        if isinstance(event, AgentEnd):
            await asyncio.sleep(0)
            listener_completed.set()

    agent.subscribe(_listener)
    run = asyncio.create_task(agent.prompt("go"))
    while agent.streaming_message is None:
        await asyncio.sleep(0)

    agent.abort()
    result = await run

    assert result.status is AgentRunStatus.ABORTED
    assert listener_completed.is_set()


@pytest.mark.asyncio
async def test_run_deadline_bounds_a_hanging_listener() -> None:
    listener_started = asyncio.Event()
    listener_settled = asyncio.Event()
    agent = _agent(
        ScriptedModel([text_events("never reached")]), run_timeout_s=0.05
    )

    async def _listener(event) -> None:
        if not isinstance(event, AgentStart):
            return
        try:
            listener_started.set()
            await asyncio.Event().wait()
        finally:
            listener_settled.set()

    agent.subscribe(_listener)
    with pytest.raises(AgentListenerTimeoutError):
        await agent.prompt("go")

    assert listener_started.is_set()
    assert listener_settled.is_set()
    await agent.wait_for_idle()


@pytest.mark.asyncio
async def test_abort_interrupts_hanging_tool_and_preserves_terminal_result() -> None:
    tool_started = asyncio.Event()
    tool_settled = asyncio.Event()

    @tool(name="waiter")
    async def _waiter() -> str:
        try:
            tool_started.set()
            await asyncio.Event().wait()
            return "unreachable"
        finally:
            tool_settled.set()

    registry = ToolRegistry().register(_waiter)
    transaction = RecordingTransaction()
    agent = Agent(
        model=ScriptedModel(
            [tool_events([tool_call_wire("c1", "waiter", {})])]
        ),
        tool_registry=registry,
        transaction_factory=lambda _run_id: transaction,
    )
    run = asyncio.create_task(agent.prompt("go"))
    await tool_started.wait()

    agent.abort()
    result = await asyncio.wait_for(run, timeout=1)

    assert result.status is AgentRunStatus.ABORTED
    assert tool_settled.is_set()
    assert [message.role for message in agent.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    tool_message = agent.messages[-1]
    assert tool_message.tool_call_id == "c1"
    assert tool_message.result.status == "cancelled"
    assert [record[0] for record in transaction.records] == [
        "input_accepted",
        "turn_frozen",
        "model_terminal",
        "tool_started",
        "tool_terminal",
        "turn_committed",
        "run_terminal",
    ]


@pytest.mark.asyncio
async def test_agent_keeps_the_callers_tool_registry_instance() -> None:
    registry = ToolRegistry()
    agent = Agent(model=ScriptedModel([text_events("ok")]), tool_registry=registry)
    # An explicitly passed (even empty) registry stays the Agent's registry.
    assert agent.tool_registry is registry

    @tool(name="late")
    def _late(text: str) -> str:
        return text

    registry.register(_late)
    result = await agent.prompt("go")
    assert isinstance(result, AgentLoopResult)
    assert result.status is AgentRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_facade_thinking_level_feeds_each_new_run() -> None:
    from qitos.core.model_capabilities import ModelCapabilities
    from qitos.core.thinking import ThinkingLevel

    model = ScriptedModel(
        [text_events("one"), text_events("two")],
        capabilities=ModelCapabilities(
            thinking_levels=(ThinkingLevel.LOW, ThinkingLevel.HIGH)
        ),
    )
    agent = _agent(model, thinking_level=ThinkingLevel.LOW)

    first = await agent.prompt("go")
    assert isinstance(first, AgentLoopResult)
    assert model.requests[0].thinking_level is ThinkingLevel.LOW

    # The property is captured per run: assigning it never rewrites a frozen
    # turn, and the next run picks the new value up.
    agent.thinking_level = ThinkingLevel.HIGH
    second = await agent.prompt("again")
    assert isinstance(second, AgentLoopResult)
    assert model.requests[1].thinking_level is ThinkingLevel.HIGH


def test_facade_thinking_level_rejects_untyped_values() -> None:
    from qitos.core.thinking import ThinkingLevel

    with pytest.raises(TypeError, match="ThinkingLevel"):
        _agent(
            ScriptedModel([text_events("unused")]),
            thinking_level="high",  # type: ignore[arg-type]
        )

    agent = _agent(ScriptedModel([text_events("unused")]))
    assert agent.thinking_level is None
    with pytest.raises(TypeError, match="ThinkingLevel"):
        agent.thinking_level = "low"  # type: ignore[assignment]
    agent.thinking_level = ThinkingLevel.LOW
    assert agent.thinking_level is ThinkingLevel.LOW
    agent.thinking_level = None
    assert agent.thinking_level is None
