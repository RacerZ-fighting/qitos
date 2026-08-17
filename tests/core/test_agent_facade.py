"""Agent façade behavior: runs, queues, abort, rejection and subscription."""

from __future__ import annotations

import asyncio

import pytest

from qitos.core.agent import Agent, AgentBusyError, AgentRunRejected, QueueMode
from qitos.core.agent_loop import AgentLoopResult, AgentRunStatus
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
    assert kinds == ["model_terminal", "turn_committed", "run_terminal"]


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
    assert "run_terminal" in kinds
    assert transaction.records[-1] == ("run_terminal", "failed")
    assert agent.error_message == "listener bug"
    await agent.wait_for_idle()


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
    await agent.wait_for_idle()


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
