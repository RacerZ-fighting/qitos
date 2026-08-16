"""Behavioral conformance for the minimal agent loop.

These tests express the loop contract without referencing the retired
AgentModule/Engine lifecycle: turn structure, steering/follow-up safe points,
ToolCall/ToolResult pairing, cancellation, deadlines, ordering and the
transaction barriers.
"""

from __future__ import annotations

import asyncio
import time
from typing import List

import pytest

from qitos.core.agent_events import (
    AgentEnd,
    AgentStart,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    TurnEnd,
    TurnStart,
)
from qitos.core.agent_loop import (
    AgentContext,
    AgentLoopConfig,
    AgentRunStatus,
    NextTurnUpdate,
    agent_loop,
    run_agent_loop,
    run_agent_loop_continue,
)
from qitos.core.cancellation import CancelToken
from qitos.core.message import (
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from qitos.core.model_request import ModelContinuation
from qitos.core.model_stream import ModelStreamEvent, ModelStreamEventType
from qitos.core.tool import tool
from qitos.core.tool_registry import ToolRegistry

from .agent_fakes import (
    RecordingTransaction,
    ScriptedModel,
    failed_events,
    make_hanging_model,
    text_events,
    tool_call_wire,
    tool_events,
)


def _event_types(events):
    return [event.type for event in events]


@tool(name="echo")
def _echo(text: str) -> str:
    return f"echo:{text}"


def _registry(*items):
    registry = ToolRegistry()
    for item in items:
        registry.register(item)
    return registry


@pytest.mark.asyncio
async def test_simple_text_run_event_sequence_and_transcript() -> None:
    model = ScriptedModel([text_events("hello")])
    context = AgentContext(system_prompt="sys", messages=[])
    events = []
    config = AgentLoopConfig(model=model, run_id="run-1")
    result = await run_agent_loop(
        [UserMessage(content="hi")], context, config, events.append
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert _event_types(events) == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert [message.role for message in context.messages] == ["user", "assistant"]
    request = model.requests[0]
    assert request.messages[0] == {"role": "system", "content": "sys"}
    assert request.messages[1] == {"role": "user", "content": "hi"}
    assert request.deadline_monotonic is None


@pytest.mark.asyncio
async def test_tool_call_round_trip_pairs_results_in_order() -> None:
    model = ScriptedModel(
        [
            tool_events(
                [
                    tool_call_wire("c1", "echo", {"text": "one"}),
                    tool_call_wire("c2", "echo", {"text": "two"}),
                ]
            ),
            text_events("done"),
        ]
    )
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    events = []
    config = AgentLoopConfig(model=model, run_id="run-2")
    result = await run_agent_loop(
        [UserMessage(content="go")], context, config, events.append
    )

    assert result.status is AgentRunStatus.COMPLETED
    roles = [message.role for message in context.messages]
    assert roles == ["user", "assistant", "tool", "tool", "assistant"]
    tool_messages = [
        message for message in context.messages if message.role == "tool"
    ]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]
    assert [m.result.output for m in tool_messages] == ["echo:one", "echo:two"]
    # Second request carries the paired assistant -> tool wire messages.
    second = model.requests[1]
    assert [msg["role"] for msg in second.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert second.messages[1]["tool_calls"][0]["id"] == "c1"
    assert second.messages[2]["tool_call_id"] == "c1"


@pytest.mark.asyncio
async def test_tool_schema_is_projected_into_request_options() -> None:
    class RecordingModel(ScriptedModel):
        def build_tool_schema_request_options(self, payload, *, protocol=None, delivery="prompt_injection"):
            self.seen_payload = payload
            return {"tools": [{"type": "function"}]}

    model = RecordingModel([text_events("ok")])
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(model=model, run_id="run-tools")
    await run_agent_loop([UserMessage(content="go")], context, config, None)
    payload = model.seen_payload
    assert any(item["function"]["name"] == "echo" for item in payload)
    assert model.requests[0].option_dict()["tools"] == [{"type": "function"}]


@pytest.mark.asyncio
async def test_unknown_tool_gets_terminal_error_without_execution() -> None:
    model = ScriptedModel(
        [tool_events([tool_call_wire("c1", "missing", {})]), text_events("ok")]
    )
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(model=model, run_id="run-3")
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.COMPLETED
    tool_message = context.messages[2]
    assert isinstance(tool_message, ToolResultMessage)
    assert tool_message.result.status == "error"
    assert tool_message.result.metadata["error_category"] == "tool_not_found"
    assert tool_message.result.metadata["started"] is False


@pytest.mark.asyncio
async def test_invalid_arguments_get_terminal_error_without_execution() -> None:
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"unknown": 1})]),
            text_events("ok"),
        ]
    )
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(model=model, run_id="run-4")
    await run_agent_loop([UserMessage(content="go")], context, config, None)

    tool_message = context.messages[2]
    assert tool_message.result.status == "error"
    assert tool_message.result.metadata["error_category"] == "invalid_tool_arguments"
    assert tool_message.result.metadata["started"] is False


@pytest.mark.asyncio
async def test_malformed_call_arguments_become_admission_error() -> None:
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", "{not json")]),
            text_events("ok"),
        ]
    )
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(model=model, run_id="run-5")
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.COMPLETED
    assistant = context.messages[1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.tool_calls[0].parse_error is not None
    tool_message = context.messages[2]
    assert tool_message.result.metadata["error_category"] == "invalid_tool_arguments"


@pytest.mark.asyncio
async def test_truncated_message_fails_whole_batch_without_execution() -> None:
    executed = []

    @tool(name="tracker")
    def _tracker(text: str) -> str:
        executed.append(text)
        return text

    model = ScriptedModel(
        [
            tool_events(
                [
                    tool_call_wire("c1", "tracker", {"text": "a"}),
                    tool_call_wire("c2", "tracker", {"text": "b"}),
                ],
                finish_reason="length",
            ),
            text_events("recovered"),
        ]
    )
    context = AgentContext(messages=[], tools=_registry(_tracker).freeze())
    config = AgentLoopConfig(model=model, run_id="run-6")
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.COMPLETED
    assert executed == []
    results = [
        message.result for message in context.messages if message.role == "tool"
    ]
    assert len(results) == 2
    assert all(r.status == "error" for r in results)
    assert all(r.metadata["error_category"] == "arguments_truncated" for r in results)


@pytest.mark.asyncio
async def test_steering_message_injected_before_next_model_call() -> None:
    steering_calls = 0

    def _drain_steering():
        nonlocal steering_calls
        steering_calls += 1
        if steering_calls == 2:
            return [UserMessage(content="steer me")]
        return []

    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "x"})]),
            text_events("done"),
        ]
    )
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(
        model=model,
        run_id="run-7",
        get_steering_messages=_drain_steering,
    )
    await run_agent_loop([UserMessage(content="go")], context, config, None)

    roles = [message.role for message in context.messages]
    assert roles == ["user", "assistant", "tool", "user", "assistant"]
    second = model.requests[1]
    # Steering enters the transcript before the following model request.
    assert second.messages[-1] == {"role": "user", "content": "steer me"}


@pytest.mark.asyncio
async def test_follow_up_message_revives_a_stopped_run() -> None:
    follow_ups = [[UserMessage(content="again")], []]
    model = ScriptedModel([text_events("first"), text_events("second")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(
        model=model,
        run_id="run-8",
        get_follow_up_messages=lambda: follow_ups.pop(0),
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.COMPLETED
    assistants = [
        message.text
        for message in context.messages
        if isinstance(message, AssistantMessage)
    ]
    assert assistants == ["first", "second"]


@pytest.mark.asyncio
async def test_abort_before_model_admission_produces_terminal_message() -> None:
    model = ScriptedModel([text_events("never")])
    token = CancelToken()
    token.request_cancel("immediate")
    context = AgentContext(messages=[])
    events = []
    config = AgentLoopConfig(model=model, run_id="run-9")
    result = await run_agent_loop(
        [UserMessage(content="go")], context, config, events.append, token
    )

    assert result.status is AgentRunStatus.ABORTED
    assert model.requests == []  # no model admission after cancellation
    assistant = context.messages[-1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.error
    # turn and message events stay balanced.
    assert _event_types(events).count("turn_start") == _event_types(events).count(
        "turn_end"
    )
    assert _event_types(events)[-1] == "agent_end"


@pytest.mark.asyncio
async def test_abort_mid_stream_settles_aborted() -> None:
    gate = asyncio.Event()
    token = CancelToken()
    model = ScriptedModel([make_hanging_model(gate, first_text="partial")])
    context = AgentContext(messages=[])
    events = []
    config = AgentLoopConfig(model=model, run_id="run-10")

    async def _watch(event) -> None:
        events.append(event)
        if isinstance(event, MessageUpdate):
            token.request_cancel("immediate")

    result = await run_agent_loop(
        [UserMessage(content="go")], context, config, _watch, token
    )

    assert result.status is AgentRunStatus.ABORTED
    assistant = context.messages[-1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.text == "partial"
    assert assistant.error
    gate.set()


@pytest.mark.asyncio
async def test_model_failure_event_produces_failed_run() -> None:
    model = ScriptedModel([failed_events("provider exploded")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(model=model, run_id="run-11")
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.FAILED
    assert "provider exploded" in (result.error or "")
    assistant = context.messages[-1]
    assert assistant.failed


@pytest.mark.asyncio
async def test_failed_assistant_messages_are_skipped_in_later_requests() -> None:
    model = ScriptedModel([failed_events("boom"), text_events("ok")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(model=model, run_id="run-12")
    first = await run_agent_loop([UserMessage(content="go")], context, config, None)
    assert first.status is AgentRunStatus.FAILED

    second = await run_agent_loop(
        [UserMessage(content="retry")], context, config, None
    )
    assert second.status is AgentRunStatus.COMPLETED
    wire = model.requests[1].messages
    assert [msg["role"] for msg in wire] == ["user", "user"]


@pytest.mark.asyncio
async def test_stream_without_terminal_event_fails_the_run() -> None:
    model = ScriptedModel(
        [[ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text="dangling")]]
    )
    context = AgentContext(messages=[])
    config = AgentLoopConfig(model=model, run_id="run-13")
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.FAILED
    assert context.messages[-1].failed


@pytest.mark.asyncio
async def test_expired_deadline_blocks_model_admission() -> None:
    model = ScriptedModel([text_events("never")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(
        model=model,
        run_id="run-14",
        deadline_monotonic=time.monotonic() - 1,
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.DEADLINE_EXCEEDED
    assert model.requests == []


@pytest.mark.asyncio
async def test_max_turns_stops_a_tool_calling_run() -> None:
    model = ScriptedModel(
        [tool_events([tool_call_wire(f"c{i}", "echo", {"text": "x"})]) for i in range(5)]
    )
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(model=model, run_id="run-15", max_turns=2)
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.MAX_TURNS
    assert len(model.requests) == 2


@pytest.mark.asyncio
async def test_should_stop_after_turn_hook_ends_run() -> None:
    model = ScriptedModel(
        [tool_events([tool_call_wire("c1", "echo", {"text": "x"})]), text_events("late")]
    )
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(
        model=model,
        run_id="run-16",
        should_stop_after_turn=lambda hook: True,
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.COMPLETED
    assert len(model.requests) == 1


@pytest.mark.asyncio
async def test_prepare_next_turn_swaps_system_prompt_and_model() -> None:
    replacement = ScriptedModel([text_events("new model")], model="second", provider_name="scripted")
    first = ScriptedModel([tool_events([tool_call_wire("c1", "echo", {"text": "x"})])])
    context = AgentContext(
        system_prompt="old", messages=[], tools=_registry(_echo).freeze()
    )
    config = AgentLoopConfig(
        model=first,
        run_id="run-17",
        prepare_next_turn=lambda hook: NextTurnUpdate(
            system_prompt="new", model=replacement
        ),
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.COMPLETED
    assert replacement.requests[0].messages[0] == {
        "role": "system",
        "content": "new",
    }
    assert context.messages[-1].model_name == "second"


@pytest.mark.asyncio
async def test_before_tool_call_block_denies_and_terminates() -> None:
    from qitos.core.tool_executor import BeforeToolCallDecision

    model = ScriptedModel(
        [tool_events([tool_call_wire("c1", "echo", {"text": "x"})])]
    )
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(
        model=model,
        run_id="run-18",
        before_tool_call=lambda hook: BeforeToolCallDecision(
            block=True, reason="policy", terminate=True
        ),
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.COMPLETED
    tool_message = context.messages[2]
    assert tool_message.result.status == "denied"
    assert tool_message.result.metadata["terminate"] is True
    # terminate stops the batch: no second model call.
    assert len(model.requests) == 1


@pytest.mark.asyncio
async def test_after_tool_call_replaces_the_result() -> None:
    from qitos.core.tool_result import ToolResult

    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "x"})]),
            text_events("done"),
        ]
    )
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(
        model=model,
        run_id="run-19",
        after_tool_call=lambda hook: ToolResult(status="success", output="override"),
    )
    await run_agent_loop([UserMessage(content="go")], context, config, None)

    tool_message = context.messages[2]
    assert tool_message.result.output == "override"


@pytest.mark.asyncio
async def test_transform_context_rewrites_the_wire_messages() -> None:
    model = ScriptedModel([text_events("ok")])
    context = AgentContext(messages=[])

    def _transform(messages):
        return [UserMessage(content="rewritten")]

    config = AgentLoopConfig(
        model=model, run_id="run-20", transform_context=_transform
    )
    await run_agent_loop([UserMessage(content="original")], context, config, None)

    wire = model.requests[0].messages
    assert [msg["content"] for msg in wire] == ["rewritten"]
    # The canonical transcript keeps the original message.
    assert context.messages[0].content == "original"


@pytest.mark.asyncio
async def test_transaction_barriers_wrap_model_and_tool_side_effects() -> None:
    model = ScriptedModel(
        [tool_events([tool_call_wire("c1", "echo", {"text": "x"})]), text_events("done")]
    )
    transaction = RecordingTransaction()
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(
        model=model, run_id="run-21", transaction=transaction
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.COMPLETED
    kinds = [record[0] for record in transaction.records]
    assert kinds == [
        "model_terminal",
        "tool_started",
        "tool_terminal",
        "turn_committed",
        "model_terminal",
        "turn_committed",
        "run_terminal",
    ]
    assert transaction.records[-1] == ("run_terminal", "completed")


@pytest.mark.asyncio
async def test_transaction_records_terminal_for_unexecuted_calls_on_abort() -> None:
    token = CancelToken()

    @tool(name="canceller")
    def _canceller(text: str) -> str:
        token.request_cancel("immediate")
        return text

    model = ScriptedModel(
        [
            tool_events(
                [
                    tool_call_wire("c1", "canceller", {"text": "x"}),
                    tool_call_wire("c2", "echo", {"text": "y"}),
                ]
            )
        ]
    )
    transaction = RecordingTransaction()
    context = AgentContext(messages=[], tools=_registry(_canceller, _echo).freeze())
    config = AgentLoopConfig(
        model=model, run_id="run-22", transaction=transaction
    )
    result = await run_agent_loop(
        [UserMessage(content="go")], context, config, None, token
    )

    assert result.status is AgentRunStatus.ABORTED
    started = [r for r in transaction.records if r[0] == "tool_started"]
    terminal = [r for r in transaction.records if r[0] == "tool_terminal"]
    assert [r[2] for r in started] == ["c1", "c2"]
    assert [r[2] for r in terminal] == ["c1", "c2"]
    assert terminal[1][3] == "cancelled"
    tool_messages = [
        message for message in context.messages if message.role == "tool"
    ]
    assert len(tool_messages) == 2
    assert tool_messages[1].result.metadata["started"] is False


@pytest.mark.asyncio
async def test_continuation_offered_after_tool_results() -> None:
    continuation = ModelContinuation(
        run_id="run-23",
        provider="scripted",
        model="scripted-model",
        protocol="native",
        response_id="resp-1",
        prefix_items=1,
        prefix_digest="d",
        settings_digest="s",
    )
    first = tool_events([tool_call_wire("c1", "echo", {"text": "x"})])
    first[-1] = ModelStreamEvent(
        type=ModelStreamEventType.COMPLETED,
        finish_reason="tool_calls",
        tool_calls=[tool_call_wire("c1", "echo", {"text": "x"})],
        continuation=continuation,
    )
    model = ScriptedModel([first, text_events("done")])
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(model=model, run_id="run-23")
    await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert model.requests[0].continuation is None
    # Provider continuation chains across the intervening Tool results.
    assert model.requests[1].continuation == continuation


@pytest.mark.asyncio
async def test_continuation_reused_when_assistant_is_tail() -> None:
    continuation = ModelContinuation(
        run_id="run-24",
        provider="scripted",
        model="scripted-model",
        protocol="native",
        response_id="resp-9",
        prefix_items=1,
        prefix_digest="d",
        settings_digest="s",
    )
    completed = ModelStreamEvent(
        type=ModelStreamEventType.COMPLETED,
        finish_reason="stop",
        continuation=continuation,
    )
    model = ScriptedModel([[completed], text_events("next")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(model=model, run_id="run-24")
    await run_agent_loop([UserMessage(content="one")], context, config, None)
    await run_agent_loop([UserMessage(content="two")], context, config, None)

    assert model.requests[1].continuation == continuation


@pytest.mark.asyncio
async def test_continue_requires_non_assistant_tail() -> None:
    model = ScriptedModel([text_events("ok"), text_events("continued")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(model=model, run_id="run-25")
    await run_agent_loop([UserMessage(content="go")], context, config, None)

    with pytest.raises(ValueError):
        await run_agent_loop_continue(context, config, None)

    context.messages.append(UserMessage(content="more"))
    result = await run_agent_loop_continue(context, config, None)
    assert result.status is AgentRunStatus.COMPLETED
    assert context.messages[-1].text == "continued"


@pytest.mark.asyncio
async def test_event_stream_yields_events_then_result() -> None:
    model = ScriptedModel([text_events("hi")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(model=model, run_id="run-26")
    stream = agent_loop([UserMessage(content="go")], context, config)

    seen = []
    async for event in stream:
        seen.append(event.type)
    result = await stream.result()

    assert seen[0] == "agent_start"
    assert seen[-1] == "agent_end"
    assert result.status is AgentRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_event_stream_consumer_cancellation_aborts_the_run() -> None:
    gate = asyncio.Event()
    model = ScriptedModel([make_hanging_model(gate, first_text="chunk")])
    context = AgentContext(messages=[])
    token = CancelToken()
    config = AgentLoopConfig(model=model, run_id="run-27")
    stream = agent_loop([UserMessage(content="go")], context, config, token)

    # Drain every event the producer can emit before it blocks on the model
    # stream: agent_start, turn_start, prompt message_start/end, then the
    # partial assistant message_start and its first message_update.
    for _ in range(6):
        await stream.__anext__()

    blocked = asyncio.Event()

    async def _consume() -> None:
        blocked.set()
        await stream.__anext__()

    consumer = asyncio.create_task(_consume())
    await blocked.wait()
    await asyncio.sleep(0)  # let the consumer block on the empty queue
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    try:
        result = await asyncio.wait_for(stream.result(), timeout=5)
    finally:
        gate.set()
    assert result.status is AgentRunStatus.ABORTED


@pytest.mark.asyncio
async def test_parallel_segments_overlap_and_results_keep_input_order() -> None:
    started: List[str] = []
    release = asyncio.Event()

    @tool(name="safe", concurrency_safe=True)
    async def _safe(text: str) -> str:
        started.append(text)
        await release.wait()
        return text

    model = ScriptedModel(
        [
            tool_events(
                [
                    tool_call_wire("c1", "safe", {"text": "a"}),
                    tool_call_wire("c2", "safe", {"text": "b"}),
                ]
            ),
            text_events("done"),
        ]
    )
    context = AgentContext(messages=[], tools=_registry(_safe).freeze())
    config = AgentLoopConfig(
        model=model, run_id="run-28", tool_execution="parallel"
    )

    async def _release_later() -> None:
        while len(started) < 2:
            await asyncio.sleep(0.01)
        release.set()

    releaser = asyncio.create_task(_release_later())
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)
    await releaser

    assert result.status is AgentRunStatus.COMPLETED
    assert started == ["a", "b"]  # both entered before either finished
    tool_messages = [
        message for message in context.messages if message.role == "tool"
    ]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]
