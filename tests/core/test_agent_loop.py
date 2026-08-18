"""Behavioral conformance for the minimal agent loop.

These tests express the loop contract without referencing the retired
AgentModule/Engine lifecycle: turn structure, steering/follow-up safe points,
ToolCall/ToolResult pairing, cancellation, deadlines, ordering and the
transaction barriers.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import List

import pytest

from qitos.core.agent_events import (
    MessageUpdate,
    TurnEnd,
)
from qitos.core.agent_loop import (
    AgentContext,
    AgentLoopConfig,
    AgentRunStatus,
    NextTurnUpdate,
    RunFinalizationDiagnosticCode,
    agent_loop,
    run_agent_loop,
    run_agent_loop_continue,
)
from qitos.core.cancellation import CancelMode, CancelToken
from qitos.core.message import (
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from qitos.core.model_request import ModelContinuation
from qitos.core.model_response import ModelUsage
from qitos.core.model_stream import ModelStreamEvent, ModelStreamEventType
from qitos.core.tool import tool
from qitos.core.tool_registry import ToolRegistry
from qitos.core.tool_result import ToolResult

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
async def test_env_permission_context_cannot_be_replaced_by_runtime_context() -> None:
    env_authority = object()
    forged_authority = object()
    observed = []

    @tool(name="inspect_permission_context")
    def _inspect_permission_context(runtime_context=None) -> str:
        observed.append(runtime_context["permission_context"])
        return "ok"

    model = ScriptedModel(
        [
            tool_events(
                [tool_call_wire("c1", "inspect_permission_context", {})]
            ),
            text_events("done"),
        ]
    )
    context = AgentContext(
        messages=[],
        tools=_registry(_inspect_permission_context).freeze(),
        env=SimpleNamespace(tool_permission_context=env_authority),
    )
    config = AgentLoopConfig(
        model=model,
        run_id="run-env-authority",
        runtime_context={"permission_context": forged_authority},
    )

    result = await run_agent_loop(
        [UserMessage(content="inspect")], context, config, None
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert observed == [env_authority]


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
async def test_duplicate_provider_call_ids_preserve_evidence_and_fail_closed() -> None:
    executed: list[str] = []

    @tool(name="tracked")
    def _tracked(text: str) -> str:
        executed.append(text)
        return text

    model = ScriptedModel(
        [
            tool_events(
                [
                    tool_call_wire("duplicate", "tracked", {"text": "a"}),
                    tool_call_wire("duplicate", "tracked", {"text": "b"}),
                ]
            )
        ]
    )
    transaction = RecordingTransaction()
    context = AgentContext(messages=[], tools=_registry(_tracked).freeze())
    config = AgentLoopConfig(
        model=model, run_id="run-duplicate", transaction=transaction
    )

    result = await run_agent_loop(
        [UserMessage(content="go")], context, config, None
    )

    assert result.status is AgentRunStatus.FAILED
    assert executed == []
    assistant = context.messages[-1]
    assert isinstance(assistant, AssistantMessage)
    assert [call.id for call in assistant.tool_calls] == ["duplicate", "duplicate"]
    assert [call.arguments["text"] for call in assistant.tool_calls] == ["a", "b"]
    assert assistant.error == "assistant tool call ids must be unique"
    assert [record[0] for record in transaction.records] == [
        "input_accepted",
        "turn_frozen",
        "model_terminal",
        "turn_committed",
        "run_terminal",
    ]


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
async def test_run_finalizer_settles_before_terminal_commit() -> None:
    order: List[str] = []

    class TerminalTransaction(RecordingTransaction):
        async def run_terminal(self, result) -> None:
            order.append("run_terminal")
            self.result = result
            await super().run_terminal(result)

    async def _finalize(run_id: str) -> None:
        assert run_id == "run-finalizer-order"
        order.append("finalizer")

    transaction = TerminalTransaction()
    config = AgentLoopConfig(
        model=ScriptedModel([text_events("done")]),
        run_id="run-finalizer-order",
        transaction=transaction,
        run_finalizer=_finalize,
    )

    result = await run_agent_loop(
        [UserMessage(content="go")], AgentContext(messages=[]), config, None
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert result.finalization_diagnostic is None
    assert order == ["finalizer", "run_terminal"]


@pytest.mark.asyncio
async def test_finalizer_failure_is_bounded_diagnostic_on_deadline_outcome() -> None:
    class TerminalTransaction(RecordingTransaction):
        async def run_terminal(self, result) -> None:
            self.result = result
            await super().run_terminal(result)

    async def _broken_finalizer(_run_id: str) -> None:
        raise RuntimeError("cleanup failed " + ("x" * 1000))

    transaction = TerminalTransaction()
    config = AgentLoopConfig(
        model=ScriptedModel([text_events("never")]),
        run_id="run-finalizer-deadline",
        deadline_monotonic=time.monotonic() - 1,
        transaction=transaction,
        run_finalizer=_broken_finalizer,
    )

    result = await run_agent_loop(
        [UserMessage(content="go")], AgentContext(messages=[]), config, None
    )

    assert result.status is AgentRunStatus.DEADLINE_EXCEEDED
    assert result.error == "model request deadline expired before admission"
    diagnostic = result.finalization_diagnostic
    assert diagnostic is not None
    assert (
        diagnostic.code
        is RunFinalizationDiagnosticCode.RESOURCE_QUIESCE_FAILED
    )
    assert diagnostic.message.startswith("cleanup failed")
    assert len(diagnostic.message) == 512
    assert transaction.result == result


@pytest.mark.asyncio
async def test_self_cancelled_finalizer_does_not_cancel_primary_outcome() -> None:
    async def _self_cancel(_run_id: str) -> None:
        raise asyncio.CancelledError()

    config = AgentLoopConfig(
        model=ScriptedModel([text_events("done")]),
        run_id="run-self-cancelled-finalizer",
        run_finalizer=_self_cancel,
    )

    result = await run_agent_loop(
        [UserMessage(content="go")], AgentContext(messages=[]), config, None
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert result.finalization_diagnostic is not None
    assert (
        result.finalization_diagnostic.code
        is RunFinalizationDiagnosticCode.RESOURCE_QUIESCE_FAILED
    )


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
async def test_max_turns_does_not_drain_queues_without_next_turn_capacity() -> None:
    steering_polls = 0
    follow_up_polls = 0

    def _steering():
        nonlocal steering_polls
        steering_polls += 1
        return []

    def _follow_up():
        nonlocal follow_up_polls
        follow_up_polls += 1
        return [UserMessage(content="still queued")]

    model = ScriptedModel([text_events("done")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(
        model=model,
        run_id="run-max-queue",
        max_turns=1,
        get_steering_messages=_steering,
        get_follow_up_messages=_follow_up,
    )

    result = await run_agent_loop(
        [UserMessage(content="go")], context, config, None
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert steering_polls == 1
    assert follow_up_polls == 0


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
async def test_after_tool_call_partial_override_replaces_single_fields() -> None:
    from qitos.core.tool_executor import AfterToolCallOverride

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
        after_tool_call=lambda hook: AfterToolCallOverride(output="override"),
    )
    await run_agent_loop([UserMessage(content="go")], context, config, None)

    tool_message = context.messages[2]
    # Field-level merge: output is replaced, the executed status is kept.
    assert tool_message.result.output == "override"
    assert tool_message.result.status == "success"


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
        "input_accepted",
        "turn_frozen",
        "model_terminal",
        "tool_started",
        "tool_terminal",
        "turn_committed",
        "turn_frozen",
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
        protocol="legacy",
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
        protocol="legacy",
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
    transaction = RecordingTransaction()
    config = AgentLoopConfig(
        model=model, run_id="run-27", transaction=transaction
    )
    stream = agent_loop([UserMessage(content="go")], context, config)

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
    # Consumer cancellation is not surfaced until the owned producer has
    # reached its durable run terminal.
    assert transaction.records[-1] == ("run_terminal", "aborted")
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


@pytest.mark.asyncio
async def test_after_step_stops_after_current_turn_commits() -> None:
    token = CancelToken()

    @tool(name="stopper")
    def _stopper(text: str) -> str:
        token.request_cancel("after_step")
        return text

    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "stopper", {"text": "x"})]),
            text_events("never reached"),
        ]
    )
    transaction = RecordingTransaction()
    context = AgentContext(messages=[], tools=_registry(_stopper).freeze())
    config = AgentLoopConfig(
        model=model, run_id="run-after-step", transaction=transaction
    )
    result = await run_agent_loop(
        [UserMessage(content="go")], context, config, None, token
    )

    assert result.status is AgentRunStatus.ABORTED
    # The current turn finished: the Tool executed and the turn committed.
    tool_message = context.messages[2]
    assert tool_message.result.status == "success"
    kinds = [record[0] for record in transaction.records]
    assert "turn_committed" in kinds
    # No further model call started after the stop request.
    assert len(model.requests) == 1
    # The step boundary is observable by the cancel owner.
    assert token.wait_for_step_complete(timeout=0) is True


def test_cancel_mode_only_strengthens() -> None:
    token = CancelToken()
    token.request_cancel("after_step")
    assert token.mode is CancelMode.AFTER_STEP
    token.request_cancel("immediate")
    token.request_cancel("after_step")
    assert token.mode is CancelMode.IMMEDIATE
    assert token.immediate_requested is True


@pytest.mark.asyncio
async def test_after_step_requested_before_admission_finishes_one_turn() -> None:
    token = CancelToken()
    token.request_cancel("after_step")
    transaction = RecordingTransaction()
    model = ScriptedModel([text_events("done")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(
        model=model, run_id="run-after-step-before", transaction=transaction
    )

    result = await run_agent_loop(
        [UserMessage(content="go")], context, config, None, token
    )

    assert result.status is AgentRunStatus.ABORTED
    assert len(model.requests) == 1
    assert [record[0] for record in transaction.records] == [
        "input_accepted",
        "turn_frozen",
        "model_terminal",
        "turn_committed",
        "run_terminal",
    ]


@pytest.mark.asyncio
async def test_after_step_waits_for_turn_boundary_hook_settlement() -> None:
    hook_started = asyncio.Event()
    release = asyncio.Event()

    async def _hook(_context) -> bool:
        hook_started.set()
        await release.wait()
        return False

    token = CancelToken()
    model = ScriptedModel([text_events("done")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(
        model=model,
        run_id="run-after-step-hook",
        should_stop_after_turn=_hook,
    )
    run = asyncio.create_task(
        run_agent_loop([UserMessage(content="go")], context, config, None, token)
    )
    await hook_started.wait()
    assert token.wait_for_step_complete(timeout=0) is False
    token.request_cancel("after_step")
    await asyncio.sleep(0)
    assert not run.done()
    release.set()

    result = await run
    assert result.status is AgentRunStatus.ABORTED
    assert len(model.requests) == 1
    assert token.wait_for_step_complete(timeout=0) is True


@pytest.mark.asyncio
async def test_step_complete_waits_for_turn_end_listener_settlement() -> None:
    turn_end_started = asyncio.Event()
    release = asyncio.Event()
    token = CancelToken()
    model = ScriptedModel([text_events("done")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(model=model, run_id="run-turn-end-settlement")

    async def _listener(event) -> None:
        if isinstance(event, TurnEnd):
            turn_end_started.set()
            await release.wait()

    run = asyncio.create_task(
        run_agent_loop([UserMessage(content="go")], context, config, _listener, token)
    )
    await turn_end_started.wait()
    assert token.wait_for_step_complete(timeout=0) is False

    release.set()
    result = await run

    assert result.status is AgentRunStatus.COMPLETED
    assert token.wait_for_step_complete(timeout=0) is True


@pytest.mark.asyncio
async def test_loop_task_cancellation_terminalizes_then_reraises() -> None:
    gate = asyncio.Event()  # never set: the model stream hangs mid-run
    model = ScriptedModel([make_hanging_model(gate, first_text="hi")])
    transaction = RecordingTransaction()
    context = AgentContext(messages=[])
    config = AgentLoopConfig(model=model, run_id="run-cancel", transaction=transaction)
    events = []
    streaming = asyncio.Event()

    def _sink(event) -> None:
        events.append(event)
        if isinstance(event, MessageUpdate):
            streaming.set()

    task = asyncio.create_task(
        run_agent_loop([UserMessage(content="go")], context, config, _sink)
    )
    await streaming.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Started work and the run reached durable terminal states first.
    kinds = [record[0] for record in transaction.records]
    assert kinds == [
        "input_accepted",
        "turn_frozen",
        "model_terminal",
        "run_terminal",
    ]
    assert transaction.records[-1] == ("run_terminal", "aborted")
    assistant = context.messages[-1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.failed and "cancelled" in str(assistant.error)


@pytest.mark.asyncio
async def test_task_cancellation_waits_for_run_finalizer_before_terminal() -> None:
    model_gate = asyncio.Event()
    finalizer_started = asyncio.Event()
    release_finalizer = asyncio.Event()
    finalizer_settled = asyncio.Event()

    class TerminalTransaction(RecordingTransaction):
        terminal_result = None

        async def run_terminal(self, result) -> None:
            self.terminal_result = result
            await super().run_terminal(result)

    async def _finalize(_run_id: str) -> None:
        finalizer_started.set()
        try:
            await release_finalizer.wait()
        finally:
            finalizer_settled.set()

    transaction = TerminalTransaction()
    config = AgentLoopConfig(
        model=ScriptedModel([make_hanging_model(model_gate, first_text="hi")]),
        run_id="run-cancel-finalizer",
        transaction=transaction,
        run_finalizer=_finalize,
    )
    streaming = asyncio.Event()

    def _sink(event) -> None:
        if isinstance(event, MessageUpdate):
            streaming.set()

    task = asyncio.create_task(
        run_agent_loop(
            [UserMessage(content="go")], AgentContext(messages=[]), config, _sink
        )
    )
    await streaming.wait()
    task.cancel()
    await finalizer_started.wait()

    assert transaction.terminal_result is None
    assert not task.done()

    release_finalizer.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finalizer_settled.is_set()
    assert transaction.terminal_result is not None
    assert transaction.terminal_result.status is AgentRunStatus.ABORTED


@pytest.mark.asyncio
async def test_owner_cancel_wins_when_finalizer_self_cancels_in_same_tick() -> None:
    finalizer_started = asyncio.Event()
    release_finalizer = asyncio.Event()

    class TerminalTransaction(RecordingTransaction):
        terminal_result = None

        async def run_terminal(self, result) -> None:
            self.terminal_result = result
            await super().run_terminal(result)

    async def _self_cancel(_run_id: str) -> None:
        finalizer_started.set()
        await release_finalizer.wait()
        raise asyncio.CancelledError()

    transaction = TerminalTransaction()
    config = AgentLoopConfig(
        model=ScriptedModel([text_events("done")]),
        run_id="run-owner-and-finalizer-cancel",
        transaction=transaction,
        run_finalizer=_self_cancel,
    )
    task = asyncio.create_task(
        run_agent_loop(
            [UserMessage(content="go")], AgentContext(messages=[]), config, None
        )
    )
    await finalizer_started.wait()

    release_finalizer.set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert transaction.terminal_result is not None
    assert transaction.terminal_result.status is AgentRunStatus.ABORTED


@pytest.mark.asyncio
async def test_loop_fault_keeps_primary_error_when_finalizer_also_fails() -> None:
    class FaultingTransaction(RecordingTransaction):
        terminal_result = None

        async def turn_frozen(self, turn, config) -> None:
            raise RuntimeError("primary persistence fault")

        async def run_terminal(self, result) -> None:
            self.terminal_result = result
            await super().run_terminal(result)

    async def _broken_finalizer(_run_id: str) -> None:
        raise RuntimeError("cleanup fault")

    transaction = FaultingTransaction()
    config = AgentLoopConfig(
        model=ScriptedModel([text_events("never")]),
        run_id="run-primary-fault-finalizer",
        transaction=transaction,
        run_finalizer=_broken_finalizer,
    )

    with pytest.raises(RuntimeError, match="primary persistence fault"):
        await run_agent_loop(
            [UserMessage(content="go")], AgentContext(messages=[]), config, None
        )

    terminal = transaction.terminal_result
    assert terminal is not None
    assert terminal.status is AgentRunStatus.FAILED
    assert terminal.error == "primary persistence fault"
    assert terminal.finalization_diagnostic is not None
    assert terminal.finalization_diagnostic.message == "cleanup fault"


@pytest.mark.asyncio
async def test_tools_registered_mid_run_are_exposed_next_turn() -> None:
    class RecordingModel(ScriptedModel):
        def build_tool_schema_request_options(
            self, payload, *, protocol=None, delivery="api_parameter"
        ):
            self.payloads.append(list(payload))
            return {}

    @tool(name="late")
    def _late(text: str) -> str:
        return text

    registry = _registry(_echo)
    model = RecordingModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "x"})]),
            text_events("done"),
        ]
    )
    model.payloads = []

    registered: List[bool] = []

    def _register_late(hook_context):
        # prepare_next_turn runs after every turn, including the last one.
        if not registered:
            registry.register(_late)
            registered.append(True)
        return None

    context = AgentContext(messages=[], tools=registry)  # live registry
    config = AgentLoopConfig(
        model=model, run_id="run-refresh", prepare_next_turn=_register_late
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.COMPLETED
    assert len(model.payloads) == 2
    assert [item["function"]["name"] for item in model.payloads[0]] == ["echo"]
    # The live registry is re-frozen per turn: the new Tool becomes visible.
    assert {item["function"]["name"] for item in model.payloads[1]} == {
        "echo",
        "late",
    }


@pytest.mark.asyncio
async def test_next_turn_update_replaces_history_for_next_request() -> None:
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "x"})]),
            text_events("done"),
        ]
    )
    config = AgentLoopConfig(
        model=model,
        run_id="run-replace",
        prepare_next_turn=lambda ctx: NextTurnUpdate(
            messages=(UserMessage(content="compressed summary"),)
        ),
    )
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    await run_agent_loop([UserMessage(content="go")], context, config, None)

    second = model.requests[1]
    assert [message["role"] for message in second.messages] == ["user"]
    assert second.messages[0]["content"] == "compressed summary"


@pytest.mark.asyncio
async def test_should_stop_observes_prepare_next_turn_context() -> None:
    observed: List[str] = []

    def _should_stop(hook_context) -> bool:
        observed.extend(
            message.content
            for message in hook_context.context.messages
            if isinstance(message, UserMessage)
            and isinstance(message.content, str)
        )
        return True

    model = ScriptedModel([text_events("done")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(
        model=model,
        run_id="run-stop-updated-context",
        prepare_next_turn=lambda _context: NextTurnUpdate(
            messages=(UserMessage(content="prepared context"),)
        ),
        should_stop_after_turn=_should_stop,
    )

    result = await run_agent_loop(
        [UserMessage(content="original")], context, config, None
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert observed == ["prepared context"]


@pytest.mark.asyncio
async def test_standalone_usage_event_accumulates() -> None:
    events = [
        ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text="hi"),
        ModelStreamEvent(
            type=ModelStreamEventType.USAGE,
            usage={"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
        ),
        ModelStreamEvent(type=ModelStreamEventType.COMPLETED, finish_reason="stop"),
    ]
    model = ScriptedModel([events])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(model=model, run_id="run-usage")
    await run_agent_loop([UserMessage(content="go")], context, config, None)

    assistant = context.messages[1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.usage is not None
    assert assistant.usage.total_tokens == 8


@pytest.mark.asyncio
async def test_request_protocol_uses_model_api_identity() -> None:
    from qitos.core.model_capabilities import ModelAPI, ModelCapabilities

    class ResponsesModel(ScriptedModel):
        schema_protocols: List[str] = []

        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(api=ModelAPI.RESPONSES)

        def build_tool_schema_request_options(
            self, payload, *, protocol=None, delivery="api_parameter"
        ):
            self.schema_protocols.append(protocol)
            return {}

    model = ResponsesModel([text_events("ok")])
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(model=model, run_id="run-proto")
    await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert model.requests[0].protocol == "responses"
    assert model.schema_protocols == ["responses"]


@pytest.mark.asyncio
async def test_after_tool_call_override_terminate_stops_the_batch() -> None:
    from qitos.core.tool_executor import AfterToolCallOverride

    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "x"})]),
            text_events("never reached"),
        ]
    )
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(
        model=model,
        run_id="run-terminate",
        after_tool_call=lambda ctx: AfterToolCallOverride(terminate=True),
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.COMPLETED
    assert context.messages[2].result.metadata["terminate"] is True
    assert len(model.requests) == 1


@pytest.mark.asyncio
async def test_hung_turn_hook_cannot_block_abort() -> None:
    hook_started = asyncio.Event()
    hook_settled = asyncio.Event()

    async def _hung(hook_context) -> bool:
        try:
            hook_started.set()
            await asyncio.Event().wait()
            return False
        finally:
            hook_settled.set()

    token = CancelToken()
    model = ScriptedModel([text_events("ok")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(
        model=model, run_id="run-hung-hook", should_stop_after_turn=_hung
    )
    run = asyncio.create_task(
        run_agent_loop([UserMessage(content="go")], context, config, None, token)
    )
    await hook_started.wait()
    token.request_cancel("immediate")
    result = await asyncio.wait_for(run, timeout=5)
    assert result.status is AgentRunStatus.ABORTED
    assert hook_settled.is_set()


# ── typed thinking level ────────────────────────────────────────────────────


def _thinking_capabilities(*levels: str):
    from qitos.core.model_capabilities import ModelCapabilities
    from qitos.core.thinking import ThinkingLevel

    return ModelCapabilities(
        thinking_levels=tuple(ThinkingLevel(level) for level in levels)
    )


@pytest.mark.asyncio
async def test_config_thinking_level_lands_on_the_turn_request() -> None:
    from qitos.core.thinking import ThinkingLevel

    model = ScriptedModel(
        [text_events("ok")],
        capabilities=_thinking_capabilities("off", "low", "high"),
    )
    context = AgentContext(messages=[])
    config = AgentLoopConfig(
        model=model, run_id="run-thinking", thinking_level=ThinkingLevel.LOW
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.COMPLETED
    assert model.requests[0].thinking_level is ThinkingLevel.LOW


@pytest.mark.asyncio
async def test_thinking_level_clamps_to_the_model_capability() -> None:
    from qitos.core.thinking import ThinkingLevel

    model = ScriptedModel(
        [text_events("ok")], capabilities=_thinking_capabilities("low")
    )
    context = AgentContext(messages=[])
    config = AgentLoopConfig(
        model=model, run_id="run-thinking-clamp", thinking_level=ThinkingLevel.MAX
    )
    await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert model.requests[0].thinking_level is ThinkingLevel.LOW


@pytest.mark.asyncio
async def test_thinking_level_drops_when_the_model_has_no_typed_support() -> None:
    from qitos.core.thinking import ThinkingLevel

    model = ScriptedModel([text_events("ok")])
    context = AgentContext(messages=[])
    config = AgentLoopConfig(
        model=model, run_id="run-thinking-none", thinking_level=ThinkingLevel.HIGH
    )
    await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert model.requests[0].thinking_level is None


@pytest.mark.asyncio
async def test_prepare_next_turn_thinking_level_applies_from_the_next_turn() -> None:
    from qitos.core.thinking import ThinkingLevel

    capabilities = _thinking_capabilities("off", "low", "high", "max")
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "x"})]),
            text_events("done"),
        ],
        capabilities=capabilities,
    )
    context = AgentContext(messages=[], tools=_registry(_echo).freeze())
    config = AgentLoopConfig(
        model=model,
        run_id="run-thinking-update",
        thinking_level=ThinkingLevel.LOW,
        prepare_next_turn=lambda hook: NextTurnUpdate(
            thinking_level=ThinkingLevel.MAX
        ),
    )
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.COMPLETED
    # The in-flight turn keeps its frozen snapshot; the replacement applies
    # to the next turn's request only.
    assert model.requests[0].thinking_level is ThinkingLevel.LOW
    assert model.requests[1].thinking_level is ThinkingLevel.MAX


@pytest.mark.asyncio
async def test_loop_config_rejects_an_untyped_thinking_level() -> None:
    model = ScriptedModel([text_events("ok")])
    with pytest.raises(TypeError, match="ThinkingLevel"):
        AgentLoopConfig(model=model, run_id="run-bad", thinking_level="high")


# ── typed ToolResult usage and added Tool names ──────────────────────────────


@tool(name="activating_echo")
def _activating_echo(text: str) -> ToolResult:
    return ToolResult(
        output={"echo": text},
        usage=ModelUsage.from_mapping({"total_tokens": 7, "cost_usd": 0.0001}),
        added_tool_names=("loaded_skill_tool",),
    )


@pytest.mark.asyncio
async def test_tool_result_usage_and_added_names_reach_the_committed_message() -> None:
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "activating_echo", {"text": "x"})]),
            text_events("done"),
        ]
    )
    context = AgentContext(
        messages=[], tools=_registry(_activating_echo).freeze()
    )
    config = AgentLoopConfig(model=model, run_id="run-result-facts")
    result = await run_agent_loop([UserMessage(content="go")], context, config, None)

    assert result.status is AgentRunStatus.COMPLETED
    tool_message = context.messages[2]
    assert isinstance(tool_message, ToolResultMessage)
    assert tool_message.usage is not None
    assert tool_message.usage.total_tokens == 7
    assert tool_message.usage["cost_usd"] == pytest.approx(0.0001)
    assert tool_message.added_tool_names == ("loaded_skill_tool",)
    assert tool_message.result.usage is tool_message.usage
    assert tool_message.result.added_tool_names == ("loaded_skill_tool",)
