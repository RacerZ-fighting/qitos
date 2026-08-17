"""Typed message contract tests: codec round trips and wire projections."""

from __future__ import annotations

import json

import pytest

from qitos.core._freeze import thaw_deep
from qitos.core.message import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    assistant_from_response,
    message_from_dict,
    message_to_dict,
    message_to_wire,
)
from qitos.core.model_request import ModelContinuation
from qitos.core.model_response import ModelResponse
from qitos.core.tool_result import ToolResult


def test_user_message_wire_and_dict_round_trip() -> None:
    message = UserMessage(content="hello", timestamp=12.5)
    assert message_to_wire(message) == {"role": "user", "content": "hello"}
    decoded = message_from_dict(message_to_dict(message))
    assert isinstance(decoded, UserMessage)
    assert decoded.content == "hello"
    assert decoded.timestamp == 12.5


def test_user_message_block_content_round_trip() -> None:
    message = UserMessage(
        content=(
            TextContent(text="look"),
            ImageContent(source={"type": "image_url", "image_url": {"url": "x"}}),
        )
    )
    wire = message_to_wire(message)
    assert wire["role"] == "user"
    assert wire["content"][0] == {"type": "text", "text": "look"}
    assert wire["content"][1]["type"] == "image_url"
    decoded = message_from_dict(message_to_dict(message))
    assert isinstance(decoded, UserMessage)
    assert isinstance(decoded.content, tuple)
    assert decoded.content[0] == TextContent(text="look")
    assert isinstance(decoded.content[1], ImageContent)


def test_tool_call_wire_shape_matches_adapter_contract() -> None:
    call = ToolCall(id="c1", name="echo", arguments={"text": "hi"})
    wire = call.to_wire()
    assert wire["type"] == "function"
    assert wire["function"]["name"] == "echo"
    assert json.loads(wire["function"]["arguments"]) == {"text": "hi"}
    decoded = ToolCall.from_wire(wire)
    assert decoded.id == "c1"
    assert dict(decoded.arguments) == {"text": "hi"}
    assert decoded.parse_error is None


def test_tool_call_from_wire_malformed_arguments_stays_in_transcript() -> None:
    call = ToolCall.from_wire(
        {
            "id": "c2",
            "type": "function",
            "function": {"name": "echo", "arguments": "{not json"},
        }
    )
    assert call.parse_error is not None
    assert dict(call.arguments) == {}
    round_tripped = ToolCall.from_dict(call.to_dict())
    assert round_tripped.parse_error == call.parse_error


def test_assistant_message_wire_projection() -> None:
    message = AssistantMessage(
        text="",
        tool_calls=(ToolCall(id="c1", name="echo", arguments={"a": 1}),),
        reasoning_content="thinking",
        native_items=({"type": "reasoning", "id": "r1"},),
    )
    wire = message_to_wire(message)
    assert wire["role"] == "assistant"
    assert wire["content"] is None
    assert wire["reasoning_content"] == "thinking"
    assert wire["tool_calls"][0]["function"]["name"] == "echo"
    assert wire["native_items"] == [{"type": "reasoning", "id": "r1"}]


def test_assistant_message_dict_round_trip_preserves_usage_and_continuation() -> None:
    continuation = ModelContinuation(
        run_id="run",
        provider="p",
        model="m",
        protocol="native",
        response_id="resp-1",
        prefix_items=2,
        prefix_digest="d",
        settings_digest="s",
    )
    message = AssistantMessage(
        text="answer",
        usage=None,
        finish_reason="stop",
        continuation=continuation,
        model_name="m",
        provider="p",
        metadata={"k": "v"},
        timestamp=99.0,
    )
    decoded = message_from_dict(message_to_dict(message))
    assert isinstance(decoded, AssistantMessage)
    assert decoded.text == "answer"
    assert decoded.continuation == continuation
    assert decoded.metadata == {"k": "v"}
    assert decoded.timestamp == 99.0


def test_tool_result_message_wire_uses_model_visible_output() -> None:
    result = ToolResult(
        status="success",
        output={"full": "payload"},
        model_output="short summary",
        call_id="c1",
    )
    message = ToolResultMessage(tool_call_id="c1", tool_name="echo", result=result)
    wire = message_to_wire(message)
    assert wire == {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "echo",
        "content": "short summary",
        "is_error": False,
    }
    decoded = message_from_dict(message_to_dict(message))
    assert isinstance(decoded, ToolResultMessage)
    assert decoded.result.to_dict() == result.to_dict()


def test_tool_result_message_non_string_output_is_json_encoded() -> None:
    result = ToolResult(status="success", output={"rows": [1, 2]}, call_id="c9")
    wire = message_to_wire(
        ToolResultMessage(tool_call_id="c9", tool_name="query", result=result)
    )
    assert json.loads(wire["content"]) == {"rows": [1, 2]}


def test_assistant_from_response_maps_tool_calls_and_reasoning() -> None:
    response = ModelResponse(
        text="working",
        reasoning_content="chain",
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "echo", "arguments": '{"a": 1}'},
            }
        ],
        model_name="m",
        provider="p",
    )
    message = assistant_from_response(response)
    assert message.text == "working"
    assert message.reasoning_content == "chain"
    assert [call.name for call in message.tool_calls] == ["echo"]
    assert dict(message.tool_calls[0].arguments) == {"a": 1}
    assert not message.failed
    assert not message.truncated


def test_assistant_truncated_detection() -> None:
    message = AssistantMessage(finish_reason="length")
    assert message.truncated
    assert not AssistantMessage(finish_reason="stop").truncated


def test_message_from_dict_rejects_unknown_role() -> None:
    with pytest.raises(ValueError):
        message_from_dict({"role": "system-ish", "content": "x"})


def test_error_tool_result_wire_carries_error_text_and_is_error() -> None:
    # Truncated-argument and admission-failure results carry no output; the
    # error text must still reach the model instead of a JSON null.
    result = ToolResult(status="error", output=None, error="boom", call_id="c1")
    message = ToolResultMessage(tool_call_id="c1", tool_name="echo", result=result)
    wire = message_to_wire(message)
    assert wire["content"] == "boom"
    assert wire["is_error"] is True


def test_message_from_dict_fails_closed_on_non_text_fields() -> None:
    with pytest.raises(ValueError):
        message_from_dict({"role": "assistant", "text": 42, "timestamp": 1.0})
    with pytest.raises(ValueError):
        message_from_dict(
            {
                "role": "tool",
                "tool_call_id": 42,
                "tool_name": "echo",
                "result": ToolResult(status="success", output="x").to_dict(),
                "timestamp": 1.0,
            }
        )
    with pytest.raises(ValueError):
        message_from_dict(
            {
                "role": "assistant",
                "text": "ok",
                "tool_calls": [{"id": 7, "name": "echo", "arguments": {}}],
                "timestamp": 1.0,
            }
        )


def test_tool_result_message_result_is_deeply_immutable() -> None:
    result = ToolResult(
        status="success",
        output={"rows": [1]},
        metadata={"nested": {"count": 1}},
        call_id="c1",
    )
    message = ToolResultMessage(tool_call_id="c1", tool_name="echo", result=result)
    with pytest.raises(TypeError):
        message.result.output["rows"] = []  # type: ignore[index]
    with pytest.raises(TypeError):
        message.result.metadata["nested"]["count"] = 2  # type: ignore[index]
    # The durable codec still round-trips the thawed plain shape.
    decoded = message_from_dict(message_to_dict(message))
    assert isinstance(decoded, ToolResultMessage)
    assert decoded.result.to_dict() == result.to_dict()


def test_assistant_metadata_is_deeply_immutable() -> None:
    message = AssistantMessage(text="hi", metadata={"nested": {"count": 1}})
    with pytest.raises(TypeError):
        message.metadata["nested"]["count"] = 2  # type: ignore[index]
    decoded = message_from_dict(message_to_dict(message))
    assert isinstance(decoded, AssistantMessage)
    assert thaw_deep(decoded.metadata) == {"nested": {"count": 1}}
