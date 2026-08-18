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
from qitos.core.model_response import ModelResponse, ModelUsage
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


def test_assistant_message_preserves_duplicate_ids_as_protocol_failure() -> None:
    first = ToolCall(id="duplicate", name="one", arguments={})
    second = ToolCall(id="duplicate", name="two", arguments={})

    message = AssistantMessage(tool_calls=(first, second))

    assert message.tool_calls == (first, second)
    assert message.failed
    assert message.error == "assistant tool call ids must be unique"
    restored = message_from_dict(message_to_dict(message))
    assert isinstance(restored, AssistantMessage)
    assert restored.tool_calls == (first, second)


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
    assistant_payload = message_to_dict(
        AssistantMessage(
            tool_calls=(ToolCall(id="c1", name="echo", arguments={}),),
            timestamp=1.0,
        )
    )
    assistant_payload["tool_calls"][0]["id"] = 7
    with pytest.raises(ValueError):
        message_from_dict(assistant_payload)


def test_message_from_dict_requires_the_exact_durable_shape() -> None:
    assistant_payload = message_to_dict(AssistantMessage(text="ok", timestamp=1.0))
    assistant_payload.pop("metadata")
    with pytest.raises(ValueError, match="fields"):
        message_from_dict(assistant_payload)

    user_payload = message_to_dict(UserMessage(content="hello", timestamp=1.0))
    user_payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        message_from_dict(user_payload)

    tool_payload = message_to_dict(
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="echo",
            result=ToolResult(output="ok"),
            timestamp=1.0,
        )
    )
    tool_payload.pop("tool_name")
    with pytest.raises(ValueError, match="fields"):
        message_from_dict(tool_payload)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("tool_calls", None),
        ("usage", []),
        ("native_items", {}),
        ("continuation", []),
        ("metadata", None),
    ],
)
def test_assistant_durable_decoder_does_not_default_invalid_values(
    field: str, invalid: object
) -> None:
    payload = message_to_dict(AssistantMessage(text="ok", timestamp=1.0))
    payload[field] = invalid

    with pytest.raises(ValueError):
        message_from_dict(payload)


def test_durable_decoder_rejects_noncanonical_nested_fields() -> None:
    assistant_payload = message_to_dict(
        AssistantMessage(
            tool_calls=(ToolCall(id="c1", name="echo", arguments={}),),
            timestamp=1.0,
        )
    )
    assistant_payload["tool_calls"][0].pop("arguments")
    with pytest.raises(ValueError, match="tool call payload fields"):
        message_from_dict(assistant_payload)

    user_payload = message_to_dict(
        UserMessage(content=(TextContent("hello"),), timestamp=1.0)
    )
    user_payload["content"][0]["unexpected"] = True
    with pytest.raises(ValueError, match="content block"):
        message_from_dict(user_payload)


def test_durable_decoder_rejects_invalid_continuation_and_timestamp() -> None:
    continuation = ModelContinuation(
        run_id="run",
        provider="provider",
        model="model",
        protocol="protocol",
        response_id="response",
        prefix_items=1,
        prefix_digest="prefix",
        settings_digest="settings",
    )
    payload = message_to_dict(
        AssistantMessage(text="ok", continuation=continuation, timestamp=1.0)
    )
    payload["continuation"]["prefix_items"] = True
    with pytest.raises(ValueError, match="prefix_items"):
        message_from_dict(payload)

    payload = message_to_dict(UserMessage(content="hello", timestamp=1.0))
    payload["timestamp"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        message_from_dict(payload)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_message_json_boundaries_reject_non_finite_numbers(number: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ToolCall(id="c1", name="echo", arguments={"value": number})
    with pytest.raises(ValueError, match="finite"):
        AssistantMessage(metadata={"value": number})
    with pytest.raises(ValueError, match="finite"):
        UserMessage(content="hello", timestamp=number)
    with pytest.raises(ValueError, match="finite"):
        ModelUsage.from_mapping({"provider_detail": number})


def test_model_usage_rejects_non_string_detail_keys() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        ModelUsage.from_mapping({1: "invalid"})


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


# ── ToolResultMessage typed usage and added Tool names ───────────────────────


def test_tool_result_message_usage_and_added_names_round_trip() -> None:
    usage = ModelUsage.from_mapping({"total_tokens": 9, "cost_usd": 0.001})
    result = ToolResult(
        output={"subagent_status": "completed"},
        usage=usage,
        added_tool_names=("skill_tool",),
    )
    message = ToolResultMessage(
        tool_call_id="c1",
        tool_name="Agent",
        result=result,
        usage=result.usage,
        added_tool_names=result.added_tool_names,
        timestamp=5.0,
    )

    payload = message_to_dict(message)
    assert payload["usage"] == {"total_tokens": 9, "cost_usd": 0.001}
    assert payload["added_tool_names"] == ["skill_tool"]

    decoded = message_from_dict(payload)
    assert isinstance(decoded, ToolResultMessage)
    assert decoded.usage is not None
    assert decoded.usage.total_tokens == 9
    assert decoded.added_tool_names == ("skill_tool",)
    assert message_to_dict(decoded) == payload


def test_tool_result_message_omits_absent_facts_and_decodes_legacy_shape() -> None:
    message = ToolResultMessage(
        tool_call_id="c1",
        tool_name="echo",
        result=ToolResult(output="ok"),
        timestamp=1.0,
    )

    payload = message_to_dict(message)
    assert "usage" not in payload
    assert "added_tool_names" not in payload

    decoded = message_from_dict(payload)
    assert isinstance(decoded, ToolResultMessage)
    assert decoded.usage is None
    assert decoded.added_tool_names == ()


def test_tool_result_message_validates_typed_facts() -> None:
    result = ToolResult(output="ok")
    with pytest.raises(TypeError, match="ModelUsage"):
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="t",
            result=result,
            usage={"total_tokens": 1},  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="added_tool_names"):
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="t",
            result=result,
            added_tool_names=("a", ""),
        )
    with pytest.raises(ValueError, match="unique"):
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="t",
            result=result,
            added_tool_names=("a", "a"),
        )


def test_tool_result_message_decoder_fails_closed_on_invalid_fact_fields() -> None:
    base = message_to_dict(
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="t",
            result=ToolResult(output="ok"),
            timestamp=1.0,
        )
    )

    with pytest.raises(ValueError, match="usage"):
        message_from_dict({**base, "usage": "total_tokens"})
    with pytest.raises(ValueError, match="added_tool_names"):
        message_from_dict({**base, "added_tool_names": "a"})
    with pytest.raises(ValueError, match="added_tool_names"):
        message_from_dict({**base, "added_tool_names": [1]})
    with pytest.raises(ValueError, match="fields"):
        message_from_dict({**base, "unexpected": True})
