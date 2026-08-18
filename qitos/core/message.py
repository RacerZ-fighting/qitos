"""Typed provider-neutral messages for the minimal agent loop.

Messages are the canonical transcript contract: user input, durable runtime
context, assistant output (including Tool calls) and Tool results. Provider
adapters keep consuming the existing wire mappings, so each message owns a
one-way ``to_wire`` projection plus a strict ``to_dict``/``from_dict`` round
trip used by persistence.

Invariants:
- Every uniquely identified call that reaches Tool admission is paired with
  exactly one terminal ``ToolResultMessage``; transcripts never delete a call.
- Duplicate raw call ids remain in a failed assistant message as pre-admission
  protocol evidence. They are not executed because no unambiguous result
  pairing or durable record identity exists.
- Assistant tool-call arguments that are not a valid JSON object stay in the
  transcript with ``parse_error`` set; the loop turns them into an admission
  error result instead of executing or dropping the call.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .model_request import ModelContinuation
from .model_response import ModelResponse, ModelUsage
from ._freeze import thaw_deep
from .tool_result import ToolResult


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text content must be a string")


@dataclass(frozen=True, slots=True)
class ImageContent:
    """One provider-neutral image part kept as its wire mapping."""

    source: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.source, Mapping):
            raise TypeError("image content must be a mapping")
        object.__setattr__(self, "source", _freeze_json(self.source))


UserContent = Union[str, Tuple[Union[TextContent, ImageContent], ...]]

_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One typed Tool call requested by the assistant.

    ``parse_error`` records that the provider emitted arguments that are not a
    JSON object; ``arguments`` is then empty and the loop must turn the call
    into an admission error ToolResult without executing any Tool.
    """

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    parse_error: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("tool call id must be non-empty text")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tool call name must be non-empty text")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool call arguments must be a mapping")
        object.__setattr__(self, "arguments", _freeze_json(self.arguments))
        if self.parse_error is not None and not isinstance(self.parse_error, str):
            raise TypeError("tool call parse_error must be text or None")

    def to_wire(self) -> Dict[str, Any]:
        """Project to the OpenAI-style function-call mapping adapters consume."""

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(
                    _thaw_json(self.arguments), ensure_ascii=False
                ),
            },
        }

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "ToolCall":
        """Decode one adapter-emitted function-call mapping.

        Malformed arguments never raise here: they surface as ``parse_error``
        so the call stays in the transcript and reaches a terminal ToolResult.
        """

        if not isinstance(value, Mapping):
            raise TypeError("tool call must be a mapping")
        function = value.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("tool call is missing its function payload")
        name = function.get("name")
        call_id = value.get("id")
        raw_arguments = function.get("arguments", "{}")
        arguments: Mapping[str, Any] = MappingProxyType({})
        parse_error: Optional[str] = None
        if isinstance(raw_arguments, str):
            try:
                decoded = json.loads(raw_arguments or "{}")
            except ValueError:
                decoded = None
            if isinstance(decoded, dict):
                arguments = decoded
            else:
                parse_error = "tool call arguments are not a valid JSON object"
        elif isinstance(raw_arguments, Mapping):
            arguments = raw_arguments
        else:
            parse_error = "tool call arguments are not a valid JSON object"
        return cls(
            id=str(call_id or ""),
            name=str(name or ""),
            arguments=arguments,
            parse_error=parse_error,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "arguments": _thaw_json(self.arguments),
        }
        if self.parse_error is not None:
            payload["parse_error"] = self.parse_error
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolCall":
        if not isinstance(value, Mapping):
            raise ValueError("tool call payload must be a mapping")
        required = {"id", "name", "arguments"}
        optional = {"parse_error"}
        fields = set(value)
        if not required.issubset(fields) or not fields.issubset(required | optional):
            raise ValueError("tool call payload fields are invalid")
        raw_arguments = value["arguments"]
        if not isinstance(raw_arguments, dict):
            raise ValueError("tool call arguments must be a mapping")
        raw_error = value.get("parse_error")
        if raw_error is not None and not isinstance(raw_error, str):
            raise ValueError("tool call parse_error must be text or null")
        raw_id = value.get("id")
        raw_name = value.get("name")
        if not isinstance(raw_id, str) or not isinstance(raw_name, str):
            raise ValueError("tool call id and name must be text")
        return cls(
            id=raw_id,
            name=raw_name,
            arguments=raw_arguments,
            parse_error=raw_error,
        )


@dataclass(frozen=True, slots=True)
class UserMessage:
    content: UserContent
    timestamp: float = field(default_factory=time.time)
    role: str = field(default="user", init=False)

    def __post_init__(self) -> None:
        _validate_message_timestamp(self.timestamp)
        if isinstance(self.content, str):
            return
        if not isinstance(self.content, tuple) or not self.content:
            raise TypeError("user content must be text or a non-empty block tuple")
        for block in self.content:
            if not isinstance(block, (TextContent, ImageContent)):
                raise TypeError("user content blocks must be text or image")


@dataclass(frozen=True, slots=True)
class ContextMessage:
    """Model-visible runtime context that is not a user instruction.

    The canonical role stays provider-neutral. OpenAI-style protocols project
    it as a developer message; protocols without a developer role preserve
    the ordered content through their documented contextual-user fallback.
    Products remain responsible for the state and delta represented by the
    text -- this message is only its durable model-history projection.
    """

    content: str
    timestamp: float = field(default_factory=time.time)
    role: str = field(default="context", init=False)

    def __post_init__(self) -> None:
        _validate_message_timestamp(self.timestamp)
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("context content must be non-empty text")


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """One terminal assistant transaction in the canonical transcript."""

    text: str = ""
    tool_calls: Tuple[ToolCall, ...] = ()
    reasoning_content: Optional[str] = None
    usage: Optional[ModelUsage] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    native_items: Optional[Tuple[Mapping[str, Any], ...]] = None
    continuation: Optional[ModelContinuation] = None
    model_name: Optional[str] = None
    provider: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    timestamp: float = field(default_factory=time.time)
    role: str = field(default="assistant", init=False)

    def __post_init__(self) -> None:
        _validate_message_timestamp(self.timestamp)
        if not isinstance(self.text, str):
            raise TypeError("assistant text must be a string")
        if not isinstance(self.tool_calls, tuple) or not all(
            isinstance(call, ToolCall) for call in self.tool_calls
        ):
            raise TypeError("assistant tool_calls must be a tuple of ToolCall")
        call_ids = [call.id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            # A duplicate id is a provider/model protocol failure, but the raw
            # calls remain canonical evidence. The loop sees ``failed`` and
            # stops before Tool admission, so no ambiguous side effect runs.
            if self.error is None:
                object.__setattr__(
                    self,
                    "error",
                    "assistant tool call ids must be unique",
                )
        if self.reasoning_content is not None and not isinstance(
            self.reasoning_content, str
        ):
            raise TypeError("reasoning_content must be text or None")
        if self.usage is not None and not isinstance(self.usage, ModelUsage):
            raise TypeError("usage must be a ModelUsage or None")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be text or None")
        if self.native_items is not None:
            if not isinstance(self.native_items, tuple) or not all(
                isinstance(item, Mapping) for item in self.native_items
            ):
                raise TypeError("native_items must be a tuple of mappings or None")
            object.__setattr__(
                self,
                "native_items",
                tuple(_freeze_json(item) for item in self.native_items),
            )
        if self.continuation is not None and not isinstance(
            self.continuation, ModelContinuation
        ):
            raise TypeError("continuation must be a ModelContinuation or None")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    @property
    def failed(self) -> bool:
        """Return whether this message is a terminal model failure record."""

        return self.error is not None

    @property
    def truncated(self) -> bool:
        """Return whether the provider cut this response at the token limit."""

        return self.finish_reason in _TRUNCATED_FINISH_REASONS


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """The unique terminal result paired with one ``ToolCall``.

    ``usage`` and ``added_tool_names`` mirror the typed accounting and
    Tool-activation facts of the committed ``ToolResult`` so the durable
    transcript carries them without re-reading the result payload.
    """

    tool_call_id: str
    tool_name: str
    result: ToolResult
    usage: Optional[ModelUsage] = None
    added_tool_names: Tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)
    role: str = field(default="tool", init=False)

    def __post_init__(self) -> None:
        _validate_message_timestamp(self.timestamp)
        if not isinstance(self.tool_call_id, str) or not self.tool_call_id:
            raise ValueError("tool result message tool_call_id must be non-empty")
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise ValueError("tool result message tool_name must be non-empty")
        if not isinstance(self.result, ToolResult):
            raise TypeError("tool result message result must be a ToolResult")
        if self.usage is not None and not isinstance(self.usage, ModelUsage):
            raise TypeError("tool result message usage must be a ModelUsage or None")
        if not isinstance(self.added_tool_names, tuple) or not all(
            isinstance(name, str) and name.strip() for name in self.added_tool_names
        ):
            raise TypeError(
                "tool result message added_tool_names must contain non-empty strings"
            )
        if len(self.added_tool_names) != len(set(self.added_tool_names)):
            raise ValueError("tool result message added_tool_names must be unique")
        # Durable messages hold a deeply immutable result snapshot: listeners
        # receive the same object, and mutation must never make the journal's
        # tool terminal record and the committed turn payload disagree.
        object.__setattr__(self, "result", self.result.frozen())

    @property
    def is_error(self) -> bool:
        return self.result.status != "success"


Message = Union[UserMessage, ContextMessage, AssistantMessage, ToolResultMessage]


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("message payload mapping keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("message payload numbers must be finite")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError("message payloads must contain JSON-compatible values")


def _validate_message_timestamp(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("message timestamp must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError("message timestamp must be finite")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def assistant_from_response(
    response: ModelResponse,
    *,
    error: Optional[str] = None,
) -> AssistantMessage:
    """Build the canonical assistant message from one completed transaction.

    ``error`` marks the message as a terminal failure record (provider FAILED
    event or transport-level failure); the partial content is still preserved.
    """

    tool_calls: List[ToolCall] = []
    for index, item in enumerate(list(response.tool_calls or [])):
        if not isinstance(item, Mapping):
            continue
        try:
            tool_calls.append(ToolCall.from_wire(item))
        except (TypeError, ValueError):
            function = item.get("function") if isinstance(item, Mapping) else None
            name = ""
            if isinstance(function, Mapping):
                name = str(function.get("name") or "")
            tool_calls.append(
                ToolCall(
                    id=str(item.get("id") or f"call_{index}"),
                    name=name or "unknown",
                    arguments=MappingProxyType({}),
                    parse_error="tool call payload is malformed",
                )
            )
    return AssistantMessage(
        text=response.text or "",
        tool_calls=tuple(tool_calls),
        reasoning_content=response.reasoning_content,
        usage=response.usage if isinstance(response.usage, ModelUsage) else None,
        finish_reason=response.finish_reason,
        error=error,
        native_items=(
            tuple(item for item in response.native_items if isinstance(item, Mapping))
            if isinstance(response.native_items, list)
            else None
        ),
        continuation=response.continuation,
        model_name=response.model_name,
        provider=response.provider,
        metadata=dict(response.metadata or {}),
    )


def message_to_wire(message: Message) -> Dict[str, Any]:
    """Project one typed message into the adapter-facing wire mapping."""

    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            content: Any = message.content
        else:
            content = [
                (
                    {"type": "text", "text": block.text}
                    if isinstance(block, TextContent)
                    else thaw_deep(block.source)
                )
                for block in message.content
            ]
        return {"role": "user", "content": content}
    if isinstance(message, ContextMessage):
        return {"role": "developer", "content": message.content}
    if isinstance(message, AssistantMessage):
        payload: Dict[str, Any] = {
            "role": "assistant",
            "content": message.text if message.text.strip() else None,
        }
        if message.reasoning_content:
            payload["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            payload["tool_calls"] = [call.to_wire() for call in message.tool_calls]
        if message.native_items:
            payload["native_items"] = [
                _thaw_json(item) for item in message.native_items
            ]
        return payload
    if isinstance(message, ToolResultMessage):
        visible = message.result.model_visible_output
        if isinstance(visible, str):
            content = visible
        elif visible is not None:
            content = json.dumps(thaw_deep(visible), ensure_ascii=False, default=str)
        else:
            # An erroring result may carry no output at all (e.g. truncated
            # tool-call arguments); the error text must still reach the model
            # instead of a JSON ``null`` (Pi keeps the error text and isError).
            content = message.result.error or ""
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "name": message.tool_name,
            "content": content,
            "is_error": message.is_error,
        }
    raise TypeError(f"unsupported message type: {type(message).__name__}")


def message_to_dict(message: Message) -> Dict[str, Any]:
    """Encode one message for durable storage (strict, lossless)."""

    if isinstance(message, UserMessage):
        content: Any
        if isinstance(message.content, str):
            content = message.content
        else:
            content = [
                (
                    {"type": "text", "text": block.text}
                    if isinstance(block, TextContent)
                    else {"type": "image", "source": thaw_deep(block.source)}
                )
                for block in message.content
            ]
        return {
            "role": "user",
            "content": content,
            "timestamp": message.timestamp,
        }
    if isinstance(message, ContextMessage):
        return {
            "role": "context",
            "content": message.content,
            "timestamp": message.timestamp,
        }
    if isinstance(message, AssistantMessage):
        payload: Dict[str, Any] = {
            "role": "assistant",
            "text": message.text,
            "tool_calls": [call.to_dict() for call in message.tool_calls],
            "reasoning_content": message.reasoning_content,
            "usage": message.usage.to_dict() if message.usage is not None else None,
            "finish_reason": message.finish_reason,
            "error": message.error,
            "native_items": (
                [_thaw_json(item) for item in message.native_items]
                if message.native_items is not None
                else None
            ),
            "continuation": (
                message.continuation.to_dict()
                if message.continuation is not None
                else None
            ),
            "model_name": message.model_name,
            "provider": message.provider,
            "metadata": thaw_deep(message.metadata),
            "timestamp": message.timestamp,
        }
        return payload
    if isinstance(message, ToolResultMessage):
        tool_payload: Dict[str, Any] = {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "result": message.result.to_dict(),
            "timestamp": message.timestamp,
        }
        if message.usage is not None:
            tool_payload["usage"] = message.usage.to_dict()
        if message.added_tool_names:
            tool_payload["added_tool_names"] = list(message.added_tool_names)
        return tool_payload
    raise TypeError(f"unsupported message type: {type(message).__name__}")


def message_from_dict(value: Mapping[str, Any]) -> Message:
    """Decode one persisted message; unknown shapes fail closed."""

    if not isinstance(value, Mapping):
        raise ValueError("message payload must be a mapping")
    role = value.get("role")
    if role == "user":
        required = {"role", "content", "timestamp"}
        if set(value) != required:
            raise ValueError("user message fields are invalid")
        raw_content = value["content"]
        if isinstance(raw_content, str):
            content: UserContent = raw_content
        elif isinstance(raw_content, list):
            blocks: List[Union[TextContent, ImageContent]] = []
            for item in raw_content:
                if not isinstance(item, dict):
                    raise ValueError("user content blocks must be mappings")
                block_type = item.get("type")
                if (
                    block_type == "text"
                    and set(item) == {"type", "text"}
                    and isinstance(item["text"], str)
                ):
                    blocks.append(TextContent(text=item["text"]))
                elif (
                    block_type == "image"
                    and set(item) == {"type", "source"}
                    and isinstance(item["source"], dict)
                ):
                    blocks.append(ImageContent(source=item["source"]))
                else:
                    raise ValueError("user content block type is unsupported")
            content = tuple(blocks)
        else:
            raise ValueError("user message content is invalid")
        return UserMessage(
            content=content,
            timestamp=_message_timestamp(value),
        )
    if role == "context":
        required = {"role", "content", "timestamp"}
        if set(value) != required:
            raise ValueError("context message fields are invalid")
        raw_content = value["content"]
        if not isinstance(raw_content, str) or not raw_content.strip():
            raise ValueError("context message content must be non-empty text")
        return ContextMessage(
            content=raw_content,
            timestamp=_message_timestamp(value),
        )
    if role == "assistant":
        required = {
            "role",
            "text",
            "tool_calls",
            "reasoning_content",
            "usage",
            "finish_reason",
            "error",
            "native_items",
            "continuation",
            "model_name",
            "provider",
            "metadata",
            "timestamp",
        }
        if set(value) != required:
            raise ValueError("assistant message fields are invalid")
        raw_calls = value["tool_calls"]
        if not isinstance(raw_calls, list):
            raise ValueError("assistant tool_calls must be a list")
        raw_usage = value["usage"]
        if raw_usage is not None and not isinstance(raw_usage, dict):
            raise ValueError("assistant usage must be a mapping or null")
        raw_native = value["native_items"]
        if raw_native is not None and (
            not isinstance(raw_native, list)
            or not all(isinstance(item, dict) for item in raw_native)
        ):
            raise ValueError("assistant native_items must be a list of mappings")
        raw_continuation = value["continuation"]
        if raw_continuation is not None and not isinstance(raw_continuation, dict):
            raise ValueError("assistant continuation must be a mapping or null")
        raw_metadata = value["metadata"]
        if not isinstance(raw_metadata, dict):
            raise ValueError("assistant metadata must be a mapping")
        raw_text = value["text"]
        if not isinstance(raw_text, str):
            raise ValueError("assistant text must be text")
        return AssistantMessage(
            text=raw_text,
            tool_calls=tuple(ToolCall.from_dict(item) for item in raw_calls),
            reasoning_content=_optional_text(value["reasoning_content"]),
            usage=(
                ModelUsage.from_mapping(raw_usage)
                if isinstance(raw_usage, Mapping)
                else None
            ),
            finish_reason=_optional_text(value["finish_reason"]),
            error=_optional_text(value["error"]),
            native_items=(
                tuple(raw_native) if raw_native is not None else None
            ),
            continuation=(
                _continuation_from_dict(raw_continuation)
                if isinstance(raw_continuation, Mapping)
                else None
            ),
            model_name=_optional_text(value["model_name"]),
            provider=_optional_text(value["provider"]),
            metadata=raw_metadata,
            timestamp=_message_timestamp(value),
        )
    if role == "tool":
        required = {"role", "tool_call_id", "tool_name", "result", "timestamp"}
        optional = {"usage", "added_tool_names"}
        fields = set(value)
        if not required.issubset(fields) or not fields.issubset(required | optional):
            raise ValueError("tool result message fields are invalid")
        raw_result = value["result"]
        if not isinstance(raw_result, dict):
            raise ValueError("tool result message result must be a mapping")
        raw_call_id = value["tool_call_id"]
        raw_tool_name = value["tool_name"]
        if not isinstance(raw_call_id, str) or not isinstance(raw_tool_name, str):
            raise ValueError("tool result message ids must be text")
        raw_usage = value.get("usage")
        if raw_usage is not None and not isinstance(raw_usage, dict):
            raise ValueError("tool result message usage must be a mapping or null")
        raw_added_tool_names = value.get("added_tool_names", [])
        if not isinstance(raw_added_tool_names, list) or not all(
            isinstance(name, str) for name in raw_added_tool_names
        ):
            raise ValueError("tool result message added_tool_names must be text")
        return ToolResultMessage(
            tool_call_id=raw_call_id,
            tool_name=raw_tool_name,
            result=ToolResult.from_dict(raw_result),
            usage=(
                ModelUsage.from_mapping(raw_usage)
                if isinstance(raw_usage, Mapping)
                else None
            ),
            added_tool_names=tuple(raw_added_tool_names),
            timestamp=_message_timestamp(value),
        )
    raise ValueError(f"unsupported message role: {role!r}")


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("message text fields must be text or null")
    return value


def _message_timestamp(value: Mapping[str, Any]) -> float:
    raw = value["timestamp"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("message timestamp must be numeric")
    timestamp = float(raw)
    if not math.isfinite(timestamp):
        raise ValueError("message timestamp must be finite")
    return timestamp


def _continuation_from_dict(value: Mapping[str, Any]) -> ModelContinuation:
    fields = {
        "run_id",
        "provider",
        "model",
        "protocol",
        "response_id",
        "prefix_items",
        "prefix_digest",
        "settings_digest",
    }
    if set(value) != fields:
        raise ValueError("assistant continuation fields are invalid")
    text_fields = fields - {"prefix_items"}
    if any(not isinstance(value[name], str) for name in text_fields):
        raise ValueError("assistant continuation text fields must be text")
    prefix_items = value["prefix_items"]
    if isinstance(prefix_items, bool) or not isinstance(prefix_items, int):
        raise ValueError("assistant continuation prefix_items must be an integer")
    return ModelContinuation.from_dict(value)


__all__ = [
    "AssistantMessage",
    "ContextMessage",
    "ImageContent",
    "Message",
    "TextContent",
    "ToolCall",
    "ToolResultMessage",
    "UserContent",
    "UserMessage",
    "assistant_from_response",
    "message_from_dict",
    "message_to_dict",
    "message_to_wire",
]
