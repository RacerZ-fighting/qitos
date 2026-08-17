"""Typed provider-neutral messages for the minimal agent loop.

Messages are the canonical transcript contract: user input, assistant output
(including Tool calls) and Tool results. Provider adapters keep consuming the
existing wire mappings, so each message owns a one-way ``to_wire`` projection
plus a strict ``to_dict``/``from_dict`` round trip used by persistence.

Invariants:
- A ``ToolCall`` carried by an assistant message is paired with exactly one
  terminal ``ToolResultMessage`` produced by the loop; transcripts never
  delete a call.
- Assistant tool-call arguments that are not a valid JSON object stay in the
  transcript with ``parse_error`` set; the loop turns them into an admission
  error result instead of executing or dropping the call.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .model_request import ModelContinuation
from .model_response import ModelResponse, ModelUsage
from ._freeze import freeze_deep, thaw_deep
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
        object.__setattr__(self, "source", freeze_deep(self.source))


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
        if not set(value) <= {"id", "name", "arguments", "parse_error"}:
            raise ValueError("tool call payload fields are invalid")
        raw_arguments = value.get("arguments", {})
        if not isinstance(raw_arguments, Mapping):
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
        if isinstance(self.content, str):
            return
        if not isinstance(self.content, tuple) or not self.content:
            raise TypeError("user content must be text or a non-empty block tuple")
        for block in self.content:
            if not isinstance(block, (TextContent, ImageContent)):
                raise TypeError("user content blocks must be text or image")


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
        if not isinstance(self.text, str):
            raise TypeError("assistant text must be a string")
        if not isinstance(self.tool_calls, tuple) or not all(
            isinstance(call, ToolCall) for call in self.tool_calls
        ):
            raise TypeError("assistant tool_calls must be a tuple of ToolCall")
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
        object.__setattr__(self, "metadata", freeze_deep(self.metadata))

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
    """The unique terminal result paired with one ``ToolCall``."""

    tool_call_id: str
    tool_name: str
    result: ToolResult
    timestamp: float = field(default_factory=time.time)
    role: str = field(default="tool", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tool_call_id, str) or not self.tool_call_id:
            raise ValueError("tool result message tool_call_id must be non-empty")
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise ValueError("tool result message tool_name must be non-empty")
        if not isinstance(self.result, ToolResult):
            raise TypeError("tool result message result must be a ToolResult")
        # Durable messages hold a deeply immutable result snapshot: listeners
        # receive the same object, and mutation must never make the journal's
        # tool terminal record and the committed turn payload disagree.
        object.__setattr__(self, "result", self.result.frozen())

    @property
    def is_error(self) -> bool:
        return self.result.status != "success"


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError("message payloads must contain JSON-compatible values")


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
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "result": message.result.to_dict(),
            "timestamp": message.timestamp,
        }
    raise TypeError(f"unsupported message type: {type(message).__name__}")


def message_from_dict(value: Mapping[str, Any]) -> Message:
    """Decode one persisted message; unknown shapes fail closed."""

    if not isinstance(value, Mapping):
        raise ValueError("message payload must be a mapping")
    role = value.get("role")
    if role == "user":
        allowed = {"role", "content", "timestamp"}
        if not set(value) <= allowed:
            raise ValueError("user message fields are invalid")
        raw_content = value.get("content")
        if isinstance(raw_content, str):
            content: UserContent = raw_content
        elif isinstance(raw_content, Sequence) and not isinstance(
            raw_content, (str, bytes)
        ):
            blocks: List[Union[TextContent, ImageContent]] = []
            for item in raw_content:
                if not isinstance(item, Mapping):
                    raise ValueError("user content blocks must be mappings")
                block_type = item.get("type")
                if block_type == "text" and isinstance(item.get("text"), str):
                    blocks.append(TextContent(text=item["text"]))
                elif block_type == "image" and isinstance(item.get("source"), Mapping):
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
    if role == "assistant":
        allowed = {
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
        if not set(value) <= allowed:
            raise ValueError("assistant message fields are invalid")
        raw_calls = value.get("tool_calls") or []
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
            raise ValueError("assistant tool_calls must be a sequence")
        raw_usage = value.get("usage")
        raw_native = value.get("native_items")
        if raw_native is not None and (
            not isinstance(raw_native, Sequence)
            or isinstance(raw_native, (str, bytes))
            or not all(isinstance(item, Mapping) for item in raw_native)
        ):
            raise ValueError("assistant native_items must be a sequence of mappings")
        raw_continuation = value.get("continuation")
        raw_metadata = value.get("metadata") or {}
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("assistant metadata must be a mapping")
        raw_text = value.get("text")
        if not isinstance(raw_text, str):
            raise ValueError("assistant text must be text")
        return AssistantMessage(
            text=raw_text,
            tool_calls=tuple(ToolCall.from_dict(item) for item in raw_calls),
            reasoning_content=_optional_text(value.get("reasoning_content")),
            usage=(
                ModelUsage.from_mapping(raw_usage)
                if isinstance(raw_usage, Mapping)
                else None
            ),
            finish_reason=_optional_text(value.get("finish_reason")),
            error=_optional_text(value.get("error")),
            native_items=(
                tuple(raw_native) if raw_native is not None else None
            ),
            continuation=(
                ModelContinuation.from_dict(raw_continuation)
                if isinstance(raw_continuation, Mapping)
                else None
            ),
            model_name=_optional_text(value.get("model_name")),
            provider=_optional_text(value.get("provider")),
            metadata=raw_metadata,
            timestamp=_message_timestamp(value),
        )
    if role == "tool":
        allowed = {"role", "tool_call_id", "tool_name", "result", "timestamp"}
        if not set(value) <= allowed:
            raise ValueError("tool result message fields are invalid")
        raw_result = value.get("result")
        if not isinstance(raw_result, Mapping):
            raise ValueError("tool result message result must be a mapping")
        raw_call_id = value.get("tool_call_id")
        raw_tool_name = value.get("tool_name")
        if not isinstance(raw_call_id, str) or not isinstance(raw_tool_name, str):
            raise ValueError("tool result message ids must be text")
        return ToolResultMessage(
            tool_call_id=raw_call_id,
            tool_name=raw_tool_name,
            result=ToolResult.from_dict(raw_result),
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
    raw = value.get("timestamp")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("message timestamp must be numeric")
    return float(raw)


__all__ = [
    "AssistantMessage",
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
