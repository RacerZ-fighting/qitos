"""Discriminated events emitted by asynchronous model Providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .model_request import ModelContinuation
from .model_response import ModelUsage, normalize_model_usage


class ModelStreamEventType(str, Enum):
    """Provider-neutral event kinds with non-overlapping semantics."""

    TEXT_DELTA = "text.delta"
    REASONING_DELTA = "reasoning.delta"
    TOOL_CALL_DELTA = "tool_call.delta"
    OUTPUT_ITEM = "output.item"
    USAGE = "usage"
    LIFECYCLE = "lifecycle"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class ModelStreamEvent:
    """One validated event in a logical Provider transaction.

    ``event_type`` retains the raw Provider event name for diagnostics. ``type``
    is the stable QitOS discriminant consumed by Engine and cache code.
    """

    type: ModelStreamEventType
    text: str = ""
    reasoning_content: Optional[str] = None
    usage: ModelUsage | Mapping[str, Any] | None = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    native_items: Optional[List[Dict[str, Any]]] = None
    event_type: Optional[str] = None
    event_metadata: Dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    continuation: ModelContinuation | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, ModelStreamEventType):
            raise TypeError("type must be a ModelStreamEventType")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if self.reasoning_content is not None and not isinstance(
            self.reasoning_content, str
        ):
            raise TypeError("reasoning_content must be a string or None")
        self.usage = normalize_model_usage(self.usage)
        if self.tool_calls is not None and not isinstance(self.tool_calls, list):
            raise TypeError("tool_calls must be a list or None")
        if self.native_items is not None and not isinstance(self.native_items, list):
            raise TypeError("native_items must be a list or None")
        if not isinstance(self.event_metadata, dict):
            raise TypeError("event_metadata must be a dictionary")
        if self.continuation is not None and not isinstance(
            self.continuation, ModelContinuation
        ):
            raise TypeError("continuation must be a ModelContinuation or None")
        if self.type is not ModelStreamEventType.COMPLETED:
            self._forbid(
                bool(self.finish_reason),
                "only completed events may contain a finish reason",
            )
            self._forbid(
                self.continuation is not None,
                "only completed events may contain a continuation",
            )
        if self.type is not ModelStreamEventType.FAILED:
            self._forbid(bool(self.error), "only failed events may contain an error")

        has_text = bool(self.text)
        has_reasoning = bool(self.reasoning_content)
        has_usage = self.usage is not None
        has_tools = bool(self.tool_calls)
        has_items = bool(self.native_items)
        if self.type is ModelStreamEventType.TEXT_DELTA:
            self._require(has_text, "text delta must contain text")
            self._forbid(
                has_reasoning or has_usage or has_tools or has_items,
                "text delta contains fields owned by another event type",
            )
        elif self.type is ModelStreamEventType.REASONING_DELTA:
            self._require(
                has_reasoning,
                "reasoning delta must contain reasoning_content",
            )
            self._forbid(
                has_text or has_usage or has_tools or has_items,
                "reasoning delta contains fields owned by another event type",
            )
        elif self.type is ModelStreamEventType.TOOL_CALL_DELTA:
            self._require(
                bool(self.event_type),
                "tool-call delta must retain its Provider event type",
            )
            self._forbid(
                has_text or has_reasoning or has_usage or has_tools or has_items,
                "tool-call delta contains completed model content",
            )
        elif self.type is ModelStreamEventType.OUTPUT_ITEM:
            self._require(has_items, "output-item event must contain native_items")
            self._forbid(
                has_text or has_reasoning or has_usage or has_tools,
                "output-item event contains fields owned by another event type",
            )
        elif self.type is ModelStreamEventType.USAGE:
            self._require(has_usage, "usage event must contain typed usage")
            self._forbid(
                has_text or has_reasoning or has_tools or has_items,
                "usage event contains fields owned by another event type",
            )
        elif self.type is ModelStreamEventType.LIFECYCLE:
            self._require(
                bool(self.event_type),
                "lifecycle event must retain its Provider event type",
            )
            self._forbid(
                has_text
                or has_reasoning
                or has_usage
                or has_tools
                or has_items,
                "lifecycle event contains model content",
            )
        elif self.type is ModelStreamEventType.COMPLETED:
            self._forbid(bool(self.error), "completed event must not contain an error")
        elif self.type is ModelStreamEventType.FAILED:
            self._require(bool(self.error), "failed event must contain an error")
            self._forbid(
                has_text
                or has_reasoning
                or has_usage
                or has_tools
                or has_items,
                "failed event must not contain completed model content",
            )

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    @staticmethod
    def _forbid(condition: Any, message: str) -> None:
        if condition:
            raise ValueError(message)

    @property
    def done(self) -> bool:
        """Return whether this event successfully completed the transaction."""

        return self.type is ModelStreamEventType.COMPLETED

    @property
    def is_final(self) -> bool:
        """Return whether this is the unique success or failure terminal."""

        return self.type in {
            ModelStreamEventType.COMPLETED,
            ModelStreamEventType.FAILED,
        }


__all__ = ["ModelStreamEvent", "ModelStreamEventType"]
