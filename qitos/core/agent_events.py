"""Events emitted by the minimal agent loop.

The event set mirrors the loop's observable boundaries: run, turn, message
and Tool execution. Events are observational; the canonical transcript and
the transaction boundary remain the only recovery truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Awaitable, Callable, Mapping, Union

from .message import AssistantMessage, Message
from .model_stream import ModelStreamEvent
from .tool_result import ToolResult


@dataclass(frozen=True, slots=True)
class AgentStart:
    type: str = field(default="agent_start", init=False)


@dataclass(frozen=True, slots=True)
class AgentEnd:
    """Terminal event of one run; ``messages`` holds the run's new messages."""

    messages: tuple[Message, ...]
    type: str = field(default="agent_end", init=False)


@dataclass(frozen=True, slots=True)
class TurnStart:
    turn: int
    type: str = field(default="turn_start", init=False)


@dataclass(frozen=True, slots=True)
class TurnEnd:
    turn: int
    message: AssistantMessage
    tool_results: tuple[Message, ...]
    type: str = field(default="turn_end", init=False)


@dataclass(frozen=True, slots=True)
class MessageStart:
    message: Message
    type: str = field(default="message_start", init=False)


@dataclass(frozen=True, slots=True)
class MessageUpdate:
    """One streamed assistant delta with the partial message accumulated so far."""

    message: AssistantMessage
    stream_event: ModelStreamEvent
    type: str = field(default="message_update", init=False)


@dataclass(frozen=True, slots=True)
class MessageEnd:
    message: Message
    type: str = field(default="message_end", init=False)


@dataclass(frozen=True, slots=True)
class ToolExecutionStart:
    tool_call_id: str
    tool_name: str
    args: Mapping[str, Any]
    type: str = field(default="tool_execution_start", init=False)


@dataclass(frozen=True, slots=True)
class ToolExecutionUpdate:
    """One intermediate progress payload reported by a running Tool."""

    tool_call_id: str
    tool_name: str
    args: Mapping[str, Any]
    partial_result: Any
    type: str = field(default="tool_execution_update", init=False)


@dataclass(frozen=True, slots=True)
class ToolExecutionEnd:
    tool_call_id: str
    tool_name: str
    result: ToolResult
    is_error: bool
    type: str = field(default="tool_execution_end", init=False)


AgentEvent = Union[
    AgentStart,
    AgentEnd,
    TurnStart,
    TurnEnd,
    MessageStart,
    MessageUpdate,
    MessageEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    ToolExecutionEnd,
]

EventSink = Callable[[AgentEvent], Union[Awaitable[None], None]]


async def emit_to(sink: EventSink | None, event: AgentEvent) -> None:
    """Deliver one event; a missing sink drops it by design.

    Sinks may be synchronous (queue push) or asynchronous (listener fan-out);
    awaitables are awaited so async listeners stay inside the run settlement.
    """

    if sink is None:
        return
    outcome = sink(event)
    if inspect.isawaitable(outcome):
        await outcome


__all__ = [
    "AgentEnd",
    "AgentEvent",
    "AgentStart",
    "EventSink",
    "MessageEnd",
    "MessageStart",
    "MessageUpdate",
    "ToolExecutionEnd",
    "ToolExecutionStart",
    "ToolExecutionUpdate",
    "TurnEnd",
    "TurnStart",
    "emit_to",
]
