"""Shared fakes for minimal-loop behavior tests."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any, Deque, Iterable, List, Mapping, Optional, Sequence

from qitos.core.agent_loop import TurnTransactionBoundary
from qitos.core.message import AssistantMessage, ToolCall
from qitos.core.model_capabilities import ModelCapabilities
from qitos.core.model_request import ModelRequest
from qitos.core.model_stream import ModelStreamEvent, ModelStreamEventType
from qitos.core.tool_result import ToolResult
from qitos.models.base import Model


def text_events(
    text: str,
    *,
    finish_reason: str = "stop",
    usage: Optional[Mapping[str, Any]] = None,
) -> List[ModelStreamEvent]:
    events: List[ModelStreamEvent] = []
    if text:
        events.append(
            ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text=text)
        )
    events.append(
        ModelStreamEvent(
            type=ModelStreamEventType.COMPLETED,
            finish_reason=finish_reason,
            usage=usage,
        )
    )
    return events


def tool_call_wire(
    call_id: str, name: str, arguments: Any
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": (
                arguments
                if isinstance(arguments, str)
                else json.dumps(arguments, ensure_ascii=False)
            ),
        },
    }


def tool_events(
    calls: Sequence[dict[str, Any]],
    *,
    text: str = "",
    finish_reason: str = "tool_calls",
) -> List[ModelStreamEvent]:
    events: List[ModelStreamEvent] = []
    if text:
        events.append(
            ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text=text)
        )
    events.append(
        ModelStreamEvent(
            type=ModelStreamEventType.COMPLETED,
            finish_reason=finish_reason,
            tool_calls=list(calls),
        )
    )
    return events


def failed_events(error: str) -> List[ModelStreamEvent]:
    return [ModelStreamEvent(type=ModelStreamEventType.FAILED, error=error)]


class ScriptedModel(Model):
    """Model that replays scripted event sequences, one per request.

    A response entry may be a list of events or an async-generator factory
    receiving the request (for cancellation/hang scenarios).
    """

    def __init__(
        self,
        responses: Iterable[Any],
        *,
        model: str = "scripted-model",
        provider_name: str = "scripted",
        capabilities: Optional[ModelCapabilities] = None,
    ) -> None:
        super().__init__(model=model, provider_name=provider_name)
        self._responses: Deque[Any] = deque(responses)
        self._capabilities = capabilities
        self.requests: List[ModelRequest] = []

    @property
    def capabilities(self) -> ModelCapabilities:
        if self._capabilities is not None:
            return self._capabilities
        return super().capabilities

    async def stream(self, request: ModelRequest) -> Any:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("scripted model ran out of responses")
        item = self._responses.popleft()
        if callable(item):
            async for event in item(request):
                yield event
            return
        for event in item:
            yield event


class RecordingTransaction(TurnTransactionBoundary):
    """In-memory TurnTransactionBoundary capturing barrier order."""

    def __init__(self) -> None:
        self.records: List[tuple[str, Any]] = []

    async def input_accepted(self, prompts: tuple) -> None:
        self.records.append(("input_accepted", len(prompts)))

    async def turn_frozen(self, turn: int, config: Any) -> None:
        self.records.append(("turn_frozen", turn, tuple(config.tool_names)))

    async def model_terminal(
        self, turn: int, request: ModelRequest, message: AssistantMessage
    ) -> None:
        self.records.append(("model_terminal", turn, message.error))

    async def tool_started(self, turn: int, call: ToolCall) -> None:
        self.records.append(("tool_started", turn, call.id))

    async def tool_terminal(
        self, turn: int, call: ToolCall, result: ToolResult
    ) -> None:
        self.records.append(("tool_terminal", turn, call.id, result.status))

    async def turn_committed(self, turn: int, new_messages: tuple) -> None:
        self.records.append(("turn_committed", turn, len(new_messages)))

    async def run_terminal(self, result: Any) -> None:
        self.records.append(("run_terminal", result.status.value))


def make_hanging_model(
    gate: asyncio.Event, *, first_text: str = ""
) -> Any:
    """Response factory that streams one delta then waits for ``gate``."""

    async def _response(request: ModelRequest) -> Any:
        if first_text:
            yield ModelStreamEvent(
                type=ModelStreamEventType.TEXT_DELTA, text=first_text
            )
        await gate.wait()

    return _response
