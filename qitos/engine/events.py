"""Structured engine events for streaming and observability."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from .states import RuntimePhase


class EngineEventType(str, Enum):
    STEP_START = "step_start"
    STEP_END = "step_end"
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    DECIDE = "decide"
    ACT = "act"
    REDUCE = "reduce"
    CRITIC = "critic"
    CHECK_STOP = "check_stop"
    HANDOFF = "handoff"
    DELEGATE = "delegate"
    FANOUT = "fanout"
    ERROR = "error"
    INTERRUPT = "interrupt"
    RUN_START = "run_start"
    RUN_END = "run_end"
    STEP_STREAM = "step_stream"  # Token-level streaming chunk


@dataclass
class EngineEvent:
    """Structured event emitted by Engine/AsyncEngine during execution."""

    event_type: EngineEventType
    step_id: int = 0
    agent_id: Optional[str] = None
    phase: Optional[RuntimePhase] = None
    ok: bool = True
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    ts: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "event_type": self.event_type.value,
            "step_id": self.step_id,
            "ok": self.ok,
            "ts": self.ts,
        }
        if self.agent_id is not None:
            d["agent_id"] = self.agent_id
        if self.phase is not None:
            d["phase"] = self.phase.value
        if self.payload:
            d["payload"] = self.payload
        if self.error is not None:
            d["error"] = self.error
        return d


class EventStream:
    """Async-compatible event stream for consuming engine events.

    Usage::

        stream = EventStream()
        engine = Engine(agent, ...)
        # Subscribe before starting
        async for event in stream:
            print(event)

        # Producer side (engine):
        stream.emit(EngineEvent(event_type=EngineEventType.STEP_START, ...))
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Optional[EngineEvent]] = asyncio.Queue(maxsize=4096)
        self._subscribers: List[asyncio.Queue[Optional[EngineEvent]]] = []
        self._closed = False
        self._loop: Optional[asyncio.AbstractEventLoop]
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._loop_lock = threading.Lock()

    def _bind_running_loop(self) -> None:
        if self._loop is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        with self._loop_lock:
            if self._loop is None:
                self._loop = loop

    def _dispatch(self, callback: Callable[[], None]) -> None:
        """Run queue mutations on the consumer loop when it is active."""

        loop = self._loop
        if loop is None or not loop.is_running():
            callback()
            return
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            callback()
        elif not loop.is_closed():
            loop.call_soon_threadsafe(callback)

    @staticmethod
    def _put_event(
        queue: asyncio.Queue[Optional[EngineEvent]],
        event: Optional[EngineEvent],
    ) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    @staticmethod
    def _put_terminal(queue: asyncio.Queue[Optional[EngineEvent]]) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:  # pragma: no cover - defensive race guard
            pass

    def _emit_now(self, event: EngineEvent) -> None:
        if self._closed:
            return
        for queue in self._subscribers:
            self._put_event(queue, event)
        self._put_event(self._queue, event)

    def _close_now(self) -> None:
        if self._closed:
            return
        self._closed = True
        for queue in self._subscribers:
            self._put_terminal(queue)
        self._put_terminal(self._queue)

    def emit(self, event: EngineEvent) -> None:
        """Emit an event to all subscribers (thread-safe for sync callers)."""
        self._dispatch(lambda: self._emit_now(event))

    def emit_sync(self, event: EngineEvent) -> None:
        """Emit from a sync context (safe to call from Engine.run)."""
        self.emit(event)

    def close(self) -> None:
        """Signal end of stream."""
        self._dispatch(self._close_now)

    async def __aiter__(self) -> AsyncIterator[EngineEvent]:
        self._bind_running_loop()
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    def subscribe(self) -> asyncio.Queue[Optional[EngineEvent]]:
        """Create a new subscriber queue for fan-out consumption."""
        self._bind_running_loop()
        q: asyncio.Queue[Optional[EngineEvent]] = asyncio.Queue(maxsize=1024)
        self._subscribers.append(q)
        return q


__all__ = ["EngineEvent", "EngineEventType", "EventStream"]
