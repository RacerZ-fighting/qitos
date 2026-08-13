"""Async run inbox used by the Engine loop."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import Callable
from typing import Literal

from ..core.runtime_input import RuntimeInput


RuntimeWaitOutcome = Literal["event", "cancelled", "timeout", "closed"]


class _RuntimeInbox:
    """Thread-safe, idempotent queue owned by one Engine instance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run_id = ""
        self._open = False
        self._events: deque[RuntimeInput] = deque()
        self._seen_event_ids: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake_event: asyncio.Event | None = None

    def open(
        self,
        run_id: str,
        *,
        recovered: tuple[RuntimeInput, ...] = (),
    ) -> None:
        """Open an inbox for one run, optionally with durable pending events."""

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        with self._lock:
            self._run_id = run_id
            self._open = True
            self._events = deque(recovered)
            self._seen_event_ids = {event.event_id for event in recovered}
            self._loop = loop
            self._wake_event = asyncio.Event() if loop is not None else None

    def post(self, run_id: str, event: RuntimeInput) -> bool:
        """Queue an event, rejecting wrong-run, late, and duplicate posts."""

        with self._lock:
            if not self._open or run_id != self._run_id:
                return False
            if event.event_id in self._seen_event_ids:
                return False
            self._seen_event_ids.add(event.event_id)
            self._events.append(event)
            self._notify_waiter_locked()
            return True

    def accepts(self, run_id: str, event_id: str) -> bool:
        """Return whether a post could currently enter this run's queue."""

        with self._lock:
            return (
                self._open
                and run_id == self._run_id
                and event_id not in self._seen_event_ids
            )

    def drain(self, run_id: str) -> list[RuntimeInput]:
        """Remove all events queued for the active run."""

        with self._lock:
            if not self._open or run_id != self._run_id:
                return []
            events = list(self._events)
            self._events.clear()
            if self._wake_event is not None:
                self._wake_event.clear()
            return events

    def has_events(self, run_id: str) -> bool:
        """Return whether the active run has input waiting at its next safe point."""

        with self._lock:
            return self._open and run_id == self._run_id and bool(self._events)

    async def wait(
        self,
        run_id: str,
        *,
        timeout_seconds: float | None,
        cancelled: Callable[[], bool],
    ) -> RuntimeWaitOutcome:
        """Await input, cancellation, deadline, or close without blocking the loop."""

        while True:
            with self._lock:
                if not self._open or run_id != self._run_id:
                    return "closed"
                if self._events:
                    return "event"
                if cancelled():
                    return "cancelled"
                loop = asyncio.get_running_loop()
                if self._loop is None:
                    self._loop = loop
                elif self._loop is not loop:
                    raise RuntimeError("runtime inbox cannot move between event loops")
                if self._wake_event is None:
                    self._wake_event = asyncio.Event()
                wake_event = self._wake_event
            try:
                if timeout_seconds is None:
                    await wake_event.wait()
                else:
                    await asyncio.wait_for(
                        wake_event.wait(), timeout=max(0.0, timeout_seconds)
                    )
            except asyncio.TimeoutError:
                return "timeout"
            finally:
                wake_event.clear()

    def wake(self) -> None:
        """Wake waiters so they can observe cancellation."""

        with self._lock:
            self._notify_waiter_locked()

    def close(self, run_id: str | None = None) -> None:
        """Close the current inbox and reject future events for that run."""

        with self._lock:
            if run_id is not None and run_id != self._run_id:
                return
            self._open = False
            self._events.clear()
            self._notify_waiter_locked()

    def _notify_waiter_locked(self) -> None:
        loop = self._loop
        wake_event = self._wake_event
        if loop is None or wake_event is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(wake_event.set)
        except RuntimeError:
            return


__all__ = ["RuntimeWaitOutcome"]
