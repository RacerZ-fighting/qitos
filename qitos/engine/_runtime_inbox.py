"""Thread-safe run inbox used by the Engine loop."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Literal

from ..core.runtime_input import RuntimeInput


RuntimeWaitOutcome = Literal["event", "cancelled", "timeout", "closed"]


class _RuntimeInbox:
    """Thread-safe, idempotent queue owned by one Engine instance."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._run_id = ""
        self._open = False
        self._events: deque[RuntimeInput] = deque()
        self._seen_event_ids: set[str] = set()

    def open(self, run_id: str) -> None:
        """Open an empty inbox for exactly one run."""

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        with self._condition:
            self._run_id = run_id
            self._open = True
            self._events.clear()
            self._seen_event_ids.clear()
            self._condition.notify_all()

    def post(self, run_id: str, event: RuntimeInput) -> bool:
        """Queue an event, rejecting wrong-run, late, and duplicate posts."""

        with self._condition:
            if not self._open or run_id != self._run_id:
                return False
            if event.event_id in self._seen_event_ids:
                return False
            self._seen_event_ids.add(event.event_id)
            self._events.append(event)
            self._condition.notify_all()
            return True

    def drain(self, run_id: str) -> list[RuntimeInput]:
        """Remove all events queued for the active run."""

        with self._condition:
            if not self._open or run_id != self._run_id:
                return []
            events = list(self._events)
            self._events.clear()
            return events

    def wait(
        self,
        run_id: str,
        *,
        timeout_seconds: float | None,
        cancelled: Callable[[], bool],
    ) -> RuntimeWaitOutcome:
        """Block without polling until input, cancellation, deadline, or close."""

        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + max(0.0, timeout_seconds)
        )
        with self._condition:
            while True:
                if not self._open or run_id != self._run_id:
                    return "closed"
                if self._events:
                    return "event"
                if cancelled():
                    return "cancelled"
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if remaining == 0.0:
                    return "timeout"
                self._condition.wait(remaining)

    def wake(self) -> None:
        """Wake waiters so they can observe cancellation."""

        with self._condition:
            self._condition.notify_all()

    def close(self, run_id: str | None = None) -> None:
        """Close the current inbox and reject future events for that run."""

        with self._condition:
            if run_id is not None and run_id != self._run_id:
                return
            self._open = False
            self._events.clear()
            self._condition.notify_all()


__all__ = ["RuntimeWaitOutcome"]
