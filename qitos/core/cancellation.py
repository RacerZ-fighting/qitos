"""Cancellation support for agent runs.

A run owner hands one :class:`CancelToken` to the agent loop, which checks it
at turn boundaries, Tool admission and retry backoff safe points.

Modes
-----
- ``"immediate"`` — signal the loop to stop right away.
  The current turn may be mid-execution; partial results are preserved.
- ``"after_step"`` — wait for the current turn to finish before stopping.
"""

from __future__ import annotations

import asyncio
import threading
from enum import Enum


class CancelMode(str, Enum):
    """Cancellation mode for agent runs."""

    NONE = "none"
    IMMEDIATE = "immediate"
    AFTER_STEP = "after_step"


class CancelToken:
    """Thread-safe cancellation signal shared between a run handle and the loop.

    The loop checks ``token.is_cancel_requested`` at each turn boundary and
    Tool admission point. Setting the mode to ``"immediate"`` causes the
    next check to stop the run; ``"after_step"`` waits until the turn finishes.

    ``wait_cancelled`` is the async-native bridge used while awaiting model
    streams; it never parks a thread the way ``asyncio.to_thread(token.wait)``
    would, so a completed run leaves no blocked waiter behind.
    """

    def __init__(self) -> None:
        self._mode = CancelMode.NONE
        self._lock = threading.Lock()
        self._cancel_requested = threading.Event()
        self._step_complete = threading.Event()
        self._async_waiters: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Event]
        ] = []

    @property
    def mode(self) -> CancelMode:
        with self._lock:
            return self._mode

    @property
    def is_cancel_requested(self) -> bool:
        with self._lock:
            return self._mode != CancelMode.NONE

    def request_cancel(self, mode: str = "immediate") -> None:
        """Signal the run to cancel.

        Parameters
        ----------
        mode : str
            ``"immediate"`` or ``"after_step"``.
        """
        with self._lock:
            self._mode = CancelMode(mode)
            self._cancel_requested.set()
            waiters = list(self._async_waiters)
            self._async_waiters.clear()
        for loop, event in waiters:
            loop.call_soon_threadsafe(event.set)

    def clear(self) -> None:
        """Reset the token (called at the start of each run)."""
        with self._lock:
            self._mode = CancelMode.NONE
            self._cancel_requested.clear()
        self._step_complete.clear()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until cancellation is requested or the timeout elapses."""

        return self._cancel_requested.wait(timeout=timeout)

    async def wait_cancelled(self) -> bool:
        """Wait asynchronously until cancellation is requested."""

        if self.is_cancel_requested:
            return True
        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        waiter = (loop, event)
        with self._lock:
            if self._mode != CancelMode.NONE:
                return True
            self._async_waiters.append(waiter)
        try:
            await event.wait()
            return True
        finally:
            with self._lock:
                if waiter in self._async_waiters:
                    self._async_waiters.remove(waiter)

    def mark_step_complete(self) -> None:
        """Signal that the current step has finished."""
        self._step_complete.set()

    def wait_for_step_complete(self, timeout: float = 30.0) -> bool:
        """Wait until the current step completes or timeout expires."""
        return self._step_complete.wait(timeout=timeout)

    def reset_step_event(self) -> None:
        """Reset the step-complete event for the next step."""
        self._step_complete.clear()


__all__ = ["CancelMode", "CancelToken"]
