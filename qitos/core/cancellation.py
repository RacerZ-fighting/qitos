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

    Mode semantics (the loop and Tool executor honor them exactly):

    - ``"immediate"`` interrupts in-flight work at the next await point:
      model streams stop between chunks, Tool admission and retry backoff
      stop before the next attempt, and not-started calls are terminalized
      as cancelled.
    - ``"after_step"`` never interrupts an in-flight model stream or Tool
      call; the run stops after the current turn commits. ``is_cancel_requested``
      is still true, so turn-boundary and new-model-admission checks stop the
      run without starting further work.

    ``wait_cancelled`` wakes on any mode; ``wait_immediate`` wakes only on
    ``"immediate"``. Both are async-native bridges used while awaiting model
    streams or hooks; they never park a thread the way
    ``asyncio.to_thread(token.wait)`` would, so a completed run leaves no
    blocked waiter behind.

    ``mark_step_complete``/``reset_step_event`` bracket each loop turn so
    owners requesting ``"after_step"`` can await the current step's end via
    ``wait_for_step_complete``.
    """

    def __init__(self) -> None:
        self._mode = CancelMode.NONE
        self._lock = threading.Lock()
        self._cancel_requested = threading.Event()
        self._step_complete = threading.Event()
        self._async_waiters: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Event]
        ] = []
        self._immediate_waiters: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Event]
        ] = []

    @property
    def mode(self) -> CancelMode:
        with self._lock:
            return self._mode

    @property
    def is_cancel_requested(self) -> bool:
        """True once cancellation of any mode was requested."""

        with self._lock:
            return self._mode != CancelMode.NONE

    @property
    def immediate_requested(self) -> bool:
        """True only when ``"immediate"`` cancellation was requested."""

        with self._lock:
            return self._mode is CancelMode.IMMEDIATE

    def request_cancel(self, mode: str = "immediate") -> None:
        """Signal the run to cancel.

        Parameters
        ----------
        mode : str
            ``"immediate"`` interrupts in-flight work; ``"after_step"`` stops
            the run after the current turn commits.
        """
        with self._lock:
            self._mode = CancelMode(mode)
            self._cancel_requested.set()
            waiters = list(self._async_waiters)
            self._async_waiters.clear()
            immediate = (
                list(self._immediate_waiters)
                if self._mode is CancelMode.IMMEDIATE
                else []
            )
            if immediate:
                self._immediate_waiters.clear()
        for loop, event in waiters:
            loop.call_soon_threadsafe(event.set)
        for loop, event in immediate:
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
        """Wait asynchronously until cancellation of any mode is requested."""

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

    async def wait_immediate(self) -> bool:
        """Wait asynchronously until ``"immediate"`` cancellation is requested.

        ``"after_step"`` requests do not wake this waiter: in-flight model
        streams and Tool calls race against this method so a graceful stop
        lets the current step finish.
        """

        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        waiter = (loop, event)
        with self._lock:
            if self._mode is CancelMode.IMMEDIATE:
                return True
            self._immediate_waiters.append(waiter)
        try:
            await event.wait()
            return True
        finally:
            with self._lock:
                if waiter in self._immediate_waiters:
                    self._immediate_waiters.remove(waiter)

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
