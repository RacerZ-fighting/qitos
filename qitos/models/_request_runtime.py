"""Run-scoped deadline and cancellation for model transport adapters."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from ..core.errors import ModelRequestCancelled, ModelRequestDeadlineExceeded


@dataclass(frozen=True)
class _ModelRequestRuntime:
    """Controls one logical model request across all provider attempts."""

    deadline_monotonic: float | None
    cancelled: Callable[[], bool]


_CURRENT_RUNTIME: ContextVar[_ModelRequestRuntime | None] = ContextVar(
    "qitos_model_request_runtime",
    default=None,
)


@contextmanager
def model_request_runtime(
    *,
    deadline_monotonic: float | None,
    cancelled: Callable[[], bool],
) -> Iterator[None]:
    """Bind one Engine request budget for adapters and retry helpers."""

    token = _CURRENT_RUNTIME.set(
        _ModelRequestRuntime(
            deadline_monotonic=deadline_monotonic,
            cancelled=cancelled,
        )
    )
    try:
        yield
    finally:
        _CURRENT_RUNTIME.reset(token)


def remaining_request_seconds() -> float | None:
    """Return live time remaining for the current logical request."""

    runtime = _CURRENT_RUNTIME.get()
    deadline = None if runtime is None else runtime.deadline_monotonic
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def ensure_request_active() -> None:
    """Raise when cancellation or the Engine deadline forbids more work."""

    runtime = _CURRENT_RUNTIME.get()
    if runtime is None:
        return
    if runtime.cancelled():
        raise ModelRequestCancelled("model request cancelled")
    remaining = remaining_request_seconds()
    if remaining is not None and remaining <= 0:
        raise ModelRequestDeadlineExceeded("model request deadline expired")


def effective_request_timeout(configured_seconds: float) -> float:
    """Clamp a provider timeout to the current Engine deadline."""

    if isinstance(configured_seconds, bool) or configured_seconds <= 0:
        raise ValueError("model request timeout must be positive")
    ensure_request_active()
    remaining = remaining_request_seconds()
    if remaining is None:
        return float(configured_seconds)
    if remaining <= 0:
        raise ModelRequestDeadlineExceeded("model request deadline expired")
    return min(float(configured_seconds), remaining)


def sleep_before_retry(delay_seconds: float) -> None:
    """Wait for sync retry backoff while observing cancellation and deadline."""

    runtime = _CURRENT_RUNTIME.get()
    if runtime is None:
        time.sleep(delay_seconds)
        return
    wake_at = time.monotonic() + max(0.0, delay_seconds)
    while True:
        ensure_request_active()
        remaining_delay = wake_at - time.monotonic()
        if remaining_delay <= 0:
            return
        remaining_request = remaining_request_seconds()
        sleep_for = min(0.05, remaining_delay)
        if remaining_request is not None:
            sleep_for = min(sleep_for, remaining_request)
        if sleep_for <= 0:
            ensure_request_active()
        time.sleep(sleep_for)


async def asleep_before_retry(delay_seconds: float) -> None:
    """Wait for async retry backoff while observing cancellation and deadline."""

    runtime = _CURRENT_RUNTIME.get()
    if runtime is None:
        await asyncio.sleep(delay_seconds)
        return
    wake_at = time.monotonic() + max(0.0, delay_seconds)
    while True:
        ensure_request_active()
        remaining_delay = wake_at - time.monotonic()
        if remaining_delay <= 0:
            return
        remaining_request = remaining_request_seconds()
        sleep_for = min(0.05, remaining_delay)
        if remaining_request is not None:
            sleep_for = min(sleep_for, remaining_request)
        if sleep_for <= 0:
            ensure_request_active()
        await asyncio.sleep(sleep_for)
