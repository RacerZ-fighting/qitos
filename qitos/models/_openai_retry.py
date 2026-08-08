"""Bounded retry helpers for QitOS-owned OpenAI-compatible requests."""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

from ..core.errors import (
    ModelRequestCancelled,
    ModelRequestDeadlineExceeded,
    ModelTransportError,
)
from ._request_runtime import (
    asleep_before_retry,
    effective_request_timeout,
    ensure_request_active,
    remaining_request_seconds,
    sleep_before_retry,
)

_logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class _IncompleteStreamError(ConnectionError):
    """The provider closed a stream before its terminal event."""


@dataclass(frozen=True)
class ModelRetryPolicy:
    """One logical request's retry budget.

    The OpenAI SDK must use ``max_retries=0`` when this policy is active so a
    single visible loop owns every HTTP attempt.
    """

    max_attempts: int = 2
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 30.0
    retry_window_seconds: float = 300.0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts, int
        ):
            raise ValueError("max_attempts must be a positive integer")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if (
            isinstance(self.retry_window_seconds, bool)
            or not isinstance(self.retry_window_seconds, (int, float))
            or self.retry_window_seconds <= 0
        ):
            raise ValueError("retry_window_seconds must be positive")


def _status_code(exc: Exception) -> int | None:
    for value in (
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _header(exc: Exception, name: str) -> str | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    value = getter(name)
    return str(value).strip() if value is not None else None


def _retry_after(exc: Exception) -> float | None:
    milliseconds = _header(exc, "retry-after-ms")
    if milliseconds:
        try:
            return max(0.0, float(milliseconds) / 1000.0)
        except ValueError:
            return None
    seconds = _header(exc, "retry-after")
    if seconds:
        try:
            return max(0.0, float(seconds))
        except ValueError:
            return None
    return None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (ModelRequestCancelled, ModelRequestDeadlineExceeded)):
        return False
    if isinstance(exc, ModelTransportError):
        return exc.retryable
    should_retry = _header(exc, "x-should-retry")
    if should_retry:
        if should_retry.lower() == "false":
            return False
        if should_retry.lower() == "true":
            return True
    status = _status_code(exc)
    if status is not None:
        return status in {408, 409, 429} or status >= 500
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    # OpenAI-compatible clients may surface transport failures from a lower-level
    # HTTP stack (for example urllib3) instead of the SDK's exception classes. Keep
    # those failures inside the bounded retry policy rather than turning a transient
    # connection problem into a terminal model error.
    try:
        import httpx

        if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
            return True
    except ImportError:
        pass
    if type(exc).__name__ == "APIError":
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "stream timeout",
                "no healthy upstream",
                "service unavailable",
                "temporarily unavailable",
            )
        )
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "ReadTimeoutError",
        "RemoteProtocolError",
        "SSLError",
        "TimeoutException",
    }


def _retry_delay(
    exc: Exception, *, failed_attempt: int, policy: ModelRetryPolicy
) -> float | None:
    if failed_attempt >= policy.max_attempts or not _is_retryable(exc):
        return None
    retry_after = _retry_after(exc)
    if retry_after is not None:
        return retry_after if retry_after <= policy.max_delay_seconds else None
    base = min(
        policy.base_delay_seconds * (2 ** (failed_attempt - 1)),
        policy.max_delay_seconds,
    )
    return base * float(random.uniform(0.75, 1.25))


def _retry_delay_with_window(
    exc: Exception,
    *,
    failed_attempt: int,
    policy: ModelRetryPolicy,
    retry_deadline: float | None,
) -> tuple[float | None, float | None]:
    """Return the next delay while bounding recovery after the first failure."""
    delay = _retry_delay(exc, failed_attempt=failed_attempt, policy=policy)
    if delay is None:
        return None, retry_deadline
    now = time.monotonic()
    if retry_deadline is None:
        retry_deadline = now + policy.retry_window_seconds
    if now + delay >= retry_deadline:
        return None, retry_deadline
    return delay, retry_deadline


def _terminal(exc: Exception, attempts: int) -> ModelTransportError:
    return ModelTransportError(
        f"model request failed after {attempts} attempt(s): {exc}",
        attempts=attempts,
        retryable=_is_retryable(exc),
        status_code=_status_code(exc),
    )


def _is_transport_failure(exc: Exception) -> bool:
    """Keep local validation/programming errors outside the transport taxonomy."""
    return (
        isinstance(exc, ModelTransportError)
        or _status_code(exc) is not None
        or _is_retryable(exc)
        or type(exc).__name__ in {"APIError", "APIStatusError", "OpenAIError"}
    )


def _announce_retry(
    exc: Exception, *, attempt: int, delay: float, policy: ModelRetryPolicy
) -> None:
    _logger.warning(
        "model request retry attempt=%d/%d delay_seconds=%.3f status_code=%s error=%s",
        attempt + 1,
        policy.max_attempts,
        delay,
        _status_code(exc),
        type(exc).__name__,
    )


async def _close_stream(stream: Any) -> None:
    close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
    if callable(close):
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            _logger.debug("model stream close failed", exc_info=True)


def run_with_retry(operation: Callable[[], _T], policy: ModelRetryPolicy) -> _T:
    """Run a synchronous request with one explicit, bounded retry owner."""
    retry_deadline: float | None = None
    for attempt in range(1, policy.max_attempts + 1):
        ensure_request_active()
        try:
            result = operation()
            ensure_request_active()
            return result
        except (ModelRequestCancelled, ModelRequestDeadlineExceeded):
            raise
        except Exception as exc:
            ensure_request_active()
            delay, retry_deadline = _retry_delay_with_window(
                exc,
                failed_attempt=attempt,
                policy=policy,
                retry_deadline=retry_deadline,
            )
            if delay is None:
                if not _is_transport_failure(exc):
                    raise
                raise _terminal(exc, attempt) from exc
            _announce_retry(exc, attempt=attempt, delay=delay, policy=policy)
            sleep_before_retry(delay)
    raise AssertionError("unreachable retry loop")


async def async_run_with_retry(
    operation: Callable[[], Awaitable[_T]], policy: ModelRetryPolicy
) -> _T:
    """Run an asynchronous request with one explicit, bounded retry owner."""
    retry_deadline: float | None = None
    for attempt in range(1, policy.max_attempts + 1):
        ensure_request_active()
        try:
            result = await operation()
            ensure_request_active()
            return result
        except (ModelRequestCancelled, ModelRequestDeadlineExceeded):
            raise
        except Exception as exc:
            ensure_request_active()
            delay, retry_deadline = _retry_delay_with_window(
                exc,
                failed_attempt=attempt,
                policy=policy,
                retry_deadline=retry_deadline,
            )
            if delay is None:
                if not _is_transport_failure(exc):
                    raise
                raise _terminal(exc, attempt) from exc
            _announce_retry(exc, attempt=attempt, delay=delay, policy=policy)
            await asleep_before_retry(delay)
    raise AssertionError("unreachable retry loop")


def _close_sync_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            _logger.debug("model stream close failed", exc_info=True)


def sync_stream_with_retry(
    create_stream: Callable[[], Any], *, policy: ModelRetryPolicy
) -> Any:
    """Stream synchronously with bounded retries before the first event."""
    retry_deadline: float | None = None
    for attempt in range(1, policy.max_attempts + 1):
        ensure_request_active()
        stream: Any = None
        received_event = False
        try:
            stream = create_stream()
            for event in stream:
                ensure_request_active()
                received_event = True
                yield event
            ensure_request_active()
            return
        except (ModelRequestCancelled, ModelRequestDeadlineExceeded):
            raise
        except Exception as exc:
            ensure_request_active()
            if received_event:
                delay = None
            else:
                delay, retry_deadline = _retry_delay_with_window(
                    exc,
                    failed_attempt=attempt,
                    policy=policy,
                    retry_deadline=retry_deadline,
                )
            if delay is None:
                if not _is_transport_failure(exc):
                    raise
                raise _terminal(exc, attempt) from exc
            _close_sync_stream(stream)
            stream = None
            _announce_retry(exc, attempt=attempt, delay=delay, policy=policy)
            sleep_before_retry(delay)
        finally:
            _close_sync_stream(stream)
    raise AssertionError("unreachable retry loop")


def sync_transactional_stream_with_retry(
    create_stream: Callable[[], Iterator[_T]],
    *,
    policy: ModelRetryPolicy,
    is_complete: Callable[[_T], bool],
) -> Iterator[_T]:
    """Retry a whole stream attempt without exposing partial output.

    A failed attempt may have produced text or tool-call deltas. They remain
    private to that attempt and are discarded before retrying, so callers can
    safely execute actions only from the completed response.
    """
    retry_deadline: float | None = None
    for attempt in range(1, policy.max_attempts + 1):
        ensure_request_active()
        stream: Any = None
        buffered: list[_T] = []
        try:
            stream = create_stream()
            for event in stream:
                ensure_request_active()
                buffered.append(event)
            if not any(is_complete(event) for event in buffered):
                raise _IncompleteStreamError(
                    "model stream ended before its terminal event"
                )
        except (ModelRequestCancelled, ModelRequestDeadlineExceeded):
            raise
        except Exception as exc:
            ensure_request_active()
            delay, retry_deadline = _retry_delay_with_window(
                exc,
                failed_attempt=attempt,
                policy=policy,
                retry_deadline=retry_deadline,
            )
            if delay is None:
                if not _is_transport_failure(exc):
                    raise
                raise _terminal(exc, attempt) from exc
            _close_sync_stream(stream)
            stream = None
            _announce_retry(exc, attempt=attempt, delay=delay, policy=policy)
            sleep_before_retry(delay)
            continue
        finally:
            _close_sync_stream(stream)

        ensure_request_active()
        yield from buffered
        return
    raise AssertionError("unreachable retry loop")


async def stream_with_retry(
    create_stream: Callable[[], Awaitable[AsyncIterator[_T]]],
    *,
    policy: ModelRetryPolicy,
    idle_timeout_seconds: float,
    request_timeout_seconds: float,
) -> AsyncIterator[_T]:
    """Stream with event-idle timeout and safe retry before the first event only."""
    loop = asyncio.get_running_loop()
    retry_deadline: float | None = None
    for attempt in range(1, policy.max_attempts + 1):
        ensure_request_active()
        stream: Any = None
        received_event = False
        request_timeout = effective_request_timeout(request_timeout_seconds)
        request_remaining = remaining_request_seconds()
        request_deadline = loop.time() + (
            request_timeout
            if request_remaining is None
            else min(request_timeout, request_remaining)
        )
        try:
            stream = await asyncio.wait_for(
                create_stream(), timeout=request_timeout
            )
            iterator = stream.__aiter__()
            while True:
                ensure_request_active()
                remaining = request_deadline - loop.time()
                engine_remaining = remaining_request_seconds()
                if engine_remaining is not None:
                    remaining = min(remaining, engine_remaining)
                if remaining <= 0:
                    ensure_request_active()
                    raise TimeoutError("model stream exceeded provider timeout")
                try:
                    event = await asyncio.wait_for(
                        iterator.__anext__(),
                        timeout=min(idle_timeout_seconds, remaining),
                    )
                except StopAsyncIteration:
                    ensure_request_active()
                    return
                except asyncio.TimeoutError as exc:
                    ensure_request_active()
                    raise TimeoutError(
                        "model stream idle timeout waiting for provider event"
                    ) from exc
                received_event = True
                ensure_request_active()
                yield event
        except (ModelRequestCancelled, ModelRequestDeadlineExceeded):
            raise
        except Exception as exc:
            ensure_request_active()
            if received_event:
                delay = None
            else:
                delay, retry_deadline = _retry_delay_with_window(
                    exc,
                    failed_attempt=attempt,
                    policy=policy,
                    retry_deadline=retry_deadline,
                )
            if delay is None:
                if not _is_transport_failure(exc):
                    raise
                raise _terminal(exc, attempt) from exc
            await _close_stream(stream)
            stream = None
            _announce_retry(exc, attempt=attempt, delay=delay, policy=policy)
            await asleep_before_retry(delay)
        finally:
            await _close_stream(stream)
    raise AssertionError("unreachable retry loop")


async def transactional_stream_with_retry(
    create_stream: Callable[[], Awaitable[AsyncIterator[_T]]],
    *,
    policy: ModelRetryPolicy,
    idle_timeout_seconds: float,
    request_timeout_seconds: float,
    is_complete: Callable[[_T], bool],
) -> AsyncIterator[_T]:
    """Retry complete async stream attempts and publish only committed output.

    The request timeout covers establishing the stream. Once connected, the
    event-idle timeout detects a stalled stream while allowing a healthy long
    response to continue as long as provider events keep arriving.
    """
    retry_deadline: float | None = None
    for attempt in range(1, policy.max_attempts + 1):
        ensure_request_active()
        stream: Any = None
        buffered: list[_T] = []
        request_timeout = effective_request_timeout(request_timeout_seconds)
        try:
            stream = await asyncio.wait_for(
                create_stream(), timeout=request_timeout
            )
            iterator = stream.__aiter__()
            while True:
                ensure_request_active()
                event_timeout = idle_timeout_seconds
                engine_remaining = remaining_request_seconds()
                if engine_remaining is not None:
                    event_timeout = min(event_timeout, engine_remaining)
                try:
                    event = await asyncio.wait_for(
                        iterator.__anext__(), timeout=event_timeout
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    ensure_request_active()
                    raise TimeoutError(
                        "model stream idle timeout waiting for provider event"
                    ) from exc
                ensure_request_active()
                buffered.append(event)
            if not any(is_complete(event) for event in buffered):
                raise _IncompleteStreamError(
                    "model stream ended before its terminal event"
                )
        except (ModelRequestCancelled, ModelRequestDeadlineExceeded):
            raise
        except Exception as exc:
            ensure_request_active()
            delay, retry_deadline = _retry_delay_with_window(
                exc,
                failed_attempt=attempt,
                policy=policy,
                retry_deadline=retry_deadline,
            )
            if delay is None:
                if not _is_transport_failure(exc):
                    raise
                raise _terminal(exc, attempt) from exc
            await _close_stream(stream)
            stream = None
            _announce_retry(exc, attempt=attempt, delay=delay, policy=policy)
            await asleep_before_retry(delay)
            continue
        finally:
            await _close_stream(stream)

        ensure_request_active()
        for event in buffered:
            yield event
        return
    raise AssertionError("unreachable retry loop")


__all__ = [
    "ModelRetryPolicy",
    "async_run_with_retry",
    "run_with_retry",
    "sync_stream_with_retry",
    "sync_transactional_stream_with_retry",
    "stream_with_retry",
    "transactional_stream_with_retry",
]
