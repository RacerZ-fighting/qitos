"""Deadline, retry, and resource lifecycle for asynchronous model transports."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from ..core.errors import (
    ModelRequestDeadlineExceeded,
    ModelTransportError,
)

_logger = logging.getLogger(__name__)
EventT = TypeVar("EventT")
ResultT = TypeVar("ResultT")

_RETRYABLE_PROVIDER_ERROR_CODES = frozenset(
    {
        "allocationquota",
        "burst_rate",
        "concurrency",
        "limitrequests",
        "rate_limit_exceeded",
        "requests_rate_limit_exceeded",
        "resourceexhausted",
        "throttling_allocationquota",
        "throttling_burstrate",
        "throttling_concurrency",
        "throttling_ratequota",
        "too_many_requests",
    }
)
_NON_RETRYABLE_PROVIDER_ERROR_CODES = frozenset(
    {
        "account_quota_exceeded",
        "arrearage",
        "billing_hard_limit_reached",
        "free_allocated_quota_exceeded",
        "insufficient_quota",
        "quota_exceeded",
    }
)


def provider_error_details(value: Any) -> tuple[str | None, str | None]:
    """Extract a provider error code and message from an SDK value or body."""

    candidates: list[Any] = [value]
    seen: set[int] = set()
    fallback_message: str | None = None
    index = 0
    while index < len(candidates):
        candidate = candidates[index]
        index += 1
        identity = id(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(candidate, dict):
            nested = candidate.get("error")
            source = nested if isinstance(nested, dict) else candidate
            code = source.get("code") or source.get("error_code")
            source_type = source.get("type")
            if code is None and (
                source is not candidate
                or not any(
                    key in candidate
                    for key in ("body", "error", "response", "data", "details")
                )
            ):
                code = source_type
            message = (
                source.get("message")
                or source.get("detail")
                or source.get("error")
            )
            for key in ("body", "error", "response", "data", "details"):
                nested_value = candidate.get(key)
                if nested_value is not None:
                    candidates.append(nested_value)
        elif isinstance(candidate, (bytes, str)):
            message = (
                candidate.decode(errors="replace")
                if isinstance(candidate, bytes)
                else candidate
            )
            try:
                decoded = json.loads(message)
            except (TypeError, ValueError):
                decoded = None
            if isinstance(decoded, (dict, list)):
                candidates.append(decoded)
            code = None
        else:
            code = (
                getattr(candidate, "code", None)
                or getattr(candidate, "error_code", None)
            )
            source_type = getattr(candidate, "type", None)
            if code is None and (
                not any(
                    getattr(candidate, attribute, None) is not None
                    for attribute in ("body", "error", "response")
                )
            ):
                code = source_type
            message = getattr(candidate, "message", None) or getattr(
                candidate, "detail", None
            )
            for attribute in ("body", "error", "response"):
                nested = getattr(candidate, attribute, None)
                if nested is not None:
                    candidates.append(nested)
        if code is not None:
            return (
                str(code).strip(),
                str(message).strip() if message is not None else None,
            )
        if message is not None and fallback_message is None:
            fallback_message = str(message).strip()
    return None, fallback_message


def is_retryable_provider_error(
    code: str | None,
    message: str | None,
    *,
    default: bool = False,
) -> bool:
    """Classify transient rate limits separately from account quota failures."""

    normalized_code = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(code or "")
    ).strip("_")
    normalized_message = str(message or "").casefold()

    if any(
        marker in normalized_message
        for marker in (
            "free allocated quota exceeded",
            "free tier",
            "hour allocated quota exceeded",
            "month allocated quota exceeded",
            "week allocated quota exceeded",
            "billing hard limit",
            "account quota",
            "arrearage",
            "you exceeded your current quota",
            "insufficient balance",
        )
    ):
        return False
    explicit_token_limit = (
        "allocated quota exceeded" in normalized_message
        or "token-limit" in normalized_message
        or "token limit" in normalized_message
        or re.search(r"\b(?:tpm|tps)\b", normalized_message) is not None
    )
    if explicit_token_limit:
        return True
    if normalized_code in _NON_RETRYABLE_PROVIDER_ERROR_CODES:
        return False
    if normalized_code in _RETRYABLE_PROVIDER_ERROR_CODES:
        return True
    if any(
        marker in normalized_message
        for marker in (
            "too many requests",
            "being throttled",
            "rate limit",
            "request rate",
            "resource exhausted",
        )
    ):
        return True
    return default


def remaining_request_seconds(deadline_monotonic: float | None) -> float | None:
    """Return the remaining request budget for an absolute deadline."""

    if deadline_monotonic is None:
        return None
    return max(0.0, float(deadline_monotonic) - time.monotonic())


def ensure_request_active(deadline_monotonic: float | None) -> None:
    """Raise when the absolute request deadline has expired."""

    remaining = remaining_request_seconds(deadline_monotonic)
    if remaining is not None and remaining <= 0:
        raise ModelRequestDeadlineExceeded("model request deadline expired")


def effective_request_timeout(
    configured_seconds: float,
    deadline_monotonic: float | None,
) -> float:
    """Clamp a positive provider timeout to the remaining request budget."""

    if (
        isinstance(configured_seconds, bool)
        or not isinstance(configured_seconds, (int, float))
        or configured_seconds <= 0
    ):
        raise ValueError("model request timeout must be positive")
    ensure_request_active(deadline_monotonic)
    remaining = remaining_request_seconds(deadline_monotonic)
    if remaining is None:
        return float(configured_seconds)
    return min(float(configured_seconds), remaining)


async def await_with_deadline(
    operation: Awaitable[ResultT],
    *,
    timeout_seconds: float,
    deadline_monotonic: float | None,
) -> ResultT:
    """Await an operation using only its remaining absolute deadline."""

    timeout = effective_request_timeout(timeout_seconds, deadline_monotonic)
    try:
        return await asyncio.wait_for(operation, timeout=timeout)
    except asyncio.TimeoutError as exc:
        ensure_request_active(deadline_monotonic)
        raise TimeoutError("model provider operation timed out") from exc


async def sleep_before_retry(
    delay_seconds: float,
    *,
    deadline_monotonic: float | None,
) -> None:
    """Wait for retry backoff without granting a fresh timeout budget."""

    delay = max(0.0, float(delay_seconds))
    ensure_request_active(deadline_monotonic)
    remaining = remaining_request_seconds(deadline_monotonic)
    if remaining is not None and delay >= remaining:
        raise ModelRequestDeadlineExceeded(
            "model retry backoff would exceed request deadline"
        )
    await asyncio.sleep(delay)
    ensure_request_active(deadline_monotonic)


class _IncompleteStreamError(ConnectionError):
    """The provider closed a stream without a terminal event."""


@dataclass(frozen=True, slots=True)
class ModelRetryPolicy:
    """Retry budget for one logical model request."""

    max_attempts: int = 2
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 30.0
    retry_window_seconds: float = 300.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        for name in (
            "base_delay_seconds",
            "max_delay_seconds",
            "retry_window_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")


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
    if isinstance(exc, ModelRequestDeadlineExceeded):
        return False
    if isinstance(exc, ModelTransportError):
        return exc.retryable
    status = _status_code(exc)
    if status == 429:
        code, message = provider_error_details(exc)
        should_retry = _header(exc, "x-should-retry")
        return is_retryable_provider_error(
            code,
            message,
            default=should_retry.casefold() != "false"
            if should_retry is not None
            else True,
        )
    should_retry = _header(exc, "x-should-retry")
    if should_retry:
        if should_retry.casefold() == "false":
            return False
        if should_retry.casefold() == "true":
            return True
    if status is not None:
        return status in {408, 409, 429} or status >= 500
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    try:
        import httpx

        if isinstance(exc, httpx.TransportError):
            return True
    except ImportError:
        pass
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


def _is_transport_failure(exc: Exception) -> bool:
    return (
        isinstance(exc, ModelTransportError)
        or _status_code(exc) is not None
        or _is_retryable(exc)
        or type(exc).__name__ in {"APIError", "APIStatusError", "OpenAIError"}
    )


def _retry_delay(
    exc: Exception,
    *,
    failed_attempt: int,
    policy: ModelRetryPolicy,
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


def _next_retry(
    exc: Exception,
    *,
    failed_attempt: int,
    policy: ModelRetryPolicy,
    retry_deadline: float | None,
) -> tuple[float | None, float | None]:
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


def _announce_retry(
    exc: Exception,
    *,
    attempt: int,
    delay: float,
    policy: ModelRetryPolicy,
) -> None:
    _logger.warning(
        "model request retry attempt=%d/%d delay_seconds=%.3f "
        "status_code=%s error=%s",
        attempt + 1,
        policy.max_attempts,
        delay,
        _status_code(exc),
        type(exc).__name__,
    )


async def close_async_resource(resource: Any) -> None:
    """Best-effort close one async transport resource.

    Cancellation is never swallowed. Ordinary close failures are diagnostic
    only because they must not replace the request's primary failure.
    """

    if resource is None:
        return
    async_close = getattr(resource, "aclose", None)
    if callable(async_close):
        try:
            await async_close()
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.debug("model stream close failed", exc_info=True)
        return
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    try:
        if inspect.iscoroutinefunction(close):
            await close()
        else:
            result = await asyncio.to_thread(close)
            if inspect.isawaitable(result):
                await result
    except asyncio.CancelledError:
        raise
    except Exception:
        _logger.debug("model stream close failed", exc_info=True)


async def transactional_stream_with_retry(
    create_stream: Callable[[], Awaitable[AsyncIterator[EventT]]],
    *,
    policy: ModelRetryPolicy,
    connection_timeout_seconds: float,
    event_idle_timeout_seconds: float,
    deadline_monotonic: float | None,
    is_terminal: Callable[[EventT], bool],
) -> AsyncIterator[EventT]:
    """Publish live deltas and retry only before an attempt becomes observable.

    Non-terminal events are yielded immediately. The single terminal event is held
    until provider EOF so callers never observe a successful terminal followed by a
    late provider event. Once any event has arrived, failure ends the request instead
    of replaying the attempt and duplicating visible output or tool-call deltas.
    """

    retry_deadline: float | None = None
    for attempt in range(1, policy.max_attempts + 1):
        ensure_request_active(deadline_monotonic)
        stream: AsyncIterator[EventT] | None = None
        terminal: list[EventT] = []
        published = False
        terminal_seen = False
        try:
            stream = await await_with_deadline(
                create_stream(),
                timeout_seconds=connection_timeout_seconds,
                deadline_monotonic=deadline_monotonic,
            )
            iterator = stream.__aiter__()
            while True:
                ensure_request_active(deadline_monotonic)
                event_timeout = effective_request_timeout(
                    event_idle_timeout_seconds,
                    deadline_monotonic,
                )
                try:
                    event = await asyncio.wait_for(
                        iterator.__anext__(),
                        timeout=event_timeout,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    ensure_request_active(deadline_monotonic)
                    raise TimeoutError(
                        "model stream idle timeout waiting for provider event"
                    ) from exc
                if terminal_seen:
                    raise ModelTransportError(
                        "model stream emitted an event after its terminal chunk",
                        attempts=1,
                        retryable=False,
                    )
                terminal_seen = is_terminal(event)
                if terminal_seen:
                    terminal.append(event)
                    continue
                published = True
                yield event
            if not terminal_seen:
                raise _IncompleteStreamError(
                    "model stream ended before its terminal event"
                )
        except asyncio.CancelledError:
            raise
        except ModelRequestDeadlineExceeded:
            raise
        except Exception as exc:
            ensure_request_active(deadline_monotonic)
            delay = None
            if not published and not terminal_seen:
                delay, retry_deadline = _next_retry(
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
            await close_async_resource(stream)
            stream = None
            await sleep_before_retry(
                delay,
                deadline_monotonic=deadline_monotonic,
            )
            continue
        finally:
            await close_async_resource(stream)

        ensure_request_active(deadline_monotonic)
        yield terminal[0]
        return
    raise AssertionError("unreachable retry loop")


__all__ = [
    "ModelRetryPolicy",
    "await_with_deadline",
    "close_async_resource",
    "effective_request_timeout",
    "ensure_request_active",
    "is_retryable_provider_error",
    "provider_error_details",
    "remaining_request_seconds",
    "sleep_before_retry",
    "transactional_stream_with_retry",
]
