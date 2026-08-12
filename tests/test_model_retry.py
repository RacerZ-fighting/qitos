"""Behavior tests for the single asynchronous model retry owner."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from qitos.core.errors import ModelRequestDeadlineExceeded, ModelTransportError
from qitos.models.transport import (
    ModelRetryPolicy,
    transactional_stream_with_retry,
)
from qitos.models.base import ModelStreamChunk


class _AttemptStream(AsyncIterator[ModelStreamChunk]):
    def __init__(
        self,
        events: list[ModelStreamChunk],
        *,
        failure: Exception | None = None,
    ) -> None:
        self._events = iter(events)
        self._failure = failure
        self.closed = False

    def __aiter__(self) -> _AttemptStream:
        return self

    async def __anext__(self) -> ModelStreamChunk:
        try:
            return next(self._events)
        except StopIteration:
            if self._failure is not None:
                failure, self._failure = self._failure, None
                raise failure
            raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


async def _collect(
    create_stream: Any,
    *,
    policy: ModelRetryPolicy,
    deadline_monotonic: float | None = None,
) -> list[ModelStreamChunk]:
    return [
        chunk
        async for chunk in transactional_stream_with_retry(
            create_stream,
            policy=policy,
            connection_timeout_seconds=1.0,
            event_idle_timeout_seconds=1.0,
            deadline_monotonic=deadline_monotonic,
            is_terminal=lambda item: item.done,
        )
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"max_attempts": True},
        {"base_delay_seconds": 0},
        {"max_delay_seconds": -1},
        {"retry_window_seconds": False},
    ],
)
def test_retry_policy_rejects_invalid_budgets(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ModelRetryPolicy(**kwargs)


@pytest.mark.asyncio
async def test_failed_attempt_is_discarded_before_success_is_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[_AttemptStream] = []

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("qitos.models.transport.asyncio.sleep", no_sleep)

    async def create_stream() -> AsyncIterator[ModelStreamChunk]:
        failed = not attempts
        stream = _AttemptStream(
            [
                ModelStreamChunk(
                    text="discarded" if failed else "committed",
                    event_type="text.delta",
                ),
                *(
                    []
                    if failed
                    else [
                        ModelStreamChunk(
                            done=True,
                            event_type="model.completed",
                            finish_reason="stop",
                        )
                    ]
                ),
            ],
            failure=TimeoutError("stalled") if failed else None,
        )
        attempts.append(stream)
        return stream

    chunks = await _collect(create_stream, policy=ModelRetryPolicy(max_attempts=2))

    assert [chunk.text for chunk in chunks if chunk.text] == ["committed"]
    assert chunks[-1].done is True
    assert len(attempts) == 2
    assert all(stream.closed for stream in attempts)


@pytest.mark.asyncio
async def test_incomplete_stream_retries_and_never_publishes_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("qitos.models.transport.asyncio.sleep", no_sleep)

    async def create_stream() -> AsyncIterator[ModelStreamChunk]:
        nonlocal attempts
        attempts += 1
        return _AttemptStream([ModelStreamChunk(text=f"attempt-{attempts}")])

    with pytest.raises(ModelTransportError) as exc_info:
        await _collect(create_stream, policy=ModelRetryPolicy(max_attempts=2))

    assert attempts == 2
    assert exc_info.value.attempts == 2
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_nonretryable_status_fails_after_one_attempt() -> None:
    attempts = 0

    class ForbiddenError(Exception):
        status_code = 403

    async def create_stream() -> AsyncIterator[ModelStreamChunk]:
        nonlocal attempts
        attempts += 1
        raise ForbiddenError("forbidden")

    with pytest.raises(ModelTransportError) as exc_info:
        await _collect(create_stream, policy=ModelRetryPolicy(max_attempts=3))

    assert attempts == 1
    assert exc_info.value.status_code == 403
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_event_after_terminal_is_rejected() -> None:
    stream = _AttemptStream(
        [
            ModelStreamChunk(done=True, finish_reason="stop"),
            ModelStreamChunk(text="late"),
        ]
    )

    async def create_stream() -> AsyncIterator[ModelStreamChunk]:
        return stream

    with pytest.raises(ModelTransportError) as exc_info:
        await _collect(create_stream, policy=ModelRetryPolicy(max_attempts=1))

    assert "after its terminal" in str(exc_info.value)
    assert stream.closed is True


@pytest.mark.asyncio
async def test_backoff_cannot_cross_absolute_deadline() -> None:
    attempts = 0

    async def create_stream() -> AsyncIterator[ModelStreamChunk]:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("provider unavailable")

    with pytest.raises(ModelRequestDeadlineExceeded):
        await _collect(
            create_stream,
            policy=ModelRetryPolicy(
                max_attempts=3,
                base_delay_seconds=0.5,
            ),
            deadline_monotonic=time.monotonic() + 0.01,
        )

    assert attempts == 1


@pytest.mark.asyncio
async def test_cancellation_closes_stream_and_is_not_retried() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    attempts = 0

    class BlockingStream(AsyncIterator[ModelStreamChunk]):
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self) -> BlockingStream:
            return self

        async def __anext__(self) -> ModelStreamChunk:
            entered.set()
            await release.wait()
            raise StopAsyncIteration

        async def aclose(self) -> None:
            self.closed = True

    stream = BlockingStream()

    async def create_stream() -> AsyncIterator[ModelStreamChunk]:
        nonlocal attempts
        attempts += 1
        return stream

    task = asyncio.create_task(
        _collect(create_stream, policy=ModelRetryPolicy(max_attempts=3))
    )
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert attempts == 1
    assert stream.closed is True


@pytest.mark.asyncio
async def test_successful_stream_is_closed_once_complete() -> None:
    stream = _AttemptStream(
        [
            ModelStreamChunk(text="ok"),
            ModelStreamChunk(done=True, finish_reason="stop"),
        ]
    )

    async def create_stream() -> AsyncIterator[ModelStreamChunk]:
        return stream

    chunks = await _collect(create_stream, policy=ModelRetryPolicy(max_attempts=1))

    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert chunks[-1].done is True
    assert stream.closed is True
