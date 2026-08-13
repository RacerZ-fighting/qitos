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
from qitos.models.base import ModelStreamEvent
from qitos.models import ModelStreamEventType


class _AttemptStream(AsyncIterator[ModelStreamEvent]):
    def __init__(
        self,
        events: list[ModelStreamEvent],
        *,
        failure: Exception | None = None,
    ) -> None:
        self._events = iter(events)
        self._failure = failure
        self.closed = False

    def __aiter__(self) -> _AttemptStream:
        return self

    async def __anext__(self) -> ModelStreamEvent:
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
) -> list[ModelStreamEvent]:
    return [
        chunk
        async for chunk in transactional_stream_with_retry(
            create_stream,
            policy=policy,
            connection_timeout_seconds=1.0,
            event_idle_timeout_seconds=1.0,
            deadline_monotonic=deadline_monotonic,
            is_terminal=lambda item: item.is_final,
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
async def test_failure_before_first_event_retries_then_publishes_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[_AttemptStream] = []

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("qitos.models.transport.asyncio.sleep", no_sleep)

    async def create_stream() -> AsyncIterator[ModelStreamEvent]:
        failed = not attempts
        stream = _AttemptStream(
            (
                []
                if failed
                else [
                    ModelStreamEvent(
                        type=ModelStreamEventType.TEXT_DELTA,
                        text="committed",
                        event_type="text.delta",
                    ),
                    ModelStreamEvent(
                        type=ModelStreamEventType.COMPLETED,
                        event_type="model.completed",
                        finish_reason="stop",
                    ),
                ]
            ),
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
async def test_empty_incomplete_stream_retries_without_publishing_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("qitos.models.transport.asyncio.sleep", no_sleep)

    async def create_stream() -> AsyncIterator[ModelStreamEvent]:
        nonlocal attempts
        attempts += 1
        return _AttemptStream([])

    with pytest.raises(ModelTransportError) as exc_info:
        await _collect(create_stream, policy=ModelRetryPolicy(max_attempts=2))

    assert attempts == 2
    assert exc_info.value.attempts == 2
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_failure_after_first_event_is_not_retried_or_duplicated() -> None:
    attempts = 0
    published: list[ModelStreamEvent] = []

    async def create_stream() -> AsyncIterator[ModelStreamEvent]:
        nonlocal attempts
        attempts += 1
        return _AttemptStream(
            [
                ModelStreamEvent(
                    type=ModelStreamEventType.TEXT_DELTA,
                    text="visible",
                    event_type="text.delta",
                )
            ],
            failure=TimeoutError("stalled"),
        )

    with pytest.raises(ModelTransportError) as exc_info:
        async for chunk in transactional_stream_with_retry(
            create_stream,
            policy=ModelRetryPolicy(max_attempts=3),
            connection_timeout_seconds=1.0,
            event_idle_timeout_seconds=1.0,
            deadline_monotonic=None,
            is_terminal=lambda item: item.is_final,
        ):
            published.append(chunk)

    assert [chunk.text for chunk in published] == ["visible"]
    assert attempts == 1
    assert exc_info.value.attempts == 1


@pytest.mark.asyncio
async def test_nonterminal_event_is_visible_before_provider_terminal() -> None:
    release_terminal = asyncio.Event()
    terminal_wait_started = asyncio.Event()

    class BlockingTerminalStream(AsyncIterator[ModelStreamEvent]):
        def __init__(self) -> None:
            self._index = 0
            self.closed = False

        def __aiter__(self) -> BlockingTerminalStream:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            if self._index == 0:
                self._index += 1
                return ModelStreamEvent(
                    type=ModelStreamEventType.TEXT_DELTA,
                    text="live",
                    event_type="text.delta",
                )
            if self._index == 1:
                self._index += 1
                terminal_wait_started.set()
                await release_terminal.wait()
                return ModelStreamEvent(type=ModelStreamEventType.COMPLETED, finish_reason="stop")
            raise StopAsyncIteration

        async def aclose(self) -> None:
            self.closed = True

    provider_stream = BlockingTerminalStream()

    async def create_stream() -> AsyncIterator[ModelStreamEvent]:
        return provider_stream

    stream = transactional_stream_with_retry(
        create_stream,
        policy=ModelRetryPolicy(max_attempts=1),
        connection_timeout_seconds=1.0,
        event_idle_timeout_seconds=1.0,
        deadline_monotonic=None,
        is_terminal=lambda item: item.is_final,
    )
    iterator = stream.__aiter__()

    first = await asyncio.wait_for(iterator.__anext__(), timeout=0.1)
    assert first.text == "live"

    terminal_task = asyncio.create_task(iterator.__anext__())
    await asyncio.wait_for(terminal_wait_started.wait(), timeout=0.1)
    assert terminal_task.done() is False
    release_terminal.set()
    terminal = await asyncio.wait_for(terminal_task, timeout=0.1)

    assert terminal.done is True
    assert provider_stream.closed is True


@pytest.mark.asyncio
async def test_nonretryable_status_fails_after_one_attempt() -> None:
    attempts = 0

    class ForbiddenError(Exception):
        status_code = 403

    async def create_stream() -> AsyncIterator[ModelStreamEvent]:
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
            ModelStreamEvent(type=ModelStreamEventType.COMPLETED, finish_reason="stop"),
            ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text="late"),
        ]
    )

    async def create_stream() -> AsyncIterator[ModelStreamEvent]:
        return stream

    published: list[ModelStreamEvent] = []
    with pytest.raises(ModelTransportError) as exc_info:
        async for chunk in transactional_stream_with_retry(
            create_stream,
            policy=ModelRetryPolicy(max_attempts=1),
            connection_timeout_seconds=1.0,
            event_idle_timeout_seconds=1.0,
            deadline_monotonic=None,
            is_terminal=lambda item: item.is_final,
        ):
            published.append(chunk)

    assert "after its terminal" in str(exc_info.value)
    assert published == []
    assert stream.closed is True


@pytest.mark.asyncio
async def test_backoff_cannot_cross_absolute_deadline() -> None:
    attempts = 0

    async def create_stream() -> AsyncIterator[ModelStreamEvent]:
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

    class BlockingStream(AsyncIterator[ModelStreamEvent]):
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self) -> BlockingStream:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            entered.set()
            await release.wait()
            raise StopAsyncIteration

        async def aclose(self) -> None:
            self.closed = True

    stream = BlockingStream()

    async def create_stream() -> AsyncIterator[ModelStreamEvent]:
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
            ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text="ok"),
            ModelStreamEvent(type=ModelStreamEventType.COMPLETED, finish_reason="stop"),
        ]
    )

    async def create_stream() -> AsyncIterator[ModelStreamEvent]:
        return stream

    chunks = await _collect(create_stream, policy=ModelRetryPolicy(max_attempts=1))

    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert chunks[-1].done is True
    assert stream.closed is True
