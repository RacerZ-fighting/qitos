"""Behavioral tests for model transport retry classification."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from qitos.core.errors import ModelTransportError
from qitos.models._openai_retry import (
    ModelRetryPolicy,
    async_run_with_retry,
    sync_transactional_stream_with_retry,
)
from qitos.models.base import ModelStreamChunk
from qitos.models.openai import AsyncOpenAICompatibleModel, OpenAICompatibleModel


def test_lower_level_transport_errors_are_retried(monkeypatch) -> None:
    class ReadTimeoutError(Exception):
        pass

    class SSLError(Exception):
        pass

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("qitos.models._openai_retry.asyncio.sleep", no_sleep)

    async def retry_once(error: Exception) -> tuple[str, int]:
        attempts = 0

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise error
            return "ok"

        result = await async_run_with_retry(operation, ModelRetryPolicy(max_attempts=2))
        return result, attempts

    for error in (ReadTimeoutError("read timed out"), SSLError("TLS failed")):
        result, attempts = asyncio.run(retry_once(error))
        assert result == "ok"
        assert attempts == 2


def test_sync_openai_stream_uses_one_bounded_retry_owner(monkeypatch) -> None:
    class ReadTimeoutError(Exception):
        pass

    attempts = 0
    client_kwargs: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ReadTimeoutError("read timed out")
            assert kwargs["stream"] is True
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="ok", tool_calls=None),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content=""),
                                finish_reason="stop",
                            )
                        ],
                        usage=None,
                    ),
                ]
            )

    class Client:
        def __init__(self, **kwargs):
            client_kwargs.update(kwargs)
            self.chat = SimpleNamespace(completions=Completions())

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: Client(**kwargs)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr("qitos.models._openai_retry.time.sleep", lambda _: None)

    model = OpenAICompatibleModel(
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        max_attempts=2,
    )
    chunks = list(model.stream([{"role": "user", "content": "go"}]))

    assert attempts == 2
    assert client_kwargs["max_retries"] == 0
    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert chunks[-1].done is True


def test_sync_openai_stream_does_not_retry_after_first_event(monkeypatch) -> None:
    attempts = 0
    closes = 0

    class PartialStream:
        def __iter__(self):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="partial", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            raise TimeoutError("stream stalled")

        def close(self) -> None:
            nonlocal closes
            closes += 1

    class Completions:
        def create(self, **_kwargs):
            nonlocal attempts
            attempts += 1
            return PartialStream()

    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: Client(**kwargs)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = OpenAICompatibleModel(
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        max_attempts=2,
    )
    chunks = model.stream([{"role": "user", "content": "go"}])

    assert next(chunks).text == "partial"
    with pytest.raises(ModelTransportError) as exc_info:
        list(chunks)

    assert attempts == 1
    assert closes == 1
    assert exc_info.value.attempts == 1


def test_transactional_stream_retries_midstream_without_publishing_partial_output(
    monkeypatch,
) -> None:
    attempts = 0
    closes = 0

    class AttemptStream:
        def __init__(self, *, fail: bool) -> None:
            self._fail = fail

        def __iter__(self):
            text = "discarded" if self._fail else "committed"
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=text, tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            if self._fail:
                raise TimeoutError("stream stalled after output")
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )

        def close(self) -> None:
            nonlocal closes
            closes += 1

    class Completions:
        def create(self, **_kwargs):
            nonlocal attempts
            attempts += 1
            return AttemptStream(fail=attempts == 1)

    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: Client(**kwargs)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr("qitos.models._openai_retry.time.sleep", lambda _: None)

    model = OpenAICompatibleModel(
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        max_attempts=2,
    )

    chunks = list(model.transactional_stream([{"role": "user", "content": "go"}]))

    assert attempts == 2
    assert closes == 2
    assert "".join(chunk.text for chunk in chunks) == "committed"
    assert chunks[-1].done is True


def test_async_transactional_stream_retries_midstream_without_partial_output(
    monkeypatch,
) -> None:
    attempts = 0

    class AttemptStream:
        def __init__(self, *, fail: bool) -> None:
            self._fail = fail
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self._index += 1
            if self._index == 1:
                text = "discarded" if self._fail else "committed"
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=text, tool_calls=None),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                )
            if self._fail:
                raise TimeoutError("stream stalled after output")
            if self._index == 2:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )
            raise StopAsyncIteration

    class Completions:
        async def create(self, **_kwargs):
            nonlocal attempts
            attempts += 1
            return AttemptStream(fail=attempts == 1)

    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    async def no_sleep(_: float) -> None:
        return None

    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = lambda **kwargs: Client(**kwargs)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr("qitos.models._openai_retry.asyncio.sleep", no_sleep)

    model = AsyncOpenAICompatibleModel(
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        max_attempts=2,
    )

    async def collect() -> list[ModelStreamChunk]:
        return [
            chunk
            async for chunk in model.atransactional_stream(
                [{"role": "user", "content": "go"}]
            )
        ]

    chunks = asyncio.run(collect())

    assert attempts == 2
    assert "".join(chunk.text for chunk in chunks) == "committed"
    assert chunks[-1].done is True


def test_transactional_retry_window_stops_repeated_fast_failures(
    monkeypatch,
) -> None:
    attempts = 0
    clock = 0.0

    def create_stream():
        nonlocal attempts
        attempts += 1
        raise TimeoutError("provider unavailable")

    def sleep(delay: float) -> None:
        nonlocal clock
        clock += delay

    monkeypatch.setattr("qitos.models._openai_retry.random.uniform", lambda *_: 1.0)
    monkeypatch.setattr("qitos.models._openai_retry.time.monotonic", lambda: clock)
    monkeypatch.setattr("qitos.models._openai_retry.time.sleep", sleep)

    with pytest.raises(ModelTransportError):
        list(
            sync_transactional_stream_with_retry(
                create_stream,
                policy=ModelRetryPolicy(
                    max_attempts=10,
                    retry_window_seconds=60.0,
                ),
                is_complete=lambda _: True,
            )
        )

    assert attempts < 10
    assert clock < 60.0


def test_sync_openai_stream_does_not_retry_nonretryable_status(monkeypatch) -> None:
    attempts = 0

    class APIStatusError(Exception):
        status_code = 403

    class Completions:
        def create(self, **_kwargs):
            nonlocal attempts
            attempts += 1
            raise APIStatusError("forbidden")

    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: Client(**kwargs)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = OpenAICompatibleModel(
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        max_attempts=2,
    )

    with pytest.raises(ModelTransportError) as exc_info:
        list(model.stream([{"role": "user", "content": "go"}]))

    assert attempts == 1
    assert exc_info.value.status_code == 403
    assert exc_info.value.retryable is False


def test_rate_limit_uses_only_qitos_retry_budget(monkeypatch) -> None:
    attempts = 0

    class APIStatusError(Exception):
        status_code = 429

    class Completions:
        def create(self, **_kwargs):
            nonlocal attempts
            attempts += 1
            raise APIStatusError("rate limited")

    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: Client(**kwargs)
    fake_openai.APIError = APIStatusError
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr("qitos.models._openai_retry.time.sleep", lambda _: None)

    model = OpenAICompatibleModel(
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        max_attempts=2,
    )

    with pytest.raises(ModelTransportError) as exc_info:
        list(model.stream([{"role": "user", "content": "go"}]))

    assert attempts == 2
    assert exc_info.value.attempts == 2
    assert exc_info.value.retryable is True


def test_stream_options_fallback_is_reused_by_transport_retry(monkeypatch) -> None:
    stream_options_by_attempt: list[bool] = []

    class BadRequestError(Exception):
        status_code = 400

    class Completions:
        def create(self, **kwargs):
            stream_options_by_attempt.append("stream_options" in kwargs)
            if len(stream_options_by_attempt) == 1:
                raise BadRequestError("stream_options is unsupported")
            if len(stream_options_by_attempt) == 2:
                raise TimeoutError("temporary timeout")
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="ok", tool_calls=None),
                                finish_reason="stop",
                            )
                        ],
                        usage=None,
                    )
                ]
            )

    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: Client(**kwargs)
    fake_openai.BadRequestError = BadRequestError
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr("qitos.models._openai_retry.time.sleep", lambda _: None)

    model = OpenAICompatibleModel(
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
        max_attempts=2,
    )
    chunks = list(model.stream([{"role": "user", "content": "go"}]))

    assert stream_options_by_attempt == [True, False, False]
    assert "".join(chunk.text for chunk in chunks) == "ok"
