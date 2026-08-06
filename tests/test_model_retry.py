"""Behavioral tests for model transport retry classification."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

from qitos.models._openai_retry import ModelRetryPolicy, async_run_with_retry
from qitos.models.openai import OpenAICompatibleModel


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
