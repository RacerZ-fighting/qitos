from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from qitos.harness import build_model_for_preset, resolve_reasoning


class _AsyncStream(AsyncIterator[Any]):
    def __init__(self, items: list[Any]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> _AsyncStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        return None


async def _drain(model: Any) -> list[Any]:
    return [
        chunk async for chunk in model.stream([{"role": "user", "content": "answer"}])
    ]


@pytest.mark.parametrize(
    ("requested", "sent"),
    [
        ("low", "low"),
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "max"),
        ("max", "max"),
    ],
)
@pytest.mark.asyncio
async def test_kimi_reasoning_effort_reaches_chat_completion_request(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    sent: str,
) -> None:
    captured: dict[str, object] = {}

    class Client:
        def __init__(self, **_: Any) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kwargs: object) -> Any:
            captured.update(kwargs)
            return _AsyncStream(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content="ok",
                                    reasoning_content=None,
                                    tool_calls=None,
                                ),
                                finish_reason="stop",
                            )
                        ],
                        usage=None,
                    )
                ]
            )

        async def aclose(self) -> None:
            return None

    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = build_model_for_preset(
        family_id="kimi",
        model_name="kimi-k3",
        api_key="test-key",
        base_url="https://example.test/v1",
        reasoning_effort=requested,
        max_attempts=1,
    )

    chunks = await _drain(model)

    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert captured["reasoning_effort"] == sent


@pytest.mark.asyncio
async def test_model_reasoning_reaches_responses_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Client:
        def __init__(self, **_: Any) -> None:
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: object) -> Any:
            captured.update(kwargs)
            return _AsyncStream(
                [
                    {"type": "response.output_text.delta", "delta": "ok"},
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "response-1",
                            "status": "completed",
                            "model": "kimi-k3",
                            "output": [
                                {
                                    "type": "message",
                                    "content": [{"type": "output_text", "text": "ok"}],
                                }
                            ],
                        },
                    },
                ]
            )

        async def aclose(self) -> None:
            return None

    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = build_model_for_preset(
        family_id="kimi",
        model_name="kimi-k3",
        api_key="test-key",
        base_url="https://example.test/v1",
        api_mode="responses",
        reasoning_effort="xhigh",
        max_attempts=1,
    )

    chunks = await _drain(model)

    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert captured["reasoning"] == {"effort": "max"}
    assert "include" not in captured


@pytest.mark.asyncio
async def test_gpt56_reasoning_and_opaque_continuation_reach_responses_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Client:
        def __init__(self, **_: Any) -> None:
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: object) -> Any:
            captured.update(kwargs)
            return _AsyncStream(
                [
                    {"type": "response.output_text.delta", "delta": "ok"},
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "response-1",
                            "status": "completed",
                            "model": "gpt-5.6-luna",
                            "output": [
                                {
                                    "type": "message",
                                    "content": [{"type": "output_text", "text": "ok"}],
                                }
                            ],
                        },
                    },
                ]
            )

        async def aclose(self) -> None:
            return None

    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = build_model_for_preset(
        model_name="gpt-5.6-luna",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        api_mode="responses",
        reasoning_effort="max",
        default_request_kwargs={"include": ["file_search_call.results"]},
        max_attempts=1,
    )

    await _drain(model)

    assert captured["reasoning"] == {"effort": "max"}
    assert captured["include"] == [
        "file_search_call.results",
        "reasoning.encrypted_content",
    ]
    assert model.qitos_harness_metadata["reasoning"]["resolved"] == "max"


def test_older_openai_reasoning_policy_does_not_claim_gpt56_max() -> None:
    resolution = resolve_reasoning(
        family_id="openai",
        model_name="gpt-5.5",
        api_mode="responses",
        requested="max",
    )

    assert resolution.resolved.value == "xhigh"


@pytest.mark.asyncio
async def test_stream_preserves_reasoning_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        def __init__(self, **_: Any) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **_: Any) -> Any:
            return _AsyncStream(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    reasoning_content="check premise",
                                    tool_calls=None,
                                ),
                                finish_reason=None,
                            )
                        ],
                        usage=None,
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content="ok",
                                    reasoning_content=None,
                                    tool_calls=None,
                                ),
                                finish_reason="stop",
                            )
                        ],
                        usage=None,
                    ),
                ]
            )

        async def aclose(self) -> None:
            return None

    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = build_model_for_preset(
        family_id="kimi",
        model_name="kimi-k3",
        api_key="test-key",
        base_url="https://example.test/v1",
        max_attempts=1,
    )

    chunks = await _drain(model)

    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert "".join(chunk.reasoning_content or "" for chunk in chunks) == (
        "check premise"
    )
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "stop"
