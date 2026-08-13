from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from qitos.harness import build_model_for_preset, resolve_reasoning
from qitos.models import AnthropicModel, ModelRequest


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
    request = ModelRequest(
        run_id="reasoning-test",
        transaction_id="reasoning-test:0",
        provider=model.provider_name,
        model=model.model,
        protocol=model.capabilities.api.value,
        messages=({"role": "user", "content": "answer"},),
    )
    return [
        chunk async for chunk in model.stream(request)
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


@pytest.mark.parametrize(
    ("requested", "budget"),
    [
        ("low", 1_024),
        ("medium", 2_048),
        ("high", 4_096),
        ("xhigh", 6_144),
        ("max", 6_144),
    ],
)
def test_anthropic_45_reasoning_effort_maps_to_manual_thinking_budget(
    requested: str,
    budget: int,
) -> None:
    resolution = resolve_reasoning(
        family_id="anthropic",
        model_name="claude-sonnet-4-5",
        api_mode="messages",
        requested=requested,
        max_output_tokens=8_192,
    )

    assert resolution.resolved.value == requested
    assert resolution.request_options == {
        "thinking": {"type": "enabled", "budget_tokens": budget}
    }
    assert resolution.effective_budget_tokens == budget


def test_anthropic_manual_thinking_preserves_visible_output_room() -> None:
    with pytest.raises(
        ValueError,
        match="Anthropic manual thinking requires max_output_tokens >= 2048",
    ):
        resolve_reasoning(
            family_id="anthropic",
            model_name="claude-sonnet-4-5",
            api_mode="messages",
            requested="low",
            max_output_tokens=2_047,
        )


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
def test_kimi_reasoning_maps_to_anthropic_compatible_fields(
    requested: str,
    sent: str,
) -> None:
    resolution = resolve_reasoning(
        family_id="kimi",
        model_name="kimi-k3",
        api_mode="messages",
        requested=requested,
    )

    assert resolution.resolved.value == sent
    assert resolution.request_options == {
        "thinking": {"type": "enabled"},
        "output_config": {"effort": sent},
    }
    assert resolution.effective_budget_tokens is None


def test_kimi_preset_can_use_anthropic_compatible_transport() -> None:
    model = build_model_for_preset(
        family_id="kimi",
        model_name="kimi-k3",
        api_key="test-key",
        base_url="https://example.test",
        api_mode="messages",
        adapter_kind="anthropic",
        reasoning_effort="medium",
        max_attempts=1,
    )

    assert isinstance(model, AnthropicModel)
    assert model.provider_name == "kimi"
    assert model.default_request_kwargs == {
        "thinking": {"type": "enabled"},
        "output_config": {"effort": "high"},
    }
    assert model.qitos_harness_metadata["family_preset"] == "kimi"
    assert model.qitos_harness_metadata["adapter_kind"] == "anthropic"
    assert model.qitos_harness_metadata["resolution_source"] == "explicit_adapter"
    assert model.qitos_harness_metadata["native_tool_call_preferred"] is True


def test_anthropic_preset_builds_native_messages_model() -> None:
    model = build_model_for_preset(
        family_id="anthropic",
        model_name="claude-sonnet-4-5",
        api_key="test-key",
        reasoning_effort="medium",
        max_tokens=8_192,
        max_attempts=1,
        default_request_kwargs={"thinking": {"type": "disabled", "budget_tokens": 999}},
    )

    assert isinstance(model, AnthropicModel)
    assert model.default_request_kwargs == {
        "thinking": {"type": "enabled", "budget_tokens": 2_048}
    }
    assert model.qitos_harness_metadata["adapter_kind"] == "anthropic"
    assert model.qitos_harness_metadata["reasoning"]["resolved"] == "medium"
    assert model.qitos_harness_metadata["reasoning"]["effective_budget_tokens"] == 2_048
    assert model.qitos_harness_metadata["native_tool_call_preferred"] is True
    assert model.qitos_harness_metadata["effective_tool_delivery"] == "api_parameter"
    assert model.qitos_protocol.tool_schema_delivery == "api_parameter"
    assert model.build_tool_schema_request_options(
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read one file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        delivery="api_parameter",
    ) == {
        "tools": [
            {
                "name": "read_file",
                "description": "Read one file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]
    }


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
