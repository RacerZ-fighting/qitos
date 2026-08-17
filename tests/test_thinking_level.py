"""Contracts for the typed provider-neutral thinking level (Pi parity)."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from qitos.core.model_capabilities import ModelCapabilities
from qitos.core.model_request import ModelRequest
from qitos.core.thinking import (
    THINKING_LEVEL_ORDER,
    ThinkingLevel,
    clamp_thinking_level,
    thinking_request_options,
)
from qitos.models import AnthropicModel, OpenAICompatibleModel, OpenAIModel


def test_thinking_level_values_round_trip_through_the_constructor() -> None:
    # Each level's wire value is part of the Model boundary contract; the
    # enum constructor must accept exactly that value.
    for name, value in (
        ("OFF", "off"),
        ("MINIMAL", "minimal"),
        ("LOW", "low"),
        ("MEDIUM", "medium"),
        ("HIGH", "high"),
        ("XHIGH", "xhigh"),
        ("MAX", "max"),
    ):
        level = ThinkingLevel[name]
        assert level.value == value
        assert ThinkingLevel(value) is level


def test_thinking_level_order_starts_at_off_and_rises_to_max() -> None:
    order = THINKING_LEVEL_ORDER
    assert order[0] is ThinkingLevel.OFF
    assert order[-1] is ThinkingLevel.MAX
    assert order.index(ThinkingLevel.LOW) < order.index(ThinkingLevel.MEDIUM)
    assert order.index(ThinkingLevel.HIGH) < order.index(ThinkingLevel.XHIGH)


@pytest.mark.parametrize(
    ("requested", "supported", "expected"),
    [
        # Exact hits pass through, including off.
        ("off", ("off", "low"), "off"),
        ("high", ("low", "high", "max"), "high"),
        # Below the minimum clamps up to the minimum.
        ("off", ("low", "high"), "low"),
        ("minimal", ("medium",), "medium"),
        # Between supported levels resolves upward first, not by distance.
        ("medium", ("low", "high"), "high"),
        ("low", ("off", "max"), "max"),
        # Above the maximum falls back down to the nearest supported level.
        ("max", ("low", "high"), "high"),
        ("xhigh", ("medium",), "medium"),
        # A single supported level absorbs every request.
        ("max", ("off",), "off"),
        ("off", ("max",), "max"),
    ],
)
def test_clamp_thinking_level_nearest_up_then_down(
    requested: str,
    supported: tuple[str, ...],
    expected: str,
) -> None:
    level = ThinkingLevel(requested)
    supported_levels = tuple(ThinkingLevel(item) for item in supported)

    assert clamp_thinking_level(level, supported_levels) is ThinkingLevel(expected)


def test_clamp_thinking_level_empty_support_and_absent_request_yield_none() -> None:
    assert clamp_thinking_level(ThinkingLevel.HIGH, ()) is None
    assert clamp_thinking_level(None, (ThinkingLevel.HIGH,)) is None
    assert clamp_thinking_level(None, ()) is None


def test_clamp_thinking_level_rejects_untyped_inputs() -> None:
    with pytest.raises(TypeError, match="ThinkingLevel"):
        clamp_thinking_level("high", (ThinkingLevel.HIGH,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ThinkingLevel"):
        clamp_thinking_level(ThinkingLevel.HIGH, ("high",))  # type: ignore[arg-type]


def _request(**kwargs: Any) -> ModelRequest:
    return ModelRequest(
        run_id="run-1",
        transaction_id="run-1:0",
        provider="openai",
        model="gpt-test",
        protocol="responses",
        messages=({"role": "user", "content": "hi"},),
        **kwargs,
    )


def test_model_request_typing_validation_and_durable_round_trip() -> None:
    request = _request(thinking_level=ThinkingLevel.XHIGH)

    snapshot = request.to_dict()
    assert snapshot["thinking_level"] == "xhigh"
    restored = ModelRequest.from_dict(snapshot)
    assert restored.thinking_level is ThinkingLevel.XHIGH
    assert restored.request_digest == request.request_digest

    assert _request().thinking_level is None
    assert ModelRequest.from_dict(_request().to_dict()).thinking_level is None
    assert _request(
        continuation=None, thinking_level=ThinkingLevel.LOW
    ).without_continuation().thinking_level is ThinkingLevel.LOW

    with pytest.raises(TypeError, match="ThinkingLevel"):
        _request(thinking_level="high")  # type: ignore[arg-type]

    tampered = _request().to_dict()
    tampered["thinking_level"] = "extreme"
    with pytest.raises(ValueError, match="thinking_level"):
        ModelRequest.from_dict(tampered)
    tampered = _request().to_dict()
    tampered["thinking_level"] = 3
    with pytest.raises(ValueError, match="thinking_level"):
        ModelRequest.from_dict(tampered)


def test_model_request_digest_tracks_the_thinking_level() -> None:
    base = _request()
    low = _request(thinking_level=ThinkingLevel.LOW)
    high = _request(thinking_level=ThinkingLevel.HIGH)

    assert base.request_digest != low.request_digest
    assert low.request_digest != high.request_digest
    assert low.request_digest == _request(
        thinking_level=ThinkingLevel.LOW
    ).request_digest


def test_model_capabilities_thinking_levels_validation() -> None:
    assert ModelCapabilities().thinking_levels == ()

    capabilities = ModelCapabilities(
        thinking_levels=(ThinkingLevel.LOW, ThinkingLevel.HIGH)
    )
    assert capabilities.thinking_levels == (ThinkingLevel.LOW, ThinkingLevel.HIGH)

    with pytest.raises(TypeError, match="ThinkingLevel"):
        ModelCapabilities(thinking_levels=("low",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        ModelCapabilities(
            thinking_levels=(ThinkingLevel.LOW, ThinkingLevel.LOW)
        )


def test_openai_adapters_declare_the_full_typed_thinking_range() -> None:
    chat = OpenAICompatibleModel(
        api_key="test-key", base_url="https://provider.example/v1", model="chat-test"
    )
    responses = OpenAIModel(api_key="test-key", model="gpt-test")
    anthropic = AnthropicModel(api_key="test-key", model="claude-test")

    for model in (chat, responses, anthropic):
        supported = model.capabilities.thinking_levels
        # Every requested level survives clamping against the declared set.
        for level in THINKING_LEVEL_ORDER:
            assert clamp_thinking_level(level, supported) is level


@pytest.mark.parametrize(
    ("level", "wire_format", "api_mode", "expected"),
    [
        ("high", "openai_effort", "responses", {"reasoning": {"effort": "high"}}),
        ("minimal", "openai_effort", "responses", {"reasoning": {"effort": "minimal"}}),
        ("off", "openai_effort", "responses", {"reasoning": {"effort": "none"}}),
        ("xhigh", "openai_effort", "chat_completions", {"reasoning_effort": "xhigh"}),
        ("off", "openai_effort", "chat_completions", {"reasoning_effort": "none"}),
        (
            "low",
            "glm_effort",
            "chat_completions",
            {
                "reasoning_effort": "low",
                "extra_body": {"thinking": {"type": "enabled"}},
            },
        ),
        (
            "off",
            "glm_effort",
            "chat_completions",
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        (
            "high",
            "enable_thinking",
            "chat_completions",
            {"extra_body": {"enable_thinking": True}},
        ),
        (
            "off",
            "enable_thinking",
            "chat_completions",
            {"extra_body": {"enable_thinking": False}},
        ),
        (
            "off",
            "thinking_object",
            "chat_completions",
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        ("off", "anthropic_manual_thinking", "messages", {"thinking": {"type": "disabled"}}),
        (
            "xhigh",
            "kimi_anthropic_thinking",
            "messages",
            {"thinking": {"type": "enabled"}, "output_config": {"effort": "xhigh"}},
        ),
        (
            "off",
            "kimi_anthropic_thinking",
            "messages",
            {"thinking": {"type": "disabled"}},
        ),
        ("high", "provider_default", "responses", {}),
    ],
)
def test_thinking_request_options_wire_encodings(
    level: str,
    wire_format: str,
    api_mode: str,
    expected: dict[str, Any],
) -> None:
    assert (
        thinking_request_options(
            ThinkingLevel(level), wire_format=wire_format, api_mode=api_mode
        )
        == expected
    )


@pytest.mark.parametrize(
    ("level", "budget"),
    [
        ("minimal", 1_024),
        ("low", 1_024),
        ("medium", 2_048),
        ("high", 4_096),
        ("xhigh", 8_192),
        ("max", 16_384),
    ],
)
def test_anthropic_manual_thinking_budgets(level: str, budget: int) -> None:
    options = thinking_request_options(
        ThinkingLevel(level),
        wire_format="anthropic_manual_thinking",
        api_mode="messages",
    )
    assert options == {"thinking": {"type": "enabled", "budget_tokens": budget}}


def test_anthropic_manual_thinking_reserves_visible_output_room() -> None:
    options = thinking_request_options(
        ThinkingLevel.MAX,
        wire_format="anthropic_manual_thinking",
        api_mode="messages",
        max_output_tokens=8_192,
    )
    assert options["thinking"]["budget_tokens"] == 8_192 - 2_048

    with pytest.raises(ValueError, match="max_output_tokens"):
        thinking_request_options(
            ThinkingLevel.LOW,
            wire_format="anthropic_manual_thinking",
            api_mode="messages",
            max_output_tokens=1_024,
        )


class _AsyncStream(AsyncIterator[Any]):
    def __init__(self, items: list[Any]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> "_AsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        return None


def _typed_request(model: Any, level: ThinkingLevel | None) -> ModelRequest:
    return ModelRequest(
        run_id="thinking-test",
        transaction_id="thinking-test:0",
        provider=model.provider_name,
        model=model.model,
        protocol=model.capabilities.api.value,
        messages=({"role": "user", "content": "answer"},),
        thinking_level=level,
    )


def _chat_client(captured: dict[str, Any]) -> type:
    class Client:
        def __init__(self, **_: Any) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kwargs: Any) -> Any:
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

    return Client


def _responses_client(captured: dict[str, Any]) -> type:
    class Client:
        def __init__(self, **_: Any) -> None:
            self.responses = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return _AsyncStream(
                [
                    {"type": "response.output_text.delta", "delta": "ok"},
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "response-1",
                            "status": "completed",
                            "model": "gpt-test",
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

    return Client


def _anthropic_client(captured: dict[str, Any]) -> type:
    class Client:
        def __init__(self, **_: Any) -> None:
            self.messages = SimpleNamespace(create=self.create)

        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return _AsyncStream(
                [
                    {"type": "message_start", "message": {"id": "msg_1"}},
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "ok"},
                    },
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 2},
                    },
                    {"type": "message_stop"},
                ]
            )

        async def aclose(self) -> None:
            return None

    return Client


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (ThinkingLevel.LOW, "low"),
        (ThinkingLevel.MINIMAL, "minimal"),
        (ThinkingLevel.MAX, "max"),
        (ThinkingLevel.OFF, "none"),
    ],
)
@pytest.mark.asyncio
async def test_chat_completions_typed_level_overrides_default_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    level: ThinkingLevel,
    expected: str,
) -> None:
    captured: dict[str, Any] = {}
    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = _chat_client(captured)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = OpenAICompatibleModel(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="chat-test",
        default_request_kwargs={"reasoning_effort": "high"},
        max_attempts=1,
    )

    chunks = [chunk async for chunk in model.stream(_typed_request(model, level))]

    assert "".join(chunk.text for chunk in chunks) == "ok"
    assert captured["reasoning_effort"] == expected


@pytest.mark.asyncio
async def test_chat_completions_unset_level_keeps_default_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = _chat_client(captured)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = OpenAICompatibleModel(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="chat-test",
        default_request_kwargs={"reasoning_effort": "high"},
        max_attempts=1,
    )

    await _collect(model, None)

    assert captured["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_responses_typed_level_overrides_only_the_effort_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = _responses_client(captured)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = OpenAIModel(
        api_key="test-key",
        model="gpt-test",
        default_request_kwargs={
            "reasoning": {"effort": "low", "summary": "detailed"},
        },
        max_attempts=1,
    )

    await _collect(model, ThinkingLevel.XHIGH)

    # Nested merge: the typed field owns exactly the effort wire key.
    assert captured["reasoning"] == {"effort": "xhigh", "summary": "detailed"}


@pytest.mark.asyncio
async def test_responses_typed_off_disables_reasoning_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = _responses_client(captured)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = OpenAIModel(
        api_key="test-key",
        model="gpt-test",
        default_request_kwargs={"reasoning": {"effort": "high"}},
        max_attempts=1,
    )

    await _collect(model, ThinkingLevel.OFF)

    assert captured["reasoning"] == {"effort": "none"}


@pytest.mark.asyncio
async def test_responses_unset_level_keeps_default_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fake_openai = ModuleType("openai")
    fake_openai.AsyncOpenAI = _responses_client(captured)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    model = OpenAIModel(
        api_key="test-key",
        model="gpt-test",
        default_request_kwargs={"reasoning": {"effort": "high"}},
        max_attempts=1,
    )

    await _collect(model, None)

    assert captured["reasoning"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_anthropic_typed_level_overrides_default_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fake_anthropic = ModuleType("anthropic")
    fake_anthropic.AsyncAnthropic = _anthropic_client(captured)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    model = AnthropicModel(
        api_key="test-key",
        model="claude-test",
        max_tokens=8_192,
        default_request_kwargs={"thinking": {"type": "enabled", "budget_tokens": 1_024}},
        max_attempts=1,
    )

    await _collect(model, ThinkingLevel.MAX)

    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 8_192 - 2_048}


@pytest.mark.asyncio
async def test_anthropic_typed_off_disables_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fake_anthropic = ModuleType("anthropic")
    fake_anthropic.AsyncAnthropic = _anthropic_client(captured)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    model = AnthropicModel(
        api_key="test-key",
        model="claude-test",
        max_tokens=8_192,
        default_request_kwargs={"thinking": {"type": "enabled", "budget_tokens": 4_096}},
        max_attempts=1,
    )

    await _collect(model, ThinkingLevel.OFF)

    assert captured["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_anthropic_unset_level_keeps_default_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fake_anthropic = ModuleType("anthropic")
    fake_anthropic.AsyncAnthropic = _anthropic_client(captured)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    model = AnthropicModel(
        api_key="test-key",
        model="claude-test",
        max_tokens=8_192,
        default_request_kwargs={"thinking": {"type": "enabled", "budget_tokens": 2_048}},
        max_attempts=1,
    )

    await _collect(model, None)

    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 2_048}


@pytest.mark.asyncio
async def test_kimi_messages_transport_uses_the_output_config_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    fake_anthropic = ModuleType("anthropic")
    fake_anthropic.AsyncAnthropic = _anthropic_client(captured)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    model = AnthropicModel(
        api_key="test-key",
        base_url="https://api.moonshot.example/anthropic",
        model="kimi-k3",
        provider_name="kimi",
        default_request_kwargs={
            "thinking": {"type": "enabled"},
            "output_config": {"effort": "low"},
        },
        max_attempts=1,
    )

    await _collect(model, ThinkingLevel.MAX)

    assert captured["thinking"] == {"type": "enabled"}
    assert captured["output_config"] == {"effort": "max"}


async def _collect(model: Any, level: ThinkingLevel | None) -> list[Any]:
    return [chunk async for chunk in model.stream(_typed_request(model, level))]
