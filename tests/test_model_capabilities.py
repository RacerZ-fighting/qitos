from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError
import pytest

from qitos.cache import CachedModel, InMemoryCache
from qitos.models import (
    AnthropicModel,
    Model,
    ModelAPI,
    ModelCapabilities,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    OpenAICompatibleModel,
    OpenAIModel,
    ReasoningCapability,
)


class _LegacyModel(Model):
    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = request
        yield ModelStreamEvent(type=ModelStreamEventType.COMPLETED)


def test_unclassified_model_reports_conservative_capabilities() -> None:
    model = _LegacyModel(model="custom")

    assert model.capabilities == ModelCapabilities()
    assert model.capabilities.api is ModelAPI.LEGACY
    assert model.capabilities.native_tool_calls is False
    assert model.capabilities.continuation is False
    assert model.capabilities.hosted_tools == ()


def test_responses_capabilities_claim_validated_continuation() -> None:
    model = OpenAIModel(api_key="test-key", model="gpt-test")

    assert model.capabilities.api is ModelAPI.RESPONSES
    assert model.capabilities.native_tool_calls is True
    assert model.capabilities.reasoning == (
        ReasoningCapability.SUMMARY,
        ReasoningCapability.OPAQUE_REPLAY,
    )
    assert model.capabilities.opaque_replay is True
    assert model.capabilities.continuation is True
    assert model.capabilities.usage is True
    assert model.capabilities.prompt_cache_usage is True
    assert model.capabilities.hosted_tools == ()


def test_chat_compatibility_capabilities_stay_explicitly_narrower() -> None:
    model = OpenAICompatibleModel(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="chat-test",
    )

    assert model.capabilities.api is ModelAPI.CHAT_COMPLETIONS
    assert model.capabilities.reasoning == (ReasoningCapability.SUMMARY,)
    assert model.capabilities.opaque_replay is False
    assert model.capabilities.continuation is False


def test_anthropic_capabilities_describe_native_messages_contract() -> None:
    model = AnthropicModel(api_key="test-key", model="claude-test")

    assert model.capabilities.api is ModelAPI.ANTHROPIC_MESSAGES
    assert model.capabilities.native_tool_calls is True
    assert model.capabilities.reasoning == (ReasoningCapability.THINKING,)
    assert model.capabilities.opaque_replay is True
    assert model.capabilities.usage is True
    assert model.capabilities.prompt_cache_usage is True
    assert model.capabilities.multimodal_input is True


def test_model_capabilities_are_immutable_and_reject_duplicate_facts() -> None:
    capabilities = ModelCapabilities(api=ModelAPI.RESPONSES)

    with pytest.raises(FrozenInstanceError):
        capabilities.api = ModelAPI.LEGACY  # type: ignore[misc]
    with pytest.raises(ValueError, match="reasoning capabilities must be unique"):
        ModelCapabilities(
            reasoning=(
                ReasoningCapability.SUMMARY,
                ReasoningCapability.SUMMARY,
            )
        )
    with pytest.raises(ValueError, match="hosted_tools must be unique"):
        ModelCapabilities(hosted_tools=("web_search", "web_search"))


def test_cached_model_preserves_wrapped_transport_capabilities() -> None:
    wrapped = OpenAIModel(api_key="test-key", model="gpt-test")

    assert CachedModel(wrapped, InMemoryCache()).capabilities == wrapped.capabilities
