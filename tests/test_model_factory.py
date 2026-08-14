"""Tests for explicit, instance-scoped model construction."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from qitos.models import (
    Model,
    ModelFactory,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    OpenAICompatibleModel,
    OpenAIModel,
    builtin_model_factory,
)


class _TestModel(Model):
    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = request
        yield ModelStreamEvent(
            type=ModelStreamEventType.COMPLETED, finish_reason="stop"
        )


def test_factory_instances_are_isolated_and_duplicate_registration_fails() -> None:
    first = ModelFactory()
    second = ModelFactory()
    first.register("test", _TestModel)

    assert isinstance(first.create("TEST", model="one"), _TestModel)
    assert first.provider_names == ("test",)
    assert second.provider_names == ()
    with pytest.raises(ValueError, match="already registered"):
        first.register("test", _TestModel)
    with pytest.raises(ValueError, match="Unknown model provider"):
        second.create("test")


def test_factory_rejects_empty_names_and_non_model_results() -> None:
    factory = ModelFactory()
    with pytest.raises(ValueError, match="non-empty"):
        factory.register(" ", _TestModel)

    factory.register("bad", lambda **_: object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="expected Model"):
        factory.create("bad")


def test_builtin_factory_composes_canonical_provider_classes() -> None:
    factory = builtin_model_factory()

    assert factory.provider_names == (
        "anthropic",
        "azure",
        "gemini",
        "litellm",
        "lmstudio",
        "ollama",
        "openai",
        "openai-compatible",
        "vllm",
    )
    lmstudio = factory.create("lmstudio", model="local")
    vllm = factory.create("vllm", model="local")
    assert isinstance(lmstudio, OpenAICompatibleModel)
    assert isinstance(vllm, OpenAICompatibleModel)
    assert lmstudio.base_url == "http://localhost:1234/v1"
    assert vllm.base_url == "http://localhost:8000/v1"


def test_builtin_factory_preserves_provider_specific_defaults() -> None:
    factory = builtin_model_factory()

    openai = factory.create("openai", model="gpt-test", api_key="test-key")
    lmstudio = factory.create(
        "lmstudio",
        model="local",
        base_url="http://custom.test/v1",
    )

    assert isinstance(openai, OpenAIModel)
    assert openai.api_mode == "responses"
    assert type(lmstudio) is OpenAICompatibleModel
    assert lmstudio.base_url == "http://custom.test/v1"
    assert lmstudio.api_mode == "chat_completions"
