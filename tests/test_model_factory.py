"""Tests for explicit, instance-scoped model construction."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from qitos.config import ModelConfig, build_model
from qitos.models import (
    Model,
    ModelFactory,
    ModelStreamChunk,
    OpenAICompatibleModel,
    OpenAIModel,
    builtin_model_factory,
)


class _TestModel(Model):
    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        deadline_monotonic: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelStreamChunk]:
        _ = messages, deadline_monotonic, kwargs
        yield ModelStreamChunk(done=True, finish_reason="stop")


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


def test_build_model_keeps_openai_responses_as_provider_default() -> None:
    model = build_model(
        ModelConfig(
            provider="openai",
            model="gpt-test",
            api_key="test-key",
        )
    )

    assert isinstance(model, OpenAIModel)
    assert model.api_mode == "responses"


def test_build_model_maps_local_alias_to_canonical_openai_compatible() -> None:
    model = build_model(
        ModelConfig(
            provider="lmstudio",
            model="local",
            base_url="http://custom.test/v1",
        )
    )

    assert type(model) is OpenAICompatibleModel
    assert model.base_url == "http://custom.test/v1"
    assert model.api_mode == "chat_completions"


def test_environment_resolution_belongs_to_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "QITOS_MODEL_PROVIDER",
        "MODEL_PROVIDER",
        "OPENAI_API_KEY",
        "QITOS_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LITELLM_MODEL",
        "OLLAMA_HOST",
        "OLLAMA_BASE_URL",
        "LM_STUDIO_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    assert ModelConfig.from_env() is None

    monkeypatch.setenv("QITOS_MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("QITOS_MODEL", "gemini-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    config = ModelConfig.from_env()

    assert config is not None
    assert config.provider == "gemini"
    assert config.model == "gemini-test"
    assert config.api_key == "google-key"
    assert config.to_dict()["api_key"] == "***REDACTED***"

    monkeypatch.delenv("QITOS_MODEL_PROVIDER")
    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.setenv("QITOS_API_KEY", "compatible-key")
    config = ModelConfig.from_env()

    assert config is not None
    assert config.provider == "openai"
    assert config.api_key == "compatible-key"


def test_build_model_accepts_an_explicit_isolated_factory() -> None:
    factory = ModelFactory({"custom": _TestModel})

    model = build_model(
        ModelConfig(provider="custom", model="custom-model"),
        factory=factory,
    )

    assert isinstance(model, _TestModel)
    assert model.model == "custom-model"
