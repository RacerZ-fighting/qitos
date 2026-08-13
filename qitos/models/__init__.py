"""Canonical QitOS model providers and explicit factory composition."""

from __future__ import annotations

import os
from typing import Any

from ..core.model_capabilities import (
    ModelAPI,
    ModelCapabilities,
    ReasoningCapability,
)
from ..core.model_request import ModelContinuation, ModelRequest
from .anthropic import AnthropicModel
from .base import (
    Model,
    ModelFactory,
    ModelStreamChunk,
)
from .context_registry import infer_context_window
from .gemini import GeminiModel
from .litellm import LiteLLMModel
from .local import OllamaModel
from .openai import AzureOpenAIModel, OpenAICompatibleModel, OpenAIModel
from .profile_registry import (
    ModelProfile,
    infer_default_protocol,
    infer_model_profile,
    known_model_profiles,
)


def _local_openai_model(
    *,
    default_base_url: str,
    default_api_key: str,
    **kwargs: Any,
) -> Model:
    params = dict(kwargs)
    params.setdefault("base_url", default_base_url)
    params.setdefault("api_key", default_api_key)
    params.setdefault("api_mode", "chat_completions")
    return OpenAICompatibleModel(**params)


def builtin_model_factory() -> ModelFactory:
    """Build an isolated factory containing QitOS's shipped providers."""

    factory = ModelFactory()
    factory.register("openai", OpenAIModel)
    factory.register("openai-compatible", OpenAICompatibleModel)
    factory.register("azure", AzureOpenAIModel)
    factory.register("anthropic", AnthropicModel)
    factory.register("gemini", GeminiModel)
    factory.register("litellm", LiteLLMModel)
    factory.register("ollama", OllamaModel)
    factory.register(
        "lmstudio",
        lambda **kwargs: _local_openai_model(
            default_base_url=os.getenv(
                "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
            ),
            default_api_key="lm-studio",
            **kwargs,
        ),
    )
    factory.register(
        "vllm",
        lambda **kwargs: _local_openai_model(
            default_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            default_api_key="vllm",
            **kwargs,
        ),
    )
    return factory


__all__ = [
    "AnthropicModel",
    "AzureOpenAIModel",
    "GeminiModel",
    "LiteLLMModel",
    "Model",
    "ModelAPI",
    "ModelCapabilities",
    "ModelContinuation",
    "ModelFactory",
    "ModelProfile",
    "ModelRequest",
    "ModelStreamChunk",
    "ReasoningCapability",
    "OllamaModel",
    "OpenAICompatibleModel",
    "OpenAIModel",
    "builtin_model_factory",
    "infer_context_window",
    "infer_default_protocol",
    "infer_model_profile",
    "known_model_profiles",
]
