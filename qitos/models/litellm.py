"""Asynchronous LiteLLM adapter for its OpenAI-compatible stream."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional

from .transport import (
    ModelRetryPolicy,
    effective_request_timeout,
    transactional_stream_with_retry,
)
from .base import Model, ModelStreamChunk
from ..core.model_request import ModelRequest
from .openai import ChatEventStream, _to_openai_messages


class LiteLLMModel(Model):
    """Use LiteLLM's native async completion transport without SDK retries."""

    provider_name = "litellm"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float | None = 0.7,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        context_window: Optional[int] = None,
        max_attempts: int = 2,
        stream_idle_timeout: float = 60.0,
        retry_window_seconds: float = 300.0,
    ) -> None:
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
        )
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")
        self.api_version = api_version or os.getenv("LITELLM_API_VERSION")
        self.custom_llm_provider = custom_llm_provider or os.getenv("LITELLM_PROVIDER")
        if isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(stream_idle_timeout, bool) or stream_idle_timeout <= 0:
            raise ValueError("stream_idle_timeout must be positive")
        self.timeout = float(timeout)
        self.stream_idle_timeout = float(stream_idle_timeout)
        self.retry_policy = ModelRetryPolicy(
            max_attempts=max_attempts,
            retry_window_seconds=retry_window_seconds,
        )

    async def _open_stream(
        self,
        messages: List[Dict[str, Any]],
        request_kwargs: Dict[str, Any],
        *,
        deadline_monotonic: float | None,
    ) -> AsyncIterator[ModelStreamChunk]:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError(
                "LiteLLM support requires the qitos models extra"
            ) from exc

        payload: Dict[str, Any] = {
            **request_kwargs,
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "stream": True,
            "num_retries": 0,
        }
        payload.setdefault("max_tokens", self.max_tokens)
        payload.setdefault("stream_options", {"include_usage": True})
        payload.setdefault(
            "timeout",
            effective_request_timeout(self.timeout, deadline_monotonic),
        )
        if self.temperature is not None:
            payload.setdefault("temperature", self.temperature)
        if self.api_key:
            payload.setdefault("api_key", self.api_key)
        if self.api_base:
            payload.setdefault("api_base", self.api_base)
        if self.api_version:
            payload.setdefault("api_version", self.api_version)
        if self.custom_llm_provider:
            payload.setdefault("custom_llm_provider", self.custom_llm_provider)

        response = await litellm.acompletion(**payload)
        if not hasattr(response, "__aiter__"):
            raise TypeError("LiteLLM streaming response must be an async iterator")
        return ChatEventStream(
            response,
            None,
            provider=self.provider_name,
            model=self.model,
        )

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamChunk]:
        """Stream one committed LiteLLM transaction."""

        self.validate_request(request)
        request_kwargs = request.option_dict()

        async def create_stream() -> AsyncIterator[ModelStreamChunk]:
            return await self._open_stream(
                request.message_dicts(),
                dict(request_kwargs),
                deadline_monotonic=request.deadline_monotonic,
            )

        async for chunk in transactional_stream_with_retry(
            create_stream,
            policy=self.retry_policy,
            connection_timeout_seconds=self.timeout,
            event_idle_timeout_seconds=self.stream_idle_timeout,
            deadline_monotonic=request.deadline_monotonic,
            is_terminal=lambda item: item.done,
        ):
            yield chunk

    def supports_tool_schema_delivery(
        self, delivery: str, protocol: Any = None
    ) -> bool:
        _ = protocol
        return str(delivery or "prompt_injection") in {
            "prompt_injection",
            "api_parameter",
            "hybrid",
        }

    def build_tool_schema_request_options(
        self,
        tool_schema_payload: Optional[List[Dict[str, Any]]],
        *,
        protocol: Any = None,
        delivery: str = "prompt_injection",
    ) -> Dict[str, Any]:
        _ = protocol
        if str(delivery or "prompt_injection") not in {"api_parameter", "hybrid"}:
            return {}
        tools = list(tool_schema_payload or [])
        return {"tools": tools} if tools else {}


__all__ = ["LiteLLMModel"]
