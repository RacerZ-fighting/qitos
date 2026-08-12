"""Asynchronous local-model providers."""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from collections.abc import AsyncIterator
from typing import Any, Deque, Dict, List, Optional

from ..core.errors import ModelTransportError
from .transport import (
    ModelRetryPolicy,
    close_async_resource,
    effective_request_timeout,
    transactional_stream_with_retry,
)
from .base import Model, ModelStreamChunk


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _ollama_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Project QitOS history onto fields accepted by Ollama chat."""

    projected: List[Dict[str, Any]] = []
    for message in messages:
        payload: Dict[str, Any] = {
            "role": str(message.get("role") or "user"),
            "content": message.get("content") or "",
        }
        for key in ("images", "tool_calls"):
            if message.get(key) is not None:
                payload[key] = message[key]
        tool_name = message.get("tool_name") or message.get("name")
        if tool_name:
            payload["tool_name"] = str(tool_name)
        projected.append(payload)
    return projected


class _OllamaEventStream(AsyncIterator[ModelStreamChunk]):
    """Normalize and own one connected Ollama chat stream."""

    def __init__(self, responses: Any, client: Any, *, model: str) -> None:
        self._responses = responses
        self._client = client
        self._iterator = responses.__aiter__()
        self._model = model
        self._pending: Deque[ModelStreamChunk] = deque()
        self._tool_calls: List[Dict[str, Any]] = []
        self._finished = False
        self._closed = False

    def __aiter__(self) -> _OllamaEventStream:
        return self

    async def __anext__(self) -> ModelStreamChunk:
        while True:
            if self._pending:
                return self._pending.popleft()
            if self._finished:
                raise StopAsyncIteration

            try:
                response = await self._iterator.__anext__()
            except StopAsyncIteration as exc:
                raise ModelTransportError(
                    "Ollama stream ended before its done event",
                    attempts=1,
                    retryable=True,
                ) from exc

            message = _field(response, "message", {})
            text = str(_field(message, "content", "") or "")
            reasoning = str(_field(message, "thinking", "") or "")
            if text or reasoning:
                self._pending.append(
                    ModelStreamChunk(
                        text=text,
                        reasoning_content=reasoning or None,
                        event_type=("text.delta" if text else "reasoning.delta"),
                    )
                )
            for raw_call in list(_field(message, "tool_calls", []) or []):
                function = _field(raw_call, "function", {})
                name = str(_field(function, "name", "") or "").strip()
                if not name:
                    continue
                arguments = _field(function, "arguments", {}) or {}
                if not isinstance(arguments, dict):
                    arguments = {"input": arguments}
                call_id = str(_field(raw_call, "id", "") or "").strip()
                if not call_id:
                    call_id = f"ollama_call_{len(self._tool_calls) + 1}"
                self._tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                arguments,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                )
                self._pending.append(
                    ModelStreamChunk(
                        event_type="tool_call.done",
                        event_metadata={
                            "index": len(self._tool_calls) - 1,
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments,
                        },
                    )
                )

            if bool(_field(response, "done", False)):
                prompt_tokens = _field(response, "prompt_eval_count")
                completion_tokens = _field(response, "eval_count")
                usage = None
                if prompt_tokens is not None or completion_tokens is not None:
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": int(prompt_tokens or 0)
                        + int(completion_tokens or 0),
                    }
                self._finished = True
                self._pending.append(
                    ModelStreamChunk(
                        done=True,
                        usage=usage,
                        tool_calls=self._tool_calls or None,
                        event_type="ollama.chat.completed",
                        event_metadata={
                            "provider": "ollama",
                            "model": self._model,
                            "api_mode": "ollama_chat",
                        },
                        finish_reason=(
                            str(_field(response, "done_reason"))
                            if _field(response, "done_reason") is not None
                            else None
                        ),
                    )
                )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._finished = True
        await close_async_resource(self._responses)
        await close_async_resource(self._client)


class OllamaModel(Model):
    """Ollama Chat adapter using the official asynchronous client."""

    provider_name = "ollama"

    def __init__(
        self,
        model: str = "llama3",
        host: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float | None = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        format: str | Dict[str, Any] | None = None,
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
        self.host = (
            host
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or "http://localhost:11434"
        ).rstrip("/")
        if isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(stream_idle_timeout, bool) or stream_idle_timeout <= 0:
            raise ValueError("stream_idle_timeout must be positive")
        self.timeout = float(timeout)
        self.stream_idle_timeout = float(stream_idle_timeout)
        self.format = format
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
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                "Ollama support requires the qitos models extra"
            ) from exc

        timeout = effective_request_timeout(self.timeout, deadline_monotonic)
        client = ollama.AsyncClient(host=self.host, timeout=timeout)
        options = dict(request_kwargs.pop("options", {}) or {})
        if self.temperature is not None:
            options.setdefault("temperature", self.temperature)
        options.setdefault("num_predict", self.max_tokens)
        request_kwargs.pop("tool_choice", None)
        request_kwargs.pop("model", None)
        request_kwargs.pop("messages", None)
        request_kwargs.pop("stream", None)
        try:
            responses = await client.chat(
                model=self.model,
                messages=_ollama_messages(messages),
                stream=True,
                options=options,
                format=request_kwargs.pop("format", self.format),
                **request_kwargs,
            )
        except (asyncio.CancelledError, Exception):
            await close_async_resource(client)
            raise
        if not hasattr(responses, "__aiter__"):
            await close_async_resource(client)
            raise TypeError("Ollama streaming response must be an async iterator")
        return _OllamaEventStream(responses, client, model=self.model)

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        deadline_monotonic: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelStreamChunk]:
        """Stream one committed Ollama chat transaction."""

        request_kwargs = dict(kwargs)

        async def create_stream() -> AsyncIterator[ModelStreamChunk]:
            return await self._open_stream(
                messages,
                dict(request_kwargs),
                deadline_monotonic=deadline_monotonic,
            )

        async for chunk in transactional_stream_with_retry(
            create_stream,
            policy=self.retry_policy,
            connection_timeout_seconds=self.timeout,
            event_idle_timeout_seconds=self.stream_idle_timeout,
            deadline_monotonic=deadline_monotonic,
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


__all__ = ["OllamaModel"]
