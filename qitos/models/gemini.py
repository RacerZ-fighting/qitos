"""Native asynchronous Google Gemini provider."""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass
from typing import Any, Deque, Dict, List, Optional, cast

from ..core.errors import ModelTransportError
from ..core.multimodal import content_to_text
from .transport import (
    ModelRetryPolicy,
    close_async_resource,
    effective_request_timeout,
    transactional_stream_with_retry,
)
from .base import Model, ModelStreamChunk
from ..core.model_request import ModelRequest


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _native_value(value: Any) -> Any:
    """Convert one Google SDK value to JSON-compatible builtins."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _native_value(asdict(cast(Any, value)))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json", exclude_none=True)
        except TypeError:
            dumped = model_dump(exclude_none=True)
        return _native_value(dumped)
    if isinstance(value, dict):
        return {str(key): _native_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native_value(item) for item in value]
    values = getattr(value, "__dict__", None)
    if isinstance(values, dict):
        return {
            str(key): _native_value(item)
            for key, item in values.items()
            if not str(key).startswith("_") and item is not None
        }
    raise TypeError(f"unsupported Gemini value: {type(value).__name__}")


def _enum_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)


def _usage_payload(usage: Any) -> Optional[Dict[str, Any]]:
    if usage is None:
        return None
    prompt = _field(usage, "prompt_token_count")
    if prompt is None:
        prompt = _field(usage, "promptTokenCount")
    completion = _field(usage, "candidates_token_count")
    if completion is None:
        completion = _field(usage, "candidatesTokenCount")
    total = _field(usage, "total_token_count")
    if total is None:
        total = _field(usage, "totalTokenCount")
    thoughts = _field(usage, "thoughts_token_count")
    if thoughts is None:
        thoughts = _field(usage, "thoughtsTokenCount")
    cached = _field(usage, "cached_content_token_count")
    if cached is None:
        cached = _field(usage, "cachedContentTokenCount")
    if prompt is None and completion is None and total is None:
        return None
    result = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    if thoughts is not None:
        result["reasoning_tokens"] = thoughts
    if cached is not None:
        result["cached_tokens"] = cached
    return result


def _gemini_tools(
    tool_schema_payload: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    declarations: List[Dict[str, Any]] = []
    for item in list(tool_schema_payload or []):
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        parameters = function.get("parameters")
        if not name or not isinstance(parameters, dict):
            continue
        declaration: Dict[str, Any] = {
            "name": name,
            "parameters_json_schema": dict(parameters),
        }
        description = function.get("description")
        if description:
            declaration["description"] = str(description)
        declarations.append(declaration)
    return [{"function_declarations": declarations}] if declarations else []


def _gemini_tool_config(tool_choice: Any) -> Optional[Dict[str, Any]]:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        mode = {
            "auto": "AUTO",
            "required": "ANY",
            "none": "NONE",
        }.get(tool_choice.strip().lower())
        return {"function_calling_config": {"mode": mode}} if mode else None
    if not isinstance(tool_choice, dict):
        return None
    function = tool_choice.get("function")
    if tool_choice.get("type") != "function" or not isinstance(function, dict):
        return None
    name = str(function.get("name") or "").strip()
    if not name:
        return None
    return {
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": [name],
        }
    }


class _GeminiEventStream(AsyncIterator[ModelStreamChunk]):
    """Normalize and own one connected Gemini GenerateContent stream."""

    def __init__(self, responses: Any, client: Any, *, model: str) -> None:
        self._responses = responses
        self._client = client
        self._iterator = responses.__aiter__()
        self._model = model
        self._pending: Deque[ModelStreamChunk] = deque()
        self._native_items: List[Dict[str, Any]] = []
        self._tool_calls: List[Dict[str, Any]] = []
        self._usage: Optional[Dict[str, Any]] = None
        self._finish_reason: Optional[str] = None
        self._response_id: Optional[str] = None
        self._finished = False
        self._closed = False

    def __aiter__(self) -> _GeminiEventStream:
        return self

    async def __anext__(self) -> ModelStreamChunk:
        if self._pending:
            return self._pending.popleft()
        if self._finished:
            raise StopAsyncIteration

        while True:
            try:
                response = await self._iterator.__anext__()
            except StopAsyncIteration:
                self._finished = True
                return self._terminal_chunk()

            response_id = _field(response, "response_id")
            if response_id is None:
                response_id = _field(response, "responseId")
            if response_id:
                self._response_id = str(response_id)

            usage = _field(response, "usage_metadata")
            if usage is None:
                usage = _field(response, "usageMetadata")
            normalized_usage = _usage_payload(usage)
            if normalized_usage is not None:
                self._usage = normalized_usage

            candidates = list(_field(response, "candidates", []) or [])
            if not candidates:
                feedback = _field(response, "prompt_feedback")
                if feedback is None:
                    feedback = _field(response, "promptFeedback")
                if feedback is not None:
                    raise ModelTransportError(
                        f"Gemini rejected the prompt: {_native_value(feedback)}",
                        attempts=1,
                        retryable=False,
                    )
                continue

            candidate = candidates[0]
            finish_reason = _field(candidate, "finish_reason")
            if finish_reason is None:
                finish_reason = _field(candidate, "finishReason")
            if finish_reason is not None:
                self._finish_reason = _enum_value(finish_reason)

            content = _field(candidate, "content", {})
            for part in list(_field(content, "parts", []) or []):
                self._consume_part(part)
            if self._pending:
                return self._pending.popleft()

    def _consume_part(self, part: Any) -> None:
        native = _native_value(part)
        if isinstance(native, dict):
            self._native_items.append(dict(native))

        text = str(_field(part, "text", "") or "")
        is_thought = bool(_field(part, "thought", False))
        if text:
            self._pending.append(
                ModelStreamChunk(
                    text="" if is_thought else text,
                    reasoning_content=text if is_thought else None,
                    event_type=("reasoning.delta" if is_thought else "text.delta"),
                )
            )

        function_call = _field(part, "function_call")
        if function_call is None:
            function_call = _field(part, "functionCall")
        if function_call is None:
            return
        name = str(_field(function_call, "name", "") or "").strip()
        if not name:
            return
        arguments = _native_value(_field(function_call, "args", {}) or {})
        if not isinstance(arguments, dict):
            arguments = {"input": arguments}
        call_id = str(_field(function_call, "id", "") or "").strip()
        if not call_id:
            call_id = f"gemini_call_{len(self._tool_calls) + 1}"
        tool_call = {
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
        self._tool_calls.append(tool_call)
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

    def _terminal_chunk(self) -> ModelStreamChunk:
        if self._finish_reason is None and not self._native_items:
            raise ModelTransportError(
                "Gemini stream ended without a candidate",
                attempts=1,
                retryable=True,
            )
        return ModelStreamChunk(
            done=True,
            usage=self._usage,
            tool_calls=self._tool_calls or None,
            native_items=self._native_items or None,
            event_type="generate_content.completed",
            event_metadata={
                "provider": "gemini",
                "model": self._model,
                "api_mode": "gemini_generate_content",
                "response_id": self._response_id,
            },
            finish_reason=self._finish_reason,
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._finished = True
        await close_async_resource(self._responses)
        await close_async_resource(self._client)


class GeminiModel(Model):
    """Google Gemini GenerateContent adapter using the official async SDK."""

    provider_name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
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
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be configured")
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL")
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

    def _system_text(self, messages: List[Dict[str, Any]]) -> str:
        parts = [str(self.system_prompt)] if self.system_prompt else []
        parts.extend(
            content_to_text(message.get("content")).strip()
            for message in messages
            if str(message.get("role") or "") == "system"
            and content_to_text(message.get("content")).strip()
        )
        return "\n\n".join(parts).strip()

    def _gemini_contents(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        call_names: Dict[str, str] = {}
        for message in messages:
            for call in list(message.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if (
                    isinstance(function, dict)
                    and call.get("id")
                    and function.get("name")
                ):
                    call_names[str(call["id"])] = str(function["name"])

        contents: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            if role == "system":
                continue
            if role == "tool":
                call_id = str(message.get("tool_call_id") or "").strip()
                name = str(message.get("name") or call_names.get(call_id) or "").strip()
                if not name:
                    raise ValueError(
                        f"Gemini tool result {call_id!r} is missing its function name"
                    )
                response: Any = message.get("content")
                if not isinstance(response, dict):
                    response = {"result": content_to_text(response)}
                contents.append(
                    {
                        "role": "tool",
                        "parts": [
                            {
                                "function_response": {
                                    "id": call_id or None,
                                    "name": name,
                                    "response": response,
                                }
                            }
                        ],
                    }
                )
                continue

            mapped_role = "model" if role == "assistant" else "user"
            native_parts = [
                dict(item)
                for item in list(message.get("native_items") or [])
                if isinstance(item, dict)
                and any(
                    key in item
                    for key in (
                        "text",
                        "function_call",
                        "functionCall",
                        "inline_data",
                        "inlineData",
                        "file_data",
                        "fileData",
                        "thought_signature",
                        "thoughtSignature",
                    )
                )
                and "type" not in item
            ]
            parts = native_parts or [{"text": content_to_text(message.get("content"))}]
            if mapped_role == "model":
                native_names = {
                    str(
                        (
                            item.get("function_call") or item.get("functionCall") or {}
                        ).get("name")
                        or ""
                    )
                    for item in native_parts
                    if isinstance(
                        item.get("function_call") or item.get("functionCall"), dict
                    )
                }
                for call in list(message.get("tool_calls") or []):
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function")
                    if not isinstance(function, dict):
                        continue
                    name = str(function.get("name") or "").strip()
                    if not name or name in native_names:
                        continue
                    raw_arguments = function.get("arguments") or "{}"
                    try:
                        arguments = (
                            json.loads(raw_arguments)
                            if isinstance(raw_arguments, str)
                            else dict(raw_arguments)
                        )
                    except (TypeError, ValueError, json.JSONDecodeError):
                        arguments = {"_raw": str(raw_arguments)}
                    parts.append(
                        {
                            "function_call": {
                                "id": call.get("id"),
                                "name": name,
                                "args": arguments,
                            }
                        }
                    )
            contents.append({"role": mapped_role, "parts": parts})
        return contents

    def _generation_config(
        self,
        messages: List[Dict[str, Any]],
        request_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        config = dict(request_kwargs.pop("config", {}) or {})
        if self.temperature is not None:
            config.setdefault("temperature", self.temperature)
        config.setdefault("max_output_tokens", self.max_tokens)
        system = self._system_text(messages)
        if system:
            config.setdefault("system_instruction", system)
        tools = _gemini_tools(request_kwargs.pop("tools", None))
        if tools:
            config.setdefault("tools", tools)
            config.setdefault("automatic_function_calling", {"disable": True})
        tool_config = _gemini_tool_config(request_kwargs.pop("tool_choice", None))
        if tool_config:
            config.setdefault("tool_config", tool_config)
        config.update(request_kwargs)
        return config

    async def _open_stream(
        self,
        messages: List[Dict[str, Any]],
        request_kwargs: Dict[str, Any],
        *,
        deadline_monotonic: float | None,
    ) -> AsyncIterator[ModelStreamChunk]:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support requires the qitos models extra"
            ) from exc

        timeout = effective_request_timeout(self.timeout, deadline_monotonic)
        http_options: Dict[str, Any] = {
            "timeout": max(1, int(timeout * 1000)),
            "retry_options": {"attempts": 1},
        }
        if self.base_url:
            http_options["base_url"] = self.base_url
        owner = genai.Client(
            api_key=self.api_key,
            http_options=cast(Any, http_options),
        )
        client = owner.aio
        try:
            responses = await client.models.generate_content_stream(
                model=self.model,
                contents=cast(Any, self._gemini_contents(messages)),
                config=cast(Any, self._generation_config(messages, request_kwargs)),
            )
        except (asyncio.CancelledError, Exception):
            await close_async_resource(client)
            raise
        return _GeminiEventStream(responses, client, model=self.model)

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamChunk]:
        """Stream one committed Gemini GenerateContent transaction."""

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


__all__ = ["GeminiModel"]
