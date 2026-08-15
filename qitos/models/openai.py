"""Async OpenAI Responses and Chat Completions providers."""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, cast

from ..core.errors import ModelContinuationRejected, ModelTransportError
from ..core.model_capabilities import (
    ModelAPI,
    ModelCapabilities,
    ReasoningCapability,
)
from ..core.model_request import ModelRequest
from ..core.model_stream import ModelStreamEventType
from ..core.multimodal import (
    content_to_text,
    ensure_data_url,
    file_to_data_url,
    has_nontext_content,
    normalize_content_block,
    normalize_messages,
)
from ._openai_responses import (
    _normalize_api_mode,
    _normalize_request_kwargs,
    _open_responses_stream,
    _to_responses_input,
)
from .transport import (
    ModelRetryPolicy,
    close_async_resource,
    effective_request_timeout,
    transactional_stream_with_retry,
)
from .base import (
    Model,
    ModelStreamEvent,
)

GLM_TOKENIZER_ENV_VARS = ("QITOS_GLM_TOKENIZER_PATH", "GLM_TOKENIZER_PATH")
_MANAGED_WEB_SEARCH_FALLBACK_OPTION = "_qitos_managed_web_search_tools"
_QWEN_PROVIDER_NAME = "qwen"
_OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _usage_payload(usage: Any) -> Optional[Dict[str, Any]]:
    """Normalize Chat Completions usage, including cache reporting."""

    if usage is None:
        return None
    prompt_tokens = _field(usage, "prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = _field(usage, "input_tokens")
    completion_tokens = _field(usage, "completion_tokens")
    if completion_tokens is None:
        completion_tokens = _field(usage, "output_tokens")
    total_tokens = _field(usage, "total_tokens")
    cached = _field(usage, "cached_tokens")
    if cached is None:
        cached = _field(_field(usage, "prompt_tokens_details"), "cached_tokens")
    if cached is None:
        cached = _field(_field(usage, "input_tokens_details"), "cached_tokens")
    if cached is None:
        cached = _field(usage, "prompt_cache_hit_tokens")
    if cached is None:
        cached = _field(usage, "cache_read_input_tokens")
    cache_write = _field(usage, "cache_write_tokens")
    if cache_write is None:
        cache_write = _field(usage, "cache_write_input_tokens")
    if cache_write is None:
        cache_write = _field(usage, "cache_creation_input_tokens")
    reasoning = _field(usage, "reasoning_tokens")
    if reasoning is None:
        reasoning = _field(
            _field(usage, "output_tokens_details"),
            "reasoning_tokens",
        )
    if all(
        value is None
        for value in (
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cached,
            cache_write,
            reasoning,
        )
    ):
        return None
    result = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached,
    }
    if cache_write is not None:
        result["cache_write_tokens"] = cache_write
    if reasoning is not None:
        result["reasoning_tokens"] = reasoning
    return result


class ChatStreamAccumulator:
    """Normalize one Chat Completions stream attempt."""

    def __init__(self, *, provider: str, model: str) -> None:
        self._provider = provider
        self._model = model
        self._tool_calls: List[Dict[str, Any]] = []
        self._usage: Optional[Dict[str, Any]] = None
        self._finish_reason: Optional[str] = None

    def consume(self, chunk: Any) -> List[ModelStreamEvent]:
        events: List[ModelStreamEvent] = []
        choices = list(_field(chunk, "choices", []) or [])
        if not choices:
            usage = _usage_payload(_field(chunk, "usage"))
            if usage:
                self._usage = usage
            return events

        choice = choices[0]
        delta = _field(choice, "delta", {})
        text = str(_field(delta, "content", "") or "")
        reasoning = _field(delta, "reasoning_content")
        if text:
            events.append(
                ModelStreamEvent(
                    type=ModelStreamEventType.TEXT_DELTA,
                    text=text,
                    event_type="text.delta",
                )
            )
        if reasoning:
            events.append(
                ModelStreamEvent(
                    type=ModelStreamEventType.REASONING_DELTA,
                    reasoning_content=str(reasoning),
                    event_type="reasoning.delta",
                )
            )
        events.extend(self._accumulate_tool_calls(_field(delta, "tool_calls")))

        finish_reason = _field(choice, "finish_reason")
        if finish_reason is not None:
            self._finish_reason = str(finish_reason)
        usage = _usage_payload(_field(chunk, "usage"))
        if usage is None:
            usage = _usage_payload(_field(choice, "usage"))
        if usage:
            self._usage = usage
        return events

    def complete(self) -> ModelStreamEvent:
        if self._finish_reason is None:
            raise ModelTransportError(
                "model stream ended before a terminal finish reason",
                attempts=1,
                retryable=True,
            )
        complete_tool_calls = [
            item
            for item in self._tool_calls
            if item.get("id") and item.get("function", {}).get("name")
        ]
        incomplete_tool_calls = [
            {
                "index": index,
                "call_id": item.get("id"),
                "name": item.get("function", {}).get("name"),
                "code": "tool_call_incomplete",
            }
            for index, item in enumerate(self._tool_calls)
            if item not in complete_tool_calls
        ]
        if self._finish_reason not in {"tool_calls", "function_call"}:
            incomplete_tool_calls.extend(
                {
                    "index": index,
                    "call_id": item.get("id"),
                    "name": item.get("function", {}).get("name"),
                    "code": "tool_call_unexpected_finish_reason",
                }
                for index, item in enumerate(complete_tool_calls)
            )
            complete_tool_calls = []
        return ModelStreamEvent(
            type=ModelStreamEventType.COMPLETED,
            usage=self._usage,
            tool_calls=complete_tool_calls or None,
            event_type="chat.completion.completed",
            event_metadata={
                "provider": self._provider,
                "model": self._model,
                "api_mode": "chat_completions",
                "invalid_tool_calls": incomplete_tool_calls,
            },
            finish_reason=self._finish_reason,
        )

    def _accumulate_tool_calls(self, deltas: Any) -> List[ModelStreamEvent]:
        events: List[ModelStreamEvent] = []
        for item in list(deltas or []):
            raw_index = _field(item, "index", len(self._tool_calls))
            index = raw_index if isinstance(raw_index, int) else len(self._tool_calls)
            while len(self._tool_calls) <= index:
                self._tool_calls.append(
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )
            tool_call = self._tool_calls[index]
            tool_call_id = _field(item, "id")
            if tool_call_id:
                tool_call["id"] = str(tool_call_id)
            tool_call_type = _field(item, "type")
            if tool_call_type:
                tool_call["type"] = str(tool_call_type)
            function = _field(item, "function")
            name = _field(function, "name")
            if name:
                tool_call["function"]["name"] = str(name)
            arguments = _field(function, "arguments")
            if arguments:
                previous = str(tool_call["function"].get("arguments") or "")
                tool_call["function"]["arguments"] = previous + str(arguments)
            metadata = {
                key: value
                for key, value in {
                    "index": index,
                    "call_id": tool_call_id,
                    "tool_type": tool_call_type,
                    "name": name,
                    "arguments_delta": arguments,
                }.items()
                if value is not None
            }
            if len(metadata) > 1:
                events.append(
                    ModelStreamEvent(
                        type=ModelStreamEventType.TOOL_CALL_DELTA,
                        event_type="tool_call.delta",
                        event_metadata=metadata,
                    )
                )
        return events


class ChatEventStream(AsyncIterator[ModelStreamEvent]):
    """Own one connected Chat Completions stream and its client."""

    def __init__(
        self, response: Any, client: Any, *, provider: str, model: str
    ) -> None:
        self._response = response
        self._client = client
        self._iterator = response.__aiter__()
        self._pending: Deque[ModelStreamEvent] = deque()
        self._accumulator = ChatStreamAccumulator(provider=provider, model=model)
        self._finished = False
        self._closed = False

    def __aiter__(self) -> ChatEventStream:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        if self._pending:
            return self._pending.popleft()
        if self._finished:
            raise StopAsyncIteration
        while True:
            try:
                chunk = await self._iterator.__anext__()
            except StopAsyncIteration:
                self._finished = True
                return self._accumulator.complete()
            self._pending.extend(self._accumulator.consume(chunk))
            if self._pending:
                return self._pending.popleft()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._finished = True
        await close_async_resource(self._response)
        await close_async_resource(self._client)


class _OwnedEventStream(AsyncIterator[ModelStreamEvent]):
    """Attach an SDK client lifecycle to a protocol event stream."""

    def __init__(self, stream: AsyncIterator[ModelStreamEvent], client: Any) -> None:
        self._stream = stream
        self._client = client
        self._closed = False

    def __aiter__(self) -> _OwnedEventStream:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        return await self._stream.__anext__()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await close_async_resource(self._stream)
        await close_async_resource(self._client)


def _wire_tool_schema(
    tool_schema_payload: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Return provider-valid function tools from registry metadata."""

    wire: List[Dict[str, Any]] = []
    for item in list(tool_schema_payload or []):
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        clean_function = {
            key: function[key]
            for key in ("name", "description", "parameters", "strict")
            if key in function and function[key] is not None
        }
        wire.append({"type": "function", "function": clean_function})
    return wire


def _prefer_hosted_web_search(
    tools: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]] | None]:
    """Replace an admitted managed search schema with the hosted tool."""

    hosted: List[Dict[str, Any]] = []
    replaced = False
    for tool in tools:
        function = tool.get("function")
        if (
            not replaced
            and tool.get("type") == "function"
            and isinstance(function, dict)
            and function.get("name") == "web_search"
        ):
            hosted.append({"type": "web_search"})
            replaced = True
        else:
            hosted.append(tool)
    return hosted, list(tools) if replaced else None


def _prefer_qwen_chat_web_search(
    tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Enable Qwen's native Chat search for an admitted managed search Tool."""

    remaining: List[Dict[str, Any]] = []
    replaced = False
    for tool in tools:
        function = tool.get("function")
        if (
            not replaced
            and tool.get("type") == "function"
            and isinstance(function, dict)
            and function.get("name") == "web_search"
        ):
            replaced = True
            continue
        remaining.append(tool)
    if not replaced:
        return {"tools": tools}
    options: Dict[str, Any] = {
        "extra_body": {"enable_search": True},
        _MANAGED_WEB_SEARCH_FALLBACK_OPTION: list(tools),
    }
    if remaining:
        options["tools"] = remaining
    return options


def _managed_chat_web_search_fallback(
    request_kwargs: Dict[str, Any],
    tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fallback = dict(request_kwargs)
    extra_body = fallback.get("extra_body")
    if isinstance(extra_body, dict):
        managed_extra_body = dict(extra_body)
        managed_extra_body.pop("enable_search", None)
        if managed_extra_body:
            fallback["extra_body"] = managed_extra_body
        else:
            fallback.pop("extra_body", None)
    fallback["tools"] = tools
    return fallback


def _provider_status_code(exc: Exception) -> int | None:
    status_code = _field(exc, "status_code")
    if status_code is None:
        status_code = _field(_field(exc, "response"), "status_code")
    return (
        status_code
        if isinstance(status_code, int) and not isinstance(status_code, bool)
        else None
    )


def _is_unsupported_hosted_web_search_error(exc: Exception) -> bool:
    """Recognize a provider's request-time rejection of hosted Web search."""

    if _provider_status_code(exc) not in {400, 404, 422}:
        return False
    detail = f"{exc} {_field(exc, 'body', '')}".casefold()
    names_web_search = any(
        name in detail for name in ("enable_search", "web_search", "web search")
    )
    rejects_capability = any(
        marker in detail
        for marker in (
            "invalid",
            "not available",
            "not supported",
            "unknown",
            "unrecognized",
            "unsupported",
        )
    )
    return names_web_search and rejects_capability


def _relocate_chat_template_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(kwargs)
    chat_template_kwargs = result.pop("chat_template_kwargs", None)
    if isinstance(chat_template_kwargs, dict) and chat_template_kwargs:
        extra_body = dict(result.pop("extra_body", None) or {})
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        result["extra_body"] = extra_body
    return result


def _merge_request_kwargs(
    defaults: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    merged = dict(defaults)
    for key, value in overrides.items():
        if key in {"extra_body", "reasoning"} and isinstance(value, dict):
            nested = dict(merged.get(key) or {})
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _is_forced_tool_choice(tool_choice: Any) -> bool:
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() == "required"
    return isinstance(tool_choice, dict)


def _disable_thinking_for_forced_tool_choice(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_forced_tool_choice(kwargs.get("tool_choice")):
        return kwargs
    result = dict(kwargs)
    disabled_thinking = False
    if "enable_thinking" in result:
        result["enable_thinking"] = False
        disabled_thinking = True
    if "thinking" in result:
        result["thinking"] = {"type": "disabled"}
        disabled_thinking = True
    extra_body = result.get("extra_body")
    if isinstance(extra_body, dict):
        patched_extra = dict(extra_body)
        if "enable_thinking" in patched_extra:
            patched_extra["enable_thinking"] = False
            disabled_thinking = True
        if "thinking" in patched_extra:
            patched_extra["thinking"] = {"type": "disabled"}
            disabled_thinking = True
        result["extra_body"] = patched_extra
    if disabled_thinking:
        result.pop("reasoning_effort", None)
    return result


def _is_unsupported_stream_options_error(exc: Exception) -> bool:
    status_code = _field(exc, "status_code")
    if status_code is None:
        status_code = _field(_field(exc, "response"), "status_code")
    if status_code not in {400, 422}:
        return False
    detail = f"{exc} {_field(exc, 'body', '')}".casefold()
    return "stream_options" in detail and any(
        marker in detail
        for marker in (
            "extra_forbidden",
            "invalid",
            "not support",
            "not permitted",
            "unknown",
            "unrecognized",
            "unsupported",
            "unexpected",
        )
    )


def _is_unsupported_prompt_cache_key_error(exc: Exception) -> bool:
    status_code = _provider_status_code(exc)
    if status_code not in {400, 404, 422, None}:
        return False
    detail = f"{exc} {_field(exc, 'body', '')}".casefold()
    if "prompt_cache_key" not in detail and "prompt cache key" not in detail:
        return False
    return any(
        marker in detail
        for marker in (
            "not supported",
            "unexpected keyword",
            "unknown parameter",
            "unrecognized",
            "unsupported",
        )
    )


def _to_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = normalize_messages(messages)
    out: List[Dict[str, Any]] = []
    for message in normalized:
        role = str(message.get("role") or "user").strip() or "user"
        content = message.get("content")
        payload: Dict[str, Any] = {"role": role}
        for key, value in message.items():
            if key in {"role", "content", "native_items"}:
                continue
            payload[key] = value
        if isinstance(content, list):
            if has_nontext_content(message):
                payload["content"] = _to_openai_content_blocks(content)
            else:
                text_blocks = [
                    str(normalize_content_block(block).get("text") or "")
                    for block in content
                    if str(normalize_content_block(block).get("type") or "text")
                    == "text"
                ]
                payload["content"] = "\n".join(part for part in text_blocks if part)
        elif content is None and role == "assistant" and payload.get("tool_calls"):
            payload["content"] = None
        else:
            payload["content"] = str(content or "")
        out.append(payload)
    return out


def _count_openai_request_tokens(
    adapter: Model,
    messages: List[Dict[str, Any]],
    request_options: Optional[Dict[str, Any]],
) -> Optional[int]:
    api_mode = str(getattr(adapter, "api_mode", "chat_completions"))
    wire_messages = (
        _to_responses_input(messages)
        if api_mode == "responses"
        else _to_openai_messages(messages)
    )
    message_payload: Any = (
        {"input": wire_messages} if api_mode == "responses" else wire_messages
    )
    message_tokens = adapter.count_tokens(message_payload)
    option_payload = dict(request_options or {})
    if api_mode == "responses":
        option_payload = _normalize_request_kwargs(option_payload)
    option_tokens = adapter.count_tokens(option_payload) if option_payload else 0
    if not isinstance(message_tokens, int) or not isinstance(option_tokens, int):
        return None
    return max(0, message_tokens) + max(0, option_tokens)


def _to_openai_content_blocks(content: List[Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for raw in content:
        block = normalize_content_block(raw)
        block_type = str(block.get("type") or "text")
        if block_type == "text":
            blocks.append({"type": "text", "text": str(block.get("text") or "")})
            continue
        detail = str(block.get("detail") or "").strip()
        if block_type == "image_url":
            image_url: Dict[str, Any] = {"url": str(block.get("url") or "")}
        elif block_type == "image_base64":
            mime_type = str(block.get("mime_type") or "image/png")
            image_url = {
                "url": ensure_data_url(
                    str(block.get("data") or ""), mime_type=mime_type
                )
            }
        elif block_type == "image_file":
            path = str(block.get("path") or "")
            mime_type = str(block.get("mime_type") or "")
            image_url = {"url": file_to_data_url(path, mime_type=mime_type or None)}
        else:
            blocks.append({"type": "text", "text": str(block)})
            continue
        if detail:
            image_url["detail"] = detail
        blocks.append({"type": "image_url", "image_url": image_url})
    return blocks


def _is_glm_model_name(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith("glm-") or normalized.startswith("zai-org/glm-")


def _glm_tokenizer_path() -> Optional[str]:
    for name in GLM_TOKENIZER_ENV_VARS:
        value = os.getenv(name, "").strip()
        if value and Path(value).exists():
            return value
    return None


@lru_cache(maxsize=4)
def _load_glm_tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
    )


def _tokenizer_count_result(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, list):
        return len(value)
    getter = getattr(value, "get", None)
    if callable(getter):
        ids = getter("input_ids")
        if isinstance(ids, list):
            return len(ids)
    return None


def _normalize_messages_for_tokenizer(payload: List[Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            messages.append({"role": "user", "content": str(item)})
            continue
        role = str(item.get("role") or "user").strip() or "user"
        content = content_to_text(item.get("content"))
        extras = {
            key: item.get(key)
            for key in ("tool_calls", "tool_call_id", "name")
            if item.get(key) not in (None, "", [])
        }
        if extras:
            content = (
                content + "\n" + json.dumps(extras, ensure_ascii=False, sort_keys=True)
            ).strip()
        messages.append({"role": role, "content": content})
    return messages


class OpenAICompatibleModel(Model):
    """OpenAI-compatible provider with an explicit wire protocol."""

    provider_name = "openai-compatible"
    responses_hosted_tools: tuple[str, ...] = ()

    def __init__(
        self,
        model: str = "default",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float | None = 0.7,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        context_window: Optional[int] = None,
        default_request_kwargs: Optional[Dict[str, Any]] = None,
        api_mode: str = "chat_completions",
        max_attempts: int = 2,
        stream_idle_timeout: float = 60.0,
        retry_window_seconds: float = 300.0,
        provider_name: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
            provider_name=provider_name,
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "dummy-key"
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        if not resolved_base_url:
            raise ValueError("OPENAI_BASE_URL must be configured")
        self.base_url: str = resolved_base_url
        if isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(stream_idle_timeout, bool) or stream_idle_timeout <= 0:
            raise ValueError("stream_idle_timeout must be positive")
        self.timeout = float(timeout)
        self.stream_idle_timeout = float(stream_idle_timeout)
        self.default_request_kwargs = dict(default_request_kwargs or {})
        self.api_mode = _normalize_api_mode(api_mode)
        self.retry_policy = ModelRetryPolicy(
            max_attempts=max_attempts,
            retry_window_seconds=retry_window_seconds,
        )

    def _request_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return _merge_request_kwargs(self.default_request_kwargs, kwargs)

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return tested Responses or compatibility-channel behavior."""

        if self.api_mode == "responses":
            return ModelCapabilities(
                api=ModelAPI.RESPONSES,
                native_tool_calls=True,
                reasoning=(
                    ReasoningCapability.SUMMARY,
                    ReasoningCapability.OPAQUE_REPLAY,
                ),
                opaque_replay=True,
                continuation=True,
                usage=True,
                prompt_cache_usage=True,
                multimodal_input=True,
                hosted_tools=self._hosted_tools(),
            )
        return ModelCapabilities(
            api=ModelAPI.CHAT_COMPLETIONS,
            native_tool_calls=True,
            reasoning=(ReasoningCapability.SUMMARY,),
            usage=True,
            prompt_cache_usage=True,
            multimodal_input=True,
            hosted_tools=self._hosted_tools(),
        )

    def _hosted_tools(self) -> tuple[str, ...]:
        if self.provider_name == _QWEN_PROVIDER_NAME:
            return ("web_search",)
        if self.api_mode == "responses":
            return self.responses_hosted_tools
        return ()

    def _attempt_request_kwargs(
        self,
        kwargs: Dict[str, Any],
        *,
        deadline_monotonic: float | None,
    ) -> Dict[str, Any]:
        attempt = dict(kwargs)
        requested_timeout = attempt.get("timeout", self.timeout)
        if isinstance(requested_timeout, bool) or not isinstance(
            requested_timeout, (int, float)
        ):
            raise ValueError("model request timeout must be numeric")
        attempt["timeout"] = effective_request_timeout(
            min(self.timeout, float(requested_timeout)),
            deadline_monotonic,
        )
        return attempt

    def _new_client(self, *, timeout: float) -> Any:
        import openai

        return openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=0,
        )

    def _chat_stream_request(
        self,
        model_request: ModelRequest,
        kwargs: Dict[str, Any],
        *,
        deadline_monotonic: float | None,
    ) -> Dict[str, Any]:
        create_kwargs = _disable_thinking_for_forced_tool_choice(
            _relocate_chat_template_kwargs(kwargs)
        )
        create_kwargs.setdefault("stream_options", {"include_usage": True})
        create_kwargs.setdefault(
            "prompt_cache_key",
            model_request.cache_affinity,
        )
        request: Dict[str, Any] = {
            **create_kwargs,
            "model": self.model,
            "messages": cast(
                Any,
                _to_openai_messages(model_request.message_dicts()),
            ),
            "stream": True,
        }
        request.setdefault("max_tokens", self.max_tokens)
        if self.temperature is not None:
            request.setdefault("temperature", self.temperature)
        return self._attempt_request_kwargs(
            request,
            deadline_monotonic=deadline_monotonic,
        )

    async def _open_chat_stream(
        self,
        request_kwargs: Dict[str, Any],
        managed_web_search_tools: List[Dict[str, Any]] | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        client = self._new_client(timeout=float(request_kwargs["timeout"]))
        candidate = dict(request_kwargs)
        stream_options_fallback = False
        cache_affinity_fallback = False
        managed_search_fallback = False
        try:
            while True:
                try:
                    response = await client.chat.completions.create(**candidate)
                    break
                except Exception as exc:
                    if (
                        not stream_options_fallback
                        and "stream_options" in candidate
                        and _is_unsupported_stream_options_error(exc)
                    ):
                        candidate.pop("stream_options", None)
                        stream_options_fallback = True
                        continue
                    if (
                        not cache_affinity_fallback
                        and "prompt_cache_key" in candidate
                        and _is_unsupported_prompt_cache_key_error(exc)
                    ):
                        candidate.pop("prompt_cache_key", None)
                        cache_affinity_fallback = True
                        continue
                    if (
                        not managed_search_fallback
                        and isinstance(managed_web_search_tools, list)
                        and _is_unsupported_hosted_web_search_error(exc)
                    ):
                        candidate = _managed_chat_web_search_fallback(
                            candidate,
                            managed_web_search_tools,
                        )
                        managed_search_fallback = True
                        continue
                    raise
        except (asyncio.CancelledError, Exception):
            await close_async_resource(client)
            raise
        return ChatEventStream(
            response,
            client,
            provider=self.provider_name,
            model=self.model,
        )

    async def _open_stream(
        self,
        request: ModelRequest,
        request_kwargs: Dict[str, Any],
    ) -> AsyncIterator[ModelStreamEvent]:
        deadline_monotonic = request.deadline_monotonic
        managed_web_search_tools = request_kwargs.pop(
            _MANAGED_WEB_SEARCH_FALLBACK_OPTION,
            None,
        )
        if self.api_mode == "chat_completions":
            chat_request = self._chat_stream_request(
                request,
                request_kwargs,
                deadline_monotonic=deadline_monotonic,
            )
            return await self._open_chat_stream(
                chat_request,
                managed_web_search_tools=(
                    managed_web_search_tools
                    if isinstance(managed_web_search_tools, list)
                    else None
                ),
            )

        client = self._new_client(
            timeout=effective_request_timeout(self.timeout, deadline_monotonic)
        )
        response_kwargs = self._attempt_request_kwargs(
            request_kwargs,
            deadline_monotonic=deadline_monotonic,
        )
        response_kwargs.setdefault("prompt_cache_key", request.cache_affinity)
        cache_affinity_fallback = False
        managed_search_fallback = False
        try:
            while True:
                try:
                    stream = await _open_responses_stream(
                        self,
                        client,
                        request,
                        provider=self.provider_name,
                        request_kwargs=dict(response_kwargs),
                    )
                    break
                except Exception as exc:
                    if (
                        not cache_affinity_fallback
                        and "prompt_cache_key" in response_kwargs
                        and _is_unsupported_prompt_cache_key_error(exc)
                    ):
                        response_kwargs.pop("prompt_cache_key", None)
                        cache_affinity_fallback = True
                        continue
                    if (
                        not managed_search_fallback
                        and isinstance(managed_web_search_tools, list)
                        and _is_unsupported_hosted_web_search_error(exc)
                    ):
                        response_kwargs["tools"] = managed_web_search_tools
                        managed_search_fallback = True
                        continue
                    raise
        except (asyncio.CancelledError, Exception):
            await close_async_resource(client)
            raise
        return _OwnedEventStream(stream, client)

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream one committed Responses or Chat transaction."""

        self.validate_request(request)
        effective_request = request
        continuation_fallback = False
        while True:
            request_kwargs = self._request_kwargs(effective_request.option_dict())

            async def create_stream() -> AsyncIterator[ModelStreamEvent]:
                return await self._open_stream(
                    effective_request,
                    dict(request_kwargs),
                )

            published_content = False
            try:
                async for chunk in transactional_stream_with_retry(
                    create_stream,
                    policy=self.retry_policy,
                    connection_timeout_seconds=self.timeout,
                    event_idle_timeout_seconds=self.stream_idle_timeout,
                    deadline_monotonic=effective_request.deadline_monotonic,
                    is_terminal=lambda item: item.is_final,
                ):
                    if (
                        chunk.text
                        or chunk.reasoning_content
                        or chunk.tool_calls
                        or chunk.native_items
                        or "function_call" in str(chunk.event_type or "")
                    ):
                        published_content = True
                    if (
                        chunk.type is ModelStreamEventType.COMPLETED
                        and continuation_fallback
                    ):
                        chunk.event_metadata["continuation_fallback"] = True
                    yield chunk
                return
            except ModelContinuationRejected:
                if (
                    published_content
                    or effective_request.continuation is None
                    or continuation_fallback
                ):
                    raise
                effective_request = effective_request.without_continuation()
                continuation_fallback = True

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
        wire = _wire_tool_schema(tool_schema_payload)
        if not wire:
            return {}
        if "web_search" not in self.capabilities.hosted_tools:
            return {"tools": wire}
        if self.api_mode == "chat_completions":
            return _prefer_qwen_chat_web_search(wire)
        hosted, fallback = _prefer_hosted_web_search(wire)
        options: Dict[str, Any] = {"tools": hosted}
        if fallback is not None:
            options[_MANAGED_WEB_SEARCH_FALLBACK_OPTION] = fallback
        return options

    def count_request_tokens(
        self,
        messages: List[Dict[str, Any]],
        request_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        return _count_openai_request_tokens(self, messages, request_options)

    def supports_multimodal_input(self) -> bool:
        return True

    def count_tokens(self, messages_or_text: Any) -> Optional[int]:
        if self._should_use_glm_tokenizer():
            value = self._count_tokens_with_glm_tokenizer(messages_or_text)
            if isinstance(value, int) and value >= 0:
                return value
        return super().count_tokens(messages_or_text)

    def _should_use_glm_tokenizer(self) -> bool:
        metadata = dict(getattr(self, "qitos_harness_metadata", {}) or {})
        if str(metadata.get("family_preset") or "").strip().lower() == "glm":
            return True
        return _is_glm_model_name(self.model)

    def _count_tokens_with_glm_tokenizer(self, payload: Any) -> Optional[int]:
        path = _glm_tokenizer_path()
        if not path:
            return None
        try:
            tokenizer = _load_glm_tokenizer(path)
            if isinstance(payload, list):
                encoded = tokenizer.apply_chat_template(
                    _normalize_messages_for_tokenizer(payload),
                    tokenize=True,
                    add_generation_prompt=False,
                )
            else:
                encoded = tokenizer.encode(
                    self._stringify_token_payload(payload),
                    add_special_tokens=False,
                )
            return _tokenizer_count_result(encoded)
        except Exception:
            return None


class OpenAIModel(OpenAICompatibleModel):
    """Official OpenAI provider; Responses is the default protocol."""

    provider_name = "openai"
    responses_hosted_tools = ("web_search",)

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        context_window: Optional[int] = None,
        default_request_kwargs: Optional[Dict[str, Any]] = None,
        api_mode: str = "responses",
        max_attempts: int = 2,
        stream_idle_timeout: float = 60.0,
        retry_window_seconds: float = 300.0,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY must be configured")
        super().__init__(
            model=model,
            api_key=resolved_api_key,
            base_url=base_url
            or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            context_window=context_window,
            default_request_kwargs=default_request_kwargs,
            api_mode=api_mode,
            max_attempts=max_attempts,
            stream_idle_timeout=stream_idle_timeout,
            retry_window_seconds=retry_window_seconds,
        )

    def _hosted_tools(self) -> tuple[str, ...]:
        if self.base_url.rstrip("/") != _OFFICIAL_OPENAI_BASE_URL:
            return ()
        return super()._hosted_tools()


class AzureOpenAIModel(OpenAICompatibleModel):
    """Azure OpenAI Chat Completions provider."""

    provider_name = "azure"

    def __init__(
        self,
        deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        system_prompt: Optional[str] = None,
        temperature: float | None = 0.7,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        context_window: Optional[int] = None,
        max_attempts: int = 2,
        stream_idle_timeout: float = 60.0,
    ) -> None:
        resolved_endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        resolved_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not resolved_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be configured")
        self.endpoint = resolved_endpoint
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or ""
        self.api_version = api_version or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
        )
        super().__init__(
            model=self.deployment or "azure",
            api_key=resolved_api_key,
            base_url=resolved_endpoint,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            context_window=context_window,
            api_mode="chat_completions",
            max_attempts=max_attempts,
            stream_idle_timeout=stream_idle_timeout,
        )

    def _new_client(self, *, timeout: float) -> Any:
        import openai

        return openai.AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            timeout=timeout,
            max_retries=0,
        )


__all__ = ["AzureOpenAIModel", "OpenAICompatibleModel", "OpenAIModel"]
