"""Native asynchronous Anthropic Messages provider."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional, cast

from ..core.errors import ModelTransportError
from ..core.model_capabilities import (
    ModelAPI,
    ModelCapabilities,
    ReasoningCapability,
)
from ..core.model_request import ModelRequest
from ..core.model_stream import ModelStreamEventType
from ..core.thinking import THINKING_LEVEL_ORDER, thinking_request_options
from ..core.multimodal import content_to_text, normalize_content_block
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

_ANTHROPIC_BLOCK_TYPES = {
    "server_tool_use",
    "text",
    "thinking",
    "redacted_thinking",
    "tool_use",
    "tool_result",
    "web_search_tool_result",
}
_ANTHROPIC_HOSTED_WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
}
_MANAGED_WEB_SEARCH_FALLBACK_OPTION = "_qitos_managed_web_search_tools"
_OFFICIAL_ANTHROPIC_BASE_URL = "https://api.anthropic.com"


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _native_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _native_value(model_dump(exclude_none=True))
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
    raise TypeError(f"unsupported Anthropic value: {type(value).__name__}")


def _anthropic_tools(
    tool_schema_payload: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
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
        tool: Dict[str, Any] = {
            "name": name,
            "input_schema": dict(parameters),
        }
        description = function.get("description")
        if description:
            tool["description"] = str(description)
        tools.append(tool)
    return tools


def _prefer_hosted_web_search(
    tools: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]] | None]:
    """Replace one admitted managed search schema with Anthropic's server tool."""

    hosted: List[Dict[str, Any]] = []
    replaced = False
    for tool in tools:
        if not replaced and tool.get("name") == "web_search":
            hosted.append(dict(_ANTHROPIC_HOSTED_WEB_SEARCH_TOOL))
            replaced = True
        else:
            hosted.append(tool)
    return hosted, list(tools) if replaced else None


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
    """Recognize a request-time rejection before any stream output exists."""

    if _provider_status_code(exc) not in {400, 404, 422}:
        return False
    detail = f"{exc} {_field(exc, 'body', '')}".casefold()
    names_web_search = "web_search" in detail or "web search" in detail
    rejects_capability = any(
        marker in detail
        for marker in (
            "disabled",
            "invalid",
            "not available",
            "not enabled",
            "not supported",
            "unknown",
            "unrecognized",
            "unsupported",
        )
    )
    return names_web_search and rejects_capability


def _is_unsupported_cache_affinity_error(exc: Exception) -> bool:
    if _provider_status_code(exc) not in {400, 404, 422, None}:
        return False
    detail = f"{exc} {_field(exc, 'body', '')}".casefold()
    if "metadata" not in detail and "user_id" not in detail:
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


def _anthropic_tool_choice(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return {"type": "auto"}
        if normalized == "required":
            return {"type": "any"}
        if normalized == "none":
            return {"type": "none"}
        return value
    if not isinstance(value, dict):
        return value
    function = value.get("function")
    if value.get("type") == "function" and isinstance(function, dict):
        name = str(function.get("name") or "").strip()
        if name:
            return {"type": "tool", "name": name}
    return value


def _merge_anthropic_request_kwargs(
    defaults: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge request defaults without sharing nested provider options."""

    merged = dict(defaults)
    for key, value in overrides.items():
        if key in {"thinking", "output_config"} and isinstance(value, dict):
            current = merged.get(key)
            if (
                key == "thinking"
                and isinstance(current, dict)
                and value.get("type") is not None
                and value.get("type") != current.get("type")
            ):
                merged[key] = dict(value)
                continue
            nested = dict(merged.get(key) or {})
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


class _AnthropicEventStream(AsyncIterator[ModelStreamEvent]):
    """Normalize and own one connected Anthropic message stream."""

    def __init__(self, events: Any, client: Any, *, provider: str, model: str) -> None:
        self._events = events
        self._client = client
        self._iterator = events.__aiter__()
        self._provider = provider
        self._model = model
        self._blocks: Dict[int, Dict[str, Any]] = {}
        self._input_json: Dict[int, str] = {}
        self._stopped_blocks: set[int] = set()
        self._tool_errors: Dict[int, str] = {}
        self._usage: Dict[str, Any] = {}
        self._message_id: str | None = None
        self._finish_reason: str | None = None
        self._finished = False
        self._closed = False

    def __aiter__(self) -> _AnthropicEventStream:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        if self._finished:
            raise StopAsyncIteration
        while True:
            try:
                event = await self._iterator.__anext__()
            except StopAsyncIteration as exc:
                raise ModelTransportError(
                    "Anthropic stream ended before message_stop",
                    attempts=1,
                    retryable=True,
                ) from exc

            event_type = str(_field(event, "type", "") or "")
            raw_index = _field(event, "index", 0)
            index = raw_index if isinstance(raw_index, int) else 0

            if event_type == "message_start":
                message = _field(event, "message", {})
                message_id = _field(message, "id")
                if message_id is not None:
                    self._message_id = str(message_id)
                self._record_usage(_field(message, "usage"))
                continue

            if event_type == "content_block_start":
                block = _native_value(_field(event, "content_block", {}))
                if isinstance(block, dict):
                    self._blocks[index] = dict(block)
                    block_type = block.get("type")
                    if block_type in {"server_tool_use", "tool_use"}:
                        self._input_json[index] = ""
                    if block_type == "tool_use":
                        return ModelStreamEvent(
                            type=ModelStreamEventType.TOOL_CALL_DELTA,
                            event_type="tool_call.start",
                            event_metadata={
                                "index": index,
                                "call_id": block.get("id"),
                                "name": block.get("name"),
                            },
                        )
                    if block_type in {"server_tool_use", "web_search_tool_result"}:
                        return ModelStreamEvent(
                            type=ModelStreamEventType.LIFECYCLE,
                            event_type="native_item.start",
                            event_metadata={
                                "index": index,
                                "item_type": block_type,
                                "call_id": block.get("id")
                                or block.get("tool_use_id"),
                                "name": block.get("name"),
                            },
                        )
                continue

            if event_type == "content_block_delta":
                delta = _field(event, "delta", {})
                delta_type = str(_field(delta, "type", "") or "")
                block = self._blocks.setdefault(index, {})
                if delta_type == "text_delta":
                    text = str(_field(delta, "text", "") or "")
                    block["text"] = str(block.get("text") or "") + text
                    if text:
                        return ModelStreamEvent(
                            type=ModelStreamEventType.TEXT_DELTA,
                            text=text,
                            event_type="text.delta",
                            event_metadata={"index": index},
                        )
                    continue
                if delta_type == "thinking_delta":
                    thinking = str(_field(delta, "thinking", "") or "")
                    block["thinking"] = str(block.get("thinking") or "") + thinking
                    if thinking:
                        return ModelStreamEvent(
                            type=ModelStreamEventType.REASONING_DELTA,
                            reasoning_content=thinking,
                            event_type="reasoning.delta",
                            event_metadata={"index": index},
                        )
                    continue
                if delta_type == "signature_delta":
                    signature = str(_field(delta, "signature", "") or "")
                    block["signature"] = str(block.get("signature") or "") + signature
                    continue
                if delta_type == "citations_delta":
                    citation = _native_value(_field(delta, "citation", {}))
                    if isinstance(citation, dict):
                        citations = block.setdefault("citations", [])
                        if isinstance(citations, list):
                            citations.append(citation)
                    continue
                if delta_type == "input_json_delta":
                    arguments = str(_field(delta, "partial_json", "") or "")
                    self._input_json[index] = (
                        self._input_json.get(index, "") + arguments
                    )
                    if block.get("type") != "tool_use":
                        continue
                    return ModelStreamEvent(
                        type=ModelStreamEventType.TOOL_CALL_DELTA,
                        event_type="tool_call.delta",
                        event_metadata={
                            "index": index,
                            "call_id": block.get("id"),
                            "name": block.get("name"),
                            "arguments_delta": arguments,
                        },
                    )
                continue

            if event_type == "content_block_stop":
                self._stopped_blocks.add(index)
                self._finish_block(index, completed=True)
                continue

            if event_type == "message_delta":
                delta = _field(event, "delta", {})
                stop_reason = _field(delta, "stop_reason")
                if stop_reason is not None:
                    self._finish_reason = str(stop_reason)
                self._record_usage(_field(event, "usage"))
                continue

            if event_type == "message_stop":
                for block_index in list(self._blocks):
                    self._finish_block(
                        block_index,
                        completed=block_index in self._stopped_blocks,
                    )
                self._finished = True
                return self._terminal_chunk()

            if event_type == "error":
                self._finished = True
                return ModelStreamEvent(
                    type=ModelStreamEventType.FAILED,
                    event_type="message.error",
                    error=f"Anthropic stream failed: {_field(event, 'error')}",
                )

    def _finish_block(self, index: int, *, completed: bool) -> None:
        block = self._blocks.get(index)
        if not isinstance(block, dict):
            return
        block_type = block.get("type")
        if block_type not in {"server_tool_use", "tool_use"}:
            return
        is_client_tool = block_type == "tool_use"
        if index in self._tool_errors:
            return
        raw_arguments = self._input_json.get(index, "")
        if not completed:
            if is_client_tool:
                self._tool_errors[index] = "tool_call_not_completed"
            return
        if not raw_arguments:
            existing = block.setdefault("input", {})
            if is_client_tool and not isinstance(existing, dict):
                self._tool_errors[index] = "tool_call_arguments_invalid"
            return
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            if is_client_tool:
                self._tool_errors[index] = "tool_call_arguments_invalid"
        else:
            if not isinstance(parsed, dict):
                if is_client_tool:
                    self._tool_errors[index] = "tool_call_arguments_invalid"
            else:
                block["input"] = parsed

    def _record_usage(self, usage: Any) -> None:
        if usage is None:
            return
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            value = _field(usage, key)
            if isinstance(value, int) and not isinstance(value, bool):
                self._usage[key] = value

    def _normalized_usage(self) -> Optional[Dict[str, Any]]:
        if not self._usage:
            return None
        input_tokens = int(self._usage.get("input_tokens", 0))
        cache_creation = int(self._usage.get("cache_creation_input_tokens", 0))
        cache_read = int(self._usage.get("cache_read_input_tokens", 0))
        output_tokens = int(self._usage.get("output_tokens", 0))
        prompt_tokens = input_tokens + cache_creation + cache_read
        result = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        }
        if "cache_creation_input_tokens" in self._usage:
            result["cache_creation_input_tokens"] = cache_creation
        if "cache_read_input_tokens" in self._usage:
            result["cache_read_input_tokens"] = cache_read
        return result

    def _terminal_chunk(self) -> ModelStreamEvent:
        native_items: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        invalid_tool_calls: List[Dict[str, Any]] = []
        for index in sorted(self._blocks):
            block = self._blocks[index]
            if block.get("type") != "tool_use":
                native_items.append(block)
                continue
            call_id = str(block.get("id") or "").strip()
            name = str(block.get("name") or "").strip()
            protocol_error = self._tool_errors.get(index, "")
            if not protocol_error and self._finish_reason != "tool_use":
                protocol_error = "tool_call_unexpected_stop_reason"
            if not call_id or not name:
                protocol_error = protocol_error or "tool_call_invalid"
            if protocol_error:
                invalid_tool_calls.append(
                    {
                        "index": index,
                        "call_id": call_id or None,
                        "name": name or None,
                        "code": protocol_error,
                        "arguments_chars": len(self._input_json.get(index, "")),
                    }
                )
                continue
            native_items.append(block)
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            block.get("input", {}),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                }
            )
        return ModelStreamEvent(
            type=ModelStreamEventType.COMPLETED,
            usage=self._normalized_usage(),
            tool_calls=tool_calls or None,
            native_items=native_items or None,
            event_type="message.stop",
            event_metadata={
                "provider": self._provider,
                "model": self._model,
                "api_mode": "anthropic_messages",
                "id": self._message_id,
                "invalid_tool_calls": invalid_tool_calls,
            },
            finish_reason=self._finish_reason,
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._finished = True
        await close_async_resource(self._events)
        await close_async_resource(self._client)


class AnthropicModel(Model):
    """Anthropic Messages provider using the official asynchronous SDK."""

    provider_name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float | None = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        context_window: Optional[int] = None,
        default_request_kwargs: Optional[Dict[str, Any]] = None,
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
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be configured")
        self.base_url = (
            base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        ).rstrip("/")
        if isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(stream_idle_timeout, bool) or stream_idle_timeout <= 0:
            raise ValueError("stream_idle_timeout must be positive")
        self.timeout = float(timeout)
        self.stream_idle_timeout = float(stream_idle_timeout)
        self.default_request_kwargs = dict(default_request_kwargs or {})
        self.retry_policy = ModelRetryPolicy(
            max_attempts=max_attempts,
            retry_window_seconds=retry_window_seconds,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return behavior covered by the native Messages adapter contracts."""

        return ModelCapabilities(
            api=ModelAPI.ANTHROPIC_MESSAGES,
            native_tool_calls=True,
            reasoning=(ReasoningCapability.THINKING,),
            thinking_levels=THINKING_LEVEL_ORDER,
            opaque_replay=True,
            usage=True,
            prompt_cache_usage=True,
            multimodal_input=True,
            hosted_tools=("web_search",) if self._uses_official_api() else (),
        )

    def _uses_official_api(self) -> bool:
        return (
            self.provider_name == "anthropic"
            and self.base_url == _OFFICIAL_ANTHROPIC_BASE_URL
        )

    def _system_text(self, messages: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        if self.system_prompt:
            parts.append(str(self.system_prompt))
        for message in messages:
            if str(message.get("role") or "") != "system":
                continue
            content = content_to_text(message.get("content")).strip()
            if content:
                parts.append(content)
        return "\n\n".join(parts).strip()

    def _anthropic_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            if role == "system":
                continue
            if role == "tool":
                call_id = str(message.get("tool_call_id") or "").strip()
                if not call_id:
                    continue
                tool_result_block: Dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": content_to_text(message.get("content")),
                }
                if message.get("is_error"):
                    tool_result_block["is_error"] = True
                content: List[Dict[str, Any]] = [tool_result_block]
                self._append_message(converted, "user", content)
                continue

            mapped_role = "assistant" if role == "assistant" else "user"
            native = [
                dict(item)
                for item in list(message.get("native_items") or [])
                if isinstance(item, dict)
                and str(item.get("type") or "") in _ANTHROPIC_BLOCK_TYPES
                and str(item.get("type") or "") != "tool_result"
            ]
            blocks = native or self._content_blocks(message.get("content"))
            if role == "developer" and not native:
                blocks = [
                    {
                        "type": "text",
                        "text": (
                            "<runtime-context>\n"
                            f"{content_to_text(message.get('content'))}\n"
                            "</runtime-context>"
                        ),
                    }
                ]
            if mapped_role == "assistant":
                native_call_ids = {
                    str(item.get("id") or "")
                    for item in native
                    if item.get("type") == "tool_use"
                }
                for call in list(message.get("tool_calls") or []):
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function")
                    if not isinstance(function, dict):
                        continue
                    call_id = str(call.get("id") or "").strip()
                    name = str(function.get("name") or "").strip()
                    if not call_id or not name or call_id in native_call_ids:
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
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": arguments,
                        }
                    )
            if not blocks:
                blocks = [{"type": "text", "text": ""}]
            self._append_message(converted, mapped_role, blocks)
        return converted

    @staticmethod
    def _append_message(
        messages: List[Dict[str, Any]],
        role: str,
        content: List[Dict[str, Any]],
    ) -> None:
        if messages and messages[-1].get("role") == role:
            previous = messages[-1].get("content")
            if isinstance(previous, list):
                previous.extend(content)
                return
        messages.append({"role": role, "content": content})

    @staticmethod
    def _content_blocks(content: Any) -> List[Dict[str, Any]]:
        if not isinstance(content, list):
            return [{"type": "text", "text": content_to_text(content)}]
        blocks: List[Dict[str, Any]] = []
        for raw in content:
            block = normalize_content_block(raw)
            block_type = str(block.get("type") or "text")
            if block_type == "text":
                blocks.append({"type": "text", "text": str(block.get("text") or "")})
            elif block_type == "image_base64":
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": str(block.get("mime_type") or "image/png"),
                            "data": str(block.get("data") or ""),
                        },
                    }
                )
            else:
                blocks.append({"type": "text", "text": content_to_text([block])})
        return blocks

    async def _open_stream(
        self,
        messages: List[Dict[str, Any]],
        request_kwargs: Dict[str, Any],
        *,
        cache_affinity: str,
        deadline_monotonic: float | None,
    ) -> AsyncIterator[ModelStreamEvent]:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic support requires the qitos models extra"
            ) from exc

        managed_web_search_tools = request_kwargs.pop(
            _MANAGED_WEB_SEARCH_FALLBACK_OPTION,
            None,
        )
        timeout = effective_request_timeout(self.timeout, deadline_monotonic)
        client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=0,
        )
        payload: Dict[str, Any] = {
            **request_kwargs,
            "model": self.model,
            "messages": cast(Any, self._anthropic_messages(messages)),
            "stream": True,
            "timeout": timeout,
        }
        payload.setdefault("max_tokens", self.max_tokens)
        thinking = payload.get("thinking")
        thinking_enabled = isinstance(thinking, dict) and str(
            thinking.get("type") or ""
        ) not in {"", "disabled"}
        if thinking_enabled:
            # Anthropic thinking uses the provider's default sampling behavior.
            payload.pop("temperature", None)
        elif self.temperature is not None:
            payload.setdefault("temperature", self.temperature)
        system = self._system_text(messages)
        if system:
            payload.setdefault("system", system)
        if "tool_choice" in payload:
            payload["tool_choice"] = _anthropic_tool_choice(payload["tool_choice"])
        raw_metadata = payload.get("metadata")
        affinity_injected = raw_metadata is None or (
            isinstance(raw_metadata, dict) and "user_id" not in raw_metadata
        )
        if affinity_injected:
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            metadata["user_id"] = cache_affinity
            payload["metadata"] = metadata
        candidate = dict(payload)
        cache_affinity_fallback = False
        managed_search_fallback = False
        try:
            while True:
                try:
                    events = await client.messages.create(**candidate)
                    break
                except Exception as exc:
                    if (
                        affinity_injected
                        and not cache_affinity_fallback
                        and _is_unsupported_cache_affinity_error(exc)
                    ):
                        candidate_metadata = candidate.get("metadata")
                        if isinstance(candidate_metadata, dict):
                            fallback_metadata = dict(candidate_metadata)
                            fallback_metadata.pop("user_id", None)
                            if fallback_metadata:
                                candidate["metadata"] = fallback_metadata
                            else:
                                candidate.pop("metadata", None)
                        cache_affinity_fallback = True
                        continue
                    if (
                        not managed_search_fallback
                        and isinstance(managed_web_search_tools, list)
                        and _is_unsupported_hosted_web_search_error(exc)
                    ):
                        candidate["tools"] = managed_web_search_tools
                        managed_search_fallback = True
                        continue
                    raise
        except (asyncio.CancelledError, Exception):
            await close_async_resource(client)
            raise
        return _AnthropicEventStream(
            events,
            client,
            provider=self.provider_name,
            model=self.model,
        )

    def _typed_thinking_kwargs(self, request: ModelRequest) -> Dict[str, Any]:
        """Project a typed request thinking level onto the Messages wire.

        The typed ``ModelRequest.thinking_level`` is the only runtime
        mutation channel for reasoning: it overrides construction-time and
        per-request reasoning kwargs for exactly this transport's wire keys
        (``thinking`` and, on the Kimi Messages variant, ``output_config``).
        ``None`` leaves the configured defaults untouched. The Kimi variant
        is selected by provider identity, mirroring the harness policy that
        resolves the construction-time default for the same adapter.
        """

        if request.thinking_level is None:
            return {}
        wire_format = (
            "kimi_anthropic_thinking"
            if self.provider_name.strip().lower() == "kimi"
            else "anthropic_manual_thinking"
        )
        return thinking_request_options(
            request.thinking_level,
            wire_format=wire_format,
            api_mode="messages",
            max_output_tokens=self.max_tokens,
        )

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream one committed Anthropic Messages transaction."""

        self.validate_request(request)
        request_kwargs = _merge_anthropic_request_kwargs(
            self.default_request_kwargs,
            request.option_dict(),
        )
        thinking_kwargs = self._typed_thinking_kwargs(request)
        if thinking_kwargs:
            request_kwargs = _merge_anthropic_request_kwargs(
                request_kwargs,
                thinking_kwargs,
            )

        async def create_stream() -> AsyncIterator[ModelStreamEvent]:
            return await self._open_stream(
                request.message_dicts(),
                dict(request_kwargs),
                cache_affinity=request.cache_affinity,
                deadline_monotonic=request.deadline_monotonic,
            )

        async for chunk in transactional_stream_with_retry(
            create_stream,
            policy=self.retry_policy,
            connection_timeout_seconds=self.timeout,
            event_idle_timeout_seconds=self.stream_idle_timeout,
            deadline_monotonic=request.deadline_monotonic,
            is_terminal=lambda item: item.is_final,
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
        tools = _anthropic_tools(tool_schema_payload)
        if not tools:
            return {}
        if "web_search" not in self.capabilities.hosted_tools:
            return {"tools": tools}
        hosted, fallback = _prefer_hosted_web_search(tools)
        options: Dict[str, Any] = {"tools": hosted}
        if fallback is not None:
            options[_MANAGED_WEB_SEARCH_FALLBACK_OPTION] = fallback
        return options

    def supports_multimodal_input(self) -> bool:
        return True


__all__ = ["AnthropicModel"]
