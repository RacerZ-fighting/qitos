"""
OpenAI Model Implementation

OpenAI API-based model calling implementation.
Supports environment variable configuration: OPENAI_API_KEY, OPENAI_BASE_URL
"""

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, cast

from ..core.model_response import ModelResponse
from ..core.multimodal import (
    content_to_text,
    ensure_data_url,
    file_to_data_url,
    has_nontext_content,
    normalize_content_block,
    normalize_messages,
)
from ._openai_responses import (
    _async_responses_completion,
    _async_responses_stream,
    _normalize_api_mode,
    _responses_completion,
    _responses_stream,
    _normalize_request_kwargs,
    _to_responses_input,
)
from ._openai_retry import (
    ModelRetryPolicy,
    async_run_with_retry,
    run_with_retry,
    sync_stream_with_retry,
    sync_transactional_stream_with_retry,
    stream_with_retry,
    transactional_stream_with_retry,
)
from .base import Model, ModelStreamChunk


GLM_TOKENIZER_ENV_VARS = ("QITOS_GLM_TOKENIZER_PATH", "GLM_TOKENIZER_PATH")


def _usage_payload(usage: Any) -> Optional[Dict[str, Any]]:
    """Normalize OpenAI/SGLang usage, including optional cache reporting."""
    if usage is None:
        return None

    def field(name: str) -> Any:
        return usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)

    prompt_tokens = field("prompt_tokens")
    completion_tokens = field("completion_tokens")
    total_tokens = field("total_tokens")
    details = field("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached,
    }


def _stream_timeout(idle_timeout: float) -> Any:
    """Use the provider read timeout as the synchronous stream-idle watchdog."""
    import httpx

    return httpx.Timeout(float(idle_timeout))


class _ChatStreamAccumulator:
    """Normalize one Chat Completions stream attempt."""

    def __init__(self, adapter: Model) -> None:
        self._adapter = adapter
        self._tool_calls: List[Dict[str, Any]] = []
        self._usage: Optional[Dict[str, Any]] = None
        self.finished = False

    def consume(self, chunk: Any) -> Iterator[ModelStreamChunk]:
        if not chunk.choices:
            usage = _usage_payload(getattr(chunk, "usage", None))
            if usage:
                self._usage = usage
            return

        choice = chunk.choices[0]
        delta = choice.delta
        text = delta.content or ""
        reasoning = getattr(delta, "reasoning_content", None)
        if text or reasoning:
            yield ModelStreamChunk(
                text=text,
                reasoning_content=str(reasoning) if reasoning else None,
                done=False,
            )
        self._accumulate_tool_calls(getattr(delta, "tool_calls", None))
        if choice.finish_reason is None:
            return

        usage = _usage_payload(getattr(chunk, "usage", None))
        if usage:
            self._usage = usage
        self.finished = True

    def complete(self) -> ModelStreamChunk | None:
        if not self.finished:
            return None
        self._adapter._set_last_usage(self._usage)
        return ModelStreamChunk(
            text="",
            done=True,
            usage=self._usage,
            tool_calls=self._tool_calls or None,
        )

    def _accumulate_tool_calls(self, deltas: Any) -> None:
        for item in list(deltas or []):
            index = getattr(item, "index", len(self._tool_calls))
            while len(self._tool_calls) <= index:
                self._tool_calls.append(
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )
            tool_call = self._tool_calls[index]
            tool_call_id = getattr(item, "id", None)
            if tool_call_id:
                tool_call["id"] = tool_call_id
            tool_call_type = getattr(item, "type", None)
            if tool_call_type:
                tool_call["type"] = tool_call_type
            function = getattr(item, "function", None)
            if not function:
                continue
            name = getattr(function, "name", None)
            if name:
                tool_call["function"]["name"] = name
            arguments = getattr(function, "arguments", None)
            if arguments:
                previous = tool_call["function"].get("arguments", "")
                tool_call["function"]["arguments"] = previous + arguments


def _wire_tool_schema(
    tool_schema_payload: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Return provider-valid function tools from richer registry metadata."""
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


def _relocate_chat_template_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Move ``chat_template_kwargs`` from top-level kwargs into ``extra_body``.

    The OpenAI Python SDK does not accept ``chat_template_kwargs`` as a
    top-level parameter.  vLLM-compatible serving endpoints expect it inside
    ``extra_body`` instead.  Calling code that merges ``default_request_kwargs``
    often places it at the top level, so we relocate it here.
    """
    result = dict(kwargs)
    ctk = result.pop("chat_template_kwargs", None)
    if isinstance(ctk, dict) and ctk:
        extra_body = dict(result.pop("extra_body", None) or {})
        extra_body["chat_template_kwargs"] = ctk
        result["extra_body"] = extra_body
    return result


def _merge_request_kwargs(
    defaults: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge model defaults with per-call overrides without losing nested options."""
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
    """Return whether a provider explicitly rejected ``stream_options``."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code not in {400, 422}:
        return False
    detail = f"{exc} {getattr(exc, 'body', '')}".casefold()
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
    """Count the adapter's actual Chat or Responses input shape."""

    api_mode = str(getattr(adapter, "api_mode", "chat_completions"))
    wire_messages = (
        _to_responses_input(messages)
        if api_mode == "responses"
        else _to_openai_messages(messages)
    )
    # A Responses item is not a chat-template message. Wrapping it in the
    # provider's ``input`` field makes tokenizer-backed compatible adapters
    # serialize every native field instead of discarding call/reasoning data as
    # unknown chat-message metadata.
    message_payload: Any = (
        {"input": wire_messages} if api_mode == "responses" else wire_messages
    )
    message_tokens = adapter.count_tokens(message_payload)
    option_payload = dict(request_options or {})
    if api_mode == "responses":
        option_payload = _normalize_request_kwargs(option_payload)
    option_tokens = (
        adapter.count_tokens(option_payload) if option_payload else 0
    )
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
            if detail:
                image_url["detail"] = detail
            blocks.append({"type": "image_url", "image_url": image_url})
            continue
        if block_type == "image_base64":
            mime_type = str(block.get("mime_type") or "image/png")
            image_url = {"url": ensure_data_url(str(block.get("data") or ""), mime_type=mime_type)}
            if detail:
                image_url["detail"] = detail
            blocks.append({"type": "image_url", "image_url": image_url})
            continue
        if block_type == "image_file":
            path = str(block.get("path") or "")
            mime_type = str(block.get("mime_type") or "")
            image_url = {
                "url": file_to_data_url(path, mime_type=mime_type or None)
            }
            if detail:
                image_url["detail"] = detail
            blocks.append({"type": "image_url", "image_url": image_url})
            continue
        blocks.append({"type": "text", "text": str(block)})
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
        extras: Dict[str, Any] = {}
        for key in ("tool_calls", "tool_call_id", "name"):
            if key in item and item.get(key) not in (None, "", []):
                extras[key] = item.get(key)
        if extras:
            content = (
                content
                + "\n"
                + json.dumps(extras, ensure_ascii=False, sort_keys=True)
            ).strip()
        messages.append({"role": role, "content": content})
    return messages


class OpenAIModel(Model):
    """
    OpenAI model calling implementation

    Environment variable configuration:
    - OPENAI_API_KEY: OpenAI API key
    - OPENAI_BASE_URL: OpenAI API base URL (optional, default https://api.openai.com/v1)

    Output format:
    - If model returns tool_calls: Convert to "Action: tool_name(args)" format
    - If model returns content: Return directly
    - Supports function calling format

    Example:
        llm = OpenAIModel(model="gpt-4")
        result = llm([{"role": "user", "content": "Help me search for Python tutorials"}])
        # Returns: "Action: search(query='Python tutorials')"
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120,
        context_window: Optional[int] = None,
        default_request_kwargs: Optional[Dict[str, Any]] = None,
        api_mode: str = "chat_completions",
    ):
        """
        Initialize OpenAI model

        Args:
            model: Model name, default gpt-4
            api_key: API key, default read from environment variable
            base_url: API base URL, default read from environment variable
            system_prompt: System prompt
            temperature: Temperature parameter (0.0-1.0)
            max_tokens: Maximum output token count
            timeout: Request timeout (seconds)
            context_window: Total model context window
            default_request_kwargs: Extra kwargs merged into every API call
            api_mode: OpenAI transport, ``chat_completions`` or ``responses``
        """
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
        )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.timeout = timeout
        self.default_request_kwargs = default_request_kwargs or {}
        self.api_mode = _normalize_api_mode(api_mode)

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Please set environment variable or pass api_key parameter."
            )

    def _call_api(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """
        Call OpenAI API

        Args:
            messages: OpenAI-style messages list

        Returns:
            Text that can be parsed by parse_tool_calls()
        """
        import openai

        try:
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
            )

            if self.api_mode == "responses":
                response = _responses_completion(
                    self, client, messages, provider="openai", **kwargs
                )
            else:
                response = self._chat_completion(client, messages, **kwargs)
            if isinstance(response, ModelResponse):
                return response.text
            return self._parse_response(response)

        except openai.APIError as e:
            return f"API Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _parse_response(self, response) -> str:
        """
        Parse OpenAI response and convert to target format

        Args:
            response: OpenAI API response object

        Returns:
            Text in parse_tool_calls compatible format
        """
        choice = response.choices[0]
        message = choice.message

        # Prioritize processing tool_calls
        if message.tool_calls:
            return self._format_tool_calls(message.tool_calls)

        # Return content
        if message.content:
            return message.content.strip()

        return ""

    def _chat_completion(
        self, client: Any, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Any:
        safe_kwargs = _relocate_chat_template_kwargs(kwargs)
        response = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=cast(Any, _to_openai_messages(messages)),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **safe_kwargs,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt >= 2:
                    raise
                time.sleep(2 ** attempt)
        if response is None:
            assert last_error is not None
            raise last_error
        self._set_last_usage(self._usage_from_response(response))
        return response

    def call_raw(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        import openai

        self._last_usage = None
        client = openai.OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )
        if self.api_mode == "responses":
            return _responses_completion(
                self, client, messages, provider="openai", **kwargs
            )
        return self._chat_completion(client, messages, **kwargs)

    def stream(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Iterator[ModelStreamChunk]:
        """Stream OpenAI response as chunks, yielding token-level text."""
        import openai

        self._last_usage = None
        try:
            client = openai.OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
            if self.api_mode == "responses":
                yield from _responses_stream(
                    self, client, messages, provider="openai", **kwargs
                )
                return
            response = client.chat.completions.create(
                model=self.model,
                messages=cast(Any, _to_openai_messages(messages)),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs,
            )
            accumulated_tool_calls: List[Dict[str, Any]] = []
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = delta.content or ""
                if text:
                    yield ModelStreamChunk(text=text, done=False)
                # Accumulate streaming tool call deltas
                delta_tool_calls = getattr(delta, "tool_calls", None)
                if delta_tool_calls:
                    for dtc in delta_tool_calls:
                        idx = getattr(dtc, "index", len(accumulated_tool_calls))
                        while len(accumulated_tool_calls) <= idx:
                            accumulated_tool_calls.append(
                                {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                            )
                        tc = accumulated_tool_calls[idx]
                        tc_id = getattr(dtc, "id", None)
                        if tc_id:
                            tc["id"] = tc_id
                        tc_type = getattr(dtc, "type", None)
                        if tc_type:
                            tc["type"] = tc_type
                        fn = getattr(dtc, "function", None)
                        if fn:
                            fn_name = getattr(fn, "name", None)
                            if fn_name:
                                tc["function"]["name"] = fn_name
                            fn_args = getattr(fn, "arguments", None)
                            if fn_args:
                                tc["function"]["arguments"] = tc["function"].get("arguments", "") + fn_args
                if chunk.choices[0].finish_reason is not None:
                    usage_data = None
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_data = {
                            "prompt_tokens": getattr(chunk.usage, "prompt_tokens", None),
                            "completion_tokens": getattr(chunk.usage, "completion_tokens", None),
                            "total_tokens": getattr(chunk.usage, "total_tokens", None),
                            "cached_tokens": getattr(getattr(chunk.usage, "prompt_tokens_details", None), "cached_tokens", None),
                        }
                        self._set_last_usage(usage_data)
                    yield ModelStreamChunk(
                        text="", done=True, usage=usage_data,
                        tool_calls=accumulated_tool_calls if accumulated_tool_calls else None,
                    )
        except openai.APIError as e:
            yield ModelStreamChunk(text=f"API Error: {str(e)}", done=True)
        except Exception as e:
            yield ModelStreamChunk(text=f"Error: {str(e)}", done=True)

    def _usage_from_response(self, response: Any) -> Optional[Dict[str, Any]]:
        return _usage_payload(getattr(response, "usage", None))

    def _format_tool_calls(self, tool_calls) -> str:
        """
        Convert OpenAI tool_calls format to parse_tool_calls compatible format

        Args:
            tool_calls: OpenAI tool_calls list

        Returns:
            Formatted tool call text
        """
        parts = []

        for i, call in enumerate(tool_calls):
            function = call.function
            name = function.name
            args = function.arguments

            try:
                args_dict = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args_dict = {"raw_args": args}

            if len(tool_calls) > 1:
                parts.append(f"Action {i + 1}: {name}")
            else:
                parts.append(f"Action: {name}")

            if args_dict:
                args_str = ", ".join(
                    f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                    for k, v in args_dict.items()
                )
                parts[-1] += f"({args_str})"

        return "\n".join(parts)

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
        return {"tools": wire}

    def count_request_tokens(
        self,
        messages: List[Dict[str, Any]],
        request_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        return _count_openai_request_tokens(self, messages, request_options)

    def supports_multimodal_input(self) -> bool:
        return True


class OpenAICompatibleModel(Model):
    """
    OpenAI compatible interface model

    Supports any service compatible with OpenAI API format, such as:
    - Azure OpenAI
    - Anthropic (via compatible endpoints)
    - LM Studio
    - LocalAI
    - Tongyi Qianwen
    - Zhipu AI

    Example:
        llm = OpenAICompatibleModel(
            model="qwen-turbo",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    """

    def __init__(
        self,
        model: str = "default",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float | None = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120,
        context_window: Optional[int] = None,
        default_request_kwargs: Optional[Dict[str, Any]] = None,
        api_mode: str = "chat_completions",
        max_attempts: int = 2,
        stream_idle_timeout: float = 60.0,
        retry_window_seconds: float = 300.0,
    ):
        """
        Initialize compatible model

        Args:
            model: Model name
            api_key: API key
            base_url: API base URL
            system_prompt: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum output token count
            timeout: Request timeout
            context_window: Total model context window
            default_request_kwargs: Extra kwargs merged into every API call
                (e.g. {"chat_template_kwargs": {"thinking": True}})
            api_mode: OpenAI transport, ``chat_completions`` or ``responses``
            max_attempts: Total transport attempts, including the initial request
            stream_idle_timeout: Maximum seconds between Responses stream events
            retry_window_seconds: Maximum recovery window after the first
                retryable failure
        """
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
        )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "dummy-key"
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        self.timeout = timeout
        self.default_request_kwargs = default_request_kwargs or {}
        self.api_mode = _normalize_api_mode(api_mode)
        if isinstance(stream_idle_timeout, bool) or stream_idle_timeout <= 0:
            raise ValueError("stream_idle_timeout must be positive")
        self.retry_policy = ModelRetryPolicy(
            max_attempts=max_attempts,
            retry_window_seconds=retry_window_seconds,
        )
        self.stream_idle_timeout = float(stream_idle_timeout)

        if not self.base_url:
            raise ValueError(
                "OPENAI_BASE_URL not set. Please set environment variable or pass base_url parameter."
            )

    def _request_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply model-level request defaults to every invocation path."""
        return _merge_request_kwargs(self.default_request_kwargs, kwargs)

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
        except Exception:
            return None

        try:
            if isinstance(payload, list):
                messages = _normalize_messages_for_tokenizer(payload)
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                return _tokenizer_count_result(encoded)
            text = self._stringify_token_payload(payload)
            encoded = tokenizer.encode(text, add_special_tokens=False)
            return _tokenizer_count_result(encoded)
        except Exception:
            return None

    def _call_api(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """
        Call OpenAI compatible API

        Args:
            messages: OpenAI-style messages list

        Returns:
            Text that can be parsed by parse_tool_calls()
        """
        import openai

        kwargs = self._request_kwargs(kwargs)
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )
        if self.api_mode == "responses":
            response = run_with_retry(
                lambda: _responses_completion(
                    self, client, messages, provider="openai-compatible", **kwargs
                ),
                self.retry_policy,
            )
        else:
            response = self._chat_completion(client, messages, **kwargs)
        if isinstance(response, ModelResponse):
            return response.text
        return self._parse_response(response)

    def _parse_response(self, response) -> str:
        """
        Parse response
        """
        choice = response.choices[0]
        message = choice.message

        if message.tool_calls:
            return self._format_tool_calls(message.tool_calls)

        if message.content:
            return message.content.strip()

        return ""

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
        return {"tools": wire}

    def count_request_tokens(
        self,
        messages: List[Dict[str, Any]],
        request_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        return _count_openai_request_tokens(self, messages, request_options)

    def supports_multimodal_input(self) -> bool:
        return True

    def _format_tool_calls(self, tool_calls) -> str:
        """
        Format tool calls
        """
        import json

        parts = []

        for i, call in enumerate(tool_calls):
            function = call.function
            name = function.name
            args = function.arguments or "{}"

            try:
                args_dict = json.loads(args)
            except json.JSONDecodeError:
                args_dict = {"raw": args}

            if len(tool_calls) > 1:
                parts.append(f"Action {i + 1}: {name}")
            else:
                parts.append(f"Action: {name}")

            if args_dict:
                args_str = ", ".join(
                    f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                    for k, v in args_dict.items()
                )
                parts[-1] += f"({args_str})"

        return "\n".join(parts)

    def _chat_completion(
        self, client: Any, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Any:
        safe_kwargs = _disable_thinking_for_forced_tool_choice(
            _relocate_chat_template_kwargs(kwargs)
        )
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": cast(Any, _to_openai_messages(messages)),
            "max_tokens": self.max_tokens,
            **safe_kwargs,
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        response = run_with_retry(
            lambda: client.chat.completions.create(**request_kwargs),
            self.retry_policy,
        )
        self._set_last_usage(self._usage_from_response(response))
        return response

    def call_raw(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        import openai

        self._last_usage = None
        kwargs = self._request_kwargs(kwargs)
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )
        if self.api_mode == "responses":
            return run_with_retry(
                lambda: _responses_completion(
                    self, client, messages, provider="openai-compatible", **kwargs
                ),
                self.retry_policy,
            )
        return self._chat_completion(client, messages, **kwargs)

    def _usage_from_response(self, response: Any) -> Optional[Dict[str, Any]]:
        return _usage_payload(getattr(response, "usage", None))

    def _stream_client(self) -> Any:
        import openai

        return openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=_stream_timeout(self.stream_idle_timeout),
            max_retries=0,
        )

    def _chat_stream_request(
        self, messages: List[Dict[str, Any]], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        create_kwargs = _disable_thinking_for_forced_tool_choice(
            _relocate_chat_template_kwargs(kwargs)
        )
        if "stream_options" not in create_kwargs:
            create_kwargs["stream_options"] = {"include_usage": True}
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": cast(Any, _to_openai_messages(messages)),
            "max_tokens": self.max_tokens,
            "stream": True,
            **create_kwargs,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        return request

    def _chat_stream_once(
        self, request_kwargs: Dict[str, Any]
    ) -> Iterator[ModelStreamChunk]:
        client = self._stream_client()
        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if "stream_options" not in request_kwargs:
                raise
            if not _is_unsupported_stream_options_error(exc):
                raise
            request_kwargs.pop("stream_options", None)
            response = client.chat.completions.create(**request_kwargs)

        try:
            yield from self._chat_stream_chunks(response)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    def _chat_stream_chunks(self, response: Any) -> Iterator[ModelStreamChunk]:
        accumulator = _ChatStreamAccumulator(self)
        for chunk in response:
            yield from accumulator.consume(chunk)
        completed = accumulator.complete()
        if completed is not None:
            yield completed

    def _responses_stream_once(
        self,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
        *,
        require_completed: bool,
    ) -> Iterator[ModelStreamChunk]:
        yield from _responses_stream(
            self,
            self._stream_client(),
            messages,
            provider="openai-compatible",
            require_completed=require_completed,
            **kwargs,
        )

    def stream(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Iterator[ModelStreamChunk]:
        """Stream live chunks, retrying only before output becomes observable."""
        self._last_usage = None
        kwargs = self._request_kwargs(kwargs)
        if self.api_mode == "responses":
            yield from sync_stream_with_retry(
                lambda: self._responses_stream_once(
                    messages, dict(kwargs), require_completed=False
                ),
                policy=self.retry_policy,
            )
            return

        request_kwargs = self._chat_stream_request(messages, dict(kwargs))
        yield from sync_stream_with_retry(
            lambda: self._chat_stream_once(request_kwargs),
            policy=self.retry_policy,
        )

    def transactional_stream(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Iterator[ModelStreamChunk]:
        """Publish one complete attempt and retry discarded partial attempts."""
        self._last_usage = None
        kwargs = self._request_kwargs(kwargs)
        if self.api_mode == "responses":
            yield from sync_transactional_stream_with_retry(
                lambda: self._responses_stream_once(
                    messages, dict(kwargs), require_completed=True
                ),
                policy=self.retry_policy,
                is_complete=lambda chunk: chunk.done,
            )
            return

        request_kwargs = self._chat_stream_request(messages, dict(kwargs))
        yield from sync_transactional_stream_with_retry(
            lambda: self._chat_stream_once(request_kwargs),
            policy=self.retry_policy,
            is_complete=lambda chunk: chunk.done,
        )


class AzureOpenAIModel(OpenAICompatibleModel):
    """
    Azure OpenAI model implementation

    Specifically optimized for Azure OpenAI service

    Environment variable configuration:
    - AZURE_OPENAI_API_KEY: Azure API key
    - AZURE_OPENAI_ENDPOINT: Azure endpoint URL
    - AZURE_OPENAI_DEPLOYMENT: Deployment name
    - AZURE_OPENAI_API_VERSION: API version (default 2024-02-15-preview)

    Example:
        llm = AzureOpenAIModel(
            deployment="gpt-4",
            api_version="2024-02-15-preview"
        )
    """

    def __init__(
        self,
        deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 60,
        context_window: Optional[int] = None,
    ):
        """
        Initialize Azure OpenAI model

        Args:
            deployment: Deployment name (used as model)
            api_key: API key, default read from environment variable
            endpoint: Endpoint URL, default read from environment variable
            api_version: API version
            system_prompt: System prompt
            temperature: Temperature parameter
            max_tokens: Maximum output token count
            timeout: Request timeout
            context_window: Total model context window
        """
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")

        if not endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT not set. Please set environment variable or pass endpoint parameter."
            )

        base_url = (
            f"{endpoint.rstrip('/')}/openai/deployments/{deployment or 'default'}"
        )

        super().__init__(
            model=deployment or "azure",
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            context_window=context_window,
        )

        self.api_version = api_version
        self.deployment = deployment
        self.endpoint = endpoint

    def _call_api(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """
        Call Azure OpenAI API (adds api_version parameter)
        """
        import openai

        try:
            client = openai.AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                timeout=self.timeout,
            )

            response = client.chat.completions.create(
                model=self.deployment or "",
                messages=cast(Any, _to_openai_messages(messages)),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )
            self._set_last_usage(self._usage_from_response(response))

            return self._parse_response(response)

        except openai.APIError as e:
            return f"API Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"


class AsyncOpenAICompatibleModel(OpenAICompatibleModel):
    """
    Async version of OpenAICompatibleModel using openai.AsyncOpenAI.

    Supports any service compatible with the OpenAI API format.
    Use ``await model.acall(messages)`` for non-blocking calls.

    Example::

        llm = AsyncOpenAICompatibleModel(
            model="qwen-turbo",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        result = await llm.acall([{"role": "user", "content": "Hello"}])
    """

    async def _acall_api(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """Async call to OpenAI-compatible API."""
        import openai

        kwargs = self._request_kwargs(kwargs)
        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )
        if self.api_mode == "responses":
            response = await _async_responses_completion(
                self, client, messages, provider="openai-compatible", **kwargs
            )
        else:
            response = await self._achat_completion(client, messages, **kwargs)
        if isinstance(response, ModelResponse):
            return response.text
        return self._parse_response(response)

    async def _achat_completion(
        self, client: Any, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Any:
        safe_kwargs = _disable_thinking_for_forced_tool_choice(
            _relocate_chat_template_kwargs(kwargs)
        )
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": cast(Any, _to_openai_messages(messages)),
            "max_tokens": self.max_tokens,
            **safe_kwargs,
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        response = await async_run_with_retry(
            lambda: client.chat.completions.create(**request_kwargs),
            self.retry_policy,
        )
        self._set_last_usage(self._usage_from_response(response))
        return response

    async def acall_raw(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        """Async version of call_raw returning provider-native response."""
        import openai

        self._last_usage = None
        kwargs = self._request_kwargs(kwargs)
        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )
        if self.api_mode == "responses":
            return await _async_responses_completion(
                self, client, messages, provider="openai-compatible", **kwargs
            )
        return await self._achat_completion(client, messages, **kwargs)

    async def astream(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[ModelStreamChunk]:
        """Async stream OpenAI-compatible response as chunks."""
        import openai

        self._last_usage = None
        kwargs = self._request_kwargs(kwargs)
        if self.api_mode == "responses":
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
            )
            async for chunk in _async_responses_stream(
                self, client, messages, provider="openai-compatible", **kwargs
            ):
                yield chunk
            return
        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )
        safe_kwargs = _disable_thinking_for_forced_tool_choice(
            _relocate_chat_template_kwargs(kwargs)
        )
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": cast(Any, _to_openai_messages(messages)),
            "max_tokens": self.max_tokens,
            "stream": True,
            **safe_kwargs,
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature

        async def create_stream() -> AsyncIterator[Any]:
            return await client.chat.completions.create(**request_kwargs)

        async for chunk in stream_with_retry(
            create_stream,
            policy=self.retry_policy,
            idle_timeout_seconds=self.stream_idle_timeout,
            request_timeout_seconds=float(self.timeout),
        ):
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    self._set_last_usage(_usage_payload(chunk.usage))
                continue
            delta = chunk.choices[0].delta
            text = delta.content or ""
            reasoning = getattr(delta, "reasoning_content", None)
            if text or reasoning:
                yield ModelStreamChunk(
                    text=text,
                    reasoning_content=str(reasoning) if reasoning else None,
                    done=False,
                )
            if chunk.choices[0].finish_reason is not None:
                usage_data = _usage_payload(getattr(chunk, "usage", None))
                self._set_last_usage(usage_data)
                yield ModelStreamChunk(text="", done=True, usage=usage_data)

    async def _atransactional_stream_once(
        self,
        messages: List[Dict[str, Any]],
        request_kwargs: Dict[str, Any],
    ) -> AsyncIterator[ModelStreamChunk]:
        import openai

        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=_stream_timeout(self.stream_idle_timeout),
            max_retries=0,
        )
        if self.api_mode == "responses":
            async for chunk in _async_responses_stream(
                self,
                client,
                messages,
                provider="openai-compatible",
                retry_events=False,
                require_completed=True,
                **request_kwargs,
            ):
                yield chunk
            return

        try:
            response = await client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if "stream_options" not in request_kwargs:
                raise
            if not _is_unsupported_stream_options_error(exc):
                raise
            request_kwargs.pop("stream_options", None)
            response = await client.chat.completions.create(**request_kwargs)

        accumulator = _ChatStreamAccumulator(self)
        try:
            async for item in response:
                for chunk in accumulator.consume(item):
                    yield chunk
            completed = accumulator.complete()
            if completed is not None:
                yield completed
        finally:
            close = getattr(response, "aclose", None)
            if callable(close):
                await close()

    async def atransactional_stream(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[ModelStreamChunk]:
        """Publish one complete attempt and retry discarded partial attempts."""
        self._last_usage = None
        kwargs = self._request_kwargs(kwargs)
        request_kwargs = (
            dict(kwargs)
            if self.api_mode == "responses"
            else self._chat_stream_request(messages, dict(kwargs))
        )

        async def create_stream() -> AsyncIterator[ModelStreamChunk]:
            return self._atransactional_stream_once(messages, request_kwargs)

        async for chunk in transactional_stream_with_retry(
            create_stream,
            policy=self.retry_policy,
            idle_timeout_seconds=self.stream_idle_timeout,
            request_timeout_seconds=float(self.timeout),
            is_complete=lambda item: item.done,
        ):
            yield chunk


class AsyncOpenAIModel(OpenAIModel):
    """
    Async version of OpenAIModel using openai.AsyncOpenAI.

    Example::

        llm = AsyncOpenAIModel(model="gpt-4")
        result = await llm.acall([{"role": "user", "content": "Hello"}])
    """

    async def _acall_api(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """Async call to OpenAI API."""
        import openai

        try:
            client = openai.AsyncOpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
            if self.api_mode == "responses":
                response = await _async_responses_completion(
                    self, client, messages, provider="openai", **kwargs
                )
            else:
                response = await self._achat_completion(client, messages, **kwargs)
            if isinstance(response, ModelResponse):
                return response.text
            return self._parse_response(response)
        except openai.APIError as e:
            return f"API Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _achat_completion(
        self, client: Any, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Any:
        response = await client.chat.completions.create(
            model=self.model,
            messages=cast(Any, _to_openai_messages(messages)),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        self._set_last_usage(self._usage_from_response(response))
        return response

    async def acall_raw(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        """Async version of call_raw returning provider-native response."""
        import openai

        self._last_usage = None
        client = openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )
        if self.api_mode == "responses":
            return await _async_responses_completion(
                self, client, messages, provider="openai", **kwargs
            )
        return await self._achat_completion(client, messages, **kwargs)

    async def astream(self, messages: List[Dict[str, Any]], **kwargs: Any) -> AsyncIterator[ModelStreamChunk]:
        """Async stream OpenAI response as chunks."""
        import openai

        self._last_usage = None
        try:
            client = openai.AsyncOpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
            if self.api_mode == "responses":
                async for chunk in _async_responses_stream(
                    self, client, messages, provider="openai", **kwargs
                ):
                    yield chunk
                return
            response = await client.chat.completions.create(
                model=self.model,
                messages=cast(Any, _to_openai_messages(messages)),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs,
            )
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = delta.content or ""
                if text:
                    yield ModelStreamChunk(text=text, done=False)
                if chunk.choices[0].finish_reason is not None:
                    usage_data = None
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_data = {
                            "prompt_tokens": getattr(chunk.usage, "prompt_tokens", None),
                            "completion_tokens": getattr(chunk.usage, "completion_tokens", None),
                            "total_tokens": getattr(chunk.usage, "total_tokens", None),
                        }
                        self._set_last_usage(usage_data)
                    yield ModelStreamChunk(text="", done=True, usage=usage_data)
        except openai.APIError as e:
            yield ModelStreamChunk(text=f"API Error: {str(e)}", done=True)
        except Exception as e:
            yield ModelStreamChunk(text=f"Error: {str(e)}", done=True)


# Register to factory
from .base import ModelFactory

ModelFactory.register("openai")(OpenAIModel)
ModelFactory.register("azure")(AzureOpenAIModel)
ModelFactory.register("openai-compatible")(OpenAICompatibleModel)
ModelFactory.register("async-openai")(AsyncOpenAIModel)
ModelFactory.register("async-openai-compatible")(AsyncOpenAICompatibleModel)
