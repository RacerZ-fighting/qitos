"""OpenAI Responses API protocol conversion and transport helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, cast

from ..core.errors import ModelTransportError
from ..core.model_response import ModelResponse
from ._openai_retry import ModelRetryPolicy, stream_with_retry
from .base import Model, ModelStreamChunk

OPENAI_API_MODES = {"chat_completions", "responses"}
_RESPONSES_REQUIREMENT = (
    "api_mode='responses' requires openai>=1.66.0 and an endpoint "
    "that implements POST /v1/responses"
)


def _normalize_api_mode(value: Any) -> str:
    mode = str(value or "chat_completions").strip().lower()
    if mode not in OPENAI_API_MODES:
        allowed = ", ".join(sorted(OPENAI_API_MODES))
        raise ValueError(f"Unsupported api_mode {value!r}; expected one of: {allowed}")
    return mode


def _native_value(value: Any) -> Any:
    """Convert SDK models and simple objects to JSON-compatible native values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _native_value(asdict(cast(Any, value)))
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
    return value


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert canonical QitOS history to Responses API input items."""
    items: List[Dict[str, Any]] = []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            continue

        native_items: List[Dict[str, Any]] = []
        native_call_ids: set[str] = set()
        native_output_ids: set[str] = set()
        has_native_message = False
        seen_native_transactions: set[tuple[str, str]] = set()
        for raw_item in raw_message.get("native_items") or []:
            native_item = _native_value(raw_item)
            if not isinstance(native_item, dict):
                continue
            item_type = str(native_item.get("type") or "").strip()
            call_id = str(native_item.get("call_id") or "").strip()
            if item_type in {"function_call", "function_call_output"} and call_id:
                transaction_key = (item_type, call_id)
                if transaction_key in seen_native_transactions:
                    continue
                seen_native_transactions.add(transaction_key)
                if item_type == "function_call":
                    native_call_ids.add(call_id)
                else:
                    native_output_ids.add(call_id)
            if item_type == "message":
                has_native_message = True
            native_items.append(dict(native_item))

        role = str(raw_message.get("role") or "user").strip() or "user"
        if role == "tool":
            call_id = str(raw_message.get("tool_call_id") or "").strip()
            items.extend(native_items)
            if call_id and call_id not in native_output_ids:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": str(raw_message.get("content") or ""),
                    }
                )
            continue

        content = raw_message.get("content")
        emitted_content = content is not None and not has_native_message
        if emitted_content:
            message = {"role": role, "content": content}
            items.append(cast(Dict[str, Any], _native_value(message)))

        items.extend(native_items)
        emitted_call = False
        if role == "assistant":
            for tool_call in raw_message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                call_id = str(tool_call.get("id") or "").strip()
                name = str(function.get("name") or "").strip()
                if not call_id or not name or call_id in native_call_ids:
                    continue
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": str(function.get("arguments") or "{}"),
                    }
                )
                native_call_ids.add(call_id)
                emitted_call = True

        if not emitted_content and not native_items and not emitted_call:
            items.append({"role": role, "content": ""})
    return items


def _to_responses_tools(
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    normalized: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if tool.get("type") == "function" and isinstance(function, dict):
            payload: Dict[str, Any] = {"type": "function"}
            for key in ("name", "description", "parameters", "strict"):
                if key in function:
                    payload[key] = _native_value(function[key])
            normalized.append(payload)
        else:
            normalized.append(cast(Dict[str, Any], _native_value(tool)))
    return normalized or None


def _to_responses_tool_choice(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return tool_choice
    function = tool_choice.get("function")
    if tool_choice.get("type") == "function" and isinstance(function, dict):
        name = str(function.get("name") or "").strip()
        if name:
            return {"type": "function", "name": name}
    return _native_value(tool_choice)


def _normalize_request_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    request_kwargs = dict(kwargs)
    response_format = request_kwargs.pop("response_format", None)
    if response_format is not None:
        request_kwargs["text"] = {"format": _native_value(response_format)}
    tools = request_kwargs.pop("tools", None)
    if tools is not None:
        request_kwargs["tools"] = _to_responses_tools(tools)
    if "tool_choice" in request_kwargs:
        request_kwargs["tool_choice"] = _to_responses_tool_choice(
            request_kwargs["tool_choice"]
        )
    return request_kwargs


def _responses_usage(response: Any) -> Optional[Dict[str, Any]]:
    usage = _native_value(_field(response, "usage"))
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    result: Dict[str, Any] = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    for key in ("input_tokens_details", "output_tokens_details"):
        if isinstance(usage.get(key), dict):
            result[key] = dict(usage[key])
    return result


def _response_output_text(response: Any, native_items: List[Dict[str, Any]]) -> str:
    output_text = _field(response, "output_text")
    if output_text:
        return str(output_text)
    parts: List[str] = []
    for item in native_items:
        if item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"output_text", "text"} and block.get("text"):
                parts.append(str(block["text"]))
    return "".join(parts)


def _model_response_from_responses(
    response: Any,
    *,
    provider: str,
) -> ModelResponse:
    native_output = _native_value(_field(response, "output"))
    native_items = (
        [dict(item) for item in native_output if isinstance(item, dict)]
        if isinstance(native_output, list)
        else []
    )
    tool_calls: List[Dict[str, Any]] = []
    for item in native_items:
        if item.get("type") != "function_call":
            continue
        call_id = str(item.get("call_id") or "").strip()
        name = str(item.get("name") or "").strip()
        if not call_id or not name:
            continue
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": str(item.get("arguments") or "{}"),
                },
                "metadata": {
                    "response_item_id": item.get("id"),
                    "status": item.get("status"),
                },
            }
        )
    status = _field(response, "status")
    metadata = {
        key: value
        for key, value in {
            "id": _field(response, "id"),
            "status": status,
            "previous_response_id": _field(response, "previous_response_id"),
            "api_mode": "responses",
        }.items()
        if value is not None
    }
    return ModelResponse(
        text=_response_output_text(response, native_items),
        raw=response,
        usage=_responses_usage(response),
        finish_reason=str(status) if status is not None else None,
        tool_calls=tool_calls or None,
        model_name=(
            str(_field(response, "model"))
            if _field(response, "model") is not None
            else None
        ),
        provider=provider,
        metadata=metadata,
        native_items=native_items or None,
    )


def _responses_create(client: Any) -> Any:
    create = getattr(getattr(client, "responses", None), "create", None)
    if not callable(create):
        raise RuntimeError(_RESPONSES_REQUIREMENT)
    return create


def _request_payload(
    adapter: Model,
    messages: List[Dict[str, Any]],
    kwargs: Dict[str, Any],
    *,
    stream: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": adapter.model,
        "input": cast(Any, _to_responses_input(messages)),
        "temperature": adapter.temperature,
        "max_output_tokens": adapter.max_tokens,
    }
    if stream:
        payload["stream"] = True
    payload.update(_normalize_request_kwargs(kwargs))
    return payload


def _responses_completion(
    adapter: Model,
    client: Any,
    messages: List[Dict[str, Any]],
    *,
    provider: str,
    **kwargs: Any,
) -> ModelResponse:
    response = _responses_create(client)(**_request_payload(adapter, messages, kwargs))
    normalized = _model_response_from_responses(response, provider=provider)
    adapter._set_last_usage(normalized.usage)
    return normalized


async def _async_responses_completion(
    adapter: Model,
    client: Any,
    messages: List[Dict[str, Any]],
    *,
    provider: str,
    **kwargs: Any,
) -> ModelResponse:
    response = await _responses_create(client)(
        **_request_payload(adapter, messages, kwargs)
    )
    normalized = _model_response_from_responses(response, provider=provider)
    adapter._set_last_usage(normalized.usage)
    return normalized


def _event_metadata(event: Any) -> Dict[str, Any]:
    return {
        key: value
        for key in ("sequence_number", "output_index", "content_index", "item_id")
        if (value := _field(event, key)) is not None
    }


def _event_chunk(
    event: Any,
) -> tuple[Optional[ModelStreamChunk], Optional[Dict[str, Any]], Any]:
    event_type = str(_field(event, "type", "") or "")
    metadata = _event_metadata(event)
    if event_type == "response.output_text.delta":
        delta = str(_field(event, "delta", "") or "")
        return (
            (
                ModelStreamChunk(
                    text=delta,
                    event_type=event_type,
                    event_metadata=metadata,
                )
                if delta
                else None
            ),
            None,
            None,
        )
    if event_type in {
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
    }:
        metadata["delta"] = str(_field(event, "delta", "") or "")
        arguments = _field(event, "arguments")
        if arguments is not None:
            metadata["arguments"] = str(arguments)
        return (
            ModelStreamChunk(
                text="",
                event_type=event_type,
                event_metadata=metadata,
            ),
            None,
            None,
        )
    if event_type == "response.output_item.done":
        item = _native_value(_field(event, "item"))
        if isinstance(item, dict):
            native_item = dict(item)
            return (
                ModelStreamChunk(
                    text="",
                    native_items=[native_item],
                    event_type=event_type,
                    event_metadata=metadata,
                ),
                native_item,
                None,
            )
    if event_type == "response.completed":
        return None, None, _field(event, "response")
    if event_type in {"response.failed", "error"}:
        raise ModelTransportError(
            f"model stream failed: {_field(event, 'error')}",
            attempts=1,
            retryable=False,
        )
    return None, None, None


def _final_stream_chunk(
    adapter: Model,
    response: Any,
    completed_items: List[Dict[str, Any]],
    *,
    provider: str,
) -> ModelStreamChunk:
    if response is None:
        normalized = _model_response_from_responses(
            {"status": "completed", "output": completed_items},
            provider=provider,
        )
        return ModelStreamChunk(
            text="",
            done=True,
            tool_calls=normalized.tool_calls,
            native_items=normalized.native_items,
            event_type="response.completed",
            event_metadata=dict(normalized.metadata),
        )
    normalized = _model_response_from_responses(response, provider=provider)
    adapter._set_last_usage(normalized.usage)
    return ModelStreamChunk(
        text="",
        done=True,
        usage=normalized.usage,
        tool_calls=normalized.tool_calls,
        native_items=normalized.native_items,
        event_type="response.completed",
        event_metadata=dict(normalized.metadata),
    )


def _responses_stream(
    adapter: Model,
    client: Any,
    messages: List[Dict[str, Any]],
    *,
    provider: str,
    require_completed: bool = False,
    **kwargs: Any,
) -> Iterator[ModelStreamChunk]:
    events = _responses_create(client)(
        **_request_payload(adapter, messages, kwargs, stream=True)
    )
    completed_items: List[Dict[str, Any]] = []
    completed_response: Any = None
    for event in events:
        chunk, item, response = _event_chunk(event)
        if isinstance(item, dict):
            completed_items.append(item)
        if response is not None:
            completed_response = response
        if chunk is not None:
            yield chunk
            if chunk.done:
                return
    if require_completed and completed_response is None:
        raise ModelTransportError(
            "model stream ended before response.completed",
            attempts=1,
            retryable=True,
        )
    yield _final_stream_chunk(
        adapter, completed_response, completed_items, provider=provider
    )


async def _async_responses_stream(
    adapter: Model,
    client: Any,
    messages: List[Dict[str, Any]],
    *,
    provider: str,
    retry_events: bool = True,
    require_completed: bool = False,
    **kwargs: Any,
) -> AsyncIterator[ModelStreamChunk]:
    payload = _request_payload(adapter, messages, kwargs, stream=True)
    policy = getattr(adapter, "retry_policy", None)
    if retry_events and isinstance(policy, ModelRetryPolicy):
        events = stream_with_retry(
            lambda: _responses_create(client)(**payload),
            policy=policy,
            idle_timeout_seconds=float(getattr(adapter, "stream_idle_timeout", 60.0)),
            request_timeout_seconds=float(getattr(adapter, "timeout", 120.0)),
        )
    else:
        events = await _responses_create(client)(**payload)
    completed_items: List[Dict[str, Any]] = []
    completed_response: Any = None
    async for event in events:
        chunk, item, response = _event_chunk(event)
        if isinstance(item, dict):
            completed_items.append(item)
        if response is not None:
            completed_response = response
        if chunk is not None:
            yield chunk
            if chunk.done:
                return
    if require_completed and completed_response is None:
        raise ModelTransportError(
            "model stream ended before response.completed",
            attempts=1,
            retryable=True,
        )
    yield _final_stream_chunk(
        adapter, completed_response, completed_items, provider=provider
    )


__all__ = [
    "_async_responses_completion",
    "_async_responses_stream",
    "_model_response_from_responses",
    "_normalize_api_mode",
    "_responses_completion",
    "_responses_stream",
    "_to_responses_input",
    "_to_responses_tool_choice",
    "_to_responses_tools",
]
