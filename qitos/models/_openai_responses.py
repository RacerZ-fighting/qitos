"""OpenAI Responses request projection and stream normalization."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, cast

from ..core.errors import ModelContinuationRejected, ModelTransportError
from ..core.model_request import (
    ModelContinuation,
    ModelRequest,
    model_json_digest,
)
from ..core.model_stream import ModelStreamEventType
from ..core.model_response import ModelResponse
from .transport import close_async_resource
from .base import Model, ModelStreamEvent

OPENAI_API_MODES = {"chat_completions", "responses"}
_RESPONSES_REQUIREMENT = (
    "api_mode='responses' requires an OpenAI client and endpoint that "
    "implement POST /v1/responses"
)
_OPAQUE_REASONING_INCLUDE = "reasoning.encrypted_content"
_RESPONSES_ITEM_TYPES = {
    "message",
    "reasoning",
    "function_call",
    "function_call_output",
    "computer_call",
    "computer_call_output",
    "web_search_call",
    "file_search_call",
}


def _normalize_api_mode(value: Any) -> str:
    mode = str(value or "chat_completions").strip().lower()
    if mode not in OPENAI_API_MODES:
        allowed = ", ".join(sorted(OPENAI_API_MODES))
        raise ValueError(f"Unsupported api_mode {value!r}; expected one of: {allowed}")
    return mode


def _native_value(value: Any) -> Any:
    """Convert SDK values to builtins at the provider boundary."""

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
    raise TypeError(f"unsupported provider value: {type(value).__name__}")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Project QitOS history into ordered Responses input items."""

    items: List[Dict[str, Any]] = []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            continue

        native_items: List[Dict[str, Any]] = []
        native_call_ids: set[str] = set()
        native_output_ids: set[str] = set()
        has_native_message = False
        seen_transactions: set[tuple[str, str]] = set()
        for raw_item in raw_message.get("native_items") or []:
            native_item = _native_value(raw_item)
            if not isinstance(native_item, dict):
                continue
            item_type = str(native_item.get("type") or "").strip()
            if item_type not in _RESPONSES_ITEM_TYPES:
                continue
            call_id = str(native_item.get("call_id") or "").strip()
            if item_type in {"function_call", "function_call_output"} and call_id:
                transaction = (item_type, call_id)
                if transaction in seen_transactions:
                    continue
                seen_transactions.add(transaction)
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
            items.append(
                cast(
                    Dict[str, Any],
                    _native_value({"role": role, "content": content}),
                )
            )
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
            value = _native_value(tool)
            if isinstance(value, dict):
                normalized.append(value)
    return normalized or None


def _to_responses_tool_choice(
    tool_choice: Any,
    *,
    hosted_web_search: bool = False,
) -> Any:
    if not isinstance(tool_choice, dict):
        return tool_choice
    function = tool_choice.get("function")
    if tool_choice.get("type") == "function" and isinstance(function, dict):
        name = str(function.get("name") or "").strip()
        if name:
            if hosted_web_search and name == "web_search":
                return {"type": "web_search"}
            return {"type": "function", "name": name}
    return _native_value(tool_choice)


def _normalize_request_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    request_kwargs = dict(kwargs)
    response_format = request_kwargs.pop("response_format", None)
    if response_format is not None:
        request_kwargs["text"] = {"format": _native_value(response_format)}
    tools = request_kwargs.pop("tools", None)
    hosted_web_search = False
    if tools is not None:
        normalized_tools = _to_responses_tools(tools)
        request_kwargs["tools"] = normalized_tools
        hosted_web_search = any(
            isinstance(tool, dict) and tool.get("type") == "web_search"
            for tool in normalized_tools or []
        )
    if "tool_choice" in request_kwargs:
        request_kwargs["tool_choice"] = _to_responses_tool_choice(
            request_kwargs["tool_choice"],
            hosted_web_search=hosted_web_search,
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
    for key in (
        "input_tokens_details",
        "output_tokens_details",
        "prompt_tokens_details",
        "cached_tokens",
        "prompt_cache_hit_tokens",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
        "cache_creation_input_tokens",
    ):
        if isinstance(usage.get(key), dict):
            result[key] = dict(usage[key])
        elif usage.get(key) is not None:
            result[key] = usage[key]
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
            block_type = str(block.get("type") or "")
            if block_type in {"output_text", "text"} and block.get("text"):
                parts.append(str(block["text"]))
            elif block_type == "refusal" and block.get("refusal"):
                parts.append(str(block["refusal"]))
    return "".join(parts)


def _response_reasoning_text(native_items: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in native_items:
        if item.get("type") != "reasoning":
            continue
        for block in item.get("summary") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"summary_text", "text"} and block.get("text"):
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
    status = _field(response, "status")
    terminal_status = str(status or "").strip().casefold()
    tool_calls: List[Dict[str, Any]] = []
    for item in native_items:
        if item.get("type") != "function_call":
            continue
        item_status = str(item.get("status") or "").strip().casefold()
        if terminal_status == "incomplete" or item_status not in {"", "completed"}:
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
    incomplete_details = _native_value(_field(response, "incomplete_details"))
    incomplete_reason = (
        str(incomplete_details.get("reason"))
        if isinstance(incomplete_details, dict) and incomplete_details.get("reason")
        else None
    )
    metadata = {
        key: value
        for key, value in {
            "id": _field(response, "id"),
            "status": status,
            "previous_response_id": _field(response, "previous_response_id"),
            "incomplete_details": incomplete_details,
            "api_mode": "responses",
        }.items()
        if value is not None
    }
    return ModelResponse(
        text=_response_output_text(response, native_items),
        usage=_responses_usage(response),
        finish_reason=(
            incomplete_reason or (str(status) if status is not None else None)
        ),
        tool_calls=tool_calls or None,
        model_name=(
            str(_field(response, "model"))
            if _field(response, "model") is not None
            else None
        ),
        provider=provider,
        metadata=metadata,
        reasoning_content=_response_reasoning_text(native_items) or None,
        native_items=native_items or None,
    )


def _responses_create(client: Any) -> Any:
    create = getattr(getattr(client, "responses", None), "create", None)
    if not callable(create):
        raise RuntimeError(_RESPONSES_REQUIREMENT)
    return create


def _request_payload(
    adapter: Model,
    request: ModelRequest,
    kwargs: Dict[str, Any],
    *,
    provider: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = _normalize_request_kwargs(kwargs)
    payload.pop("previous_response_id", None)
    payload.update(
        {
            "model": adapter.model,
            "input": cast(Any, _to_responses_input(request.message_dicts())),
            "stream": True,
        }
    )
    payload.setdefault("max_output_tokens", adapter.max_tokens)
    if adapter.temperature is not None:
        payload.setdefault("temperature", adapter.temperature)
    family = str(getattr(adapter, "qitos_family_preset", "") or "").casefold()
    if provider.casefold() == "openai" or family == "openai":
        configured = payload.get("include")
        if isinstance(configured, (list, tuple)):
            include = list(configured)
        elif configured is None:
            include = []
        else:
            include = [configured]
        if _OPAQUE_REASONING_INCLUDE not in include:
            include.append(_OPAQUE_REASONING_INCLUDE)
        payload["include"] = include
    return payload


def _continuation_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"input", "previous_response_id", "stream", "timeout"}
    }


def _apply_continuation(
    request: ModelRequest,
    payload: Dict[str, Any],
) -> tuple[Dict[str, Any], bool, str]:
    """Use a handle only when the full canonical request proves its prefix."""

    continuation = request.continuation
    if continuation is None:
        return payload, False, "absent"
    if not continuation.belongs_to(request):
        return payload, False, "request_identity_changed"
    settings_digest = model_json_digest(_continuation_settings(payload))
    if settings_digest != continuation.settings_digest:
        return payload, False, "request_settings_changed"
    full_input = payload.get("input")
    if not isinstance(full_input, list):
        return payload, False, "input_not_incremental"
    prefix_items = continuation.prefix_items
    if prefix_items > len(full_input):
        return payload, False, "canonical_prefix_shortened"
    if model_json_digest(full_input[:prefix_items]) != continuation.prefix_digest:
        return payload, False, "canonical_prefix_changed"
    incremental = dict(payload)
    incremental["previous_response_id"] = continuation.response_id
    incremental["input"] = full_input[prefix_items:]
    return incremental, True, "applied"


def _continuation_rejected(error: Any) -> bool:
    status = _field(error, "status_code")
    if status is None:
        status = _field(_field(error, "response"), "status_code")
    detail = f"{error} {_field(error, 'body', '')}".casefold()
    mentions_handle = "previous_response_id" in detail or "previous response" in detail
    rejected = any(
        marker in detail
        for marker in (
            "expired",
            "invalid",
            "not found",
            "unknown",
            "does not exist",
        )
    )
    return bool(mentions_handle and rejected and status in {400, 404, 409, 422, None})


def _event_metadata(event: Any) -> Dict[str, Any]:
    return {
        key: value
        for key in (
            "sequence_number",
            "output_index",
            "content_index",
            "summary_index",
            "item_id",
            "call_id",
        )
        if (value := _field(event, key)) is not None
    }


def _event_response_metadata(event: Any) -> Dict[str, Any]:
    metadata = _event_metadata(event)
    response = _native_value(_field(event, "response"))
    if isinstance(response, dict):
        if response.get("id") is not None:
            metadata["response_id"] = response["id"]
        if response.get("status") is not None:
            metadata["status"] = response["status"]
    return metadata


def _function_event_key(event: Any, item: Dict[str, Any] | None = None) -> str:
    values = item or {}
    item_id = str(values.get("id") or _field(event, "item_id") or "").strip()
    if item_id:
        return f"item:{item_id}"
    call_id = str(values.get("call_id") or _field(event, "call_id") or "").strip()
    if call_id:
        return f"call:{call_id}"
    output_index = _field(event, "output_index")
    if isinstance(output_index, int) and not isinstance(output_index, bool):
        return f"output:{output_index}"
    return ""


def _missing_stream_suffix(
    complete: str,
    streamed: str,
    *,
    field_name: str,
) -> str:
    if not complete:
        return ""
    if not streamed:
        return complete
    if complete.startswith(streamed):
        return complete[len(streamed) :]
    raise ModelTransportError(
        f"completed response {field_name} does not match streamed deltas",
        attempts=1,
        retryable=False,
    )


def _native_item_identity(item: Dict[str, Any]) -> tuple[str, str] | None:
    item_id = str(item.get("id") or "").strip()
    if item_id:
        return "id", item_id
    call_id = str(item.get("call_id") or "").strip()
    item_type = str(item.get("type") or "").strip()
    if call_id and item_type:
        return item_type, call_id
    return None


def _merge_completed_output(
    response: Any,
    completed_items: List[Dict[str, Any]],
) -> Any:
    if not completed_items:
        return response
    payload = _native_value(response)
    if not isinstance(payload, dict):
        return response
    raw_output = payload.get("output")
    output = (
        [dict(item) for item in raw_output if isinstance(item, dict)]
        if isinstance(raw_output, list)
        else []
    )
    completed_by_id = {
        identity: dict(item)
        for item in completed_items
        if (identity := _native_item_identity(item)) is not None
    }
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in output:
        identity = _native_item_identity(item)
        event_item = completed_by_id.get(identity) if identity is not None else None
        merged.append({**event_item, **item} if event_item is not None else item)
        if identity is not None:
            seen.add(identity)
    for item in completed_items:
        identity = _native_item_identity(item)
        if identity is not None and identity in seen:
            continue
        if identity is None and item in merged:
            continue
        merged.append(dict(item))
    payload["output"] = merged
    return payload


class _ResponsesEventStream(AsyncIterator[ModelStreamEvent]):
    """Own and normalize one already-connected Responses stream."""

    def __init__(
        self,
        events: Any,
        *,
        provider: str,
        request: ModelRequest | None = None,
        full_input: List[Dict[str, Any]] | None = None,
        settings_digest: str = "",
        continuation_applied: bool = False,
        continuation_reason: str = "absent",
    ) -> None:
        self._events = events
        self._iterator = events.__aiter__()
        self._provider = provider
        self._completed_items: List[Dict[str, Any]] = []
        self._function_arguments: Dict[str, str] = {}
        self._streamed_text = ""
        self._streamed_reasoning = ""
        self._finished = False
        self._request = request
        self._full_input = list(full_input or [])
        self._settings_digest = settings_digest
        self._continuation_applied = continuation_applied
        self._continuation_reason = continuation_reason

    def __aiter__(self) -> _ResponsesEventStream:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        if self._finished:
            raise StopAsyncIteration
        while True:
            try:
                event = await self._iterator.__anext__()
            except StopAsyncIteration as exc:
                raise ModelTransportError(
                    "model stream ended before response.completed",
                    attempts=1,
                    retryable=True,
                ) from exc
            event_type = str(_field(event, "type", "") or "")[:128]
            if not event_type:
                raise ModelTransportError(
                    "model stream emitted an event without a type",
                    attempts=1,
                    retryable=False,
                )
            metadata = _event_metadata(event)
            if event_type == "response.output_text.delta":
                delta = str(_field(event, "delta", "") or "")
                if delta:
                    self._streamed_text += delta
                    return ModelStreamEvent(
                        type=ModelStreamEventType.TEXT_DELTA,
                        text=delta,
                        event_type=event_type,
                        event_metadata=metadata,
                    )
                continue
            if event_type in {
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            }:
                delta = str(_field(event, "delta", "") or "")
                if delta:
                    self._streamed_reasoning += delta
                    return ModelStreamEvent(
                        type=ModelStreamEventType.REASONING_DELTA,
                        reasoning_content=delta,
                        event_type=event_type,
                        event_metadata=metadata,
                    )
                continue
            if event_type in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            }:
                key = _function_event_key(event)
                if not key:
                    raise ModelTransportError(
                        f"{event_type} is missing an item identity",
                        attempts=1,
                        retryable=False,
                    )
                delta = str(_field(event, "delta", "") or "")
                if event_type.endswith(".delta"):
                    self._function_arguments[key] = (
                        self._function_arguments.get(key, "") + delta
                    )
                    metadata["arguments_delta"] = delta
                arguments = _field(event, "arguments")
                if arguments is not None:
                    completed_arguments = str(arguments)
                    streamed_arguments = self._function_arguments.get(key, "")
                    if streamed_arguments and completed_arguments != streamed_arguments:
                        raise ModelTransportError(
                            "completed function arguments do not match streamed deltas",
                            attempts=1,
                            retryable=False,
                        )
                    self._function_arguments[key] = completed_arguments
                metadata["arguments_chars"] = len(self._function_arguments.get(key, ""))
                return ModelStreamEvent(
                    type=ModelStreamEventType.TOOL_CALL_DELTA,
                    event_type=event_type,
                    event_metadata=metadata,
                )
            if event_type == "response.refusal.delta":
                delta = str(_field(event, "delta", "") or "")
                if delta:
                    self._streamed_text += delta
                    return ModelStreamEvent(
                        type=ModelStreamEventType.TEXT_DELTA,
                        text=delta,
                        event_type=event_type,
                        event_metadata=metadata,
                    )
                continue
            if event_type == "response.output_item.added":
                item = _native_value(_field(event, "item"))
                if not isinstance(item, dict):
                    raise ModelTransportError(
                        "response.output_item.added is missing a valid item",
                        attempts=1,
                        retryable=False,
                    )
                metadata["item_type"] = str(item.get("type") or "")
                if item.get("call_id") is not None:
                    metadata["call_id"] = item["call_id"]
                if item.get("type") == "function_call":
                    key = _function_event_key(event, item)
                    if key:
                        self._function_arguments.setdefault(
                            key, str(item.get("arguments") or "")
                        )
                return ModelStreamEvent(
                    type=(
                        ModelStreamEventType.TOOL_CALL_DELTA
                        if item.get("type") == "function_call"
                        else ModelStreamEventType.LIFECYCLE
                    ),
                    event_type=event_type,
                    event_metadata=metadata,
                )
            if event_type == "response.output_item.done":
                item = _native_value(_field(event, "item"))
                if not isinstance(item, dict):
                    raise ModelTransportError(
                        "response.output_item.done is missing a valid item",
                        attempts=1,
                        retryable=False,
                    )
                native_item = dict(item)
                if native_item.get("type") == "function_call":
                    key = _function_event_key(event, native_item)
                    # output_item.done is the authoritative completed item. Delta
                    # delivery may be coalesced or omitted by compatible endpoints.
                    if key:
                        self._function_arguments.pop(key, None)
                self._completed_items.append(native_item)
                return ModelStreamEvent(
                    type=ModelStreamEventType.OUTPUT_ITEM,
                    native_items=[native_item],
                    event_type=event_type,
                    event_metadata=metadata,
                )
            if event_type in {"response.completed", "response.incomplete"}:
                raw_response = _native_value(_field(event, "response"))
                if not isinstance(raw_response, dict):
                    raise ModelTransportError(
                        f"{event_type} is missing a valid response",
                        attempts=1,
                        retryable=False,
                    )
                expected_status = (
                    "completed" if event_type == "response.completed" else "incomplete"
                )
                reported_status = str(raw_response.get("status") or "").casefold()
                if reported_status and reported_status != expected_status:
                    raise ModelTransportError(
                        f"{event_type} carries conflicting status {reported_status!r}",
                        attempts=1,
                        retryable=False,
                    )
                raw_response["status"] = expected_status
                response = _merge_completed_output(
                    raw_response,
                    self._completed_items,
                )
                normalized = _model_response_from_responses(
                    response,
                    provider=self._provider,
                )
                text_suffix = _missing_stream_suffix(
                    normalized.text,
                    self._streamed_text,
                    field_name="text",
                )
                reasoning_suffix = _missing_stream_suffix(
                    normalized.reasoning_content or "",
                    self._streamed_reasoning,
                    field_name="reasoning",
                )
                self._finished = True
                continuation = None
                response_id = str(normalized.metadata.get("id") or "").strip()
                if self._request is not None and response_id:
                    prefix = self._full_input + list(normalized.native_items or [])
                    continuation = ModelContinuation(
                        run_id=self._request.run_id,
                        provider=self._request.provider,
                        model=self._request.model,
                        protocol=self._request.protocol,
                        response_id=response_id,
                        prefix_items=len(prefix),
                        prefix_digest=model_json_digest(prefix),
                        settings_digest=self._settings_digest,
                    )
                terminal_metadata = dict(normalized.metadata)
                terminal_metadata.update(
                    {
                        "continuation_applied": self._continuation_applied,
                        "continuation_reason": self._continuation_reason,
                    }
                )
                return ModelStreamEvent(
                    type=ModelStreamEventType.COMPLETED,
                    text=text_suffix,
                    reasoning_content=reasoning_suffix or None,
                    usage=normalized.usage,
                    tool_calls=normalized.tool_calls,
                    native_items=normalized.native_items,
                    event_type=event_type,
                    event_metadata=terminal_metadata,
                    finish_reason=normalized.finish_reason,
                    continuation=continuation,
                )
            if event_type in {"response.failed", "error"}:
                error = _field(event, "error") or _field(event, "response")
                if self._continuation_applied and _continuation_rejected(error):
                    raise ModelContinuationRejected(str(error)[:1000])
                self._finished = True
                return ModelStreamEvent(
                    type=ModelStreamEventType.FAILED,
                    event_type=event_type,
                    event_metadata=metadata,
                    error=f"model stream failed: {str(error)[:1000]}",
                )
            if event_type in {
                "response.created",
                "response.in_progress",
                "response.content_part.added",
                "response.content_part.done",
                "response.output_text.done",
                "response.refusal.done",
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_part.done",
                "response.reasoning_summary_text.done",
                "response.reasoning_text.done",
            }:
                return ModelStreamEvent(
                    type=ModelStreamEventType.LIFECYCLE,
                    event_type=event_type,
                    event_metadata=_event_response_metadata(event),
                )
            metadata["unrecognized_provider_event"] = True
            return ModelStreamEvent(
                type=ModelStreamEventType.LIFECYCLE,
                event_type=event_type,
                event_metadata=metadata,
            )

    async def aclose(self) -> None:
        self._finished = True
        await close_async_resource(self._events)


async def _open_responses_stream(
    adapter: Model,
    client: Any,
    request: ModelRequest,
    *,
    provider: str,
    request_kwargs: Dict[str, Any],
) -> AsyncIterator[ModelStreamEvent]:
    """Establish one Responses stream and return its owning iterator."""

    full_payload = _request_payload(
        adapter,
        request,
        request_kwargs,
        provider=provider,
    )
    raw_input = full_payload.get("input")
    full_input = (
        [dict(item) for item in raw_input if isinstance(item, dict)]
        if isinstance(raw_input, list)
        else []
    )
    settings_digest = model_json_digest(_continuation_settings(full_payload))
    payload, continuation_applied, continuation_reason = _apply_continuation(
        request,
        full_payload,
    )
    try:
        events = await _responses_create(client)(**payload)
    except Exception as exc:
        if continuation_applied and _continuation_rejected(exc):
            raise ModelContinuationRejected(str(exc)[:1000]) from exc
        raise
    return _ResponsesEventStream(
        events,
        provider=provider,
        request=request,
        full_input=full_input,
        settings_digest=settings_digest,
        continuation_applied=continuation_applied,
        continuation_reason=continuation_reason,
    )


__all__ = [
    "_model_response_from_responses",
    "_normalize_api_mode",
    "_open_responses_stream",
    "_to_responses_input",
    "_to_responses_tool_choice",
    "_to_responses_tools",
]
