"""Cache complete model stream transactions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional

from ..core.errors import ModelTransportError
from ..core.model_capabilities import ModelCapabilities
from ..core.model_request import ModelRequest
from ..core.model_stream import ModelStreamEventType
from ..core.model_response import ModelUsage, ModelUsageSource
from ..models.base import Model, ModelStreamEvent
from .backends import CacheBackend

_logger = logging.getLogger(__name__)


class CachedModel(Model):
    """Cache only complete provider-neutral model transactions.

    Cache serialization lives here because this is the sole durable owner of
    cached model chunks. Provider SDK objects are forbidden at the Model
    boundary and are never pickled.
    """

    def __init__(
        self,
        wrapped: Model,
        backend: CacheBackend,
        enabled: bool = True,
        ttl: Optional[float] = None,
    ) -> None:
        super().__init__(
            model=wrapped.model,
            system_prompt=wrapped.system_prompt,
            temperature=wrapped.temperature,
            max_tokens=wrapped.max_tokens,
            context_window=wrapped.context_window,
        )
        self._wrapped = wrapped
        self._backend = backend
        self._enabled = enabled
        self._ttl = ttl
        self.provider_name = wrapped.provider_name
        self._hits = 0
        self._misses = 0

    def _cache_key(
        self,
        messages: List[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> str:
        canonical = json.dumps(
            {
                "provider": self._wrapped.provider_name,
                "model": self._wrapped.model,
                "messages": messages,
                "kwargs": _json_safe(kwargs),
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Yield a cached complete stream or commit one successful miss."""

        self.validate_request(request)

        if not self._enabled:
            async for chunk in self._wrapped.stream(request):
                yield chunk
            return

        key = self._cache_key(request.message_dicts(), request.option_dict())
        cached = await asyncio.to_thread(self._backend.get, key)
        if cached is not None:
            try:
                cached_chunks = _decode_chunks(cached)
            except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
                try:
                    await asyncio.to_thread(self._backend.delete, key)
                except OSError:
                    _logger.debug(
                        "invalid model cache entry could not be deleted", exc_info=True
                    )
            else:
                self._hits += 1
                for chunk in cached_chunks:
                    yield chunk
                return

        self._misses += 1
        committed_chunks: List[ModelStreamEvent] = []
        async for chunk in self._wrapped.stream(request):
            committed_chunks.append(chunk)
        _validate_complete_chunks(committed_chunks)
        try:
            encoded = json.dumps(
                [_encode_chunk(chunk) for chunk in committed_chunks],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError):
            _logger.debug("model stream is not JSON-cacheable", exc_info=True)
        else:
            try:
                await asyncio.to_thread(self._backend.set, key, encoded, self._ttl)
            except OSError:
                _logger.debug("model cache write failed", exc_info=True)
        for chunk in committed_chunks:
            yield chunk

    async def close(self) -> None:
        await self._wrapped.close()

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._wrapped.capabilities

    def supports_tool_schema_delivery(
        self, delivery: str, protocol: Any = None
    ) -> bool:
        return self._wrapped.supports_tool_schema_delivery(delivery, protocol)

    def build_tool_schema_request_options(
        self,
        tool_schema_payload: Optional[List[Dict[str, Any]]],
        *,
        protocol: Any = None,
        delivery: str = "prompt_injection",
    ) -> Dict[str, Any]:
        return self._wrapped.build_tool_schema_request_options(
            tool_schema_payload,
            protocol=protocol,
            delivery=delivery,
        )

    def supports_multimodal_input(self) -> bool:
        return self._wrapped.supports_multimodal_input()

    def count_tokens(self, messages_or_text: Any) -> Optional[int]:
        return self._wrapped.count_tokens(messages_or_text)

    def count_request_tokens(
        self,
        messages: List[Dict[str, Any]],
        request_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        return self._wrapped.count_request_tokens(messages, request_options)

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}


def _validate_complete_chunks(chunks: List[ModelStreamEvent]) -> None:
    terminal_indexes = [
        index
        for index, chunk in enumerate(chunks)
        if chunk.type is ModelStreamEventType.COMPLETED
    ]
    if terminal_indexes != [len(chunks) - 1]:
        raise ModelTransportError(
            "cached model transaction must end with exactly one terminal chunk",
            attempts=1,
            retryable=False,
        )


def _decode_chunks(payload: bytes) -> List[ModelStreamEvent]:
    raw = json.loads(payload.decode("utf-8"))
    if not isinstance(raw, list):
        raise TypeError("cached model transaction must be a list")
    chunks = [_decode_chunk(item) for item in raw if isinstance(item, dict)]
    if len(chunks) != len(raw):
        raise TypeError("cached model transaction contains a non-object chunk")
    _validate_complete_chunks(chunks)
    return chunks


def _encode_chunk(chunk: ModelStreamEvent) -> Dict[str, Any]:
    usage = chunk.usage
    return {
        "type": chunk.type.value,
        "text": chunk.text,
        "usage": usage.to_dict() if isinstance(usage, ModelUsage) else usage,
        "usage_source": usage.source.value if isinstance(usage, ModelUsage) else None,
        "tool_calls": chunk.tool_calls,
        "native_items": chunk.native_items,
        "event_type": chunk.event_type,
        "event_metadata": chunk.event_metadata,
        "reasoning_content": chunk.reasoning_content,
        "finish_reason": chunk.finish_reason,
        "error": chunk.error,
    }


def _decode_chunk(item: Dict[str, Any]) -> ModelStreamEvent:
    values = dict(item)
    try:
        values["type"] = ModelStreamEventType(str(values.get("type") or ""))
    except ValueError as exc:
        raise ValueError("cached model stream event type is invalid") from exc
    source_value = values.pop("usage_source", None)
    usage = values.get("usage")
    if isinstance(usage, dict) and source_value is not None:
        try:
            source = ModelUsageSource(str(source_value))
        except ValueError as exc:
            raise ValueError("cached model usage source is invalid") from exc
        values["usage"] = ModelUsage.from_mapping(usage, source=source)
    return ModelStreamEvent(**values)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


__all__ = ["CachedModel"]
