"""Completed provider-neutral model transaction used by the Engine."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Optional, cast


class ModelUsageSource(str, Enum):
    """Origin of token counts carried by a model transaction."""

    PROVIDER = "provider"
    ESTIMATE = "estimate"


def _token_count(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be a non-negative integer or None")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _nested_value(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_token_count(
    value: Mapping[str, Any],
    *paths: tuple[str, ...],
    field_name: str,
) -> int | None:
    for path in paths:
        candidate = _nested_value(value, *path)
        if candidate is not None:
            return _token_count(candidate, field_name=field_name)
    return None


def _freeze_usage_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_usage_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_usage_value(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError("model usage details must contain JSON-compatible values")


def _thaw_usage_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_usage_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_usage_value(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True, slots=True, eq=False)
class ModelUsage(Mapping[str, Any]):
    """Typed token usage with a lossless provider-detail projection.

    The mapping interface preserves compatibility with existing consumers of
    ``prompt_tokens``/``completion_tokens`` dictionaries. Typed fields give
    runtimes stable semantics without discarding provider-specific cache or
    reasoning detail.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    source: ModelUsageSource = ModelUsageSource.PROVIDER
    _details: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
        ):
            _token_count(getattr(self, name), field_name=name)
        if not isinstance(self.source, ModelUsageSource):
            raise TypeError("source must be a ModelUsageSource")
        if not isinstance(self._details, Mapping):
            raise TypeError("model usage details must be a mapping")
        details = dict(self._details) or self._canonical_projection()
        object.__setattr__(self, "_details", _freeze_usage_value(details))

    @classmethod
    def from_mapping(
        cls,
        usage: Mapping[str, Any],
        *,
        source: ModelUsageSource = ModelUsageSource.PROVIDER,
    ) -> ModelUsage:
        """Normalize known token fields while retaining the complete mapping."""

        if not isinstance(usage, Mapping):
            raise TypeError("usage must be a mapping")
        return cls(
            input_tokens=_first_token_count(
                usage,
                ("prompt_tokens",),
                ("input_tokens",),
                field_name="input_tokens",
            ),
            output_tokens=_first_token_count(
                usage,
                ("completion_tokens",),
                ("output_tokens",),
                field_name="output_tokens",
            ),
            total_tokens=_first_token_count(
                usage, ("total_tokens",), field_name="total_tokens"
            ),
            cache_read_tokens=_first_token_count(
                usage,
                ("cached_tokens",),
                ("cache_read_input_tokens",),
                ("input_tokens_details", "cached_tokens"),
                field_name="cache_read_tokens",
            ),
            cache_write_tokens=_first_token_count(
                usage,
                ("cache_write_tokens",),
                ("cache_creation_input_tokens",),
                field_name="cache_write_tokens",
            ),
            reasoning_tokens=_first_token_count(
                usage,
                ("reasoning_tokens",),
                ("output_tokens_details", "reasoning_tokens"),
                field_name="reasoning_tokens",
            ),
            source=source,
            _details=usage,
        )

    def _canonical_projection(self) -> Dict[str, Any]:
        values = {
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }
        return {key: value for key, value in values.items() if value is not None}

    def __getitem__(self, key: str) -> Any:
        return _thaw_usage_value(self._details[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._details)

    def __len__(self) -> int:
        return len(self._details)

    def to_dict(self) -> Dict[str, Any]:
        """Return a mutable copy of the lossless provider projection."""

        return {key: _thaw_usage_value(value) for key, value in self._details.items()}


def normalize_model_usage(
    usage: ModelUsage | Mapping[str, Any] | None,
) -> ModelUsage | None:
    """Normalize a compatible usage mapping into the canonical typed value."""

    if usage is None or isinstance(usage, ModelUsage):
        return usage
    if not isinstance(usage, Mapping):
        raise TypeError("model usage must be a mapping or None")
    return ModelUsage.from_mapping(usage)


def _sanitize(value: Any) -> Any:
    if value is not None and dataclasses.is_dataclass(value):
        return {str(k): _sanitize(v) for k, v in asdict(cast(Any, value)).items()}
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sanitize_native_item(value: Any) -> Any:
    """Sanitize provider items while omitting opaque reasoning continuation data."""
    if value is not None and dataclasses.is_dataclass(value):
        value = asdict(cast(Any, value))
    if isinstance(value, dict):
        return {
            str(key): _sanitize_native_item(item)
            for key, item in value.items()
            if str(key) != "encrypted_content"
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_native_item(item) for item in value]
    return _sanitize(value)


@dataclass(slots=True)
class ModelResponse:
    text: str
    usage: ModelUsage | Mapping[str, Any] | None = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    model_name: Optional[str] = None
    provider: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning_content: Optional[str] = None
    native_items: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        """Reject responses that still carry provider SDK objects."""

        if not isinstance(self.text, str):
            raise TypeError("ModelResponse.text must be a string")
        self.usage = normalize_model_usage(self.usage)
        if self.tool_calls is not None and not isinstance(self.tool_calls, list):
            raise TypeError("ModelResponse.tool_calls must be a list or None")
        if self.native_items is not None and not isinstance(self.native_items, list):
            raise TypeError("ModelResponse.native_items must be a list or None")

    def to_summary_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "text": str(self.text or ""),
            "usage": (
                self.usage.to_dict() if isinstance(self.usage, ModelUsage) else None
            ),
            "usage_source": (
                self.usage.source.value if isinstance(self.usage, ModelUsage) else None
            ),
            "finish_reason": (
                str(self.finish_reason) if self.finish_reason is not None else None
            ),
            "tool_calls": (
                _sanitize(self.tool_calls)
                if isinstance(self.tool_calls, list)
                else None
            ),
            "model_name": str(self.model_name) if self.model_name is not None else None,
            "provider": str(self.provider) if self.provider is not None else None,
            "metadata": (
                _sanitize(self.metadata) if isinstance(self.metadata, dict) else {}
            ),
            "native_items": (
                _sanitize_native_item(self.native_items)
                if isinstance(self.native_items, list)
                else None
            ),
        }
        if self.reasoning_content:
            d["reasoning_content"] = self.reasoning_content
        return d


__all__ = [
    "ModelResponse",
    "ModelUsage",
    "ModelUsageSource",
    "normalize_model_usage",
]
