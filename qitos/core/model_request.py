"""Immutable provider request and continuation contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List

from .thinking import ThinkingLevel

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "headers",
        "password",
        "proxy-authorization",
        "secret",
        "token",
    }
)


def _freeze_json(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _redact_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).strip().casefold().replace("_", "-")
            if normalized in {item.replace("_", "-") for item in _SENSITIVE_KEYS}:
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = _redact_json(item)
        return redacted
    if isinstance(value, tuple):
        return [_redact_json(item) for item in value]
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _redact_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelContinuation:
    """Opaque Provider optimization guarded by canonical-prefix evidence."""

    run_id: str
    provider: str
    model: str
    protocol: str
    response_id: str
    prefix_items: int
    prefix_digest: str
    settings_digest: str

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "provider",
            "model",
            "protocol",
            "response_id",
            "prefix_digest",
            "settings_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.prefix_items, int) or isinstance(
            self.prefix_items, bool
        ):
            raise TypeError("prefix_items must be an integer")
        if self.prefix_items < 0:
            raise ValueError("prefix_items must be non-negative")

    def belongs_to(self, request: "ModelRequest") -> bool:
        """Return whether this handle belongs to the request's stable identity."""

        return (
            self.run_id == request.run_id
            and self.provider == request.provider
            and self.model == request.model
            and self.protocol == request.protocol
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "protocol": self.protocol,
            "response_id": self.response_id,
            "prefix_items": self.prefix_items,
            "prefix_digest": self.prefix_digest,
            "settings_digest": self.settings_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelContinuation":
        return cls(
            run_id=str(value.get("run_id") or ""),
            provider=str(value.get("provider") or ""),
            model=str(value.get("model") or ""),
            protocol=str(value.get("protocol") or ""),
            response_id=str(value.get("response_id") or ""),
            prefix_items=int(value.get("prefix_items") or 0),
            prefix_digest=str(value.get("prefix_digest") or ""),
            settings_digest=str(value.get("settings_digest") or ""),
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Exact immutable input for one logical model transaction.

    Provider authentication remains owned by the configured ``Model``. The
    durable projection redacts request-option fields that could contain
    credentials; messages, tool schemas, model settings, and continuation
    identity remain reproducible.
    """

    run_id: str
    transaction_id: str
    provider: str
    model: str
    protocol: str
    messages: tuple[Mapping[str, Any], ...]
    options: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    deadline_monotonic: float | None = field(default=None, compare=False, repr=False)
    continuation: ModelContinuation | None = None
    thinking_level: ThinkingLevel | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "transaction_id", "provider", "model", "protocol"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.deadline_monotonic is not None:
            if isinstance(self.deadline_monotonic, bool) or not isinstance(
                self.deadline_monotonic, (int, float)
            ):
                raise TypeError("deadline_monotonic must be numeric or None")
            if not math.isfinite(float(self.deadline_monotonic)):
                raise ValueError("deadline_monotonic must be finite")
        if self.thinking_level is not None and not isinstance(
            self.thinking_level, ThinkingLevel
        ):
            raise TypeError("thinking_level must be a ThinkingLevel or None")
        frozen_messages: List[Mapping[str, Any]] = []
        for index, message in enumerate(self.messages):
            frozen = _freeze_json(message, path=f"messages[{index}]")
            if not isinstance(frozen, Mapping):
                raise TypeError("model messages must be mappings")
            frozen_messages.append(frozen)
        frozen_options = _freeze_json(self.options, path="options")
        if not isinstance(frozen_options, Mapping):
            raise TypeError("model request options must be a mapping")
        object.__setattr__(self, "messages", tuple(frozen_messages))
        object.__setattr__(self, "options", frozen_options)
        if self.continuation is not None and not isinstance(
            self.continuation, ModelContinuation
        ):
            raise TypeError("continuation must be a ModelContinuation or None")

    @property
    def request_digest(self) -> str:
        """Digest the stable, credential-redacted semantic request."""

        return _digest(
            {
                "provider": self.provider,
                "model": self.model,
                "protocol": self.protocol,
                "messages": self.messages,
                "options": self.options,
                "thinking_level": (
                    self.thinking_level.value
                    if self.thinking_level is not None
                    else None
                ),
            }
        )

    @property
    def cache_affinity(self) -> str:
        """Return the stable Run identity used by Provider prompt caches."""

        return self.run_id

    def message_dicts(self) -> List[Dict[str, Any]]:
        return [dict(_thaw_json(message)) for message in self.messages]

    def option_dict(self) -> Dict[str, Any]:
        return dict(_thaw_json(self.options))

    def without_continuation(self) -> "ModelRequest":
        if self.continuation is None:
            return self
        return ModelRequest(
            run_id=self.run_id,
            transaction_id=self.transaction_id,
            provider=self.provider,
            model=self.model,
            protocol=self.protocol,
            messages=self.messages,
            options=self.options,
            deadline_monotonic=self.deadline_monotonic,
            thinking_level=self.thinking_level,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the durable, credential-redacted request snapshot."""

        return {
            "run_id": self.run_id,
            "transaction_id": self.transaction_id,
            "provider": self.provider,
            "model": self.model,
            "protocol": self.protocol,
            "cache_affinity": self.cache_affinity,
            "messages": _redact_json(self.messages),
            "options": _redact_json(self.options),
            "thinking_level": (
                self.thinking_level.value
                if self.thinking_level is not None
                else None
            ),
            "request_digest": self.request_digest,
            "continuation": (
                self.continuation.to_dict()
                if self.continuation is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        deadline_monotonic: float | None = None,
    ) -> "ModelRequest":
        raw_messages = value.get("messages")
        if not isinstance(raw_messages, list) or not all(
            isinstance(message, Mapping) for message in raw_messages
        ):
            raise TypeError("persisted model request messages must be a list of mappings")
        raw_options = value.get("options")
        if not isinstance(raw_options, Mapping):
            raise TypeError("persisted model request options must be a mapping")
        raw_continuation = value.get("continuation")
        continuation = (
            ModelContinuation.from_dict(raw_continuation)
            if isinstance(raw_continuation, Mapping)
            else None
        )
        raw_thinking_level = value.get("thinking_level")
        thinking_level: ThinkingLevel | None = None
        if raw_thinking_level is not None:
            if not isinstance(raw_thinking_level, str):
                raise ValueError("persisted thinking_level must be text or null")
            try:
                thinking_level = ThinkingLevel(raw_thinking_level)
            except ValueError as exc:
                raise ValueError(
                    "persisted thinking_level is not a ThinkingLevel"
                ) from exc
        request = cls(
            run_id=str(value.get("run_id") or ""),
            transaction_id=str(value.get("transaction_id") or ""),
            provider=str(value.get("provider") or ""),
            model=str(value.get("model") or ""),
            protocol=str(value.get("protocol") or ""),
            messages=tuple(dict(message) for message in raw_messages),
            options=dict(raw_options),
            deadline_monotonic=deadline_monotonic,
            continuation=continuation,
            thinking_level=thinking_level,
        )
        expected = value.get("request_digest")
        if expected is not None and str(expected) != request.request_digest:
            raise ValueError("persisted model request digest does not match")
        return request


def model_json_digest(value: Any) -> str:
    """Return the stable digest used by Provider continuation validation."""

    return _digest(_freeze_json(value, path="value"))


__all__ = ["ModelContinuation", "ModelRequest", "model_json_digest"]
