"""Factory for provider-managed web-search capabilities."""

from __future__ import annotations

from collections.abc import Callable

from .capability import WebSearchCapability
from .kimi import (
    DEFAULT_KIMI_BASE_URL,
    DEFAULT_KIMI_MODEL,
    KimiBuiltinWebSearchCapability,
    KimiWebSearchCapability,
)
from .qwen import DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL, QwenWebSearchCapability

CapabilityBuilder = Callable[
    [str, str | None, str | None, str | None, float],
    WebSearchCapability,
]


def _build_kimi(
    api_key: str,
    search_url: str | None,
    base_url: str | None,
    model: str | None,
    timeout_seconds: float,
) -> WebSearchCapability:
    if search_url is not None:
        return KimiWebSearchCapability(
            api_key=api_key,
            search_url=search_url,
            timeout_seconds=timeout_seconds,
        )
    normalized_base = (base_url or DEFAULT_KIMI_BASE_URL).rstrip("/")
    return KimiBuiltinWebSearchCapability(
        api_key=api_key,
        base_url=normalized_base,
        model=model or DEFAULT_KIMI_MODEL,
        timeout_seconds=timeout_seconds,
    )


def _build_qwen(
    api_key: str,
    search_url: str | None,
    base_url: str | None,
    model: str | None,
    timeout_seconds: float,
) -> WebSearchCapability:
    if search_url is not None:
        raise ValueError("Qwen web search does not accept a separate search URL")
    return QwenWebSearchCapability(
        api_key=api_key,
        base_url=(base_url or DEFAULT_QWEN_BASE_URL).rstrip("/"),
        model=model or DEFAULT_QWEN_MODEL,
        timeout_seconds=timeout_seconds,
    )


_BUILDERS: dict[str, CapabilityBuilder] = {
    "kimi": _build_kimi,
    "qwen": _build_qwen,
}


def build_web_search_capability(
    *,
    provider: str,
    api_key: str,
    search_url: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 30.0,
) -> WebSearchCapability | None:
    """Build a managed search capability when the provider is supported."""

    builder = _BUILDERS.get(provider.strip().lower())
    if builder is None:
        return None
    return builder(api_key, search_url, base_url, model, timeout_seconds)


__all__ = ["build_web_search_capability"]
