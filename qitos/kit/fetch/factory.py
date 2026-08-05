"""Factory for provider-managed public web-fetch capabilities."""

from __future__ import annotations

from collections.abc import Callable

from .capability import WebFetchCapability
from .kimi import KimiWebFetchCapability

CapabilityBuilder = Callable[[str, str | None, float], WebFetchCapability | None]


def _build_kimi(
    api_key: str,
    fetch_url: str | None,
    timeout_seconds: float,
) -> WebFetchCapability | None:
    if fetch_url is None:
        return None
    return KimiWebFetchCapability(
        api_key=api_key,
        fetch_url=fetch_url,
        timeout_seconds=timeout_seconds,
    )


_BUILDERS: dict[str, CapabilityBuilder] = {"kimi": _build_kimi}


def build_web_fetch_capability(
    *,
    provider: str,
    api_key: str,
    fetch_url: str | None = None,
    timeout_seconds: float = 30.0,
) -> WebFetchCapability | None:
    """Build a managed fetch capability when the provider is supported."""

    builder = _BUILDERS.get(provider.strip().lower())
    if builder is None:
        return None
    return builder(api_key, fetch_url, timeout_seconds)


__all__ = ["build_web_fetch_capability"]
