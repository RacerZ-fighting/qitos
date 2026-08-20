"""Direct public web fetch through the runtime's own egress."""

from __future__ import annotations

from typing import Any

import httpx

from .capability import WebFetchError, WebFetchResponse
from .kimi import _MAX_CONTENT_CHARS, _bounded_response_bytes, _validate_public_url


class DirectWebFetchCapability:
    """Fetch one public URL directly, honoring the runtime's proxy environment.

    No provider account is involved: the request leaves through the runtime's
    own network stack, so ``httpx`` picks up HTTP(S)_PROXY/ALL_PROXY from the
    process environment. The same public-URL validation and size bounds as the
    managed provider apply.
    """

    def __init__(self, *, timeout_seconds: float = 30.0, client: Any = None) -> None:
        if timeout_seconds <= 0:
            raise ValueError("direct web fetch timeout must be positive")
        self._timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            trust_env=True,
            follow_redirects=True,
        )

    async def fetch(self, url: str) -> WebFetchResponse:
        normalized_url = _validate_public_url(url)
        try:
            async with self._client.stream(
                "GET",
                normalized_url,
                timeout=self._timeout_seconds,
            ) as response:
                status_code = response.status_code
                if status_code != 200:
                    raise WebFetchError(
                        "provider",
                        f"direct web fetch returned HTTP {status_code}",
                    )
                content, truncated_bytes = await _bounded_response_bytes(response)
                encoding = response.encoding or "utf-8"
                try:
                    text = content.decode(encoding, errors="replace")
                except LookupError:
                    text = content.decode("utf-8", errors="replace")
                truncated_chars = len(text) > _MAX_CONTENT_CHARS
                if truncated_chars:
                    text = text[:_MAX_CONTENT_CHARS]
                content_type = str(response.headers.get("Content-Type") or "")
        except httpx.TimeoutException as exc:
            raise WebFetchError("timeout", "direct web fetch timed out") from exc
        except httpx.RequestError as exc:
            raise WebFetchError("network", "direct web fetch request failed") from exc
        return WebFetchResponse(
            url=normalized_url,
            content=text,
            content_type=content_type,
            truncated=truncated_bytes or truncated_chars,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


__all__ = ["DirectWebFetchCapability"]
