"""Kimi managed public web-fetch capability."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

import requests

from .capability import WebFetchError, WebFetchResponse

_MAX_URL_CHARS = 2_048
_MAX_CONTENT_BYTES = 512 * 1024
_MAX_CONTENT_CHARS = 100_000
_BLOCKED_HOSTNAMES = frozenset({"localhost", "localhost.localdomain"})


class KimiWebFetchCapability:
    """Call the managed ``/fetch`` endpoint used by Kimi Code."""

    def __init__(
        self,
        *,
        api_key: str,
        fetch_url: str,
        timeout_seconds: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Kimi web fetch requires a non-empty API key")
        if not fetch_url.strip():
            raise ValueError("Kimi web fetch requires a non-empty URL")
        if timeout_seconds <= 0:
            raise ValueError("Kimi web fetch timeout must be positive")
        self._api_key = api_key
        self._fetch_url = fetch_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._session = session or requests.Session()

    def fetch(self, url: str) -> WebFetchResponse:
        normalized_url = _validate_public_url(url)
        try:
            response = self._session.post(
                self._fetch_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "text/markdown",
                    "Content-Type": "application/json",
                },
                json={"url": normalized_url},
                timeout=self._timeout_seconds,
                stream=True,
            )
        except requests.Timeout as exc:
            raise WebFetchError("timeout", "Kimi web fetch timed out") from exc
        except requests.RequestException as exc:
            raise WebFetchError("network", "Kimi web fetch request failed") from exc

        status_code = response.status_code
        if status_code != 200:
            response.close()
        if status_code in {401, 403}:
            raise WebFetchError(
                "authentication", "Kimi web fetch authentication failed"
            )
        if status_code == 402:
            raise WebFetchError("billing", "Kimi web fetch payment is required")
        if status_code == 429:
            raise WebFetchError("rate_limited", "Kimi web fetch was rate limited")
        if status_code != 200:
            raise WebFetchError(
                "provider",
                f"Kimi web fetch returned HTTP {status_code}",
            )

        content, truncated_bytes = _bounded_response_bytes(response)
        encoding = response.encoding or "utf-8"
        try:
            text = content.decode(encoding, errors="strict")
        except (LookupError, UnicodeDecodeError) as exc:
            raise WebFetchError(
                "protocol", "Kimi web fetch returned invalid text"
            ) from exc
        truncated_chars = len(text) > _MAX_CONTENT_CHARS
        if truncated_chars:
            text = text[:_MAX_CONTENT_CHARS]
        content_type = str(response.headers.get("Content-Type") or "text/markdown")
        return WebFetchResponse(
            url=normalized_url,
            content=text,
            content_type=content_type,
            truncated=truncated_bytes or truncated_chars,
        )


def _validate_public_url(url: str) -> str:
    if not isinstance(url, str):
        raise TypeError("web fetch URL must be a string")
    normalized = url.strip()
    if not normalized or len(normalized) > _MAX_URL_CHARS:
        raise ValueError(f"web fetch URL must contain 1 to {_MAX_URL_CHARS} characters")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("web fetch URL must use HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("web fetch URL must not contain credentials")
    hostname = parsed.hostname.rstrip(".").casefold()
    if (
        hostname in _BLOCKED_HOSTNAMES
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
        or hostname.endswith(".internal")
    ):
        raise PermissionError("web fetch accepts public URLs only")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise PermissionError("web fetch accepts public URLs only")
    return normalized


def _bounded_response_bytes(response: requests.Response) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    size = 0
    truncated = False
    try:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            remaining = _MAX_CONTENT_BYTES - size
            if remaining <= 0:
                truncated = True
                break
            chunks.append(chunk[:remaining])
            size += min(len(chunk), remaining)
            if len(chunk) > remaining:
                truncated = True
                break
    finally:
        response.close()
    return b"".join(chunks), truncated


__all__ = ["KimiWebFetchCapability"]
