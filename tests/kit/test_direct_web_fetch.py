"""Direct (unmanaged) web-fetch capability tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from qitos.kit.fetch import (
    DirectWebFetchCapability,
    WebFetchError,
    build_web_fetch_capability,
)


@dataclass
class _Response:
    status_code: int
    body: bytes = b""
    headers: dict[str, str] = field(
        default_factory=lambda: {"Content-Type": "text/html"}
    )
    encoding: str | None = "utf-8"
    closed: bool = False

    async def aiter_bytes(self, chunk_size: int):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset : offset + chunk_size]


class _Stream:
    def __init__(self, response: _Response) -> None:
        self._response = response

    async def __aenter__(self) -> _Response:
        return self._response

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        _ = exc_type, exc, traceback
        self._response.closed = True


@dataclass
class _Client:
    response: _Response | None = None
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def stream(self, method: str, url: str, **kwargs: Any) -> _Stream:
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return _Stream(self.response)


def test_factory_builds_direct_without_credentials() -> None:
    capability = build_web_fetch_capability(
        provider="direct",
        api_key="",
        fetch_url=None,
    )
    assert isinstance(capability, DirectWebFetchCapability)


@pytest.mark.asyncio
async def test_direct_fetch_returns_bounded_content() -> None:
    response = _Response(200, b"<html>advisory body</html>")
    client = _Client(response)
    capability = DirectWebFetchCapability(client=client)

    result = await capability.fetch("https://vendor.example/advisory")

    assert result.content == "<html>advisory body</html>"
    assert result.url == "https://vendor.example/advisory"
    assert result.content_type == "text/html"
    assert result.truncated is False
    assert response.closed is True
    assert client.calls[0]["method"] == "GET"
    assert client.calls[0]["url"] == "https://vendor.example/advisory"


@pytest.mark.asyncio
async def test_direct_fetch_rejects_non_public_urls() -> None:
    capability = DirectWebFetchCapability(client=_Client(_Response(200)))
    for url in (
        "http://127.0.0.1/internal",
        "http://localhost/status",
        "http://192.168.0.10/admin",
        "file:///etc/passwd",
        "https://user:pw@vendor.example/x",
    ):
        with pytest.raises((PermissionError, ValueError)):
            await capability.fetch(url)


@pytest.mark.asyncio
async def test_direct_fetch_maps_http_and_network_failures() -> None:
    capability = DirectWebFetchCapability(client=_Client(_Response(404)))
    with pytest.raises(WebFetchError) as http_error:
        await capability.fetch("https://vendor.example/missing")
    assert http_error.value.kind == "provider"

    import httpx

    failing = DirectWebFetchCapability(
        client=_Client(error=httpx.ConnectError("unreachable")),
    )
    with pytest.raises(WebFetchError) as network_error:
        await failing.fetch("https://vendor.example/down")
    assert network_error.value.kind == "network"


@pytest.mark.asyncio
async def test_direct_fetch_marks_truncated_char_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qitos.kit.fetch.direct as direct_module

    monkeypatch.setattr(direct_module, "_MAX_CONTENT_CHARS", 10)
    capability = DirectWebFetchCapability(
        client=_Client(_Response(200, b"x" * 42)),
    )

    result = await capability.fetch("https://vendor.example/long")

    assert result.content == "x" * 10
    assert result.truncated is True
