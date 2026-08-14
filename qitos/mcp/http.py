"""Official MCP Streamable HTTP transport integration."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from mcp.client.streamable_http import streamable_http_client

from .server import _SDKMCPServer

if TYPE_CHECKING:
    from .server import _ReadStream, _WriteStream


class MCPServerStreamableHttp(_SDKMCPServer):
    """Connect to an endpoint through the official Streamable HTTP client."""

    def __init__(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        name: str | None = None,
        *,
        read_timeout_seconds: float | None = None,
        request_timeout_seconds: float = 30.0,
        terminate_on_close: bool = True,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string")
        normalized_headers = dict(headers or {})
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in normalized_headers.items()
        ):
            raise TypeError("headers must map strings to strings")
        if (
            isinstance(request_timeout_seconds, bool)
            or not isinstance(request_timeout_seconds, (int, float))
            or request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be positive")
        if not isinstance(terminate_on_close, bool):
            raise TypeError("terminate_on_close must be a boolean")
        normalized_url = url.rstrip("/")
        super().__init__(
            name=name or f"http:{normalized_url}",
            read_timeout_seconds=read_timeout_seconds,
        )
        self._url = normalized_url
        self._headers = normalized_headers
        self._request_timeout_seconds = float(request_timeout_seconds)
        self._terminate_on_close = terminate_on_close

    def _open_transport(
        self,
    ) -> AbstractAsyncContextManager[tuple[_ReadStream, _WriteStream]]:
        return self._open_http_transport()

    @asynccontextmanager
    async def _open_http_transport(
        self,
    ) -> AsyncIterator[tuple[_ReadStream, _WriteStream]]:
        async with httpx.AsyncClient(
            headers=self._headers,
            timeout=httpx.Timeout(self._request_timeout_seconds),
        ) as client:
            async with streamable_http_client(
                self._url,
                http_client=client,
                terminate_on_close=self._terminate_on_close,
            ) as (read_stream, write_stream, _get_session_id):
                yield read_stream, write_stream


__all__ = ["MCPServerStreamableHttp"]
