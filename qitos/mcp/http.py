"""MCP server transport over Streamable HTTP."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional

from .server import (
    MCPCallToolResult,
    MCPRequestError,
    MCPServer,
    MCPToolInfo,
    _collect_mcp_tool_catalog,
)

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2024-11-05"
_SESSION_ID_HEADER = "Mcp-Session-Id"
_PROTOCOL_VERSION_HEADER = "MCP-Protocol-Version"
_EVENT_STREAM_CONTENT_TYPE = "text/event-stream"
_MAX_SSE_EVENT_BYTES = 1024 * 1024
_EVENT_STREAM_RECONNECT_DELAY_SECONDS = 0.1
_DEFAULT_CANCEL_NOTIFICATION_TIMEOUT_SECONDS = 0.25

# Try to import httpx; raise a helpful error at usage time if missing.
try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


def _require_httpx() -> None:
    if httpx is None:
        raise ImportError(
            "The httpx package is required for MCPServerStreamableHttp. "
            "Install it with: pip install httpx"
        )


class MCPServerStreamableHttp(MCPServer):
    """Connect to an MCP server via the Streamable HTTP transport.

    JSON-RPC requests and notifications use HTTP POST. When a server advertises
    Tool-list change notifications, one owned GET event stream receives idle
    notifications until cleanup. Session and protocol headers negotiated by the
    initialization response apply to all subsequent requests.

    Usage::

        server = MCPServerStreamableHttp(
            url="http://localhost:8080/mcp",
            headers={"Authorization": "Bearer token123"},
        )
        await server.connect()
        tools = await server.list_tools()
        result = await server.call_tool("search", {"query": "hello"})
        await server.cleanup()
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        *,
        cancel_notification_timeout_seconds: float = (
            _DEFAULT_CANCEL_NOTIFICATION_TIMEOUT_SECONDS
        ),
    ) -> None:
        _require_httpx()
        if (
            isinstance(cancel_notification_timeout_seconds, bool)
            or not isinstance(cancel_notification_timeout_seconds, (int, float))
            or cancel_notification_timeout_seconds <= 0
        ):
            raise ValueError("cancel_notification_timeout_seconds must be positive")
        self._url = url.rstrip("/")
        self._headers = dict(headers or {})
        self._name = name or f"http:{self._url}"
        self._client: Optional[httpx.AsyncClient] = None
        self._request_id = 0
        self._request_lock = asyncio.Lock()
        self._session_id: str | None = None
        self._protocol_version: str | None = None
        self._listener_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._closing = False
        self._notification_last_event_id: str | None = None
        self._cancel_notification_timeout_seconds = float(
            cancel_notification_timeout_seconds
        )

    @property
    def name(self) -> str:
        return self._name

    # -- lifecycle ----------------------------------------------------------- #

    async def connect(self) -> None:
        """Open the HTTP client session and complete the MCP handshake."""
        _require_httpx()
        if self._client is not None:
            raise RuntimeError("MCP server is already connected")
        self._closing = False
        self._notification_last_event_id = None
        self._client = httpx.AsyncClient(
            base_url=self._url,
            headers={
                **self._headers,
                "Accept": f"application/json, {_EVENT_STREAM_CONTENT_TYPE}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )
        try:
            # MCP initialization handshake
            init_result = await self._send_request(
                "initialize",
                params={
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "qitos-mcp-client",
                        "version": "0.1.0",
                    },
                },
            )
            logger.debug("MCP HTTP initialize response: %s", init_result)

            # Send initialized notification
            await self._send_notification("notifications/initialized")
            if _supports_tool_list_changed(init_result):
                await self._start_notification_listener()
            logger.info("MCP HTTP handshake complete for %s", self._name)
        except BaseException:
            try:
                await self.cleanup()
            except BaseException as cleanup_error:
                logger.warning(
                    "MCP HTTP cleanup after handshake failure also failed",
                    exc_info=cleanup_error,
                )
            raise

    async def cleanup(self) -> None:
        """Wait for in-flight requests, terminate the session, and close HTTP."""
        client = self._client
        cleanup = self._cleanup_task
        if client is None and cleanup is None:
            return
        if cleanup is None:
            assert client is not None
            self._closing = True
            cleanup = asyncio.create_task(
                self._cleanup_client(client, listener=self._listener_task),
                name=f"qitos-mcp-http-cleanup-{self._name}",
            )
            self._cleanup_task = cleanup
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as cancellation:
            try:
                await _settle_cleanup_task(cleanup)
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise
        finally:
            if cleanup.done() and self._cleanup_task is cleanup:
                self._cleanup_task = None

    async def _cleanup_client(
        self,
        client: httpx.AsyncClient,
        *,
        listener: asyncio.Task[None] | None,
    ) -> None:
        """Close resources owned by one connection without racing POST work."""
        cancellation: asyncio.CancelledError | None = None
        cleanup_error: BaseException | None = None
        client_closed = False

        if listener is not None:
            if not listener.done():
                listener.cancel()
            try:
                await asyncio.gather(listener, return_exceptions=True)
            except asyncio.CancelledError as exc:
                cancellation = exc

        async with self._request_lock:
            if self._session_id is not None:
                try:
                    await self._terminate_session(client)
                except asyncio.CancelledError as exc:
                    if cancellation is None:
                        cancellation = exc
                except BaseException as exc:
                    cleanup_error = exc

            try:
                await client.aclose()
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                else:
                    logger.warning(
                        "MCP HTTP client close also failed for %s",
                        self._name,
                        exc_info=exc,
                    )
            else:
                client_closed = True
            finally:
                if client_closed and self._client is client:
                    self._client = None
                self._session_id = None
                self._protocol_version = None
                self._notification_last_event_id = None
                self._listener_task = None
                self._closing = False

        logger.info("MCP HTTP client closed for %s", self._name)
        if cancellation is not None:
            raise cancellation
        if cleanup_error is not None:
            raise cleanup_error

    # -- MCP operations ------------------------------------------------------ #

    async def list_tools(self) -> List[MCPToolInfo]:
        """Request the list of tools from the MCP server."""

        async def _fetch_page(params: dict[str, Any]) -> Dict[str, Any]:
            return await self._send_request("tools/list", params=params)

        return await _collect_mcp_tool_catalog(_fetch_page)

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> MCPCallToolResult:
        """Invoke a tool on the MCP server."""
        result = await self._send_request(
            "tools/call",
            params={"name": tool_name, "arguments": arguments},
        )
        return MCPCallToolResult.from_dict(result)

    # -- internal JSON-RPC transport ----------------------------------------- #

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send_request(
        self, method: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send a JSON-RPC request via HTTP POST and return the result."""
        request_id: int | None = None
        try:
            async with self._request_lock:
                client = self._client
                if client is None or self._closing:
                    raise MCPRequestError("MCP server is not connected")

                request_id = self._next_id()
                payload = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params,
                    "id": request_id,
                }
                logger.debug("MCP HTTP -> %s", json.dumps(payload))

                async with client.stream(
                    "POST",
                    "",
                    json=payload,
                ) as response:
                    self._raise_if_session_expired(response)
                    response.raise_for_status()
                    content_type = _response_header(response, "Content-Type")
                    if content_type is not None and content_type.lower().startswith(
                        _EVENT_STREAM_CONTENT_TYPE
                    ):
                        result = await self._read_event_stream_result(
                            response,
                            request_id=request_id,
                        )
                    elif content_type is None or content_type.lower().startswith(
                        "application/json"
                    ):
                        result = await self._read_json_result(
                            response,
                            request_id=request_id,
                        )
                    else:
                        raise MCPRequestError(
                            "MCP HTTP response must be application/json or "
                            "text/event-stream",
                            error_code="MCP_PROTOCOL_ERROR",
                            error_category="mcp_protocol_error",
                        )
                    if method == "initialize":
                        self._apply_initialize_metadata(response, result)
                    return result
        except asyncio.CancelledError:
            if request_id is not None:
                await self._notify_cancelled(request_id)
            raise
        except httpx.TimeoutException as exc:
            raise asyncio.TimeoutError(f"MCP HTTP request timed out: {method}") from exc
        except httpx.HTTPError as exc:
            raise MCPRequestError(f"MCP HTTP request failed: {method}: {exc}") from exc

    async def _read_json_result(
        self,
        response: httpx.Response,
        *,
        request_id: int,
    ) -> dict[str, Any]:
        try:
            data = json.loads(await response.aread())
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise MCPRequestError(
                "MCP HTTP response is not valid JSON",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            ) from exc
        logger.debug("MCP HTTP <- %s", json.dumps(data)[:500])
        messages = data if isinstance(data, list) else [data]
        matched: dict[str, Any] | None = None
        for raw_message in messages:
            message = self._validate_server_message(raw_message)
            is_match, result = self._process_server_message(
                message,
                request_id=request_id,
            )
            if not is_match:
                continue
            if matched is not None:
                raise MCPRequestError(
                    "MCP HTTP response contains duplicate matching ids",
                    error_code="MCP_PROTOCOL_ERROR",
                    error_category="mcp_protocol_error",
                )
            assert result is not None
            matched = result
        if matched is None:
            raise MCPRequestError(
                "MCP HTTP response did not match its request id",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        return matched

    async def _read_event_stream_result(
        self,
        response: httpx.Response,
        *,
        request_id: int,
    ) -> dict[str, Any]:
        async for raw_message in self._iter_event_data(response):
            message = self._parse_event_message(raw_message)
            is_match, result = self._process_server_message(
                message,
                request_id=request_id,
            )
            if is_match:
                assert result is not None
                return result
        raise MCPRequestError(
            "MCP HTTP event stream closed before the matching response",
            error_code="MCP_PROTOCOL_ERROR",
            error_category="mcp_protocol_error",
        )

    def _process_server_message(
        self,
        message: dict[str, Any],
        *,
        request_id: int | None,
    ) -> tuple[bool, dict[str, Any] | None]:
        method = message.get("method")
        if isinstance(method, str):
            if "id" not in message:
                if method == "notifications/tools/list_changed":
                    self.notify_tools_changed()
                else:
                    logger.debug("Ignoring MCP HTTP notification %s", method)
            else:
                logger.debug("Ignoring unsupported MCP HTTP server request %s", method)
            return False, None

        response_id = message.get("id")
        if isinstance(response_id, bool) or not isinstance(response_id, int):
            raise MCPRequestError(
                "MCP JSON-RPC response id must be an integer",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        if request_id is None or response_id != request_id:
            return False, None
        return True, self._response_result(message)

    @staticmethod
    def _response_result(message: dict[str, Any]) -> dict[str, Any]:
        if "error" in message:
            if "result" in message:
                raise MCPRequestError(
                    "MCP JSON-RPC response contains result and error",
                    error_code="MCP_PROTOCOL_ERROR",
                    error_category="mcp_protocol_error",
                )
            error = message["error"]
            if (
                not isinstance(error, dict)
                or isinstance(error.get("code"), bool)
                or not isinstance(error.get("code"), int)
                or not isinstance(error.get("message"), str)
            ):
                raise MCPRequestError(
                    "MCP JSON-RPC response contains invalid error",
                    error_code="MCP_PROTOCOL_ERROR",
                    error_category="mcp_protocol_error",
                )
            raise MCPRequestError(
                "MCP JSON-RPC error " f"(code={error['code']}): {error['message']}",
                error_code="MCP_REMOTE_ERROR",
            )
        if "result" not in message:
            raise MCPRequestError(
                "MCP JSON-RPC response is missing a result",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        result = message["result"]
        if not isinstance(result, dict):
            raise MCPRequestError(
                "MCP JSON-RPC result must be an object",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        return result

    async def _notify_cancelled(self, request_id: int) -> None:
        if self._closing:
            return
        try:
            await asyncio.wait_for(
                self._send_notification(
                    "notifications/cancelled",
                    params={
                        "requestId": request_id,
                        "reason": "request cancelled by client",
                    },
                ),
                timeout=self._cancel_notification_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "MCP HTTP cancellation notification failed for %s: %s",
                self._name,
                exc,
            )

    async def _send_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a JSON-RPC notification via HTTP POST."""
        async with self._request_lock:
            client = self._client
            if client is None or self._closing:
                raise RuntimeError("MCP server is not connected")

            payload: Dict[str, Any] = {
                "jsonrpc": "2.0",
                "method": method,
            }
            if params is not None:
                payload["params"] = params

            logger.debug("MCP HTTP -> (notification) %s", json.dumps(payload))
            try:
                response = await client.post("", json=payload)
                self._raise_if_session_expired(response)
                # Successful notifications may return 200, 202, or 204. HTTP
                # failures still mean the notification was not accepted.
                response.raise_for_status()
            except httpx.TimeoutException as exc:
                raise asyncio.TimeoutError(
                    f"MCP HTTP notification timed out: {method}"
                ) from exc
            except MCPRequestError:
                raise
            except httpx.HTTPError as exc:
                raise MCPRequestError(
                    f"MCP HTTP notification failed: {method}: {exc}"
                ) from exc

    def _apply_initialize_metadata(
        self,
        response: httpx.Response,
        result: dict[str, Any],
    ) -> None:
        protocol_version = result.get("protocolVersion")
        if protocol_version is not None and not isinstance(protocol_version, str):
            raise MCPRequestError(
                "MCP initialize protocolVersion must be a string",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        session_id = _response_header(response, _SESSION_ID_HEADER)
        if session_id is not None and not session_id.strip():
            raise MCPRequestError(
                "MCP initialize returned an empty session id",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )

        self._protocol_version = protocol_version
        self._session_id = session_id
        client = self._client
        if client is None:
            return
        if protocol_version is not None:
            client.headers[_PROTOCOL_VERSION_HEADER] = protocol_version
        if session_id is not None:
            client.headers[_SESSION_ID_HEADER] = session_id

    async def _start_notification_listener(self) -> None:
        if self._listener_task is not None:
            raise RuntimeError("MCP HTTP notification listener is already started")
        ready: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        listener = asyncio.create_task(
            self._listen_for_notifications(ready),
            name=f"qitos-mcp-http-listener-{self._name}",
        )
        self._listener_task = listener
        try:
            await ready
        except BaseException:
            if not listener.done():
                listener.cancel()
            await asyncio.gather(listener, return_exceptions=True)
            self._listener_task = None
            raise

    async def _listen_for_notifications(
        self,
        ready: asyncio.Future[bool],
    ) -> None:
        while True:
            client = self._client
            if client is None:
                if not ready.done():
                    ready.set_exception(MCPRequestError("MCP server is not connected"))
                return
            headers = (
                {"Last-Event-ID": self._notification_last_event_id}
                if self._notification_last_event_id is not None
                else None
            )
            try:
                async with client.stream(
                    "GET",
                    "",
                    timeout=None,
                    headers=headers,
                ) as response:
                    if response.status_code == 405:
                        logger.info(
                            "MCP HTTP server %s does not support a GET event stream",
                            self._name,
                        )
                        if not ready.done():
                            ready.set_result(False)
                        return
                    self._raise_if_session_expired(response)
                    try:
                        response.raise_for_status()
                    except httpx.HTTPError as exc:
                        raise MCPRequestError(
                            f"MCP HTTP event stream failed: {exc}"
                        ) from exc
                    content_type = _response_header(response, "Content-Type")
                    if content_type is None or not content_type.lower().startswith(
                        _EVENT_STREAM_CONTENT_TYPE
                    ):
                        raise MCPRequestError(
                            "MCP HTTP event stream response must be text/event-stream",
                            error_code="MCP_PROTOCOL_ERROR",
                            error_category="mcp_protocol_error",
                        )
                    if not ready.done():
                        ready.set_result(True)
                    async for raw_message in self._iter_event_data(
                        response,
                        track_event_id=True,
                    ):
                        message = self._parse_event_message(raw_message)
                        self._process_server_message(message, request_id=None)
                if self._client is None:
                    return
                self.notify_tools_changed()
                logger.info(
                    "MCP HTTP event stream closed for %s; reconnecting",
                    self._name,
                )
                await asyncio.sleep(_EVENT_STREAM_RECONNECT_DELAY_SECONDS)
            except asyncio.CancelledError:
                if not ready.done():
                    ready.cancel()
                raise
            except MCPRequestError as exc:
                if not ready.done():
                    ready.set_exception(exc)
                else:
                    self.notify_tools_changed()
                    logger.warning(
                        "MCP HTTP event stream failed for %s: %s",
                        self._name,
                        exc,
                    )
                return
            except httpx.TransportError as exc:
                if not ready.done():
                    ready.set_exception(
                        MCPRequestError(f"MCP HTTP event stream failed: {exc}")
                    )
                    return
                self.notify_tools_changed()
                logger.warning(
                    "MCP HTTP event stream disconnected for %s: %s; reconnecting",
                    self._name,
                    exc,
                )
                await asyncio.sleep(_EVENT_STREAM_RECONNECT_DELAY_SECONDS)

    async def _iter_event_data(
        self,
        response: httpx.Response,
        *,
        track_event_id: bool = False,
    ) -> AsyncIterator[str]:
        event_type: str | None = None
        data_lines: list[str] = []
        event_size = 0
        async for line in response.aiter_lines():
            encoded_size = len(line.encode("utf-8"))
            event_size += encoded_size + 1
            if event_size > _MAX_SSE_EVENT_BYTES:
                raise MCPRequestError(
                    "MCP HTTP event exceeded the size limit",
                    error_code="MCP_PROTOCOL_ERROR",
                    error_category="mcp_protocol_error",
                )
            if line == "":
                if data_lines and event_type in (None, "", "message"):
                    yield "\n".join(data_lines)
                event_type = None
                data_lines.clear()
                event_size = 0
                continue
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "event":
                event_type = value
            elif field == "id" and "\x00" not in value:
                if track_event_id:
                    self._notification_last_event_id = value or None
            elif field == "data":
                data_lines.append(value)

    def _parse_event_message(self, raw: str) -> dict[str, Any]:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MCPRequestError(
                "MCP HTTP event is not valid JSON",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            ) from exc
        return self._validate_server_message(message)

    @staticmethod
    def _validate_server_message(message: Any) -> dict[str, Any]:
        if not isinstance(message, dict):
            raise MCPRequestError(
                "MCP JSON-RPC message must be an object",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        if message.get("jsonrpc") != "2.0":
            raise MCPRequestError(
                "MCP JSON-RPC version must be '2.0'",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        return message

    async def _terminate_session(self, client: httpx.AsyncClient) -> None:
        try:
            response = await client.delete("")
            if response.status_code in (404, 405):
                return
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise asyncio.TimeoutError(
                "MCP HTTP session termination timed out"
            ) from exc
        except httpx.HTTPError as exc:
            raise MCPRequestError(
                f"MCP HTTP session termination failed: {exc}"
            ) from exc

    def _raise_if_session_expired(self, response: httpx.Response) -> None:
        if response.status_code != 404 or self._session_id is None:
            return
        self.notify_tools_changed()
        raise MCPRequestError(
            "MCP HTTP session expired with 404 Not Found",
            error_code="MCP_SESSION_EXPIRED",
            error_category="mcp_transport_error",
        )


def _supports_tool_list_changed(result: dict[str, Any]) -> bool:
    capabilities = result.get("capabilities")
    if not isinstance(capabilities, dict):
        return False
    tools = capabilities.get("tools")
    return isinstance(tools, dict) and tools.get("listChanged") is True


def _response_header(response: httpx.Response, name: str) -> str | None:
    headers = response.headers
    value = headers.get(name)
    if value is not None:
        return str(value)
    folded_name = name.casefold()
    for header_name, header_value in headers.items():
        if str(header_name).casefold() == folded_name:
            return str(header_value)
    return None


async def _settle_cleanup_task(task: asyncio.Task[None]) -> None:
    """Wait through repeated caller cancellation for owned HTTP cleanup."""

    while True:
        try:
            await asyncio.shield(task)
            return
        except asyncio.CancelledError:
            if task.done():
                task.result()
                return
