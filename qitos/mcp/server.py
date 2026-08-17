"""Run-owned MCP clients backed by the official Python SDK."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any, NoReturn, TypeAlias

from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import (
    CallToolResult,
    Implementation,
    ServerNotification,
    Tool,
    ToolListChangedNotification,
)
from pydantic import ValidationError

from .. import __version__

if TYPE_CHECKING:
    from anyio.streams.memory import (
        MemoryObjectReceiveStream,
        MemoryObjectSendStream,
    )
    from mcp.shared.message import SessionMessage

    _ReadStream: TypeAlias = MemoryObjectReceiveStream[SessionMessage | Exception]
    _WriteStream: TypeAlias = MemoryObjectSendStream[SessionMessage]

_logger = logging.getLogger(__name__)
_MCPToolsChangedHandler = Callable[[], None]

_MAX_TOOL_CATALOG_PAGES = 100
_MAX_TOOL_CATALOG_ITEMS = 2_048
_MAX_PAGINATION_CURSOR_BYTES = 64 * 1024


class MCPRequestError(RuntimeError):
    """A failed MCP request with a stable QitOS classification."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "MCP_REQUEST_ERROR",
        error_category: str = "mcp_request_error",
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.error_category = error_category


class MCPServer(ABC):
    """Run-scoped MCP connection consumed by :class:`MCPRuntime`.

    Protocol payloads use the official ``mcp.types`` models directly. Concrete
    transports own their SDK session and must close it on the event loop that
    opened it.
    """

    _tools_changed_handler: _MCPToolsChangedHandler | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this connection."""

    @abstractmethod
    async def connect(self) -> None:
        """Open the transport and complete MCP initialization."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Close the session and every resource owned by the transport."""

    @abstractmethod
    async def list_tools(self) -> list[Tool]:
        """Return the bounded, complete remote Tool catalog."""

    @abstractmethod
    async def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> CallToolResult:
        """Invoke a remote Tool and return the official MCP result model."""

    def set_tools_changed_handler(
        self,
        handler: _MCPToolsChangedHandler | None,
    ) -> None:
        """Install the run owner's lightweight catalog invalidation hook."""

        self._tools_changed_handler = handler

    def notify_tools_changed(self) -> None:
        """Mark this server's catalog dirty at the next Agent turn safe point."""

        handler = self._tools_changed_handler
        if handler is not None:
            handler()


class _SDKMCPServer(MCPServer):
    """Shared official-SDK lifecycle for concrete MCP transports.

    The SDK's AnyIO context managers must exit in the same task that entered
    them. A dedicated owner task therefore holds the transport and
    ``ClientSession`` while callers use the initialized session concurrently.
    Cleanup stops admitting requests, waits for in-flight operations, and then
    asks that owner task to unwind its contexts.
    """

    def __init__(
        self,
        *,
        name: str,
        read_timeout_seconds: float | None = None,
    ) -> None:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("MCP server name must be non-empty")
        if read_timeout_seconds is not None and (
            isinstance(read_timeout_seconds, bool)
            or not isinstance(read_timeout_seconds, (int, float))
            or read_timeout_seconds <= 0
        ):
            raise ValueError("read_timeout_seconds must be positive or null")
        self._name = normalized_name
        self._read_timeout = (
            timedelta(seconds=float(read_timeout_seconds))
            if read_timeout_seconds is not None
            else None
        )
        self._session: ClientSession | None = None
        self._owner_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] | None = None
        self._close_requested: asyncio.Event | None = None
        self._request_condition = asyncio.Condition()
        self._active_requests = 0
        self._closing = False

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def _open_transport(
        self,
    ) -> AbstractAsyncContextManager[tuple[_ReadStream, _WriteStream]]:
        """Create a fresh official SDK transport context."""

    async def connect(self) -> None:
        if self._owner_task is not None:
            raise RuntimeError("MCP server is already connected")
        loop = asyncio.get_running_loop()
        ready = loop.create_future()
        close_requested = asyncio.Event()
        owner = asyncio.create_task(
            self._run_owner(ready, close_requested),
            name=f"qitos-mcp-owner-{self._name}",
        )
        self._ready = ready
        self._close_requested = close_requested
        self._owner_task = owner
        self._closing = False
        try:
            await asyncio.shield(ready)
        except asyncio.CancelledError:
            await self._abort_connect(owner)
            raise
        except BaseException:
            await asyncio.gather(owner, return_exceptions=True)
            self._reset_owner(owner)
            raise

    async def cleanup(self) -> None:
        owner = self._owner_task
        close_requested = self._close_requested
        if owner is None:
            return
        if close_requested is None:
            raise RuntimeError("MCP owner task has no close signal")
        async with self._request_condition:
            self._closing = True
        close_requested.set()
        try:
            await asyncio.shield(owner)
        except asyncio.CancelledError as cancellation:
            try:
                await _settle_task(owner)
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise
        finally:
            if owner.done():
                self._reset_owner(owner)

    async def list_tools(self) -> list[Tool]:
        try:
            async with self._session_scope() as session:
                return await _collect_tool_catalog(session)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            _raise_mcp_error(exc)

    async def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> CallToolResult:
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments must be a mapping")
        try:
            async with self._session_scope() as session:
                return await session.call_tool(tool_name, dict(arguments))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            _raise_mcp_error(exc)

    async def _run_owner(
        self,
        ready: asyncio.Future[None],
        close_requested: asyncio.Event,
    ) -> None:
        try:
            async with self._open_transport() as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=self._read_timeout,
                    message_handler=self._handle_message,
                    client_info=Implementation(name="qitos", version=__version__),
                ) as session:
                    try:
                        await session.initialize()
                    except BaseException as exc:
                        _raise_mcp_error(exc)
                    self._session = session
                    if not ready.done():
                        ready.set_result(None)
                    await close_requested.wait()
                    async with self._request_condition:
                        while self._active_requests:
                            await self._request_condition.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise
        finally:
            self._session = None

    async def _handle_message(
        self,
        message: ServerNotification | object,
    ) -> None:
        if isinstance(message, ServerNotification) and isinstance(
            message.root,
            ToolListChangedNotification,
        ):
            self.notify_tools_changed()
        elif isinstance(message, Exception):
            _logger.debug("MCP transport %s reported: %s", self._name, message)

    @asynccontextmanager
    async def _session_scope(self) -> AsyncIterator[ClientSession]:
        async with self._request_condition:
            session = self._session
            if session is None or self._closing:
                raise MCPRequestError(
                    "MCP server is not connected",
                    error_code="MCP_TRANSPORT_CLOSED",
                    error_category="mcp_transport_error",
                )
            self._active_requests += 1
        try:
            yield session
        finally:
            async with self._request_condition:
                self._active_requests -= 1
                self._request_condition.notify_all()

    async def _abort_connect(self, owner: asyncio.Task[None]) -> None:
        owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)
        self._reset_owner(owner)

    def _reset_owner(self, owner: asyncio.Task[None]) -> None:
        if self._owner_task is not owner:
            return
        self._owner_task = None
        self._ready = None
        self._close_requested = None
        self._session = None
        self._closing = False


async def _collect_tool_catalog(session: ClientSession) -> list[Tool]:
    catalog: list[Tool] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()

    for _ in range(_MAX_TOOL_CATALOG_PAGES):
        page = await session.list_tools(cursor=cursor)
        if len(page.tools) > _MAX_TOOL_CATALOG_ITEMS - len(catalog):
            raise MCPRequestError(
                f"MCP tools/list exceeded {_MAX_TOOL_CATALOG_ITEMS} tools",
                error_code="MCP_CATALOG_LIMIT",
                error_category="mcp_protocol_error",
            )
        catalog.extend(page.tools)
        next_cursor = page.nextCursor
        if next_cursor is None:
            return catalog
        if len(next_cursor.encode("utf-8")) > _MAX_PAGINATION_CURSOR_BYTES:
            raise MCPRequestError(
                "MCP tools/list pagination cursor is too large",
                error_code="MCP_CATALOG_LIMIT",
                error_category="mcp_protocol_error",
            )
        if next_cursor in seen_cursors:
            raise MCPRequestError(
                "MCP tools/list returned a repeated pagination cursor",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    raise MCPRequestError(
        f"MCP tools/list exceeded {_MAX_TOOL_CATALOG_PAGES} pages",
        error_code="MCP_CATALOG_LIMIT",
        error_category="mcp_protocol_error",
    )


def _raise_mcp_error(error: BaseException) -> NoReturn:
    if isinstance(error, MCPRequestError):
        raise error
    if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
        raise error
    if isinstance(error, McpError):
        code = error.error.code
        message = error.error.message
        if code == 408:
            raise asyncio.TimeoutError(message) from error
        if code == 32600 and message.lower() == "session terminated":
            raise MCPRequestError(
                message,
                error_code="MCP_SESSION_EXPIRED",
                error_category="mcp_transport_error",
            ) from error
        raise MCPRequestError(
            f"MCP JSON-RPC error (code={code}): {message}",
            error_code="MCP_REMOTE_ERROR",
        ) from error
    if isinstance(error, ValidationError):
        raise MCPRequestError(
            "MCP peer returned an invalid protocol message",
            error_code="MCP_PROTOCOL_ERROR",
            error_category="mcp_protocol_error",
        ) from error
    raise MCPRequestError(
        f"MCP transport failed: {error}",
        error_code="MCP_TRANSPORT_CLOSED",
        error_category="mcp_transport_error",
    ) from error


async def _settle_task(task: asyncio.Task[None]) -> None:
    """Wait through repeated caller cancellation for one owner task."""

    while True:
        try:
            await asyncio.shield(task)
            return
        except asyncio.CancelledError:
            if task.done():
                task.result()
                return


MCPServerFactory: TypeAlias = Callable[[], Sequence[MCPServer]]

__all__ = ["MCPRequestError", "MCPServer", "MCPServerFactory"]
