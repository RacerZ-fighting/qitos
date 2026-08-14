"""MCP server transport over stdio (subprocess with JSON-RPC)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from typing import Any, Dict, List, Optional

from .server import (
    MCPCallToolResult,
    MCPRequestError,
    MCPServer,
    MCPToolInfo,
    _collect_mcp_tool_catalog,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# JSON-RPC helpers
# --------------------------------------------------------------------------- #

_JSONRPC_VERSION = "2.0"
_DEFAULT_MAX_FRAME_BYTES = 8 * 1024 * 1024
_DEFAULT_CANCEL_NOTIFICATION_TIMEOUT_SECONDS = 0.25
_DEFAULT_TERMINATE_GRACE_SECONDS = 5.0

if sys.platform == "win32":
    _CREATE_NEW_PROCESS_GROUP = subprocess.CREATE_NEW_PROCESS_GROUP
    _CTRL_BREAK_EVENT = signal.CTRL_BREAK_EVENT
else:
    _CREATE_NEW_PROCESS_GROUP = 0
    _CTRL_BREAK_EVENT = 0


def _make_request(
    method: str, params: Optional[Dict[str, Any]] = None, request_id: int = 1
) -> str:
    """Build a JSON-RPC request string."""
    payload: Dict[str, Any] = {
        "jsonrpc": _JSONRPC_VERSION,
        "method": method,
        "id": request_id,
    }
    if params is not None:
        payload["params"] = params
    return json.dumps(payload)


def _make_notification(method: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Build a JSON-RPC notification (no id, server must not reply)."""
    payload: Dict[str, Any] = {
        "jsonrpc": _JSONRPC_VERSION,
        "method": method,
    }
    if params is not None:
        payload["params"] = params
    return json.dumps(payload)


def _parse_response(raw: str) -> Dict[str, Any]:
    """Parse a JSON-RPC response, raising on error objects."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MCPRequestError(
            "MCP JSON-RPC response is not valid JSON",
            error_code="MCP_PROTOCOL_ERROR",
            error_category="mcp_protocol_error",
        ) from exc
    if not isinstance(data, dict):
        raise MCPRequestError(
            "MCP JSON-RPC response must be an object",
            error_code="MCP_PROTOCOL_ERROR",
            error_category="mcp_protocol_error",
        )
    if data.get("jsonrpc") != _JSONRPC_VERSION:
        raise MCPRequestError(
            "MCP JSON-RPC version must be '2.0'",
            error_code="MCP_PROTOCOL_ERROR",
            error_category="mcp_protocol_error",
        )
    if "error" in data:
        if "result" in data:
            raise MCPRequestError(
                "MCP JSON-RPC response contains result and error",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        err = data["error"]
        if (
            not isinstance(err, dict)
            or isinstance(err.get("code"), bool)
            or not isinstance(err.get("code"), int)
            or not isinstance(err.get("message"), str)
        ):
            raise MCPRequestError(
                "MCP JSON-RPC response contains invalid error",
                error_code="MCP_PROTOCOL_ERROR",
                error_category="mcp_protocol_error",
            )
        raise MCPRequestError(
            f"MCP JSON-RPC error (code={err['code']}): {err['message']}",
            error_code="MCP_REMOTE_ERROR",
        )
    if "result" not in data:
        raise MCPRequestError(
            "MCP JSON-RPC response is missing a result",
            error_code="MCP_PROTOCOL_ERROR",
            error_category="mcp_protocol_error",
        )
    return data


# --------------------------------------------------------------------------- #
# MCPServerStdio
# --------------------------------------------------------------------------- #


class MCPServerStdio(MCPServer):
    """Connect to an MCP server launched as a local subprocess.

    The subprocess speaks the MCP protocol over its stdin/stdout using
    newline-delimited JSON-RPC.  This transport is the most common way to
    integrate with MCP servers that ship as command-line tools (e.g. language
    servers, database connectors, etc.).

    Usage::

        server = MCPServerStdio(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
        await server.connect()
        tools = await server.list_tools()
        result = await server.call_tool("read_file", {"path": "/tmp/hello.txt"})
        await server.cleanup()
    """

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        name: Optional[str] = None,
        *,
        max_frame_bytes: int = _DEFAULT_MAX_FRAME_BYTES,
        cancel_notification_timeout_seconds: float = (
            _DEFAULT_CANCEL_NOTIFICATION_TIMEOUT_SECONDS
        ),
        terminate_grace_seconds: float = _DEFAULT_TERMINATE_GRACE_SECONDS,
    ) -> None:
        if (
            isinstance(max_frame_bytes, bool)
            or not isinstance(max_frame_bytes, int)
            or max_frame_bytes <= 0
        ):
            raise ValueError("max_frame_bytes must be a positive integer")
        if (
            isinstance(cancel_notification_timeout_seconds, bool)
            or not isinstance(cancel_notification_timeout_seconds, (int, float))
            or cancel_notification_timeout_seconds <= 0
        ):
            raise ValueError("cancel_notification_timeout_seconds must be positive")
        if (
            isinstance(terminate_grace_seconds, bool)
            or not isinstance(terminate_grace_seconds, (int, float))
            or terminate_grace_seconds < 0
        ):
            raise ValueError("terminate_grace_seconds must be non-negative")
        self._command = command
        self._args: List[str] = list(args or [])
        self._env = env
        self._cwd = cwd
        self._name = name or f"stdio:{command}"
        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._write_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future[Dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._reader_error: BaseException | None = None
        self._closing = False
        self._owns_process_group = False
        self._max_frame_bytes = max_frame_bytes
        self._cancel_notification_timeout_seconds = float(
            cancel_notification_timeout_seconds
        )
        self._terminate_grace_seconds = float(terminate_grace_seconds)

    @property
    def name(self) -> str:
        return self._name

    # -- lifecycle ----------------------------------------------------------- #

    async def connect(self) -> None:
        """Launch the subprocess and complete the MCP initialization handshake."""
        if self._process is not None:
            raise RuntimeError("MCP server is already connected")
        self._reader_error = None
        self._closing = False
        env = dict(os.environ)
        if self._env:
            env.update(self._env)

        process_options: dict[str, Any] = {
            "limit": self._max_frame_bytes + 1,
        }
        if os.name == "nt":
            process_options["creationflags"] = _CREATE_NEW_PROCESS_GROUP
        else:
            process_options["start_new_session"] = True
        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env=env,
            **process_options,
        )
        self._owns_process_group = True
        self._reader_task = asyncio.create_task(
            self._read_messages(),
            name=f"qitos-mcp-reader-{self._name}",
        )
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(),
            name=f"qitos-mcp-stderr-{self._name}",
        )
        logger.info("MCP stdio process started (pid=%s)", self._process.pid)

        try:
            # MCP initialization handshake: initialize request -> initialized notification
            init_result = await self._send_request(
                "initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "qitos-mcp-client",
                        "version": "0.1.0",
                    },
                },
            )
            logger.debug("MCP initialize response: %s", init_result)

            # Send the initialized notification (no id, no response expected)
            await self._send_notification("notifications/initialized")
            logger.info("MCP stdio handshake complete for %s", self._name)
        except BaseException:
            try:
                await self.cleanup()
            except BaseException as cleanup_error:
                logger.warning(
                    "MCP stdio cleanup after handshake failure also failed",
                    exc_info=cleanup_error,
                )
            raise

    async def cleanup(self) -> None:
        """Terminate and reap the subprocess before propagating cancellation."""
        proc = self._process
        cleanup = self._cleanup_task
        if proc is None and cleanup is None:
            return
        if cleanup is None:
            assert proc is not None
            cleanup = self._ensure_cleanup_task(proc)
        elif proc is None:
            raise RuntimeError("MCP cleanup task has no owning process")
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as cancellation:
            try:
                await _settle_task(cleanup)
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise
        finally:
            if cleanup.done():
                self._finalize_cleanup_task(proc, cleanup)
        logger.info("MCP stdio process terminated for %s", self._name)

    def _ensure_cleanup_task(
        self,
        proc: asyncio.subprocess.Process,
    ) -> asyncio.Task[None]:
        cleanup = self._cleanup_task
        if cleanup is not None:
            return cleanup
        self._closing = True
        cleanup = asyncio.create_task(
            self._cleanup_process(
                proc,
                reader_task=self._reader_task,
                stderr_task=self._stderr_task,
                owns_process_group=self._owns_process_group,
            ),
            name=f"qitos-mcp-cleanup-{self._name}",
        )
        self._cleanup_task = cleanup
        cleanup.add_done_callback(
            lambda completed: self._finalize_cleanup_task(proc, completed)
        )
        return cleanup

    def _finalize_cleanup_task(
        self,
        proc: asyncio.subprocess.Process,
        cleanup: asyncio.Task[None],
    ) -> None:
        if self._cleanup_task is not cleanup:
            return
        self._cleanup_task = None
        self._closing = False
        if cleanup.cancelled():
            return
        error = cleanup.exception()
        if error is not None:
            logger.warning(
                "MCP stdio background cleanup failed for %s",
                self._name,
                exc_info=error,
            )
            return
        if self._process is proc:
            self._process = None
        self._reader_task = None
        self._stderr_task = None
        self._owns_process_group = False

    async def _cleanup_process(
        self,
        proc: asyncio.subprocess.Process,
        *,
        reader_task: asyncio.Task[None] | None,
        stderr_task: asyncio.Task[None] | None,
        owns_process_group: bool,
    ) -> None:
        try:
            try:
                await self._signal_process_tree(
                    proc,
                    owns_process_group=owns_process_group,
                    force=False,
                )
                try:
                    await asyncio.wait_for(
                        asyncio.shield(proc.wait()),
                        timeout=self._terminate_grace_seconds,
                    )
                except asyncio.TimeoutError:
                    await self._signal_process_tree(
                        proc,
                        owns_process_group=owns_process_group,
                        force=True,
                    )
                    await proc.wait()
                else:
                    # The process-group leader may exit before its descendants.
                    # A final group kill makes cleanup an ownership boundary for
                    # the complete subprocess tree.
                    if owns_process_group:
                        await self._signal_process_tree(
                            proc,
                            owns_process_group=True,
                            force=True,
                        )
            except ProcessLookupError:
                await proc.wait()
        finally:
            tasks = tuple(
                task for task in (reader_task, stderr_task) if task is not None
            )
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._fail_pending(MCPRequestError("MCP stdio transport closed"))

    async def _signal_process_tree(
        self,
        proc: asyncio.subprocess.Process,
        *,
        owns_process_group: bool,
        force: bool,
    ) -> None:
        if not owns_process_group:
            if force:
                proc.kill()
            else:
                proc.terminate()
            return
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
            except ProcessLookupError:
                if proc.returncode is None:
                    if force:
                        proc.kill()
                    else:
                        proc.terminate()
            return

        # ``CREATE_NEW_PROCESS_GROUP`` enables a group interrupt, while
        # ``taskkill /T`` provides the standard-library path that also targets
        # descendants. The forced pass is still followed by ``proc.wait()``.
        if not force:
            try:
                proc.send_signal(_CTRL_BREAK_EVENT)
                return
            except (OSError, ProcessLookupError):
                pass
        taskkill_args = ["taskkill", "/PID", str(proc.pid), "/T"]
        if force:
            taskkill_args.append("/F")
        try:
            taskkill = await asyncio.create_subprocess_exec(
                *taskkill_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await taskkill.wait()
        except (OSError, ProcessLookupError):
            if proc.returncode is None:
                if force:
                    proc.kill()
                else:
                    proc.terminate()

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
        """Send a JSON-RPC request and read the response."""
        request_id: int | None = None
        response: asyncio.Future[Dict[str, Any]] | None = None
        try:
            async with self._write_lock:
                if self._reader_error is not None:
                    raise MCPRequestError(
                        "MCP stdio response reader is unavailable",
                        error_code="MCP_TRANSPORT_CLOSED",
                        error_category="mcp_transport_error",
                    ) from self._reader_error
                process = self._process
                if process is None or process.stdin is None or self._closing:
                    raise MCPRequestError("MCP server is not connected")
                request_id = self._next_id()
                response = asyncio.get_running_loop().create_future()
                self._pending[request_id] = response
                request_str = _make_request(
                    method,
                    params=params,
                    request_id=request_id,
                )
                logger.debug("MCP -> %s", request_str)
                process.stdin.write((request_str + "\n").encode("utf-8"))
                await process.stdin.drain()
        except asyncio.CancelledError:
            if request_id is not None:
                self._pending.pop(request_id, None)
                if response is not None:
                    response.cancel()
                await self._notify_cancelled(request_id)
            raise
        except MCPRequestError:
            if request_id is not None:
                self._pending.pop(request_id, None)
                if response is not None:
                    response.cancel()
            raise
        except Exception as exc:
            if request_id is not None:
                self._pending.pop(request_id, None)
                if response is not None:
                    response.cancel()
            raise MCPRequestError(f"MCP stdio request failed: {method}: {exc}") from exc

        assert request_id is not None
        assert response is not None
        try:
            return await response
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            response.cancel()
            await self._notify_cancelled(request_id)
            raise

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
                "MCP stdio cancellation notification failed for %s: %s",
                self._name,
                exc,
            )

    async def _send_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        async with self._write_lock:
            process = self._process
            if process is None or process.stdin is None or self._closing:
                raise RuntimeError("MCP server is not connected")
            notification_str = _make_notification(method, params=params)
            logger.debug("MCP -> (notification) %s", notification_str)
            process.stdin.write((notification_str + "\n").encode("utf-8"))
            await process.stdin.drain()

    async def _read_messages(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            while True:
                try:
                    raw_line = await process.stdout.readline()
                except (ValueError, asyncio.LimitOverrunError) as exc:
                    raise MCPRequestError(
                        "MCP stdio frame exceeds the configured byte limit",
                        error_code="MCP_PROTOCOL_ERROR",
                        error_category="mcp_protocol_error",
                    ) from exc
                if not raw_line:
                    raise MCPRequestError(
                        "MCP server closed stdout unexpectedly",
                        error_code="MCP_TRANSPORT_CLOSED",
                        error_category="mcp_transport_error",
                    )
                if len(raw_line) > self._max_frame_bytes:
                    raise MCPRequestError(
                        "MCP stdio frame exceeds the configured byte limit",
                        error_code="MCP_PROTOCOL_ERROR",
                        error_category="mcp_protocol_error",
                    )
                raw = raw_line.decode("utf-8").strip()
                if not raw:
                    continue
                logger.debug("MCP <- %s", raw)
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise MCPRequestError(
                        "MCP JSON-RPC message is not valid JSON",
                        error_code="MCP_PROTOCOL_ERROR",
                        error_category="mcp_protocol_error",
                    ) from exc
                if not isinstance(message, dict):
                    raise MCPRequestError(
                        "MCP JSON-RPC message must be an object",
                        error_code="MCP_PROTOCOL_ERROR",
                        error_category="mcp_protocol_error",
                    )
                if message.get("jsonrpc") != _JSONRPC_VERSION:
                    raise MCPRequestError(
                        "MCP JSON-RPC version must be '2.0'",
                        error_code="MCP_PROTOCOL_ERROR",
                        error_category="mcp_protocol_error",
                    )
                method = message.get("method")
                if isinstance(method, str) and "id" not in message:
                    if method == "notifications/tools/list_changed":
                        self.notify_tools_changed()
                    else:
                        logger.debug("Ignoring MCP notification %s", method)
                    continue
                request_id = message.get("id")
                if isinstance(request_id, bool) or not isinstance(request_id, int):
                    raise MCPRequestError(
                        "MCP response is missing an integer id",
                        error_code="MCP_PROTOCOL_ERROR",
                        error_category="mcp_protocol_error",
                    )
                response = self._pending.pop(request_id, None)
                if response is None or response.done():
                    logger.debug("Ignoring late MCP response id=%s", request_id)
                    continue
                try:
                    parsed = _parse_response(raw)
                except Exception as exc:
                    response.set_exception(exc)
                    continue
                result = parsed.get("result", {})
                if not isinstance(result, dict):
                    response.set_exception(
                        MCPRequestError(
                            "MCP JSON-RPC result must be an object",
                            error_code="MCP_PROTOCOL_ERROR",
                            error_category="mcp_protocol_error",
                        )
                    )
                    continue
                response.set_result(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._reader_error = exc
            self._fail_pending(exc)
            if self._process is not None:
                logger.warning("MCP stdio reader failed for %s: %s", self._name, exc)
                self._ensure_cleanup_task(self._process)

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            while await process.stderr.readline():
                pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._process is not None:
                logger.debug("MCP stderr reader failed for %s: %s", self._name, exc)

    def _fail_pending(self, error: BaseException) -> None:
        pending = tuple(self._pending.values())
        self._pending.clear()
        for response in pending:
            if not response.done():
                response.set_exception(error)


async def _settle_task(task: asyncio.Task[None]) -> None:
    """Wait through repeated caller cancellation for one owned cleanup Task."""

    while True:
        try:
            await asyncio.shield(task)
            return
        except asyncio.CancelledError:
            if task.done():
                task.result()
                return
