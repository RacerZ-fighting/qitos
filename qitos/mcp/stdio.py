"""Official MCP stdio transport integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import TYPE_CHECKING

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

from .server import _SDKMCPServer

if TYPE_CHECKING:
    from .server import _ReadStream, _WriteStream


class MCPServerStdio(_SDKMCPServer):
    """Connect to a subprocess through the official MCP stdio client."""

    def __init__(
        self,
        command: str,
        args: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
        name: str | None = None,
        *,
        read_timeout_seconds: float | None = None,
    ) -> None:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        normalized_args = list(args or ())
        if not all(isinstance(value, str) for value in normalized_args):
            raise TypeError("args must contain strings")
        normalized_env = dict(env) if env is not None else None
        if normalized_env is not None and not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in normalized_env.items()
        ):
            raise TypeError("env must map strings to strings")
        super().__init__(
            name=name or f"stdio:{command}",
            read_timeout_seconds=read_timeout_seconds,
        )
        self._parameters = StdioServerParameters(
            command=command,
            args=normalized_args,
            env=normalized_env,
            cwd=cwd,
        )

    def _open_transport(
        self,
    ) -> AbstractAsyncContextManager[tuple[_ReadStream, _WriteStream]]:
        return stdio_client(self._parameters)


__all__ = ["MCPServerStdio"]
