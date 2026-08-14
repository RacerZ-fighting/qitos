"""MCP (Model Context Protocol) server integration for QitOS.

Provides transport clients for connecting to MCP servers, schema conversion
from MCP JSON Schema to QitOS ToolSpec, and a bridge that turns MCP tools
into QitOS FunctionTool instances.

Public API::

    from qitos.mcp import (
        MCPServer,
        MCPServerFactory,
        MCPServerStdio,
        MCPServerStreamableHttp,
        mcp_server_to_function_tools,
        ToolFilter,
    )
"""

from .server import MCPRequestError, MCPServer, MCPServerFactory
from .stdio import MCPServerStdio
from .http import MCPServerStreamableHttp
from .bridge import mcp_server_to_function_tools
from .filter import ToolFilter

__all__ = [
    "MCPRequestError",
    "MCPServer",
    "MCPServerFactory",
    "MCPServerStdio",
    "MCPServerStreamableHttp",
    "mcp_server_to_function_tools",
    "ToolFilter",
]
