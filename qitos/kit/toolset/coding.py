"""Coding-oriented preset toolset and registry builder.

Migration note (Phase 2): COMPLETE — all @tool decorators in coding_impl.py
have been migrated to @function_tool for richer schema inference and
needs_approval support. See qitos.core.function_tool_decorator for details.
"""

from __future__ import annotations

from qitos.core.tool_registry import ToolRegistry
from qitos.kit.tool.internal.coding_impl import CodingToolSet


class FullCodingToolSet(CodingToolSet):
    """Canonical full coding bundle with editor, shell, tasks, and web tools."""

    def __init__(
        self,
        workspace_root: str,
        *,
        shell_timeout: int = 30,
        include_notebook: bool = True,
        auto_approve: bool = False,
    ):
        super().__init__(
            workspace_root=workspace_root,
            shell_timeout=shell_timeout,
            include_notebook=include_notebook,
            enable_lsp=True,
            enable_tasks=True,
            enable_web=True,
            profile="full",
            auto_approve=auto_approve,
        )


def coding_tools(
    workspace_root: str,
    shell_timeout: int = 30,
    include_notebook: bool = True,
    *,
    auto_approve: bool = False,
) -> ToolRegistry:
    """Build a registry with the standard full coding bundle."""
    return ToolRegistry().include_toolset(
        FullCodingToolSet(
            workspace_root=workspace_root,
            shell_timeout=shell_timeout,
            include_notebook=include_notebook,
            auto_approve=auto_approve,
        )
    )


__all__ = ["CodingToolSet", "FullCodingToolSet", "coding_tools"]
