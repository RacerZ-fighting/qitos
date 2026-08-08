"""Atomic codebase discovery and search tools."""

from __future__ import annotations

from qitos.kit.tool.internal.coding_impl import CodingToolSet
from qitos.kit.tool.internal.delegating import DelegatingTool


class Glob(DelegatingTool):
    def __init__(self, workspace_root: str = "."):
        super().__init__(CodingToolSet(workspace_root=workspace_root).glob)


class Grep(DelegatingTool):
    def __init__(self, workspace_root: str = "."):
        super().__init__(CodingToolSet(workspace_root=workspace_root).grep)


class ListTree(DelegatingTool):
    def __init__(self, workspace_root: str = "."):
        super().__init__(CodingToolSet(workspace_root=workspace_root).list_tree)


__all__ = [
    "Glob",
    "Grep",
    "ListTree",
]
