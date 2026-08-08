"""Atomic file inspection and editing tools."""

from __future__ import annotations

from qitos.kit.tool.internal.coding_impl import CodingToolSet
from qitos.kit.tool.internal.delegating import DelegatingTool


class ReadFile(DelegatingTool):
    def __init__(self, workspace_root: str = "."):
        super().__init__(CodingToolSet(workspace_root=workspace_root).read_file)


class ListFiles(DelegatingTool):
    def __init__(self, workspace_root: str = "."):
        super().__init__(CodingToolSet(workspace_root=workspace_root).list_files)


class WriteFile(DelegatingTool):
    def __init__(self, workspace_root: str = "."):
        super().__init__(CodingToolSet(workspace_root=workspace_root).write_file)


class EditFile(DelegatingTool):
    def __init__(self, workspace_root: str = "."):
        super().__init__(CodingToolSet(workspace_root=workspace_root).edit_file)


class MakeDirectory(DelegatingTool):
    def __init__(self, workspace_root: str = "."):
        super().__init__(CodingToolSet(workspace_root=workspace_root).make_directory)


__all__ = [
    "EditFile",
    "ListFiles",
    "MakeDirectory",
    "ReadFile",
    "WriteFile",
]
