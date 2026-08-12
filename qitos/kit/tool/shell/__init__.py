"""Atomic shell execution tools."""

from __future__ import annotations

from qitos.kit.tool.internal.coding_impl import CodingToolSet
from qitos.kit.tool.internal.delegating import DelegatingTool


class RunCommand(DelegatingTool):
    def __init__(
        self,
        workspace_root: str = ".",
        shell_timeout: int = 30,
        *,
        auto_approve: bool = False,
    ):
        delegate = CodingToolSet(
            workspace_root=workspace_root,
            shell_timeout=shell_timeout,
            auto_approve=auto_approve,
        ).run_command
        if auto_approve:
            delegate.spec.needs_approval = False
        super().__init__(delegate)


__all__ = ["RunCommand"]
