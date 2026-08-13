"""Atomic shell execution tools."""

from __future__ import annotations

from collections.abc import Mapping

from qitos.kit.tool.internal.coding_impl import CodingToolSet
from qitos.kit.tool.internal.delegating import DelegatingTool


class RunCommand(DelegatingTool):
    def __init__(
        self,
        workspace_root: str = ".",
        shell_timeout: int = 30,
        *,
        process_env: Mapping[str, str] | None = None,
    ):
        self._toolset = CodingToolSet(
            workspace_root=workspace_root,
            shell_timeout=shell_timeout,
            profile="shell",
            process_env=process_env,
        )
        super().__init__(self._toolset.run_command)

    async def aclose(self) -> None:
        await self._toolset.ateardown({})


__all__ = ["RunCommand"]
