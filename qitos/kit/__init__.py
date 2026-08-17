"""Curated practical building blocks for common QiTOS agent authoring."""

import importlib
from .artifact import FileArtifactStore
from .env import (
    ContainerDesktopProvider,
    DesktopEnv,
    HostEnv,
    MockDesktopProvider,
    RepoEnv,
    ScreenshotEnv,
    TextWebEnv,
    TmuxEnv,
)
from .memory import MarkdownFileMemory, MemdirMemory, WindowMemory
from .tool import (
    CodingToolSet,
    EpubToolSet,
    HTMLExtractText,
    HTTPGet,
    ReportToolSet,
    SendTerminalKeys,
    TaskToolSet,
    UpdateWorkPlanTool,
    WorkspaceAwareMixin,
)
from .tool.toolset import toolset_from_tools
from .toolset.codebase import codebase_tools
from .toolset.coding import coding_tools
from .toolset.computer_use import ComputerUseToolSet, computer_use_tools
from .toolset.editor import editor_tools
from .toolset.report import report_tools
_LAZY_MODULE_EXPORTS = {
    "env",
    "evaluate",
    "memory",
    "metric",
    "state",
    "tool",
    "toolset",
}


def __getattr__(name: str):
    if name in _LAZY_MODULE_EXPORTS:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FileArtifactStore",
    "env",
    "evaluate",
    "memory",
    "metric",
    "state",
    "tool",
    "toolset",
    "CodingToolSet",
    "ComputerUseToolSet",
    "SendTerminalKeys",
    "HTTPGet",
    "HTMLExtractText",
    "ReportToolSet",
    "EpubToolSet",
    "TaskToolSet",
    "UpdateWorkPlanTool",
    "WorkspaceAwareMixin",
    "toolset_from_tools",
    "coding_tools",
    "computer_use_tools",
    "editor_tools",
    "codebase_tools",
    "report_tools",
    "MarkdownFileMemory",
    "WindowMemory",
    "MemdirMemory",
    "HostEnv",
    "DesktopEnv",
    "ContainerDesktopProvider",
    "MockDesktopProvider",
    "RepoEnv",
    "ScreenshotEnv",
    "TextWebEnv",
    "TmuxEnv",
]
