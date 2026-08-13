"""Environment abstraction contracts for QitOS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence


@dataclass
class EnvSpec:
    """Declarative environment requirement attached to a task."""

    type: str
    config: Dict[str, Any] = field(default_factory=dict)
    required_tools: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvObservation:
    """Structured environment observation payload."""

    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvStepResult:
    """Structured result emitted by one environment step."""

    observation: EnvObservation = field(default_factory=EnvObservation)
    done: bool = False
    reward: Optional[float] = None
    info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass(frozen=True)
class FileStat:
    """Backend-neutral metadata for one capability-scoped path."""

    path: str
    kind: Literal["file", "directory", "symlink", "other"]
    size: int
    modified_at: float | None = None

    @property
    def is_file(self) -> bool:
        """Whether the path is a regular file."""

        return self.kind == "file"

    @property
    def is_directory(self) -> bool:
        """Whether the path is a directory."""

        return self.kind == "directory"

    @property
    def is_symlink(self) -> bool:
        """Whether the path itself is a symbolic link."""

        return self.kind == "symlink"


@dataclass(frozen=True)
class TextFileChunk:
    """Bounded, line-oriented view of a UTF-8 text file."""

    content: str
    offset: int
    line_count: int
    total_lines: int
    size_bytes: int
    has_more: bool
    truncated: bool
    line_ending: Literal["lf", "crlf", "mixed"] = "lf"


class Env(ABC):
    """Canonical environment interface for agent-world interaction."""

    name: str = "env"
    version: str = "1.0"

    @abstractmethod
    def reset(
        self, task: Any = None, workspace: Optional[str] = None, **kwargs: Any
    ) -> EnvObservation:
        """Initialize environment state for a task and return first observation."""

    @abstractmethod
    def observe(self, state: Any = None) -> EnvObservation:
        """Return current environment observation without applying actions."""

    @abstractmethod
    def step(self, action: Any, state: Any = None) -> EnvStepResult:
        """Apply one action to environment and return step result."""

    def setup(
        self, task: Any = None, workspace: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Prepare env before reset/run."""
        return None

    def health_check(self) -> Dict[str, Any]:
        """Return health probe result used by runtime preflight."""
        return {"ok": True}

    def get_ops(self, group: str) -> Any:
        """Return concrete ops implementation for one capability group."""
        return None

    def has_ops(self, group: str) -> bool:
        """Whether this env provides one capability group."""
        return self.get_ops(group) is not None

    def is_terminal(
        self, state: Any = None, last_result: Optional[EnvStepResult] = None
    ) -> bool:
        """Return whether environment should terminate the episode."""
        if last_result is None:
            return False
        return bool(last_result.done)

    def close(self) -> None:
        """Release environment resources."""
        return None

    def teardown(self) -> None:
        """Symmetric shutdown hook called by runtime."""
        self.close()


class FileSystemCapability(ABC):
    """Root-scoped filesystem contract used by environment implementations."""

    @abstractmethod
    def resolve_path(self, path: str, *, allow_missing: bool = False) -> str:
        """Resolve a capability-relative path without escaping its root."""

    @abstractmethod
    def stat(self, path: str, *, follow_symlinks: bool = True) -> FileStat:
        """Return metadata for a capability-relative path."""

    @abstractmethod
    def read_bytes(
        self,
        path: str,
        limit: int | None = None,
        *,
        offset: int = 0,
    ) -> bytes:
        """Read raw bytes from ``offset``, optionally capped at ``limit`` bytes."""

    @abstractmethod
    def read_text(self, path: str) -> str:
        """Read UTF-8 text from file path."""

    @abstractmethod
    def read_text_chunk(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 1000,
        max_bytes: int = 100 * 1024,
        max_line_bytes: int = 2000,
    ) -> TextFileChunk:
        """Read a bounded whole-line UTF-8 chunk with file-level metadata."""

    @abstractmethod
    def write_text(self, path: str, content: str) -> None:
        """Write UTF-8 text to file path."""

    @abstractmethod
    def write_bytes(self, path: str, content: bytes) -> None:
        """Write raw bytes to file path."""

    @abstractmethod
    def append_text(self, path: str, content: str) -> None:
        """Append UTF-8 text to file path."""

    @abstractmethod
    def make_directory(self, path: str, *, parents: bool = True) -> None:
        """Create a directory inside the capability root."""

    @abstractmethod
    def list_entries(self, path: str = ".") -> List[FileStat]:
        """List immediate children of one directory in stable name order."""

    @abstractmethod
    def list_files(self, path: str = ".", limit: int = 200) -> List[str]:
        """List files relative to capability root."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists within capability scope."""


class CommandCapability(ABC):
    """Command execution capability contract used by env implementations."""

    @abstractmethod
    async def arun(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Run one shell command without blocking the owning event loop."""

    @abstractmethod
    async def arun_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> Dict[str, Any]:
        """Run one argv process asynchronously without shell interpretation."""

    @abstractmethod
    def run(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Compatibility entry point for synchronous environment setup code."""

    @abstractmethod
    def run_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> Dict[str, Any]:
        """Run one process without interpreting arguments through a shell."""

    def start(
        self,
        command: str,
        *,
        cwd: str | None = None,
        stdout_path: str | None = None,
    ) -> Dict[str, Any]:
        """Start one background shell command when the provider supports it."""

        raise NotImplementedError("background commands are not supported")


class TerminalCapability(ABC):
    """Interactive terminal capability contract used by env implementations."""

    @abstractmethod
    def send_keys(
        self,
        keys: str | list[str],
        min_timeout_sec: float = 0.0,
        block: bool = False,
        max_timeout_sec: float = 180.0,
    ) -> Dict[str, Any]:
        """Send raw keystrokes to the terminal and optionally wait."""

    @abstractmethod
    def capture_screen(self) -> str:
        """Return the currently visible terminal screen."""

    @abstractmethod
    def capture_buffer(self) -> str:
        """Return the full terminal scrollback buffer when available."""

    @abstractmethod
    def get_incremental_output(self) -> str:
        """Return new output since the previous capture, or the current screen."""

    @abstractmethod
    def is_session_alive(self) -> bool:
        """Whether the interactive terminal session is still alive."""

    @abstractmethod
    def get_timestamp(self) -> float | None:
        """Return a backend-specific timestamp if available."""


class GUIObserverCapability(ABC):
    """GUI observation capability for multimodal environments."""

    @abstractmethod
    def capture_observation(self, state: Any = None) -> Dict[str, Any]:
        """Return a normalized GUI observation pack payload."""


class GUIControllerCapability(ABC):
    """GUI control capability for click/type/scroll style actions."""

    @abstractmethod
    def perform(self, action: Dict[str, Any], state: Any = None) -> Dict[str, Any]:
        """Apply one GUI action and return a structured result."""


class OCRCapability(ABC):
    """OCR capability contract for multimodal environments."""

    @abstractmethod
    def extract_text(self, source: Any) -> List[Dict[str, Any]]:
        """Extract OCR rows or spans from the provided source."""


class GroundingCapability(ABC):
    """Grounding capability contract for GUI element linking."""

    @abstractmethod
    def ground(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Return grounding metadata for a multimodal observation pack."""


__all__ = [
    "EnvSpec",
    "EnvObservation",
    "EnvStepResult",
    "Env",
    "FileStat",
    "TextFileChunk",
    "FileSystemCapability",
    "CommandCapability",
    "TerminalCapability",
    "GUIObserverCapability",
    "GUIControllerCapability",
    "OCRCapability",
    "GroundingCapability",
]
