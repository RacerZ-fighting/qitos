"""Environment abstraction contracts for QitOS."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Sequence

from .process import ProcessHandle, ProcessSnapshot, ProcessTerminalNotifier

if TYPE_CHECKING:
    from .journal import SessionJournal


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
    content_sha256: str = ""


@dataclass(frozen=True, slots=True)
class AtomicFileWrite:
    """Result of one atomic capability-scoped file replacement."""

    path: str
    size_bytes: int
    content_sha256: str
    previous_sha256: str | None
    created: bool


class FileRevisionConflictError(RuntimeError):
    """The file changed after the caller captured its expected revision."""

    def __init__(
        self,
        path: str,
        *,
        expected_sha256: str,
        current_sha256: str | None,
    ) -> None:
        self.path = path
        self.expected_sha256 = expected_sha256
        self.current_sha256 = current_sha256
        current = current_sha256 if current_sha256 is not None else "missing"
        super().__init__(
            f"file revision conflict for {path}: expected "
            f"{expected_sha256}, current {current}"
        )


def _runtime_strings(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in normalized):
        raise TypeError(f"{field_name} must contain non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


@dataclass(frozen=True, slots=True)
class RuntimeCommand:
    """One command whose availability was verified inside a runtime backend."""

    name: str
    executable: str
    available: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("runtime command name must be non-empty")
        if not isinstance(self.executable, str) or not self.executable.strip():
            raise ValueError("runtime command executable must be non-empty")
        if not isinstance(self.available, bool):
            raise TypeError("runtime command available must be a boolean")
        if not isinstance(self.detail, str):
            raise TypeError("runtime command detail must be a string")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "executable": self.executable,
            "available": self.available,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeCommand":
        name = payload.get("name")
        executable = payload.get("executable")
        available = payload.get("available")
        detail = payload.get("detail", "")
        if not isinstance(name, str):
            raise TypeError("runtime command name must be a string")
        if not isinstance(executable, str):
            raise TypeError("runtime command executable must be a string")
        if not isinstance(available, bool):
            raise TypeError("runtime command available must be a boolean")
        if not isinstance(detail, str):
            raise TypeError("runtime command detail must be a string")
        return cls(name, executable, available, detail)


@dataclass(frozen=True, slots=True)
class RuntimeLimitation:
    """Stable machine-readable reason a backend omits or bounds a facility."""

    code: str
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("runtime limitation code must be non-empty")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("runtime limitation detail must be non-empty")

    def to_dict(self) -> Dict[str, str]:
        return {"code": self.code, "detail": self.detail}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeLimitation":
        code = payload.get("code")
        detail = payload.get("detail")
        if not isinstance(code, str):
            raise TypeError("runtime limitation code must be a string")
        if not isinstance(detail, str):
            raise TypeError("runtime limitation detail must be a string")
        return cls(code=code, detail=detail)


@dataclass(frozen=True, slots=True)
class RuntimeCapabilitySnapshot:
    """Immutable facts verified for one initialized execution backend."""

    backend: str
    working_directory: str
    operation_groups: tuple[str, ...] = ()
    facilities: tuple[str, ...] = ()
    commands: tuple[RuntimeCommand, ...] = ()
    limitations: tuple[RuntimeLimitation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("runtime backend must be non-empty")
        if (
            not isinstance(self.working_directory, str)
            or not self.working_directory.strip()
        ):
            raise ValueError("runtime working_directory must be non-empty")
        if not isinstance(self.operation_groups, tuple):
            raise TypeError("runtime operation_groups must be a tuple")
        if not isinstance(self.facilities, tuple):
            raise TypeError("runtime facilities must be a tuple")
        if not isinstance(self.commands, tuple) or not all(
            isinstance(command, RuntimeCommand) for command in self.commands
        ):
            raise TypeError("runtime commands must be a tuple of RuntimeCommand")
        if not isinstance(self.limitations, tuple) or not all(
            isinstance(item, RuntimeLimitation) for item in self.limitations
        ):
            raise TypeError("runtime limitations must be a tuple of RuntimeLimitation")
        _runtime_strings(self.operation_groups, "runtime operation_groups")
        _runtime_strings(self.facilities, "runtime facilities")
        _runtime_strings(
            tuple(command.name for command in self.commands),
            "runtime command names",
        )
        _runtime_strings(
            tuple(item.code for item in self.limitations),
            "runtime limitation codes",
        )

    def has_operation_group(self, group: str) -> bool:
        return group in self.operation_groups

    def has_facility(self, facility: str) -> bool:
        return facility in self.facilities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "working_directory": self.working_directory,
            "operation_groups": list(self.operation_groups),
            "facilities": list(self.facilities),
            "commands": [command.to_dict() for command in self.commands],
            "limitations": [item.to_dict() for item in self.limitations],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeCapabilitySnapshot":
        backend = payload.get("backend")
        working_directory = payload.get("working_directory")
        raw_groups = payload.get("operation_groups", [])
        raw_facilities = payload.get("facilities", [])
        raw_commands = payload.get("commands", [])
        raw_limitations = payload.get("limitations", [])
        if not isinstance(backend, str):
            raise TypeError("runtime backend must be a string")
        if not isinstance(working_directory, str):
            raise TypeError("runtime working_directory must be a string")
        for value, name in (
            (raw_groups, "operation_groups"),
            (raw_facilities, "facilities"),
            (raw_commands, "commands"),
            (raw_limitations, "limitations"),
        ):
            if not isinstance(value, list):
                raise TypeError(f"runtime {name} must be an array")
        if any(not isinstance(value, str) for value in raw_groups):
            raise TypeError("runtime operation_groups must contain strings")
        if any(not isinstance(value, str) for value in raw_facilities):
            raise TypeError("runtime facilities must contain strings")
        if any(not isinstance(value, Mapping) for value in raw_commands):
            raise TypeError("runtime commands must contain objects")
        if any(not isinstance(value, Mapping) for value in raw_limitations):
            raise TypeError("runtime limitations must contain objects")
        return cls(
            backend=backend,
            working_directory=working_directory,
            operation_groups=tuple(raw_groups),
            facilities=tuple(raw_facilities),
            commands=tuple(RuntimeCommand.from_dict(value) for value in raw_commands),
            limitations=tuple(
                RuntimeLimitation.from_dict(value) for value in raw_limitations
            ),
        )


class RuntimeCapabilityUnavailableError(RuntimeError):
    """Raised when a selected backend does not provide a required facility."""

    def __init__(self, facility: str, *, backend: str) -> None:
        if not isinstance(facility, str) or not facility.strip():
            raise ValueError("facility must be non-empty")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("backend must be non-empty")
        self.facility = facility
        self.backend = backend
        super().__init__(
            f"runtime backend {backend!r} does not provide facility {facility!r}"
        )


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

    async def ainitialize(
        self, task: Any = None, workspace: Optional[str] = None, **kwargs: Any
    ) -> EnvObservation:
        """Initialize the backend without blocking the owning event loop."""

        def _initialize() -> EnvObservation:
            self.setup(task=task, workspace=workspace, **kwargs)
            return self.reset(task=task, workspace=workspace, **kwargs)

        initialization = asyncio.create_task(asyncio.to_thread(_initialize))
        try:
            return await asyncio.shield(initialization)
        except asyncio.CancelledError as cancellation:
            # A worker thread cannot be stopped safely. Settle initialization
            # before propagating cancellation so the Run owner cannot tear down an
            # Env while its legacy setup hook is still allocating resources.
            while not initialization.done():
                try:
                    await asyncio.shield(initialization)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            try:
                initialization.result()
            except BaseException as exc:
                raise cancellation from exc
            raise

    async def ahealth_check(self) -> Dict[str, Any]:
        """Run the backend health probe outside the owning event loop."""

        return await asyncio.to_thread(self.health_check)

    def capability_snapshot(self) -> RuntimeCapabilitySnapshot | None:
        """Return immutable facts when this backend declares them.

        Existing Env implementations may still expose operations through
        ``get_ops`` without having adopted runtime capability snapshots. In
        that case ``None`` preserves the established operation resolution
        path; concrete backends that return an empty snapshot explicitly
        declare that no operation groups are available.
        """

        return None

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

    async def ateardown(self) -> None:
        """Release legacy environment resources outside the event loop."""

        await asyncio.to_thread(self.teardown)


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
    def write_text_atomic(
        self,
        path: str,
        content: str,
        *,
        expected_sha256: str | None = None,
    ) -> AtomicFileWrite:
        """Atomically replace UTF-8 text after an optional revision check.

        Implementations serialize mutations to the same canonical path within
        one capability instance. A supplied SHA-256 value is compared with the
        complete current file immediately before replacement.
        """

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
    async def arun(self, command: str, timeout: float = 30) -> Dict[str, Any]:
        """Run one shell command without blocking the owning event loop."""

    @abstractmethod
    async def arun_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: float = 30,
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

    async def astart(
        self,
        command: str,
        *,
        owner_run_id: str,
        cwd: str | None = None,
        tty: bool = False,
        journal: SessionJournal | None = None,
        terminal_notifier: ProcessTerminalNotifier | None = None,
    ) -> ProcessSnapshot:
        """Start one Run-owned background command when supported.

        A terminal notifier runs only after the process terminal fact is durable.
        It may enqueue a safe-point input, but it does not own Agent state.
        """

        _ = command, owner_run_id, cwd, tty, journal, terminal_notifier
        raise NotImplementedError("managed background commands are not supported")

    async def apoll(self, handle: ProcessHandle) -> ProcessSnapshot:
        """Return the current process state without waiting for new output."""

        _ = handle
        raise NotImplementedError("managed background commands are not supported")

    async def aread(
        self,
        handle: ProcessHandle,
        *,
        cursor: int = 0,
        wait_seconds: float = 0.0,
    ) -> ProcessSnapshot:
        """Read bounded incremental output, optionally waiting for a change."""

        _ = handle, cursor, wait_seconds
        raise NotImplementedError("managed background commands are not supported")

    async def awrite(
        self,
        handle: ProcessHandle,
        data: str,
    ) -> ProcessSnapshot:
        """Write UTF-8 input to a live process and return its new state."""

        _ = handle, data
        raise NotImplementedError("managed background commands are not supported")

    async def await_process(
        self,
        handle: ProcessHandle,
        *,
        deadline_monotonic: float | None = None,
    ) -> ProcessSnapshot:
        """Wait until terminal or until the supplied absolute deadline."""

        _ = handle, deadline_monotonic
        raise NotImplementedError("managed background commands are not supported")

    async def aterminate(self, handle: ProcessHandle) -> ProcessSnapshot:
        """Terminate a live process group and await its terminal snapshot."""

        _ = handle
        raise NotImplementedError("managed background commands are not supported")

    async def alist(
        self,
        *,
        owner_run_id: str | None = None,
    ) -> tuple[ProcessSnapshot, ...]:
        """List tracked processes in stable start order."""

        _ = owner_run_id
        raise NotImplementedError("managed background commands are not supported")

    async def arecover(
        self,
        *,
        owner_run_id: str,
        journal: SessionJournal,
    ) -> tuple[ProcessSnapshot, ...]:
        """Restore terminal observations and close interrupted ownership gaps."""

        _ = owner_run_id, journal
        return ()

    async def aquiesce(self, *, owner_run_id: str | None = None) -> None:
        """Settle resources owned by one Run without closing the capability.

        Command backends without managed background resources are already
        quiescent, so the public contract defaults to a safe no-op. Backends
        that do manage Run-owned resources must scope cleanup to
        ``owner_run_id`` when provided and remain reusable afterwards.
        """

        _ = owner_run_id

    async def aclose(self) -> None:
        """Terminate owned live processes and await all runtime Tasks."""

        return None


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
    "RuntimeCapabilitySnapshot",
    "RuntimeCapabilityUnavailableError",
    "RuntimeCommand",
    "RuntimeLimitation",
]
