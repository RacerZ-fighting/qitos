"""Host environment with filesystem + command capabilities."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import secrets
import stat as stat_module
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional

from qitos.core.action import Action
from qitos.core.env import (
    AtomicFileWrite,
    CommandCapability,
    Env,
    EnvObservation,
    EnvStepResult,
    FileRevisionConflictError,
    FileStat,
    FileSystemCapability,
    TextFileChunk,
)
from qitos.core.journal import SessionJournal
from qitos.core.process import ProcessHandle, ProcessSnapshot
from qitos.kit.env._async_process import run_process
from qitos.kit.env._file_mutation import (
    FileMutationQueue,
    normalize_expected_sha256,
)
from qitos.kit.env.managed_process import ManagedHostProcessRuntime


class HostFSCapability(FileSystemCapability):
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self._mutations = FileMutationQueue()

    def resolve_path(self, path: str, *, allow_missing: bool = False) -> str:
        return str(self._resolve(path, allow_missing=allow_missing))

    def stat(self, path: str, *, follow_symlinks: bool = True) -> FileStat:
        p = self._resolve(path, follow_symlinks=follow_symlinks)
        info = p.stat() if follow_symlinks else p.lstat()
        mode = info.st_mode
        if stat_module.S_ISREG(mode):
            kind = "file"
        elif stat_module.S_ISDIR(mode):
            kind = "directory"
        elif stat_module.S_ISLNK(mode):
            kind = "symlink"
        else:
            kind = "other"
        return FileStat(
            path=str(p.relative_to(self.root)),
            kind=kind,
            size=int(info.st_size),
            modified_at=float(info.st_mtime),
        )

    def read_bytes(
        self,
        path: str,
        limit: int | None = None,
        *,
        offset: int = 0,
    ) -> bytes:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        p = self._resolve(path)
        with p.open("rb") as handle:
            handle.seek(offset)
            return handle.read() if limit is None else handle.read(limit)

    def read_text(self, path: str) -> str:
        p = self._resolve(path)
        return p.read_text(encoding="utf-8")

    def read_text_chunk(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 1000,
        max_bytes: int = 100 * 1024,
        max_line_bytes: int = 2000,
    ) -> TextFileChunk:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit <= 0 or max_bytes <= 0 or max_line_bytes <= 0:
            raise ValueError("limit and byte bounds must be positive")

        p = self._resolve(path)
        if not p.is_file():
            raise IsADirectoryError(path)

        selected: List[str] = []
        selected_bytes = 0
        total_lines = 0
        truncated = False
        has_crlf = False
        has_lf = False
        has_lone_cr = False
        content_hash = hashlib.sha256()

        with p.open("r", encoding="utf-8", errors="strict", newline="") as handle:
            for raw_line in handle:
                content_hash.update(raw_line.encode("utf-8"))
                total_lines += 1
                if "\x00" in raw_line:
                    raise UnicodeError(f"file contains NUL bytes: {path}")
                if raw_line.endswith("\r\n"):
                    has_crlf = True
                    line = raw_line[:-2]
                elif raw_line.endswith("\n"):
                    has_lf = True
                    line = raw_line[:-1]
                else:
                    line = raw_line
                if "\r" in line:
                    has_lone_cr = True
                if total_lines <= offset or len(selected) >= limit:
                    continue

                rendered, line_truncated = _truncate_utf8(line, max_line_bytes)
                encoded_size = len(rendered.encode("utf-8"))
                separator_size = 1 if selected else 0
                remaining = max_bytes - selected_bytes - separator_size
                if remaining <= 0:
                    truncated = True
                    continue
                if encoded_size > remaining:
                    rendered, _ = _truncate_utf8(rendered, remaining)
                    encoded_size = len(rendered.encode("utf-8"))
                    line_truncated = True
                selected.append(rendered)
                selected_bytes += separator_size + encoded_size
                truncated = truncated or line_truncated
            size_bytes = int(os.fstat(handle.fileno()).st_size)

        if has_lone_cr or (has_crlf and has_lf):
            line_ending = "mixed"
        elif has_crlf:
            line_ending = "crlf"
        else:
            line_ending = "lf"
        has_more = total_lines > offset + len(selected)
        return TextFileChunk(
            content="\n".join(selected),
            offset=offset,
            line_count=len(selected),
            total_lines=total_lines,
            size_bytes=size_bytes,
            has_more=has_more,
            truncated=truncated or has_more and len(selected) < min(limit, total_lines),
            line_ending=line_ending,
            content_sha256=content_hash.hexdigest(),
        )

    def write_text(self, path: str, content: str) -> None:
        self.write_text_atomic(path, content)

    def write_text_atomic(
        self,
        path: str,
        content: str,
        *,
        expected_sha256: str | None = None,
    ) -> AtomicFileWrite:
        """Atomically replace one UTF-8 file under a revision guard."""

        if not isinstance(content, str):
            raise TypeError("content must be a string")
        expected = normalize_expected_sha256(expected_sha256)
        target = self._resolve(path, allow_missing=True)
        relative = target.relative_to(self.root)
        key = relative.as_posix()
        if key in {"", "."}:
            raise IsADirectoryError(path)
        with self._mutations.hold(key):
            return self._replace_bytes_at(
                relative,
                content.encode("utf-8"),
                expected_sha256=expected,
            )

    def write_bytes(self, path: str, content: bytes) -> None:
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        target = self._resolve(path, allow_missing=True)
        relative = target.relative_to(self.root)
        key = relative.as_posix()
        if key in {"", "."}:
            raise IsADirectoryError(path)
        with self._mutations.hold(key):
            self._replace_bytes_at(relative, content, expected_sha256=None)

    def append_text(self, path: str, content: str) -> None:
        p = self._resolve(path, allow_missing=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as handle:
            handle.write(content)

    def make_directory(self, path: str, *, parents: bool = True) -> None:
        self._resolve(path, allow_missing=True).mkdir(
            parents=parents,
            exist_ok=True,
        )

    def list_entries(self, path: str = ".") -> List[FileStat]:
        base = self._resolve(path)
        if not base.is_dir():
            raise NotADirectoryError(path)
        entries: List[FileStat] = []
        for child in sorted(base.iterdir(), key=lambda item: item.name):
            relative = str(child.relative_to(self.root))
            entries.append(self.stat(relative, follow_symlinks=False))
        return entries

    def list_files(self, path: str = ".", limit: int = 200) -> List[str]:
        base = self._resolve(path)
        if base.is_file():
            return [str(base.relative_to(self.root))]
        out: List[str] = []
        for p in sorted(base.rglob("*")):
            if p.is_file():
                out.append(str(p.relative_to(self.root)))
                if len(out) >= limit:
                    break
        return out

    def exists(self, path: str) -> bool:
        try:
            return self._resolve(path).exists()
        except Exception:
            return False

    def _resolve(
        self,
        path: str,
        *,
        allow_missing: bool = False,
        follow_symlinks: bool = True,
    ) -> Path:
        raw = Path(str(path or "."))
        if raw.is_absolute():
            raise PermissionError(f"absolute path is outside capability scope: {path}")
        if any(part == ".." for part in raw.parts):
            raise PermissionError(f"parent traversal is outside capability scope: {path}")

        candidate = self.root / raw
        if follow_symlinks:
            resolved = candidate.resolve(strict=not allow_missing)
        else:
            resolved = candidate.parent.resolve(strict=True) / candidate.name
            if not allow_missing and not resolved.exists() and not resolved.is_symlink():
                raise FileNotFoundError(path)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"path outside root: {path}") from exc
        return resolved

    def _replace_bytes_at(
        self,
        relative: Path,
        content: bytes,
        *,
        expected_sha256: str | None,
    ) -> AtomicFileWrite:
        """Commit bytes through one descriptor-confined parent directory."""

        parent_fd, name = self._open_parent_directory(relative)
        temp_name = f".qitos-write-{secrets.token_hex(8)}"
        temp_fd: int | None = None
        try:
            previous_sha256, previous_mode = self._current_revision(parent_fd, name)
            if expected_sha256 is not None and previous_sha256 != expected_sha256:
                raise FileRevisionConflictError(
                    relative.as_posix(),
                    expected_sha256=expected_sha256,
                    current_sha256=previous_sha256,
                )
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            temp_fd = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
            view = memoryview(content)
            while view:
                written = os.write(temp_fd, view)
                if written <= 0:
                    raise OSError("atomic file write made no progress")
                view = view[written:]
            if previous_mode is not None:
                os.fchmod(temp_fd, previous_mode)
            os.fsync(temp_fd)
            os.close(temp_fd)
            temp_fd = None
            os.replace(
                temp_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        finally:
            if temp_fd is not None:
                os.close(temp_fd)
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            os.close(parent_fd)
        return AtomicFileWrite(
            path=relative.as_posix(),
            size_bytes=len(content),
            content_sha256=hashlib.sha256(content).hexdigest(),
            previous_sha256=previous_sha256,
            created=previous_sha256 is None,
        )

    def _open_parent_directory(self, relative: Path) -> tuple[int, str]:
        parts = tuple(part for part in relative.parts if part not in {"", "."})
        if not parts:
            raise IsADirectoryError(relative.as_posix())
        directory_flags = os.O_RDONLY
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        current_fd = os.open(self.root, directory_flags)
        try:
            for part in parts[:-1]:
                try:
                    next_fd = os.open(part, directory_flags, dir_fd=current_fd)
                except FileNotFoundError:
                    os.mkdir(part, mode=0o755, dir_fd=current_fd)
                    next_fd = os.open(part, directory_flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            return current_fd, parts[-1]
        except BaseException:
            os.close(current_fd)
            raise

    @staticmethod
    def _current_revision(parent_fd: int, name: str) -> tuple[str | None, int | None]:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            file_fd = os.open(name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            return None, None
        try:
            info = os.fstat(file_fd)
            if not stat_module.S_ISREG(info.st_mode):
                raise IsADirectoryError(name)
            content_hash = hashlib.sha256()
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                content_hash.update(chunk)
            return content_hash.hexdigest(), stat_module.S_IMODE(info.st_mode)
        finally:
            os.close(file_fd)


class HostCommandCapability(CommandCapability):
    def __init__(
        self,
        cwd: str,
        *,
        env: Mapping[str, str] | None = None,
    ):
        self.cwd = str(Path(cwd).resolve())
        self._env = dict(env) if env is not None else None
        self._managed: ManagedHostProcessRuntime | None = None

    def _managed_runtime(self) -> ManagedHostProcessRuntime:
        runtime = self._managed
        if runtime is None:
            runtime = ManagedHostProcessRuntime(self.cwd, env=self._env)
            self._managed = runtime
        return runtime

    async def arun(self, command: str, timeout: float = 30) -> Dict[str, Any]:
        if not command or not command.strip():
            return {"status": "error", "error": "empty command"}
        try:
            result = await run_process(
                shell_command=command,
                cwd=self.cwd,
                env=self._env,
                timeout=float(timeout),
            )
            return {
                "status": "success" if result.returncode == 0 else "partial",
                "returncode": result.returncode,
                "stdout": result.stdout.decode("utf-8", errors="replace"),
                "stderr": result.stderr.decode("utf-8", errors="replace"),
                "cwd": self.cwd,
                "command": command,
            }
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "error": f"command timed out after {timeout} seconds",
                "command": command,
                "cwd": self.cwd,
            }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "command": command,
                "cwd": self.cwd,
            }

    async def arun_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: float = 30,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> Dict[str, Any]:
        args = [str(item) for item in argv]
        if not args or not args[0].strip():
            raise ValueError("argv must contain a non-empty executable")
        effective_cwd = self._resolve_cwd(cwd)
        result = await run_process(
            argv=args,
            cwd=effective_cwd,
            env=self._env,
            stdin=stdin,
            timeout=float(timeout),
        )
        return {
            "status": "success" if result.returncode == 0 else "partial",
            "returncode": result.returncode,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
            "cwd": effective_cwd,
            "argv": args,
        }

    def run(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        if not command or not command.strip():
            return {"status": "error", "error": "empty command"}
        try:
            r = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.cwd,
                env=self._env,
            )
            return {
                "status": "success" if r.returncode == 0 else "partial",
                "returncode": r.returncode,
                "stdout": r.stdout,
                "stderr": r.stderr,
                "cwd": self.cwd,
                "command": command,
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "command": command,
                "cwd": self.cwd,
            }

    def run_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> Dict[str, Any]:
        args = [str(item) for item in argv]
        if not args or not args[0].strip():
            raise ValueError("argv must contain a non-empty executable")
        effective_cwd = self._resolve_cwd(cwd)
        result = subprocess.run(
            args,
            input=stdin,
            capture_output=True,
            timeout=timeout,
            cwd=effective_cwd,
            env=self._env,
            check=False,
        )
        return {
            "status": "success" if result.returncode == 0 else "partial",
            "returncode": result.returncode,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
            "cwd": effective_cwd,
            "argv": args,
        }

    def _resolve_cwd(self, cwd: str | None) -> str:
        if cwd is None:
            return self.cwd
        raw = Path(cwd)
        if raw.is_absolute():
            candidate = raw.resolve(strict=True)
        else:
            candidate = (Path(self.cwd) / raw).resolve(strict=True)
        root = Path(self.cwd)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PermissionError(f"cwd outside command root: {cwd}") from exc
        return str(candidate)

    async def astart(
        self,
        command: str,
        *,
        owner_run_id: str,
        cwd: str | None = None,
        tty: bool = False,
        journal: SessionJournal | None = None,
    ) -> ProcessSnapshot:
        effective_cwd = self._resolve_cwd(cwd)
        return await self._managed_runtime().start(
            command,
            owner_run_id=owner_run_id,
            cwd=effective_cwd,
            tty=tty,
            journal=journal,
        )

    async def apoll(self, handle: ProcessHandle) -> ProcessSnapshot:
        return await self._managed_runtime().poll(handle)

    async def aread(
        self,
        handle: ProcessHandle,
        *,
        cursor: int = 0,
        wait_seconds: float = 0.0,
    ) -> ProcessSnapshot:
        return await self._managed_runtime().read(
            handle,
            cursor=cursor,
            wait_seconds=wait_seconds,
        )

    async def awrite(
        self,
        handle: ProcessHandle,
        data: str,
    ) -> ProcessSnapshot:
        return await self._managed_runtime().write(handle, data)

    async def await_process(
        self,
        handle: ProcessHandle,
        *,
        deadline_monotonic: float | None = None,
    ) -> ProcessSnapshot:
        return await self._managed_runtime().wait(
            handle,
            deadline_monotonic=deadline_monotonic,
        )

    async def aterminate(self, handle: ProcessHandle) -> ProcessSnapshot:
        return await self._managed_runtime().terminate(handle)

    async def alist(
        self,
        *,
        owner_run_id: str | None = None,
    ) -> tuple[ProcessSnapshot, ...]:
        return await self._managed_runtime().list(owner_run_id=owner_run_id)

    async def arecover(
        self,
        *,
        owner_run_id: str,
        journal: SessionJournal,
    ) -> tuple[ProcessSnapshot, ...]:
        return await self._managed_runtime().recover(
            owner_run_id=owner_run_id,
            journal=journal,
        )

    async def aclose(self) -> None:
        runtime = self._managed
        self._managed = None
        if runtime is not None:
            await runtime.close()


def _truncate_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    if max_bytes <= 3:
        return "." * max_bytes, True
    prefix = encoded[: max_bytes - 3].decode("utf-8", errors="ignore")
    return prefix + "...", True


class HostEnv(Env):
    """Host-based env that interprets common file/shell actions directly."""

    name = "host_env"
    version = "1.0"

    def __init__(
        self,
        workspace_root: str = ".",
        fs: Optional[FileSystemCapability] = None,
        cmd: Optional[CommandCapability] = None,
    ):
        self.workspace_root = str(Path(workspace_root).resolve())
        self.fs = fs or HostFSCapability(self.workspace_root)
        self.cmd = cmd or HostCommandCapability(self.workspace_root)
        self._last_error: Optional[str] = None

    def setup(
        self, task: Any = None, workspace: Optional[str] = None, **kwargs: Any
    ) -> None:
        if workspace:
            self.workspace_root = str(Path(workspace).resolve())
            self.fs = HostFSCapability(self.workspace_root)
            self.cmd = HostCommandCapability(self.workspace_root)
        Path(self.workspace_root).mkdir(parents=True, exist_ok=True)

    def reset(
        self, task: Any = None, workspace: Optional[str] = None, **kwargs: Any
    ) -> EnvObservation:
        if workspace:
            self.workspace_root = str(Path(workspace).resolve())
            self.fs = HostFSCapability(self.workspace_root)
            self.cmd = HostCommandCapability(self.workspace_root)
        Path(self.workspace_root).mkdir(parents=True, exist_ok=True)
        self._last_error = None
        return self.observe(state=None)

    def health_check(self) -> Dict[str, Any]:
        root = Path(self.workspace_root)
        if not root.exists():
            return {
                "ok": False,
                "message": f"workspace not found: {self.workspace_root}",
            }
        if not os.access(str(root), os.R_OK):
            return {
                "ok": False,
                "message": f"workspace not readable: {self.workspace_root}",
            }
        if not os.access(str(root), os.W_OK):
            return {
                "ok": False,
                "message": f"workspace not writable: {self.workspace_root}",
            }
        return {"ok": True, "workspace_root": self.workspace_root}

    def observe(self, state: Any = None) -> EnvObservation:
        files = self.fs.list_files(limit=200)
        return EnvObservation(
            data={
                "workspace_root": self.workspace_root,
                "file_count": len(files),
                "files": files,
                "last_error": self._last_error,
            },
            metadata={"state_step": getattr(state, "current_step", None)},
        )

    def step(self, action: Any, state: Any = None) -> EnvStepResult:
        # step() captures env transition. action execution is done by execute_action().
        return EnvStepResult(
            observation=self.observe(state=state),
            done=False,
            reward=None,
            info={"action_seen": self._to_action_name(action)},
            error=self._last_error,
        )

    def get_ops(self, group: str) -> Any:
        if group == "file":
            return self.fs
        if group == "process":
            return self.cmd
        return None

    async def ateardown(self) -> None:
        await self.cmd.aclose()
        self.close()

    def supports_action(self, action: Any) -> bool:
        name = self._to_action_name(action)
        return name in {
            "read_file",
            "write_file",
            "edit_file",
            "run_command",
            "list_files",
            "grep",
        }

    def execute_action(self, action: Any, state: Any = None) -> Any:
        act = action if isinstance(action, Action) else Action.from_dict(action)
        name = act.name
        args = act.args or {}
        try:
            if name == "read_file":
                path = str(args.get("path") or args.get("filename") or "")
                raw = self.fs.read_bytes(path)
                content = raw.decode("utf-8", errors="strict")
                return {
                    "status": "success",
                    "path": path,
                    "content": content,
                    "content_sha256": hashlib.sha256(raw).hexdigest(),
                }
            if name == "write_file":
                path = str(args.get("path") or args.get("filename") or "")
                content = str(args.get("content", ""))
                write = self.fs.write_text_atomic(
                    path,
                    content,
                    expected_sha256=args.get("expected_sha256"),
                )
                return {
                    "status": "success",
                    "path": path,
                    "size": write.size_bytes,
                    "content_sha256": write.content_sha256,
                    "previous_sha256": write.previous_sha256,
                }
            if name == "list_files":
                path = str(args.get("path", "."))
                files = self.fs.list_files(path=path, limit=int(args.get("limit", 200)))
                return {
                    "status": "success",
                    "path": path,
                    "files": files,
                    "count": len(files),
                }
            if name == "grep":
                path = str(args.get("path") or "")
                pattern = str(args.get("pattern") or "")
                return self._grep_file(
                    path=path, pattern=pattern, limit=int(args.get("limit", 50))
                )
            if name == "edit_file":
                return self._edit_file(
                    path=str(args.get("path", "")),
                    old_text=str(args.get("old_text", "")),
                    new_text=str(args.get("new_text", "")),
                    replace_all=bool(args.get("replace_all", False)),
                    expected_sha256=args.get("expected_sha256"),
                )
            if name == "run_command":
                return self.cmd.run(
                    str(args.get("command", "")), timeout=int(args.get("timeout", 30))
                )
            return {"status": "error", "error": f"unsupported action: {name}"}
        except FileRevisionConflictError as exc:
            self._last_error = str(exc)
            return {
                "status": "error",
                "error": str(exc),
                "error_category": "file_revision_conflict",
                "path": exc.path,
                "expected_sha256": exc.expected_sha256,
                "current_sha256": exc.current_sha256,
            }
        except Exception as exc:
            self._last_error = str(exc)
            return {"status": "error", "error": str(exc), "action": name}

    def _edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool,
        expected_sha256: str | None = None,
    ) -> Dict[str, Any]:
        raw = self.fs.read_bytes(path)
        text = raw.decode("utf-8", errors="strict")
        current_sha256 = hashlib.sha256(raw).hexdigest()
        if not old_text:
            return {"status": "error", "error": "old_text cannot be empty", "path": path}
        count = text.count(old_text)
        if count == 0:
            return {"status": "error", "error": "text not found", "path": path}
        if count > 1 and not replace_all:
            return {
                "status": "error",
                "error": "text replacement must be unique",
                "path": path,
                "occurrences": count,
            }
        updated = text.replace(old_text, new_text, -1 if replace_all else 1)
        write = self.fs.write_text_atomic(
            path,
            updated,
            expected_sha256=expected_sha256 or current_sha256,
        )
        return {
            "status": "success",
            "path": path,
            "replacements": count if replace_all else 1,
            "previous_sha256": write.previous_sha256,
            "content_sha256": write.content_sha256,
        }

    def _grep_file(self, path: str, pattern: str, limit: int = 50) -> Dict[str, Any]:
        if not pattern:
            return {"status": "error", "error": "empty pattern"}
        text = self.fs.read_text(path)
        out: List[Dict[str, Any]] = []
        for idx, line in enumerate(text.splitlines(), start=1):
            if re.search(pattern, line):
                out.append({"line": idx, "text": line})
                if len(out) >= limit:
                    break
        return {
            "status": "success",
            "path": path,
            "pattern": pattern,
            "matches": out,
            "count": len(out),
        }

    def _to_action_name(self, action: Any) -> str:
        if isinstance(action, Action):
            return action.name
        if isinstance(action, dict):
            return str(action.get("name", ""))
        return ""


__all__ = ["HostFSCapability", "HostCommandCapability", "HostEnv"]
