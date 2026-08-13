"""Docker-backed environment and capabilities."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import shlex
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Optional, Sequence

from qitos.core.env import (
    AtomicFileWrite,
    CommandCapability,
    FileRevisionConflictError,
    FileStat,
    FileSystemCapability,
    TextFileChunk,
)
from qitos.kit.env._async_process import run_process
from qitos.kit.env._file_mutation import (
    FileMutationQueue,
    normalize_expected_sha256,
)
from qitos.kit.env.host_env import HostEnv


def _run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class DockerCommandCapability(CommandCapability):
    def __init__(self, container: str, workdir: str = "/workspace"):
        self.container = container
        self.workdir = workdir

    async def arun(self, command: str, timeout: float = 30) -> Dict[str, Any]:
        if not command or not command.strip():
            return {"status": "error", "error": "empty command"}
        docker_cmd = [
            "docker",
            "exec",
            "-w",
            self.workdir,
            self.container,
            "sh",
            "-lc",
            command,
        ]
        try:
            result = await run_process(argv=docker_cmd, timeout=float(timeout))
            return {
                "status": "success" if result.returncode == 0 else "partial",
                "returncode": result.returncode,
                "stdout": result.stdout.decode("utf-8", errors="replace"),
                "stderr": result.stderr.decode("utf-8", errors="replace"),
                "command": command,
                "container": self.container,
            }
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "error": f"command timed out after {timeout} seconds",
                "command": command,
                "container": self.container,
            }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "command": command,
                "container": self.container,
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
        workdir = self.workdir if cwd is None else _docker_scoped_path(self.workdir, cwd)
        docker_argv = ["docker", "exec"]
        if stdin is not None:
            docker_argv.append("-i")
        docker_argv.extend(["-w", workdir, self.container, *args])
        result = await run_process(
            argv=docker_argv,
            stdin=stdin,
            timeout=float(timeout),
        )
        return {
            "status": "success" if result.returncode == 0 else "partial",
            "returncode": result.returncode,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
            "argv": args,
            "container": self.container,
            "cwd": workdir,
        }

    def run(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        if not command or not command.strip():
            return {"status": "error", "error": "empty command"}
        docker_cmd = [
            "docker",
            "exec",
            "-w",
            self.workdir,
            self.container,
            "sh",
            "-lc",
            command,
        ]
        try:
            r = _run(docker_cmd, timeout=timeout)
            return {
                "status": "success" if r.returncode == 0 else "partial",
                "returncode": r.returncode,
                "stdout": r.stdout,
                "stderr": r.stderr,
                "command": command,
                "container": self.container,
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "command": command,
                "container": self.container,
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
        workdir = self.workdir if cwd is None else _docker_scoped_path(self.workdir, cwd)
        docker_argv = ["docker", "exec"]
        if stdin is not None:
            docker_argv.append("-i")
        docker_argv.extend(["-w", workdir, self.container, *args])
        result = subprocess.run(
            docker_argv,
            input=stdin,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "status": "success" if result.returncode == 0 else "partial",
            "returncode": result.returncode,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": result.stderr.decode("utf-8", errors="replace"),
            "argv": args,
            "container": self.container,
            "cwd": workdir,
        }


class DockerFSCapability(FileSystemCapability):
    def __init__(self, container: str, workdir: str = "/workspace"):
        self.container = container
        self.workdir = workdir.rstrip("/") or "/workspace"
        self.cmd = DockerCommandCapability(container=container, workdir=workdir)
        self._mutations = FileMutationQueue()

    def resolve_path(self, path: str, *, allow_missing: bool = False) -> str:
        inner = self._inner_path(path)
        argv = ["realpath", "-m" if allow_missing else "-e", "--", inner]
        result = self.cmd.run_argv(argv)
        if int(result.get("returncode", 1)) != 0:
            raise FileNotFoundError(path)
        resolved = str(result.get("stdout", "")).strip()
        _relative_docker_path(self.workdir, resolved)
        return resolved

    def stat(self, path: str, *, follow_symlinks: bool = True) -> FileStat:
        inner = self._inner_path(path)
        flags = ["-L"] if follow_symlinks else []
        result = self.cmd.run_argv(
            ["stat", *flags, "-c", "%F\t%s\t%Y", "--", inner]
        )
        if int(result.get("returncode", 1)) != 0:
            raise FileNotFoundError(path)
        raw_kind, raw_size, raw_mtime = str(result.get("stdout", "")).strip().split(
            "\t", maxsplit=2
        )
        kind = (
            "file"
            if raw_kind == "regular file"
            else "directory"
            if raw_kind == "directory"
            else "symlink"
            if raw_kind == "symbolic link"
            else "other"
        )
        return FileStat(
            path=_relative_docker_path(self.workdir, inner),
            kind=kind,
            size=int(raw_size),
            modified_at=float(raw_mtime),
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
        inner = self.resolve_path(path)
        read_command = f"base64 -w 0 -- {shlex.quote(inner)}"
        if offset:
            read_command = (
                f"tail -c +{offset + 1} -- {shlex.quote(inner)} | base64 -w 0"
            )
        if limit is not None:
            source = (
                f"tail -c +{offset + 1} -- {shlex.quote(inner)}"
                if offset
                else f"cat -- {shlex.quote(inner)}"
            )
            read_command = f"{source} | head -c {int(limit)} | base64 -w 0"
        result = self.cmd.run(read_command)
        if int(result.get("returncode", 1)) != 0:
            raise RuntimeError(str(result.get("stderr", "failed to read file")))
        return base64.b64decode(str(result.get("stdout", "")), validate=True)

    def read_text(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8")

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

        info = self.stat(path)
        if not info.is_file:
            raise IsADirectoryError(path)
        inner = self.resolve_path(path)
        digest_result = self.cmd.run_argv(["sha256sum", "--", inner])
        if int(digest_result.get("returncode", 1)) != 0:
            raise RuntimeError(
                str(digest_result.get("stderr", "failed to hash file"))
            )
        content_sha256 = str(digest_result.get("stdout", "")).split(maxsplit=1)[0]
        if info.size > 0:
            text_probe = self.cmd.run(
                f"LC_ALL=C grep -Iq -- '' {shlex.quote(inner)}"
            )
            if int(text_probe.get("returncode", 1)) != 0:
                raise UnicodeError(f"file is not UTF-8 text: {path}")

        count_result = self.cmd.run_argv(
            ["awk", "END { print NR }", inner]
        )
        if int(count_result.get("returncode", 1)) != 0:
            raise RuntimeError(
                str(count_result.get("stderr", "failed to count file lines"))
            )
        total_lines = int(str(count_result.get("stdout", "0")).strip() or "0")

        start = offset + 1
        end = offset + limit
        read_command = (
            f"sed -n {start},{end}p -- {shlex.quote(inner)} | "
            f"head -c {max_bytes} | base64 -w 0"
        )
        result = self.cmd.run(read_command)
        if int(result.get("returncode", 1)) != 0:
            raise RuntimeError(str(result.get("stderr", "failed to read file chunk")))
        raw = base64.b64decode(str(result.get("stdout", "")), validate=True)
        content = raw.decode("utf-8", errors="strict")
        if "\x00" in content:
            raise UnicodeError(f"file contains NUL bytes: {path}")

        line_truncated = False
        lines: list[str] = []
        for line in content.splitlines():
            rendered, was_truncated = _truncate_utf8(line, max_line_bytes)
            line_truncated = line_truncated or was_truncated
            lines.append(rendered)
        line_count = len(lines)
        has_crlf = b"\r\n" in raw
        has_lf = b"\n" in raw.replace(b"\r\n", b"")
        has_lone_cr = b"\r" in raw.replace(b"\r\n", b"")
        line_ending = (
            "mixed"
            if has_lone_cr or has_crlf and has_lf
            else "crlf"
            if has_crlf
            else "lf"
        )
        has_more = total_lines > offset + line_count
        byte_truncated = len(raw) >= max_bytes and has_more
        return TextFileChunk(
            content="\n".join(lines),
            offset=offset,
            line_count=line_count,
            total_lines=total_lines,
            size_bytes=info.size,
            has_more=has_more,
            truncated=line_truncated or byte_truncated,
            line_ending=line_ending,
            content_sha256=content_sha256,
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
        """Atomically replace one UTF-8 file inside the container workspace."""

        if not isinstance(content, str):
            raise TypeError("content must be a string")
        return self._write_bytes_atomic(
            path,
            content.encode("utf-8"),
            expected_sha256=normalize_expected_sha256(expected_sha256),
        )

    def write_bytes(self, path: str, content: bytes) -> None:
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        self._write_bytes_atomic(path, content, expected_sha256=None)

    def _write_bytes_atomic(
        self,
        path: str,
        content: bytes,
        *,
        expected_sha256: str | None,
    ) -> AtomicFileWrite:
        inner = self.resolve_path(path, allow_missing=True)
        relative = _relative_docker_path(self.workdir, inner)
        if relative in {"", "."}:
            raise IsADirectoryError(path)
        script = """
set -eu
target=$1
expected=$2
parent=${target%/*}
mkdir -p -- "$parent"
current=
if [ -e "$target" ]; then
  current=$(sha256sum -- "$target")
  current=${current%% *}
fi
if [ -n "$expected" ] && [ "$current" != "$expected" ]; then
  printf 'QITOS_CONFLICT:%s' "$current" >&2
  exit 73
fi
temporary=$(mktemp "$parent/.qitos-write.XXXXXX")
trap 'rm -f -- "$temporary"' EXIT HUP INT TERM
cat > "$temporary"
if [ -e "$target" ]; then
  chmod --reference="$target" "$temporary" 2>/dev/null || true
fi
mv -f -- "$temporary" "$target"
trap - EXIT HUP INT TERM
printf '%s' "$current"
""".strip()
        with self._mutations.hold(relative):
            result = self.cmd.run_argv(
                [
                    "sh",
                    "-c",
                    script,
                    "qitos-atomic-write",
                    inner,
                    expected_sha256 or "",
                ],
                stdin=content,
            )
        returncode = int(result.get("returncode", 1))
        stderr = str(result.get("stderr", ""))
        if returncode == 73 and stderr.startswith("QITOS_CONFLICT:"):
            current = stderr.removeprefix("QITOS_CONFLICT:").strip() or None
            raise FileRevisionConflictError(
                relative,
                expected_sha256=expected_sha256 or "",
                current_sha256=current,
            )
        if returncode != 0:
            raise RuntimeError(stderr or "failed to atomically write file")
        previous_sha256 = str(result.get("stdout", "")).strip() or None
        return AtomicFileWrite(
            path=relative,
            size_bytes=len(content),
            content_sha256=hashlib.sha256(content).hexdigest(),
            previous_sha256=previous_sha256,
            created=previous_sha256 is None,
        )

    def append_text(self, path: str, content: str) -> None:
        inner = self.resolve_path(path, allow_missing=True)
        parent_result = self.cmd.run_argv(
            ["mkdir", "-p", "--", str(PurePosixPath(inner).parent)]
        )
        if int(parent_result.get("returncode", 1)) != 0:
            raise RuntimeError(
                str(parent_result.get("stderr", "failed to create parent directory"))
            )
        result = self.cmd.run_argv(
            [
                "dd",
                f"of={inner}",
                "oflag=append",
                "conv=notrunc",
                "status=none",
            ],
            stdin=content.encode("utf-8"),
        )
        if int(result.get("returncode", 1)) != 0:
            raise RuntimeError(str(result.get("stderr", "failed to append file")))

    def make_directory(self, path: str, *, parents: bool = True) -> None:
        inner = self.resolve_path(path, allow_missing=True)
        argv = ["mkdir", "-p", "--", inner] if parents else ["mkdir", "--", inner]
        result = self.cmd.run_argv(argv)
        if int(result.get("returncode", 1)) != 0:
            raise RuntimeError(str(result.get("stderr", "failed to create directory")))

    def list_entries(self, path: str = ".") -> list[FileStat]:
        inner = self.resolve_path(path)
        result = self.cmd.run_argv(
            ["find", inner, "-mindepth", "1", "-maxdepth", "1", "-print0"]
        )
        if int(result.get("returncode", 1)) != 0:
            raise RuntimeError(str(result.get("stderr", "failed to list directory")))
        paths = sorted(
            item for item in str(result.get("stdout", "")).split("\0") if item
        )
        return [
            self.stat(
                _relative_docker_path(self.workdir, item), follow_symlinks=False
            )
            for item in paths
        ]

    def list_files(self, path: str = ".", limit: int = 200) -> list[str]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        inner = self.resolve_path(path)
        result = self.cmd.run_argv(["find", inner, "-type", "f", "-print0"])
        if int(result.get("returncode", 1)) != 0:
            raise RuntimeError(str(result.get("stderr", "failed to list files")))
        out: list[str] = []
        for item in sorted(str(result.get("stdout", "")).split("\0")):
            if not item:
                continue
            out.append(_relative_docker_path(self.workdir, item))
            if len(out) >= limit:
                break
        return out

    def exists(self, path: str) -> bool:
        try:
            self.resolve_path(path)
        except (FileNotFoundError, PermissionError):
            return False
        return True

    def _inner_path(self, path: str) -> str:
        return _docker_scoped_path(self.workdir, path)


def _docker_scoped_path(workdir: str, path: str) -> str:
    root = PurePosixPath(workdir)
    candidate = PurePosixPath(str(path or "."))
    if candidate.is_absolute():
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PermissionError(f"path outside container workspace: {path}") from exc
        return candidate.as_posix()
    if any(part == ".." for part in candidate.parts):
        raise PermissionError(f"parent traversal is outside container workspace: {path}")
    return (root / candidate).as_posix()


def _relative_docker_path(workdir: str, path: str) -> str:
    try:
        return PurePosixPath(path).relative_to(PurePosixPath(workdir)).as_posix()
    except ValueError as exc:
        raise PermissionError(f"path outside container workspace: {path}") from exc


def _truncate_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    if max_bytes <= 3:
        return "." * max_bytes, True
    prefix = encoded[: max_bytes - 3].decode("utf-8", errors="ignore")
    return prefix + "...", True


class DockerEnv(HostEnv):
    """HostEnv-compatible action interpreter executed inside Docker.

    Supports two modes:
    1. Attach existing container: pass `container`.
    2. Auto-create ephemeral container: pass `image` and set `auto_create=True`.
    """

    name = "docker_env"
    version = "1.1"

    def __init__(
        self,
        container: Optional[str] = None,
        workspace_root: str = "/workspace",
        *,
        image: Optional[str] = None,
        host_workspace: Optional[str] = None,
        auto_create: bool = False,
        remove_on_close: bool = False,
        network: Optional[str] = None,
        extra_run_args: Optional[list[str]] = None,
        create_timeout: int = 60,
    ):
        self.container = str(container).strip() if container else ""
        self.container_workspace = workspace_root
        self.image = str(image or "").strip()
        self.host_workspace = str(host_workspace).strip() if host_workspace else ""
        self.auto_create = bool(auto_create)
        self.remove_on_close = bool(remove_on_close)
        self.network = network
        self.extra_run_args = list(extra_run_args or [])
        self.create_timeout = int(create_timeout)
        self._created_here = False

        if not self.container and self.auto_create:
            self.container = f"qitos_{Path(self.host_workspace or 'workspace').name}_{threading.get_ident()}"

        fs = DockerFSCapability(container=self.container or "", workdir=workspace_root)
        cmd = DockerCommandCapability(
            container=self.container or "", workdir=workspace_root
        )
        super().__init__(workspace_root=workspace_root, fs=fs, cmd=cmd)

    def setup(
        self, task: Any = None, workspace: Optional[str] = None, **kwargs: Any
    ) -> None:
        if workspace and not self.host_workspace:
            self.host_workspace = str(Path(workspace).resolve())
        if self.auto_create:
            self._ensure_container()
        if not self.container:
            raise ValueError(
                "DockerEnv requires `container` or `auto_create=True` with `image`"
            )

        self.fs = DockerFSCapability(
            container=self.container, workdir=self.container_workspace
        )
        self.cmd = DockerCommandCapability(
            container=self.container, workdir=self.container_workspace
        )

    def reset(self, task: Any = None, workspace: Optional[str] = None, **kwargs: Any):
        self.setup(task=task, workspace=workspace, **kwargs)
        self.workspace_root = workspace or self.container_workspace
        self._last_error = None
        return self.observe(state=None)

    def health_check(self) -> Dict[str, Any]:
        if not self.container:
            return {"ok": False, "message": "container is empty"}

        inspect = _run(["docker", "inspect", self.container], timeout=20)
        if inspect.returncode != 0:
            return {
                "ok": False,
                "message": "docker inspect failed",
                "container": self.container,
                "stderr": inspect.stderr,
            }

        probe = self.cmd.run("pwd", timeout=10)
        if int(probe.get("returncode", 1)) != 0:
            return {
                "ok": False,
                "message": "docker exec probe failed",
                "container": self.container,
                "stderr": probe.get("stderr", ""),
            }
        return {
            "ok": True,
            "container": self.container,
            "workspace_root": self.workspace_root,
        }

    def close(self) -> None:
        if not self.container:
            return
        if self.remove_on_close and self._created_here:
            _run(["docker", "rm", "-f", self.container], timeout=30)

    def _ensure_container(self) -> None:
        if not self.container:
            raise ValueError("auto_create needs container name")

        inspect = _run(["docker", "inspect", self.container], timeout=20)
        if inspect.returncode == 0:
            start = _run(["docker", "start", self.container], timeout=20)
            if start.returncode != 0:
                raise RuntimeError(
                    f"Failed to start container {self.container}: {start.stderr}"
                )
            return

        if not self.image:
            raise ValueError("auto_create requires `image`")

        run_cmd = ["docker", "run", "-d", "--name", self.container]
        if self.network:
            run_cmd += ["--network", self.network]

        if self.host_workspace:
            host = str(Path(self.host_workspace).resolve())
            run_cmd += ["-v", f"{host}:{self.container_workspace}"]

        if self.extra_run_args:
            run_cmd += list(self.extra_run_args)

        run_cmd += [self.image, "sh", "-lc", "while true; do sleep 3600; done"]
        proc = _run(run_cmd, timeout=self.create_timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to create container {self.container}: {proc.stderr}"
            )
        self._created_here = True


class DockerEnvScheduler:
    """Simple bounded scheduler for per-task DockerEnv creation.

    Useful for benchmark batch runs to control concurrent docker containers.
    """

    def __init__(self, max_active: int = 1):
        self.max_active = max(1, int(max_active))
        self._sem = threading.Semaphore(self.max_active)

    @contextmanager
    def allocate(
        self,
        *,
        image: str,
        host_workspace: str,
        workspace_root: str = "/workspace",
        network: Optional[str] = None,
        extra_run_args: Optional[list[str]] = None,
    ) -> Iterator[DockerEnv]:
        self._sem.acquire()
        env = DockerEnv(
            workspace_root=workspace_root,
            image=image,
            host_workspace=host_workspace,
            auto_create=True,
            remove_on_close=True,
            network=network,
            extra_run_args=extra_run_args,
        )
        try:
            env.setup(workspace=host_workspace)
            yield env
        finally:
            try:
                env.close()
            finally:
                self._sem.release()


__all__ = [
    "DockerCommandCapability",
    "DockerFSCapability",
    "DockerEnv",
    "DockerEnvScheduler",
]
