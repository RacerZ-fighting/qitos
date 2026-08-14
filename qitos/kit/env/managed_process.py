"""Run-owned asyncio subprocess supervision for host environments."""

from __future__ import annotations

import asyncio
import codecs
import errno
import os
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from qitos.core.journal import JournalRecordType, SessionJournal
from qitos.core.process import (
    ProcessHandle,
    ProcessNotFoundError,
    ProcessOutput,
    ProcessPersistenceError,
    ProcessSnapshot,
    ProcessStatus,
    ProcessTerminalNotifier,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_bytes(path: Path, content: bytes) -> None:
    with path.open("ab") as handle:
        handle.write(content)


class _RollingOutput:
    """Byte-addressed UTF-8 tail used for incremental process reads."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self._max_bytes = int(max_bytes)
        self._content = bytearray()
        self._total_bytes = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def append(self, content: bytes) -> None:
        if not content:
            return
        self._content.extend(content)
        self._total_bytes += len(content)
        excess = len(self._content) - self._max_bytes
        if excess <= 0:
            return
        del self._content[:excess]
        while self._content and self._content[0] & 0xC0 == 0x80:
            del self._content[0]

    def read(self, cursor: int, *, log_path: str) -> ProcessOutput:
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        retained_from = self._total_bytes - len(self._content)
        effective_cursor = min(max(cursor, retained_from), self._total_bytes)
        relative = effective_cursor - retained_from
        selected = bytes(self._content[relative:])
        skipped_for_utf8 = 0
        while selected and selected[0] & 0xC0 == 0x80:
            selected = selected[1:]
            skipped_for_utf8 += 1
        omitted = max(0, retained_from - cursor) + skipped_for_utf8
        return ProcessOutput(
            content=selected.decode("utf-8", errors="replace"),
            cursor=cursor,
            next_cursor=self._total_bytes,
            total_bytes=self._total_bytes,
            omitted_bytes=omitted,
            truncated=omitted > 0,
            log_path=log_path,
        )


@dataclass(slots=True)
class _ProcessEntry:
    handle: ProcessHandle
    command: str
    cwd: str
    tty: bool
    process: asyncio.subprocess.Process
    output: _RollingOutput
    log_path: Path
    relative_log_path: str
    started_at: str
    started_monotonic: float
    journal: SessionJournal | None
    terminal_notifier: ProcessTerminalNotifier | None
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    interaction_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    reader_task: asyncio.Task[None] | None = None
    watcher_task: asyncio.Task[None] | None = None
    status: ProcessStatus = ProcessStatus.RUNNING
    ended_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    termination_requested: bool = False
    writer_fd: int | None = None
    read_transport: asyncio.BaseTransport | None = None
    started_record_ready: asyncio.Event = field(default_factory=asyncio.Event)
    suppress_terminal_record: bool = False
    journal_error: BaseException | None = None


class ManagedHostProcessRuntime:
    """Own host subprocesses, output readers, terminal watchers, and cleanup."""

    def __init__(
        self,
        workspace_root: str,
        *,
        env: Mapping[str, str] | None = None,
        max_processes: int = 16,
        max_tracked: int = 64,
        max_output_bytes: int = 64 * 1024,
        terminate_grace_seconds: float = 0.5,
    ) -> None:
        if max_processes <= 0 or max_tracked < max_processes:
            raise ValueError("process limits are invalid")
        if terminate_grace_seconds < 0:
            raise ValueError("terminate_grace_seconds must be non-negative")
        self.workspace_root = Path(workspace_root).resolve()
        self._env = dict(env) if env is not None else None
        self._max_processes = int(max_processes)
        self._max_tracked = int(max_tracked)
        self._max_output_bytes = int(max_output_bytes)
        self._terminate_grace_seconds = float(terminate_grace_seconds)
        self._entries: dict[str, _ProcessEntry] = {}
        self._detached: dict[str, ProcessSnapshot] = {}
        self._recovered_runs: set[str] = set()
        self._lock = asyncio.Lock()
        self._recovery_lock = asyncio.Lock()
        self._start_condition = asyncio.Condition(self._lock)
        self._starting = 0
        self._closed = False

    async def start(
        self,
        command: str,
        *,
        owner_run_id: str,
        cwd: str,
        tty: bool = False,
        journal: SessionJournal | None = None,
        terminal_notifier: ProcessTerminalNotifier | None = None,
    ) -> ProcessSnapshot:
        async with self._start_condition:
            if self._closed:
                raise RuntimeError("managed process runtime is closed")
            self._prune_finished_locked()
            active = sum(
                entry.status is ProcessStatus.RUNNING
                for entry in self._entries.values()
            )
            if active + self._starting >= self._max_processes:
                raise RuntimeError("managed process concurrency limit reached")
            self._starting += 1
        try:
            return await self._start_reserved(
                command,
                owner_run_id=owner_run_id,
                cwd=cwd,
                tty=tty,
                journal=journal,
                terminal_notifier=terminal_notifier,
            )
        finally:
            async with self._start_condition:
                self._starting -= 1
                self._start_condition.notify_all()

    async def _start_reserved(
        self,
        command: str,
        *,
        owner_run_id: str,
        cwd: str,
        tty: bool,
        journal: SessionJournal | None,
        terminal_notifier: ProcessTerminalNotifier | None,
    ) -> ProcessSnapshot:
        text = str(command or "").strip()
        if not text:
            raise ValueError("command must be non-empty")
        handle = ProcessHandle(
            process_id=f"proc_{uuid4().hex[:16]}",
            owner_run_id=owner_run_id,
        )
        log_dir = self.workspace_root / ".qitos" / "processes"
        relative_log = str(Path(".qitos") / "processes" / f"{handle.process_id}.log")
        log_path = log_dir / f"{handle.process_id}.log"
        await asyncio.to_thread(log_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(log_path.write_bytes, b"")

        process: asyncio.subprocess.Process | None = None
        writer_fd: int | None = None
        read_transport: asyncio.BaseTransport | None = None
        try:
            process, reader, writer_fd, read_transport = await self._spawn(
                text,
                cwd=cwd,
                tty=tty,
            )
            loop = asyncio.get_running_loop()
            entry = _ProcessEntry(
                handle=handle,
                command=text,
                cwd=cwd,
                tty=bool(tty),
                process=process,
                output=_RollingOutput(self._max_output_bytes),
                log_path=log_path,
                relative_log_path=relative_log,
                started_at=_utc_now(),
                started_monotonic=loop.time(),
                journal=journal,
                terminal_notifier=terminal_notifier,
                writer_fd=writer_fd,
                read_transport=read_transport,
            )
            async with self._lock:
                if self._closed:
                    raise RuntimeError("managed process runtime closed during spawn")
                self._entries[handle.process_id] = entry
                entry.reader_task = asyncio.create_task(
                    self._read_output(entry, reader),
                    name=f"qitos-process-reader-{handle.process_id}",
                )
                entry.watcher_task = asyncio.create_task(
                    self._watch_terminal(entry),
                    name=f"qitos-process-watcher-{handle.process_id}",
                )
            if journal is not None:
                try:
                    await journal.append(
                        JournalRecordType.PROCESS_STARTED,
                        {
                            "handle": handle.to_dict(),
                            "command": text,
                            "cwd": cwd,
                            "pid": process.pid,
                            "tty": bool(tty),
                            "started_at": entry.started_at,
                            "log_path": relative_log,
                        },
                        record_id=(
                            f"{owner_run_id}:process:{handle.process_id}:started"
                        ),
                    )
                except BaseException:
                    entry.suppress_terminal_record = True
                    entry.started_record_ready.set()
                    await self._terminate_entry(entry)
                    async with self._lock:
                        self._entries.pop(handle.process_id, None)
                    raise
            entry.started_record_ready.set()
            return await self._snapshot(entry, cursor=0)
        except BaseException:
            if process is not None and process.returncode is None:
                await self._terminate_raw(process)
            if writer_fd is not None:
                try:
                    os.close(writer_fd)
                except OSError:
                    pass
            if read_transport is not None:
                read_transport.close()
            raise

    async def poll(self, handle: ProcessHandle) -> ProcessSnapshot:
        entry, detached = await self._lookup(handle)
        if detached is not None:
            return detached
        assert entry is not None
        return await self._snapshot(entry, cursor=0)

    async def read(
        self,
        handle: ProcessHandle,
        *,
        cursor: int = 0,
        wait_seconds: float = 0.0,
    ) -> ProcessSnapshot:
        if wait_seconds < 0:
            raise ValueError("wait_seconds must be non-negative")
        entry, detached = await self._lookup(handle)
        if detached is not None:
            return self._read_detached(detached, cursor=cursor)
        assert entry is not None
        async with entry.condition:
            if (
                wait_seconds > 0
                and entry.status is ProcessStatus.RUNNING
                and entry.output.total_bytes <= cursor
            ):
                try:
                    await asyncio.wait_for(
                        entry.condition.wait(),
                        timeout=float(wait_seconds),
                    )
                except asyncio.TimeoutError:
                    pass
            return self._snapshot_locked(entry, cursor=cursor)

    async def write(self, handle: ProcessHandle, data: str) -> ProcessSnapshot:
        entry, detached = await self._lookup(handle)
        if detached is not None:
            raise RuntimeError("process stdin is closed")
        assert entry is not None
        if not isinstance(data, str):
            raise TypeError("process input must be a string")
        async with entry.interaction_lock:
            async with entry.condition:
                if (
                    entry.status is not ProcessStatus.RUNNING
                    or entry.termination_requested
                ):
                    raise RuntimeError("process stdin is closed")
                writer_fd = entry.writer_fd
                stdin = entry.process.stdin
                tty = entry.tty
            if tty:
                if writer_fd is None:
                    raise RuntimeError("process terminal input is unavailable")
                await self._write_fd(writer_fd, data.encode("utf-8"))
            else:
                if stdin is None or stdin.is_closing():
                    raise RuntimeError("process stdin is closed")
                stdin.write(data.encode("utf-8"))
                await stdin.drain()
        return await self._snapshot(entry, cursor=0)

    async def wait(
        self,
        handle: ProcessHandle,
        *,
        deadline_monotonic: float | None = None,
    ) -> ProcessSnapshot:
        entry, detached = await self._lookup(handle)
        if detached is not None:
            return detached
        assert entry is not None
        watcher = entry.watcher_task
        if watcher is not None and not watcher.done():
            if deadline_monotonic is None:
                await asyncio.shield(watcher)
            else:
                remaining = max(
                    0.0, deadline_monotonic - asyncio.get_running_loop().time()
                )
                try:
                    await asyncio.wait_for(asyncio.shield(watcher), timeout=remaining)
                except asyncio.TimeoutError:
                    pass
        self._raise_journal_error(entry)
        return await self._snapshot(entry, cursor=0)

    async def terminate(self, handle: ProcessHandle) -> ProcessSnapshot:
        entry, detached = await self._lookup(handle)
        if detached is not None:
            return detached
        assert entry is not None
        await self._terminate_entry(entry)
        self._raise_journal_error(entry)
        return await self._snapshot(entry, cursor=0)

    async def list(
        self,
        *,
        owner_run_id: str | None = None,
    ) -> tuple[ProcessSnapshot, ...]:
        async with self._lock:
            entries = [
                entry
                for entry in self._entries.values()
                if owner_run_id is None or entry.handle.owner_run_id == owner_run_id
            ]
            detached = [
                snapshot
                for snapshot in self._detached.values()
                if owner_run_id is None or snapshot.handle.owner_run_id == owner_run_id
            ]
        snapshots = [
            await self._snapshot(entry, cursor=entry.output.total_bytes)
            for entry in entries
        ]
        snapshots.extend(detached)
        snapshots.sort(key=lambda snapshot: snapshot.started_at)
        return tuple(snapshots)

    async def recover(
        self,
        *,
        owner_run_id: str,
        journal: SessionJournal,
    ) -> tuple[ProcessSnapshot, ...]:
        """Recover this Run without reattaching or replaying a process."""

        if not isinstance(owner_run_id, str) or not owner_run_id.strip():
            raise ValueError("owner_run_id must be a non-empty string")
        async with self._recovery_lock:
            if owner_run_id in self._recovered_runs:
                return await self.list(owner_run_id=owner_run_id)
            records = await journal.replay()
            started: dict[str, dict[str, Any]] = {}
            terminal: dict[str, ProcessSnapshot] = {}
            try:
                for record in records:
                    if record.run_id != owner_run_id:
                        continue
                    if record.type is JournalRecordType.PROCESS_STARTED:
                        raw_handle = record.payload.get("handle")
                        if not isinstance(raw_handle, Mapping):
                            raise ValueError("process.started handle is invalid")
                        handle = ProcessHandle.from_dict(raw_handle)
                        if handle.owner_run_id != owner_run_id:
                            raise ValueError("process.started owner is inconsistent")
                        started[handle.process_id] = dict(record.payload)
                    elif record.type is JournalRecordType.PROCESS_TERMINAL:
                        snapshot = ProcessSnapshot.from_dict(record.payload)
                        if snapshot.handle.owner_run_id != owner_run_id:
                            raise ValueError("process.terminal owner is inconsistent")
                        if not snapshot.terminal:
                            raise ValueError("process.terminal contains a live process")
                        terminal[snapshot.handle.process_id] = snapshot
            except (TypeError, ValueError) as exc:
                raise ProcessPersistenceError(
                    "managed process journal records are invalid"
                ) from exc

            recovered = dict(terminal)
            for process_id, payload in started.items():
                if process_id in terminal:
                    continue
                snapshot = await self._lost_snapshot(payload)
                try:
                    await journal.append(
                        JournalRecordType.PROCESS_TERMINAL,
                        snapshot.to_dict(),
                        record_id=(f"{owner_run_id}:process:{process_id}:terminal"),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    raise ProcessPersistenceError(
                        "failed to persist lost process terminal record"
                    ) from exc
                recovered[process_id] = snapshot

            async with self._lock:
                for process_id, snapshot in recovered.items():
                    if process_id not in self._entries:
                        self._detached[process_id] = snapshot
                self._prune_detached_locked()
                self._recovered_runs.add(owner_run_id)
            return await self.list(owner_run_id=owner_run_id)

    async def close(self) -> None:
        async with self._start_condition:
            self._closed = True
            while self._starting:
                await self._start_condition.wait()
            entries = list(self._entries.values())
        active = [entry for entry in entries if entry.status is ProcessStatus.RUNNING]
        if active:
            await asyncio.gather(
                *(self._terminate_entry(entry) for entry in active),
                return_exceptions=False,
            )
        watchers = [
            entry.watcher_task
            for entry in entries
            if entry.watcher_task is not None and not entry.watcher_task.done()
        ]
        if watchers:
            await asyncio.gather(*watchers, return_exceptions=False)
        errors = [entry.journal_error for entry in entries if entry.journal_error]
        if errors:
            raise ProcessPersistenceError(
                f"failed to persist {len(errors)} process terminal record(s)"
            ) from errors[0]

    async def _spawn(
        self,
        command: str,
        *,
        cwd: str,
        tty: bool,
    ) -> tuple[
        asyncio.subprocess.Process,
        asyncio.StreamReader,
        int | None,
        asyncio.BaseTransport | None,
    ]:
        process_kwargs: dict[str, Any] = {
            "cwd": cwd,
            "env": None if self._env is None else dict(self._env),
        }
        if os.name == "nt":
            process_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            process_kwargs["start_new_session"] = True
        if not tty:
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                **process_kwargs,
            )
            if process.stdout is None:
                await self._terminate_raw(process)
                raise RuntimeError("process stdout pipe was not created")
            return process, process.stdout, None, None
        if os.name == "nt":
            raise NotImplementedError("PTY processes are not supported on Windows")

        import pty

        master_fd, slave_fd = pty.openpty()
        writer_fd = os.dup(master_fd)
        os.set_blocking(writer_fd, False)
        try:
            shell = os.environ.get("SHELL") or "/bin/sh"
            process = await asyncio.create_subprocess_exec(
                shell,
                "-lc",
                command,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                **process_kwargs,
            )
        except BaseException:
            os.close(master_fd)
            os.close(writer_fd)
            raise
        finally:
            os.close(slave_fd)
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        pipe = os.fdopen(master_fd, "rb", buffering=0)
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: protocol,
            pipe,
        )
        return process, reader, writer_fd, transport

    @staticmethod
    async def _write_fd(writer_fd: int, content: bytes) -> None:
        loop = asyncio.get_running_loop()
        offset = 0
        while offset < len(content):
            try:
                offset += os.write(writer_fd, content[offset:])
                continue
            except BlockingIOError:
                writable = loop.create_future()

                def _ready() -> None:
                    if not writable.done():
                        writable.set_result(None)

                loop.add_writer(writer_fd, _ready)
                try:
                    await writable
                finally:
                    loop.remove_writer(writer_fd)

    async def _read_output(
        self,
        entry: _ProcessEntry,
        reader: asyncio.StreamReader,
    ) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                chunk = await reader.read(8192)
                if not chunk:
                    break
                rendered = decoder.decode(chunk, final=False).encode("utf-8")
                await self._record_output(entry, rendered)
            final = decoder.decode(b"", final=True).encode("utf-8")
            await self._record_output(entry, final)
        except asyncio.CancelledError:
            raise
        except OSError as exc:
            # Linux PTY masters report EIO when the final slave descriptor is
            # closed. That is the PTY equivalent of EOF, not a process or
            # output-reader failure.
            if entry.tty and exc.errno == errno.EIO:
                final = decoder.decode(b"", final=True).encode("utf-8")
                await self._record_output(entry, final)
                return
            entry.error = f"process output reader failed: {exc}"
            if entry.process.returncode is None:
                self._signal_process_group(entry.process, signal.SIGKILL)
        except Exception as exc:
            entry.error = f"process output reader failed: {exc}"
            if entry.process.returncode is None:
                self._signal_process_group(entry.process, signal.SIGKILL)

    async def _record_output(self, entry: _ProcessEntry, content: bytes) -> None:
        if not content:
            return
        await asyncio.to_thread(_append_bytes, entry.log_path, content)
        async with entry.condition:
            entry.output.append(content)
            entry.condition.notify_all()

    async def _watch_terminal(self, entry: _ProcessEntry) -> None:
        try:
            exit_code = await entry.process.wait()
            if entry.reader_task is not None:
                await entry.reader_task
            async with entry.condition:
                entry.exit_code = int(exit_code)
                entry.ended_at = _utc_now()
                if entry.error:
                    entry.status = ProcessStatus.FAILED
                elif entry.termination_requested:
                    entry.status = ProcessStatus.TERMINATED
                else:
                    entry.status = ProcessStatus.EXITED
                entry.condition.notify_all()
            await entry.started_record_ready.wait()
            terminal_persisted = entry.journal is None
            if entry.journal is not None and not entry.suppress_terminal_record:
                snapshot = await self._snapshot(entry, cursor=0)
                try:
                    await entry.journal.append(
                        JournalRecordType.PROCESS_TERMINAL,
                        snapshot.to_dict(),
                        record_id=(
                            f"{entry.handle.owner_run_id}:process:"
                            f"{entry.handle.process_id}:terminal"
                        ),
                    )
                    terminal_persisted = True
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    entry.journal_error = exc
            if terminal_persisted and entry.terminal_notifier is not None:
                snapshot = await self._snapshot(entry, cursor=0)
                try:
                    await entry.terminal_notifier(snapshot)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # The process terminal remains queryable even if an active
                    # Run can no longer accept a safe-point notification.
                    pass
        finally:
            if entry.writer_fd is not None:
                try:
                    os.close(entry.writer_fd)
                except OSError:
                    pass
                entry.writer_fd = None
            if entry.read_transport is not None:
                entry.read_transport.close()
                entry.read_transport = None

    async def _terminate_entry(self, entry: _ProcessEntry) -> None:
        async with entry.condition:
            if entry.status is not ProcessStatus.RUNNING:
                watcher = entry.watcher_task
            else:
                entry.termination_requested = True
                self._signal_process_group(entry.process, signal.SIGTERM)
                watcher = entry.watcher_task
        if entry.process.returncode is None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(entry.process.wait()),
                    timeout=self._terminate_grace_seconds,
                )
            except asyncio.TimeoutError:
                self._signal_process_group(entry.process, signal.SIGKILL)
        if watcher is not None:
            await asyncio.shield(watcher)

    async def _terminate_raw(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        self._signal_process_group(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(
                asyncio.shield(process.wait()),
                timeout=self._terminate_grace_seconds,
            )
        except asyncio.TimeoutError:
            self._signal_process_group(process, signal.SIGKILL)
            await process.wait()

    @staticmethod
    def _signal_process_group(
        process: asyncio.subprocess.Process,
        sig: signal.Signals,
    ) -> None:
        if process.returncode is not None:
            return
        try:
            if os.name == "nt":
                if sig is signal.SIGKILL:
                    process.kill()
                else:
                    process.terminate()
            else:
                os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    async def _lookup(
        self,
        handle: ProcessHandle,
    ) -> tuple[_ProcessEntry | None, ProcessSnapshot | None]:
        if not isinstance(handle, ProcessHandle):
            raise TypeError("handle must be a ProcessHandle")
        async with self._lock:
            entry = self._entries.get(handle.process_id)
            detached = self._detached.get(handle.process_id)
        if entry is not None and entry.handle == handle:
            return entry, None
        if detached is not None and detached.handle == handle:
            return None, detached
        raise ProcessNotFoundError(f"unknown process handle: {handle.process_id}")

    async def _lost_snapshot(self, payload: Mapping[str, Any]) -> ProcessSnapshot:
        required = {
            "handle",
            "command",
            "cwd",
            "pid",
            "tty",
            "started_at",
            "log_path",
        }
        if set(payload) != required:
            raise ProcessPersistenceError("process.started fields are invalid")
        raw_handle = payload.get("handle")
        if not isinstance(raw_handle, Mapping):
            raise ProcessPersistenceError("process.started handle is invalid")
        handle = ProcessHandle.from_dict(raw_handle)
        for name in ("command", "cwd", "started_at", "log_path"):
            if not isinstance(payload[name], str) or not payload[name]:
                raise ProcessPersistenceError(f"process.started {name} is invalid")
        if not isinstance(payload["tty"], bool):
            raise ProcessPersistenceError("process.started tty is invalid")
        pid = payload["pid"]
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ProcessPersistenceError("process.started pid is invalid")
        relative_log = payload["log_path"]
        output = await asyncio.to_thread(self._read_log_tail, relative_log)
        return ProcessSnapshot(
            handle=handle,
            status=ProcessStatus.LOST,
            command=payload["command"],
            cwd=payload["cwd"],
            pid=None,
            tty=payload["tty"],
            started_at=payload["started_at"],
            ended_at=_utc_now(),
            exit_code=None,
            output=output,
            error="process ownership was lost before resume",
        )

    def _read_log_tail(self, relative_log: str) -> ProcessOutput:
        path = (self.workspace_root / relative_log).resolve()
        try:
            path.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ProcessPersistenceError("process log path escapes workspace") from exc
        if not relative_log or not path.exists():
            return ProcessOutput(
                content="",
                cursor=0,
                next_cursor=0,
                total_bytes=0,
                omitted_bytes=0,
                truncated=False,
                log_path=relative_log,
            )
        total_bytes = path.stat().st_size
        retained_from = max(0, total_bytes - self._max_output_bytes)
        with path.open("rb") as stream:
            stream.seek(retained_from)
            content = stream.read()
        skipped_for_utf8 = 0
        while content and content[0] & 0xC0 == 0x80:
            content = content[1:]
            skipped_for_utf8 += 1
        omitted = retained_from + skipped_for_utf8
        return ProcessOutput(
            content=content.decode("utf-8", errors="replace"),
            cursor=0,
            next_cursor=total_bytes,
            total_bytes=total_bytes,
            omitted_bytes=omitted,
            truncated=omitted > 0,
            log_path=relative_log,
        )

    @staticmethod
    def _read_detached(
        snapshot: ProcessSnapshot,
        *,
        cursor: int,
    ) -> ProcessSnapshot:
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        encoded = snapshot.output.content.encode("utf-8")
        retained_from = snapshot.output.next_cursor - len(encoded)
        effective_cursor = min(
            max(cursor, retained_from),
            snapshot.output.next_cursor,
        )
        content = encoded[effective_cursor - retained_from :]
        skipped_for_utf8 = 0
        while content and content[0] & 0xC0 == 0x80:
            content = content[1:]
            effective_cursor += 1
            skipped_for_utf8 += 1
        omitted = max(0, retained_from - cursor) + skipped_for_utf8
        return ProcessSnapshot(
            handle=snapshot.handle,
            status=snapshot.status,
            command=snapshot.command,
            cwd=snapshot.cwd,
            pid=snapshot.pid,
            tty=snapshot.tty,
            started_at=snapshot.started_at,
            ended_at=snapshot.ended_at,
            exit_code=snapshot.exit_code,
            output=ProcessOutput(
                content=content.decode("utf-8", errors="replace"),
                cursor=cursor,
                next_cursor=snapshot.output.next_cursor,
                total_bytes=snapshot.output.total_bytes,
                omitted_bytes=omitted,
                truncated=omitted > 0,
                log_path=snapshot.output.log_path,
            ),
            error=snapshot.error,
        )

    async def _snapshot(
        self,
        entry: _ProcessEntry,
        *,
        cursor: int,
    ) -> ProcessSnapshot:
        async with entry.condition:
            return self._snapshot_locked(entry, cursor=cursor)

    @staticmethod
    def _snapshot_locked(
        entry: _ProcessEntry,
        *,
        cursor: int,
    ) -> ProcessSnapshot:
        return ProcessSnapshot(
            handle=entry.handle,
            status=entry.status,
            command=entry.command,
            cwd=entry.cwd,
            pid=entry.process.pid,
            tty=entry.tty,
            started_at=entry.started_at,
            ended_at=entry.ended_at,
            exit_code=entry.exit_code,
            output=entry.output.read(cursor, log_path=entry.relative_log_path),
            error=entry.error,
        )

    @staticmethod
    def _raise_journal_error(entry: _ProcessEntry) -> None:
        if entry.journal_error is not None:
            raise ProcessPersistenceError(
                f"failed to persist terminal state for {entry.handle.process_id}"
            ) from entry.journal_error

    def _prune_finished_locked(self) -> None:
        self._prune_detached_locked(reserve=1)
        if len(self._entries) + len(self._detached) < self._max_tracked:
            return
        finished = sorted(
            (
                entry
                for entry in self._entries.values()
                if entry.status is not ProcessStatus.RUNNING
            ),
            key=lambda entry: entry.started_monotonic,
        )
        while (
            len(self._entries) + len(self._detached) >= self._max_tracked and finished
        ):
            entry = finished.pop(0)
            self._entries.pop(entry.handle.process_id, None)

    def _prune_detached_locked(self, *, reserve: int = 0) -> None:
        limit = max(0, self._max_tracked - reserve)
        ordered = sorted(
            self._detached.values(),
            key=lambda snapshot: snapshot.started_at,
        )
        while len(self._entries) + len(self._detached) > limit and ordered:
            snapshot = ordered.pop(0)
            self._detached.pop(snapshot.handle.process_id, None)


__all__ = ["ManagedHostProcessRuntime"]
