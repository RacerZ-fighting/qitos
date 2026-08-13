from __future__ import annotations

import asyncio
import os
import shlex
import sys
from pathlib import Path

import pytest

from qitos.core.journal import JournalError, JournalRecordType
from qitos.core.process import ProcessPersistenceError, ProcessStatus
from qitos.kit.env.managed_process import ManagedHostProcessRuntime
from qitos.kit.journal import JsonlSessionJournal


def _python_command(source: str) -> str:
    return shlex.join([sys.executable, "-u", "-c", source])


@pytest.mark.asyncio
async def test_managed_process_supports_incremental_output_stdin_and_wait(
    tmp_path: Path,
) -> None:
    runtime = ManagedHostProcessRuntime(str(tmp_path))
    started = await runtime.start(
        _python_command(
            "print('ready', flush=True); "
            "value = input(); "
            "print('received:' + value + ':你好', flush=True)"
        ),
        owner_run_id="run-1",
        cwd=str(tmp_path),
    )
    first = await runtime.read(
        started.handle,
        cursor=0,
        wait_seconds=2.0,
    )

    assert first.status is ProcessStatus.RUNNING
    assert "ready" in first.output.content

    await runtime.write(started.handle, "payload\n")
    terminal = await runtime.wait(
        started.handle,
        deadline_monotonic=asyncio.get_running_loop().time() + 2.0,
    )
    incremental = await runtime.read(
        started.handle,
        cursor=first.output.next_cursor,
    )

    assert terminal.status is ProcessStatus.EXITED
    assert terminal.exit_code == 0
    assert "received:payload:你好" in incremental.output.content
    assert (tmp_path / terminal.output.log_path).read_text(
        encoding="utf-8"
    ) == "ready\nreceived:payload:你好\n"

    await runtime.close()
    pending = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and task.get_name().startswith("qitos-process-")
        and not task.done()
    ]
    assert pending == []


@pytest.mark.asyncio
async def test_managed_process_bounds_memory_but_persists_complete_output(
    tmp_path: Path,
) -> None:
    runtime = ManagedHostProcessRuntime(str(tmp_path), max_output_bytes=64)
    started = await runtime.start(
        _python_command("print('α' * 200, flush=True)"),
        owner_run_id="run-1",
        cwd=str(tmp_path),
    )
    terminal = await runtime.wait(
        started.handle,
        deadline_monotonic=asyncio.get_running_loop().time() + 2.0,
    )

    assert terminal.status is ProcessStatus.EXITED
    assert terminal.output.truncated is True
    assert terminal.output.omitted_bytes > 0
    assert len(terminal.output.content.encode("utf-8")) <= 64
    assert (tmp_path / terminal.output.log_path).read_text(encoding="utf-8") == (
        "α" * 200 + "\n"
    )
    await runtime.close()


@pytest.mark.asyncio
async def test_managed_process_termination_reaps_the_process_group(
    tmp_path: Path,
) -> None:
    runtime = ManagedHostProcessRuntime(
        str(tmp_path),
        terminate_grace_seconds=0.05,
    )
    started = await runtime.start(
        _python_command(
            "import signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('waiting', flush=True); time.sleep(60)"
        ),
        owner_run_id="run-1",
        cwd=str(tmp_path),
    )
    observed = await runtime.read(
        started.handle,
        cursor=0,
        wait_seconds=2.0,
    )
    assert "waiting" in observed.output.content

    terminal = await runtime.terminate(started.handle)

    assert terminal.status is ProcessStatus.TERMINATED
    assert terminal.exit_code is not None
    await runtime.close()


@pytest.mark.asyncio
async def test_cancelled_wait_does_not_cancel_the_process_watcher(
    tmp_path: Path,
) -> None:
    runtime = ManagedHostProcessRuntime(str(tmp_path))
    started = await runtime.start(
        _python_command("import time; time.sleep(60)"),
        owner_run_id="run-1",
        cwd=str(tmp_path),
    )
    waiting = asyncio.create_task(runtime.wait(started.handle))
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert (await runtime.poll(started.handle)).status is ProcessStatus.RUNNING
    assert (await runtime.terminate(started.handle)).terminal is True
    await runtime.close()


@pytest.mark.asyncio
async def test_managed_process_enforces_active_process_limit(tmp_path: Path) -> None:
    runtime = ManagedHostProcessRuntime(str(tmp_path), max_processes=1)
    first = await runtime.start(
        _python_command("import time; time.sleep(60)"),
        owner_run_id="run-1",
        cwd=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="concurrency limit"):
        await runtime.start(
            _python_command("print('not started')"),
            owner_run_id="run-1",
            cwd=str(tmp_path),
        )

    await runtime.terminate(first.handle)
    await runtime.close()


@pytest.mark.asyncio
async def test_concurrent_starts_share_one_atomic_process_limit(
    tmp_path: Path,
) -> None:
    runtime = ManagedHostProcessRuntime(str(tmp_path), max_processes=1)
    results = await asyncio.gather(
        runtime.start(
            _python_command("import time; time.sleep(60)"),
            owner_run_id="run-1",
            cwd=str(tmp_path),
        ),
        runtime.start(
            _python_command("import time; time.sleep(60)"),
            owner_run_id="run-1",
            cwd=str(tmp_path),
        ),
        return_exceptions=True,
    )

    started = [item for item in results if not isinstance(item, BaseException)]
    rejected = [item for item in results if isinstance(item, RuntimeError)]
    assert len(started) == 1
    assert len(rejected) == 1
    assert "concurrency limit" in str(rejected[0])

    await runtime.terminate(started[0].handle)
    await runtime.close()


@pytest.mark.asyncio
async def test_managed_process_journals_one_started_and_terminal_record(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("run-1", {})
    runtime = ManagedHostProcessRuntime(str(tmp_path))
    started = await runtime.start(
        _python_command("print('done')"),
        owner_run_id="run-1",
        cwd=str(tmp_path),
        journal=journal,
    )
    terminal = await runtime.wait(
        started.handle,
        deadline_monotonic=asyncio.get_running_loop().time() + 2.0,
    )

    records = await journal.replay()
    process_records = [
        record
        for record in records
        if record.type
        in {JournalRecordType.PROCESS_STARTED, JournalRecordType.PROCESS_TERMINAL}
    ]
    assert [record.type for record in process_records] == [
        JournalRecordType.PROCESS_STARTED,
        JournalRecordType.PROCESS_TERMINAL,
    ]
    assert terminal.status is ProcessStatus.EXITED
    assert process_records[0].payload["handle"] == started.handle.to_dict()
    assert process_records[1].payload["handle"] == started.handle.to_dict()

    await runtime.close()
    await journal.close()


class _StartFailingJournal(JsonlSessionJournal):
    async def append(
        self,
        record_type: JournalRecordType,
        payload,
        *,
        record_id: str,
    ):
        if record_type is JournalRecordType.PROCESS_STARTED:
            raise JournalError("injected process.started failure")
        return await super().append(record_type, payload, record_id=record_id)


class _TerminalFailingJournal(JsonlSessionJournal):
    async def append(
        self,
        record_type: JournalRecordType,
        payload,
        *,
        record_id: str,
    ):
        if record_type is JournalRecordType.PROCESS_TERMINAL:
            raise JournalError("injected process.terminal failure")
        return await super().append(record_type, payload, record_id=record_id)


@pytest.mark.asyncio
async def test_started_record_failure_reaps_process_and_returns_no_handle(
    tmp_path: Path,
) -> None:
    journal = _StartFailingJournal(tmp_path / "journal")
    await journal.create("run-1", {})
    runtime = ManagedHostProcessRuntime(str(tmp_path))

    with pytest.raises(JournalError, match="process.started"):
        await runtime.start(
            _python_command("import time; time.sleep(60)"),
            owner_run_id="run-1",
            cwd=str(tmp_path),
            journal=journal,
        )

    assert await runtime.list() == ()
    assert [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and task.get_name().startswith("qitos-process-")
        and not task.done()
    ] == []
    await runtime.close()
    await journal.close()


@pytest.mark.asyncio
async def test_terminal_record_failure_is_reported_during_runtime_cleanup(
    tmp_path: Path,
) -> None:
    journal = _TerminalFailingJournal(tmp_path / "journal")
    await journal.create("run-1", {})
    runtime = ManagedHostProcessRuntime(str(tmp_path))
    notifications = []

    async def notify(snapshot):
        notifications.append(snapshot)
        return True

    await runtime.start(
        _python_command("import time; time.sleep(60)"),
        owner_run_id="run-1",
        cwd=str(tmp_path),
        journal=journal,
        terminal_notifier=notify,
    )

    with pytest.raises(ProcessPersistenceError, match="terminal"):
        await runtime.close()

    assert notifications == []
    await journal.close()


@pytest.mark.asyncio
async def test_rejected_terminal_notification_keeps_snapshot_queryable(
    tmp_path: Path,
) -> None:
    runtime = ManagedHostProcessRuntime(str(tmp_path))
    notified = asyncio.Event()

    async def reject(snapshot):
        assert snapshot.terminal is True
        notified.set()
        raise RuntimeError("mailbox closed")

    started = await runtime.start(
        _python_command("print('done')"),
        owner_run_id="run-1",
        cwd=str(tmp_path),
        terminal_notifier=reject,
    )
    terminal = await runtime.wait(
        started.handle,
        deadline_monotonic=asyncio.get_running_loop().time() + 2.0,
    )

    await asyncio.wait_for(notified.wait(), timeout=1)
    assert terminal.status is ProcessStatus.EXITED
    assert (await runtime.poll(started.handle)) == terminal
    await runtime.close()


@pytest.mark.skipif(os.name == "nt", reason="stdlib PTY support is POSIX-only")
@pytest.mark.asyncio
async def test_managed_pty_accepts_interactive_input(tmp_path: Path) -> None:
    runtime = ManagedHostProcessRuntime(str(tmp_path))
    started = await runtime.start(
        _python_command("value = input('prompt>'); print('echo:' + value, flush=True)"),
        owner_run_id="run-1",
        cwd=str(tmp_path),
        tty=True,
    )
    prompt = await runtime.read(started.handle, cursor=0, wait_seconds=2.0)
    assert "prompt>" in prompt.output.content

    await runtime.write(started.handle, "hello\n")
    terminal = await runtime.wait(
        started.handle,
        deadline_monotonic=asyncio.get_running_loop().time() + 2.0,
    )

    assert terminal.status is ProcessStatus.EXITED
    assert "echo:hello" in terminal.output.content
    await runtime.close()
