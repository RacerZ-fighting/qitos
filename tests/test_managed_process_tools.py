from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

import pytest

from qitos.core.journal import JournalRecordType
from qitos.core.process import ProcessStatus
from qitos.kit.env.host_env import HostCommandCapability
from qitos.kit.env.managed_process import ManagedHostProcessRuntime
from qitos.kit.journal import JsonlSessionJournal
from qitos.kit.tool.internal.coding_impl import CodingToolSet


def _python_command(source: str) -> str:
    return shlex.join([sys.executable, "-u", "-c", source])


def _runtime_context(
    run_id: str,
    *,
    journal: JsonlSessionJournal | None = None,
    timeout: float = 2.0,
) -> dict[str, object]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    def remaining_seconds() -> float:
        return max(0.0, deadline - loop.time())

    return {
        "run_id": run_id,
        "journal": journal,
        "deadline_monotonic": deadline,
        "remaining_seconds": remaining_seconds,
    }


@pytest.mark.asyncio
async def test_shell_toolset_manages_background_process_through_one_async_owner(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("run-1", {})
    tools = CodingToolSet(workspace_root=str(tmp_path), profile="shell")
    context = _runtime_context("run-1", journal=journal)

    started = await tools.run_command.execute(
        {
            "command": _python_command(
                "print('ready', flush=True); value=input(); "
                "print('echo:' + value, flush=True)"
            ),
            "run_in_background": True,
        },
        runtime_context=context,
    )
    process_id = started["process_id"]
    first = await tools.process_read.execute(
        {"process_id": process_id, "wait_seconds": 1.0},
        runtime_context=context,
    )
    await tools.process_write.execute(
        {"process_id": process_id, "data": "hello\n"},
        runtime_context=context,
    )
    terminal = await tools.process_wait.execute(
        {"process_id": process_id},
        runtime_context=context,
    )
    incremental = await tools.process_read.execute(
        {
            "process_id": process_id,
            "cursor": first["output"]["next_cursor"],
        },
        runtime_context=context,
    )
    listed = await tools.process_list.execute({}, runtime_context=context)

    assert first["status"] == ProcessStatus.RUNNING.value
    assert "ready" in first["output"]["content"]
    assert terminal["status"] == ProcessStatus.EXITED.value
    assert "echo:hello" in incremental["output"]["content"]
    assert [item["process_id"] for item in listed["processes"]] == [process_id]

    await tools.ateardown({})
    records = await journal.replay()
    lifecycle = [
        record.type
        for record in records
        if record.type
        in {JournalRecordType.PROCESS_STARTED, JournalRecordType.PROCESS_TERMINAL}
    ]
    assert lifecycle == [
        JournalRecordType.PROCESS_STARTED,
        JournalRecordType.PROCESS_TERMINAL,
    ]
    await journal.close()


@pytest.mark.asyncio
async def test_process_handles_are_bound_to_the_active_run(tmp_path: Path) -> None:
    tools = CodingToolSet(workspace_root=str(tmp_path), profile="shell")
    started = await tools.run_command.execute(
        {
            "command": _python_command("import time; time.sleep(60)"),
            "run_in_background": True,
        },
        runtime_context=_runtime_context("run-1", journal=None),
    )

    denied = await tools.process_read.execute(
        {"process_id": started["process_id"]},
        runtime_context=_runtime_context("run-2", journal=None),
    )

    assert denied["status"] == "error"
    assert "unknown process handle" in denied["message"]
    await tools.ateardown({})


@pytest.mark.asyncio
async def test_resume_marks_started_without_terminal_as_lost_once(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("run-1", {})
    log_path = Path(".qitos/processes/proc_interrupted.log")
    absolute_log = tmp_path / log_path
    absolute_log.parent.mkdir(parents=True)
    absolute_log.write_text("partial output\n", encoding="utf-8")
    await journal.append(
        JournalRecordType.PROCESS_STARTED,
        {
            "handle": {
                "process_id": "proc_interrupted",
                "owner_run_id": "run-1",
            },
            "command": "long-running-command",
            "cwd": str(tmp_path),
            "pid": 12345,
            "tty": False,
            "started_at": "2026-08-14T00:00:00+00:00",
            "log_path": str(log_path),
        },
        record_id="run-1:process:proc_interrupted:started",
    )
    runtime = ManagedHostProcessRuntime(str(tmp_path))

    first = await runtime.recover(owner_run_id="run-1", journal=journal)
    second = await runtime.recover(owner_run_id="run-1", journal=journal)
    records = await journal.replay()
    terminals = [
        record
        for record in records
        if record.type is JournalRecordType.PROCESS_TERMINAL
    ]

    assert first[0].status is ProcessStatus.LOST
    assert first[0].pid is None
    assert first[0].error is not None
    assert "partial output" in first[0].output.content
    assert second == first
    assert len(terminals) == 1
    assert terminals[0].payload == first[0].to_dict()

    await runtime.close()
    await journal.close()


@pytest.mark.asyncio
async def test_fork_does_not_inherit_parent_process_ownership(tmp_path: Path) -> None:
    parent = JsonlSessionJournal(tmp_path / "journal")
    await parent.create("parent", {})
    await parent.append(
        JournalRecordType.PROCESS_STARTED,
        {
            "handle": {
                "process_id": "proc_parent",
                "owner_run_id": "parent",
            },
            "command": "parent-command",
            "cwd": str(tmp_path),
            "pid": 12345,
            "tty": False,
            "started_at": "2026-08-14T00:00:00+00:00",
            "log_path": ".qitos/processes/proc_parent.log",
        },
        record_id="parent:process:proc_parent:started",
    )
    boundary = await parent.append(
        JournalRecordType.STATE_SNAPSHOT,
        {"step_id": 0, "state": {}},
        record_id="parent:snapshot:0",
    )
    child = await parent.fork(boundary, "child")
    runtime = ManagedHostProcessRuntime(str(tmp_path))

    recovered = await runtime.recover(owner_run_id="child", journal=child)
    child_records = await child.replay()

    assert recovered == ()
    assert all(
        record.type is not JournalRecordType.PROCESS_TERMINAL
        for record in child_records
    )

    await runtime.close()
    await child.close()
    await parent.close()


@pytest.mark.asyncio
async def test_host_capability_exposes_recovery_contract(tmp_path: Path) -> None:
    journal = JsonlSessionJournal(tmp_path / "journal")
    await journal.create("run-1", {})
    capability = HostCommandCapability(str(tmp_path))

    assert await capability.arecover(owner_run_id="run-1", journal=journal) == ()

    await capability.aclose()
    await journal.close()
