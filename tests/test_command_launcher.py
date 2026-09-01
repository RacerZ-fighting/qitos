"""Every command an environment spawns passes through its launcher."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from qitos.kit.env import CommandLauncher, HostEnv
from qitos.kit.env.managed_process import ManagedHostProcessRuntime

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="a command launcher requires a POSIX shell"
)


def _marking_launcher(tmp_path: Path, marker: str) -> CommandLauncher:
    """A launcher that records the argv it received, then runs it."""

    record = tmp_path / "launched.txt"
    script = tmp_path / "launcher.py"
    script.write_text(
        "import os, sys\n"
        f"open({str(record)!r}, 'a', encoding='utf-8').write({marker!r} + '\\n')\n"
        "os.execv(sys.argv[1], sys.argv[1:])\n",
        encoding="utf-8",
    )
    return CommandLauncher((sys.executable, str(script)))


def _launch_record(tmp_path: Path) -> list[str]:
    record = tmp_path / "launched.txt"
    if not record.is_file():
        return []
    return record.read_text(encoding="utf-8").split()


@pytest.mark.asyncio
async def test_foreground_shell_command_runs_through_the_launcher(
    tmp_path: Path,
) -> None:
    marker = "shell"
    env = HostEnv(
        workspace_root=str(tmp_path),
        command_environment={"PATH": os.defpath},
        command_launcher=_marking_launcher(tmp_path, marker),
    )

    result = await env.cmd.arun("printf ready", timeout=20)

    assert result["returncode"] == 0
    assert result["stdout"] == "ready"
    assert _launch_record(tmp_path) == [marker]


@pytest.mark.asyncio
async def test_argv_command_runs_through_the_launcher(tmp_path: Path) -> None:
    marker = "argv"
    env = HostEnv(
        workspace_root=str(tmp_path),
        command_environment={"PATH": os.defpath},
        command_launcher=_marking_launcher(tmp_path, marker),
    )

    result = await env.cmd.arun_argv([sys.executable, "-c", "print('ready')"])

    assert result["returncode"] == 0
    assert result["stdout"].strip() == "ready"
    assert _launch_record(tmp_path) == [marker]


@pytest.mark.asyncio
async def test_managed_background_process_runs_through_the_launcher(
    tmp_path: Path,
) -> None:
    marker = "managed"
    runtime = ManagedHostProcessRuntime(
        str(tmp_path),
        launcher=_marking_launcher(tmp_path, marker),
    )
    try:
        started = await runtime.start(
            "printf ready", owner_run_id="run-1", cwd=str(tmp_path)
        )
        snapshot = await runtime.wait(
            started.handle,
            deadline_monotonic=asyncio.get_running_loop().time() + 20.0,
        )
    finally:
        await runtime.close()

    assert snapshot.exit_code == 0
    assert "ready" in snapshot.output.content
    assert _launch_record(tmp_path) == [marker]


@pytest.mark.asyncio
async def test_an_environment_without_a_launcher_spawns_directly(
    tmp_path: Path,
) -> None:
    """The launcher stays opt-in: nothing wraps commands until one is set."""

    env = HostEnv(
        workspace_root=str(tmp_path),
        command_environment={"PATH": os.defpath},
    )

    result = await env.cmd.arun("printf ready", timeout=20)

    assert result["returncode"] == 0
    assert result["stdout"] == "ready"
    assert _launch_record(tmp_path) == []


def test_a_launcher_and_an_explicit_command_capability_are_exclusive(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        HostEnv(
            workspace_root=str(tmp_path),
            cmd=HostEnv(workspace_root=str(tmp_path)).cmd,
            command_launcher=_marking_launcher(tmp_path, "unused"),
        )
