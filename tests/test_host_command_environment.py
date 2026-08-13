from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import pytest

from qitos.kit.env.host_env import HostCommandCapability
from qitos.kit.tool.shell import RunCommand


def _print_environment(name: str) -> list[str]:
    return [
        sys.executable,
        "-c",
        f"import os; print(os.environ.get({name!r}, '<missing>'))",
    ]


def test_host_command_uses_inherited_environment_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    name = "QITOS_TEST_INHERITED_ENV"
    monkeypatch.setenv(name, "from-parent")

    result = HostCommandCapability(str(tmp_path)).run_argv(
        _print_environment(name)
    )

    assert result["status"] == "success"
    assert result["stdout"].strip() == "from-parent"


def test_explicit_environment_reaches_all_host_command_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inherited_name = "QITOS_TEST_FILTERED_ENV"
    configured_name = "QITOS_TEST_EXPLICIT_ENV"
    monkeypatch.setenv(inherited_name, "must-not-leak")
    configured = {configured_name: "configured"}
    capability = HostCommandCapability(str(tmp_path), env=configured)
    configured[configured_name] = "mutated-after-construction"

    argv_result = capability.run_argv(_print_environment(configured_name))
    shell_result = capability.run(
        shlex.join(_print_environment(inherited_name))
    )
    background = capability.start(
        shlex.join(_print_environment(configured_name)),
        stdout_path="background.log",
    )
    _, wait_status = os.waitpid(background["pid"], 0)

    assert os.waitstatus_to_exitcode(wait_status) == 0
    assert argv_result["stdout"].strip() == "configured"
    assert shell_result["stdout"].strip() == "<missing>"
    assert (tmp_path / "background.log").read_text(encoding="utf-8").strip() == (
        "configured"
    )


@pytest.mark.asyncio
async def test_run_command_accepts_an_explicit_process_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inherited_name = "QITOS_TEST_RUN_COMMAND_FILTERED_ENV"
    configured_name = "QITOS_TEST_RUN_COMMAND_EXPLICIT_ENV"
    monkeypatch.setenv(inherited_name, "must-not-leak")
    tool = RunCommand(
        workspace_root=str(tmp_path),
        process_env={configured_name: "configured"},
    )

    configured_result = await tool.execute(
        {"command": shlex.join(_print_environment(configured_name))}
    )
    filtered_result = await tool.execute(
        {"command": shlex.join(_print_environment(inherited_name))}
    )

    assert configured_result["status"] == "success"
    assert configured_result["stdout"].strip() == "configured"
    assert filtered_result["status"] == "success"
    assert filtered_result["stdout"].strip() == "<missing>"
