from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from qitos.core.env import RuntimeCommand
from qitos.kit.env._async_process import AsyncProcessResult
from qitos.kit.env.docker_env import DockerCommandCapability, DockerEnv
from qitos.kit.env.runtime_profile import (
    RuntimeCommandProbe,
    RuntimeProfile,
    aprobe_runtime_commands,
)


@pytest.mark.asyncio
async def test_docker_command_probe_executes_inside_selected_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    async def fake_run_process(
        *,
        argv: Sequence[str] | None = None,
        stdin: bytes | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AsyncProcessResult:
        captured.append(
            {
                "argv": tuple(argv or ()),
                "stdin": stdin,
                "timeout": timeout,
                "kwargs": kwargs,
            }
        )
        return AsyncProcessResult(0, b"nuclei 3.4.0\n", b"")

    monkeypatch.setattr(
        "qitos.kit.env.docker_env.run_process",
        fake_run_process,
    )
    profile = RuntimeProfile(
        name="kali",
        commands=(
            RuntimeCommandProbe(
                name="nuclei",
                executable="nuclei",
                argv=("nuclei", "-version"),
            ),
        ),
    )

    commands = await aprobe_runtime_commands(
        profile,
        DockerCommandCapability("runtime-container", "/workspace"),
    )

    assert commands == (RuntimeCommand("nuclei", "nuclei", True, "nuclei 3.4.0"),)
    assert captured == [
        {
            "argv": (
                "docker",
                "exec",
                "-w",
                "/workspace",
                "runtime-container",
                "nuclei",
                "-version",
            ),
            "stdin": None,
            "timeout": 5.0,
            "kwargs": {},
        }
    ]


def test_docker_snapshot_declares_only_implemented_process_facilities() -> None:
    verified = (RuntimeCommand("nuclei", "nuclei", True, "verified"),)
    snapshot = DockerEnv(
        container="runtime-container",
        commands=verified,
    ).capability_snapshot()

    assert snapshot.backend == "docker"
    assert snapshot.commands == verified
    assert snapshot.operation_groups == ("file", "process")
    assert snapshot.facilities == ("file.atomic-write", "process.foreground")
    assert snapshot.has_facility("process.background") is False
    assert snapshot.has_facility("process.pty") is False
