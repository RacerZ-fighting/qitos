from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import pytest

from qitos.core.env import (
    CommandCapability,
    RuntimeCapabilitySnapshot,
    RuntimeCommand,
    RuntimeLimitation,
)
from qitos.kit.env.runtime_profile import (
    load_runtime_profile,
    probe_runtime_commands,
)


class _ProbeCommands(CommandCapability):
    def __init__(self) -> None:
        self.argv: list[tuple[str, ...]] = []

    def run(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        raise AssertionError(f"runtime probes must not use a shell: {command}")

    async def arun(self, command: str, timeout: float = 30) -> Dict[str, Any]:
        return self.run(command, timeout=int(timeout))

    def run_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> Dict[str, Any]:
        _ = timeout, cwd, stdin
        captured = tuple(argv)
        self.argv.append(captured)
        if captured[-1] == "available":
            return {"returncode": 0, "stdout": "verified", "stderr": ""}
        return {"returncode": 9, "stdout": "", "stderr": "missing"}

    async def arun_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: float = 30,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> Dict[str, Any]:
        return self.run_argv(
            argv,
            timeout=int(timeout),
            cwd=cwd,
            stdin=stdin,
        )


def test_runtime_snapshot_round_trips_without_mutable_fields() -> None:
    snapshot = RuntimeCapabilitySnapshot(
        backend="local",
        working_directory="/workspace",
        operation_groups=("file", "process"),
        facilities=("process.foreground",),
        commands=(RuntimeCommand("shell", "sh", True, "verified"),),
        limitations=(RuntimeLimitation("bounded-output", "Output is bounded."),),
    )

    restored = RuntimeCapabilitySnapshot.from_dict(snapshot.to_dict())

    assert restored == snapshot
    with pytest.raises(TypeError, match="must be a tuple"):
        RuntimeCapabilitySnapshot(
            backend="local",
            working_directory="/workspace",
            operation_groups=["file"],  # type: ignore[arg-type]
        )


def test_runtime_profile_probes_each_command_through_backend(tmp_path: Path) -> None:
    profile_path = tmp_path / "runtime.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "test-runtime",
                "commands": [
                    {
                        "name": "present",
                        "executable": "probe",
                        "probe_argv": ["probe", "available"],
                    },
                    {
                        "name": "absent",
                        "executable": "probe",
                        "probe_argv": ["probe", "absent"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    backend = _ProbeCommands()

    profile = load_runtime_profile(profile_path)
    commands = probe_runtime_commands(profile, backend)

    assert profile.name == "test-runtime"
    assert backend.argv == [("probe", "available"), ("probe", "absent")]
    assert [command.available for command in commands] == [True, False]
    assert commands[1].detail == "missing"


def test_runtime_profile_rejects_duplicate_command_names(tmp_path: Path) -> None:
    profile_path = tmp_path / "runtime.json"
    command = {
        "name": "duplicate",
        "executable": "probe",
        "probe_argv": ["probe", "available"],
    }
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "test-runtime",
                "commands": [command, command],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain duplicates"):
        load_runtime_profile(profile_path)
