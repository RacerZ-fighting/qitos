from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from qitos.core.env import RuntimeCapabilitySnapshot
from qitos.kit.env import CapabilityEnv


@dataclass
class _Provider:
    ok: bool = True
    message: str = ""

    def health_check(self) -> dict[str, Any]:
        return {"ok": self.ok, "message": self.message}


def test_capability_env_exposes_stable_named_ops() -> None:
    file_ops = object()
    process_ops = object()
    env = CapabilityEnv(
        {"process": process_ops, "file": file_ops},
        name="attempt",
        attestation={"attempt_id": "a-1"},
        snapshot=RuntimeCapabilitySnapshot(
            backend="remote-attempt",
            working_directory="/workspace",
            operation_groups=("file", "process"),
        ),
    )

    assert env.capability_groups == ("file", "process")
    assert env.get_ops("file") is file_ops
    assert env.get_ops("process") is process_ops
    assert env.get_ops("missing") is None
    assert env.has_ops("file") is True
    assert env.capability_snapshot().backend == "remote-attempt"
    assert env.capability_snapshot().working_directory == "/workspace"
    assert env.attestation == {"attempt_id": "a-1"}


def test_capability_env_rejects_invalid_composition() -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        CapabilityEnv({}, name=" ")
    with pytest.raises(ValueError, match="group names must be non-empty"):
        CapabilityEnv({"": object()})
    with pytest.raises(ValueError, match="provider must be non-null: file"):
        CapabilityEnv({"file": None})
    with pytest.raises(ValueError, match="operation groups must match"):
        CapabilityEnv(
            {"file": object()},
            snapshot=RuntimeCapabilitySnapshot(
                backend="attempt",
                working_directory="/workspace",
                operation_groups=("process",),
            ),
        )


def test_capability_env_observation_tracks_only_action_identity() -> None:
    env = CapabilityEnv({"file": object()})

    first = env.reset(task="demo")
    assert first.data == {
        "capability_groups": ["file"],
        "last_action": None,
    }

    result = env.step({"actions": [{"name": "read_file", "args": {}}]})
    assert result.done is False
    assert result.observation.data["last_action"] == "read_file"


def test_capability_env_reports_provider_health_failures() -> None:
    env = CapabilityEnv(
        {
            "file": _Provider(),
            "process": _Provider(ok=False, message="attempt is closed"),
            "search": object(),
        }
    )

    result = env.health_check()

    assert result["ok"] is False
    assert result["failed_groups"] == ["process"]
    assert result["checks"]["file"]["ok"] is True
    assert result["checks"]["process"] == {
        "ok": False,
        "message": "attempt is closed",
    }
    assert result["checks"]["search"] == {"ok": True}
