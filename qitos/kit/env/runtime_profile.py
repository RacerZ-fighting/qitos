"""Typed runtime-profile loading and backend-local command verification."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qitos.core.env import CommandCapability, RuntimeCommand

_DEFAULT_PROBE_TIMEOUT_SECONDS = 5.0
_MAX_PROBE_DETAIL_CHARS = 500


@dataclass(frozen=True, slots=True)
class RuntimeCommandProbe:
    """Side-effect-free argv probe declared by an application profile."""

    name: str
    executable: str
    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("runtime command probe name must be non-empty")
        if not isinstance(self.executable, str) or not self.executable.strip():
            raise ValueError("runtime command probe executable must be non-empty")
        if not isinstance(self.argv, tuple) or not self.argv:
            raise TypeError("runtime command probe argv must be a non-empty tuple")
        if any(not isinstance(item, str) or not item for item in self.argv):
            raise TypeError("runtime command probe argv must contain strings")


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Application-selected profile plus commands to verify in a backend."""

    name: str
    commands: tuple[RuntimeCommandProbe, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("runtime profile name must be non-empty")
        if not isinstance(self.commands, tuple) or not all(
            isinstance(command, RuntimeCommandProbe) for command in self.commands
        ):
            raise TypeError(
                "runtime profile commands must be a tuple of RuntimeCommandProbe"
            )
        names = tuple(command.name for command in self.commands)
        if len(names) != len(set(names)):
            raise ValueError(
                "runtime profile command names must not contain duplicates"
            )


def load_runtime_profile(path: Path | None) -> RuntimeProfile:
    """Load one explicit JSON profile without probing the current machine."""

    if path is None:
        return RuntimeProfile(name="native")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid runtime profile: {path}") from exc
    root = _mapping(payload, "runtime profile")
    if root.get("schema_version") != 1:
        raise ValueError("runtime profile schema_version must be 1")
    raw_commands = root.get("commands")
    if not isinstance(raw_commands, list):
        raise TypeError("runtime profile commands must be an array")
    commands = tuple(
        _command_probe(_mapping(item, "runtime command")) for item in raw_commands
    )
    return RuntimeProfile(name=_required_string(root, "profile"), commands=commands)


def probe_runtime_commands(
    profile: RuntimeProfile,
    process: CommandCapability,
    *,
    timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> tuple[RuntimeCommand, ...]:
    """Synchronously verify commands through the selected backend provider.

    Composition roots that call this from async code must run it in a worker
    thread. Runtime execution itself remains async-native.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    return tuple(
        _probe_command(command, process, timeout_seconds)
        for command in profile.commands
    )


async def aprobe_runtime_commands(
    profile: RuntimeProfile,
    process: CommandCapability,
    *,
    timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> tuple[RuntimeCommand, ...]:
    """Asynchronously verify commands through the selected backend provider."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    verified: list[RuntimeCommand] = []
    for command in profile.commands:
        try:
            result = await process.arun_argv(
                command.argv,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            verified.append(
                RuntimeCommand(
                    command.name,
                    command.executable,
                    False,
                    _bounded(str(exc)),
                )
            )
            continue
        verified.append(_command_result(command, result))
    return tuple(verified)


def _probe_command(
    command: RuntimeCommandProbe,
    process: CommandCapability,
    timeout_seconds: float,
) -> RuntimeCommand:
    try:
        result = process.run_argv(command.argv, timeout=int(timeout_seconds))
    except Exception as exc:
        return RuntimeCommand(
            command.name,
            command.executable,
            False,
            _bounded(str(exc)),
        )
    return _command_result(command, result)


def _command_result(
    command: RuntimeCommandProbe,
    result: Mapping[str, Any],
) -> RuntimeCommand:
    returncode = result.get("returncode")
    available = (
        isinstance(returncode, int)
        and not isinstance(returncode, bool)
        and returncode == 0
    )
    output = "\n".join(
        value.strip()
        for key in ("stdout", "stderr", "error")
        if isinstance((value := result.get(key)), str) and value.strip()
    )
    detail = _bounded(output)
    if not available and not detail:
        detail = (
            f"probe exited {returncode}"
            if isinstance(returncode, int)
            else "probe did not return an exit code"
        )
    return RuntimeCommand(
        name=command.name,
        executable=command.executable,
        available=available,
        detail=detail,
    )


def _command_probe(payload: Mapping[str, Any]) -> RuntimeCommandProbe:
    raw_argv = payload.get("probe_argv")
    if not isinstance(raw_argv, list) or not raw_argv:
        raise TypeError("command probe_argv must be a non-empty array")
    if any(not isinstance(item, str) or not item for item in raw_argv):
        raise TypeError("command probe_argv must contain non-empty strings")
    executable = _required_string(payload, "executable")
    if raw_argv[0] != executable:
        raise ValueError("command probe_argv must start with its executable")
    return RuntimeCommandProbe(
        name=_required_string(payload, "name"),
        executable=executable,
        argv=tuple(raw_argv),
    )


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _bounded(value: str) -> str:
    if len(value) <= _MAX_PROBE_DETAIL_CHARS:
        return value
    return value[: _MAX_PROBE_DETAIL_CHARS - 3] + "..."


__all__ = [
    "RuntimeCommandProbe",
    "RuntimeProfile",
    "aprobe_runtime_commands",
    "load_runtime_profile",
    "probe_runtime_commands",
]
