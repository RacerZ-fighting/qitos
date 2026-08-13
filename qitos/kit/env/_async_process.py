"""Cancellation-safe asyncio subprocess primitives for environment capabilities."""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AsyncProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


async def run_process(
    *,
    argv: Sequence[str] | None = None,
    shell_command: str | None = None,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    stdin: bytes | None = None,
    timeout: float | None = None,
) -> AsyncProcessResult:
    """Run one process and reap its process group on timeout or cancellation."""

    if (argv is None) == (shell_command is None):
        raise ValueError("exactly one of argv or shell_command is required")
    process_kwargs = {
        "cwd": cwd,
        "env": None if env is None else dict(env),
        "stdin": asyncio.subprocess.PIPE if stdin is not None else None,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
    }
    if os.name != "nt":
        process_kwargs["start_new_session"] = True
    if shell_command is not None:
        process = await asyncio.create_subprocess_shell(shell_command, **process_kwargs)
    else:
        assert argv is not None
        process = await asyncio.create_subprocess_exec(
            *(str(item) for item in argv), **process_kwargs
        )
    try:
        communication = process.communicate(stdin)
        if timeout is None:
            stdout, stderr = await communication
        else:
            stdout, stderr = await asyncio.wait_for(communication, timeout=timeout)
    except BaseException:
        await _terminate_process_group(process)
        raise
    return AsyncProcessResult(
        returncode=int(process.returncode or 0),
        stdout=stdout,
        stderr=stderr,
    )


async def _terminate_process_group(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float = 0.5,
) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        return
    except asyncio.TimeoutError:
        pass
    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


__all__ = ["AsyncProcessResult", "run_process"]
