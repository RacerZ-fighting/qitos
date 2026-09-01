from __future__ import annotations

import hashlib
import os
from pathlib import PurePosixPath
from typing import Any, Sequence

import pytest

from qitos.core.env import FileRevisionConflictError
from qitos.kit.env.docker_env import DockerFSCapability
from qitos.kit.env.host_env import HostFSCapability


class _DockerAtomicCommand:
    def __init__(self, content: bytes | None = None) -> None:
        self.content = content
        self.calls: list[tuple[list[str], bytes | None]] = []

    def run_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> dict[str, Any]:
        _ = timeout, cwd
        args = [str(item) for item in argv]
        self.calls.append((args, stdin))
        if args[0] == "realpath":
            return {
                "returncode": 0,
                "stdout": str(PurePosixPath("/workspace") / args[-1]),
                "stderr": "",
            }
        if args[0] != "sh":
            raise AssertionError(f"unexpected command: {args}")
        expected = args[-1]
        current = (
            hashlib.sha256(self.content).hexdigest()
            if self.content is not None
            else ""
        )
        if expected and expected != current:
            return {
                "returncode": 73,
                "stdout": "",
                "stderr": f"QITOS_CONFLICT:{current}",
            }
        previous = current
        self.content = stdin
        return {"returncode": 0, "stdout": previous, "stderr": ""}


def test_docker_atomic_write_uses_argv_stdin_and_revision_guard() -> None:
    original = b"first\n"
    command = _DockerAtomicCommand(original)
    fs = DockerFSCapability("container")
    fs.cmd = command  # type: ignore[assignment]
    expected = hashlib.sha256(original).hexdigest()

    replaced = fs.write_text_atomic(
        "demo.txt",
        "second\n",
        expected_sha256=expected,
    )

    assert replaced.previous_sha256 == expected
    assert command.content == b"second\n"
    mutation_args, mutation_stdin = command.calls[-1]
    assert mutation_args[0:2] == ["sh", "-c"]
    assert mutation_args[-2:] == ["/workspace/demo.txt", expected]
    assert mutation_stdin == b"second\n"

    with pytest.raises(FileRevisionConflictError):
        fs.write_text_atomic(
            "demo.txt",
            "stale\n",
            expected_sha256=expected,
        )
    assert command.content == b"second\n"


@pytest.fixture
def umask_0002():
    previous = os.umask(0o002)
    try:
        yield 0o002
    finally:
        os.umask(previous)


def test_new_host_file_is_created_under_the_process_umask(tmp_path, umask_0002) -> None:
    """A Tool-written file must be usable by the accounts the umask admits.

    The runtime may run model-issued commands under a second account that shares
    the Agent's group, so a new file staged privately and never widened is a
    file that account cannot read.
    """

    fs = HostFSCapability(str(tmp_path))

    fs.write_text("payload.py", "print('hi')\n")

    mode = (tmp_path / "payload.py").stat().st_mode & 0o777
    assert mode == 0o666 & ~umask_0002


def test_replacing_a_host_file_keeps_its_own_mode(tmp_path, umask_0002) -> None:
    private = tmp_path / "secret.txt"
    private.write_text("first", encoding="utf-8")
    private.chmod(0o600)
    fs = HostFSCapability(str(tmp_path))

    fs.write_text("secret.txt", "second")

    assert private.stat().st_mode & 0o777 == 0o600
    assert private.read_text(encoding="utf-8") == "second"


def test_new_host_file_narrows_with_a_restrictive_umask(tmp_path) -> None:
    previous = os.umask(0o077)
    try:
        fs = HostFSCapability(str(tmp_path))
        fs.write_text("owned.txt", "body")
    finally:
        os.umask(previous)

    assert (tmp_path / "owned.txt").stat().st_mode & 0o777 == 0o600
