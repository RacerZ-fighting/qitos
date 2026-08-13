from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Sequence

import pytest

from qitos.core.action import Action, ActionStatus
from qitos.core.env import CommandCapability
from qitos.core.tool_registry import ToolRegistry
from qitos.engine.action_executor import ActionExecutor
from qitos.kit.env import CapabilityEnv
from qitos.kit.env.host_env import HostFSCapability
from qitos.kit.tool.internal.coding_impl import CodingToolSet


class _RecordingProcess(CommandCapability):
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    def run(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        raise AssertionError(f"search tools must not use shell strings: {command}")

    async def arun(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        return self.run(command, timeout=timeout)

    def run_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> Dict[str, Any]:
        args = [str(item) for item in argv]
        self.calls.append((args, cwd))
        assert stdin is None
        if "--files-with-matches" in args:
            stdout = "a.py\0"
        elif "--files" in args:
            stdout = "a.py\0"
        else:
            stdout = json.dumps(
                {
                    "type": "match",
                    "data": {
                        "path": {"text": "a.py"},
                        "line_number": 1,
                        "lines": {"text": "remote value\n"},
                    },
                }
            )
        return {
            "status": "success",
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
        }

    async def arun_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> Dict[str, Any]:
        return self.run_argv(
            argv,
            timeout=timeout,
            cwd=cwd,
            stdin=stdin,
        )


class _StaticProcess(_RecordingProcess):
    def __init__(
        self,
        result: Dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.result = result
        self.error = error

    def run_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> Dict[str, Any]:
        self.calls.append(([str(item) for item in argv], cwd))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _tool_context(tmp_path: Path, process: CommandCapability) -> Dict[str, Any]:
    env = CapabilityEnv(
        {
            "file": HostFSCapability(str(tmp_path)),
            "process": process,
        }
    )
    return {
        "env": env,
        "ops": {
            "file": env.get_ops("file"),
            "process": env.get_ops("process"),
        },
    }


@pytest.mark.asyncio
async def test_coding_tools_use_selected_environment_instead_of_local_fallback(
    tmp_path: Path,
) -> None:
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    (local / "src").mkdir(parents=True)
    (remote / "src").mkdir(parents=True)
    (local / "src" / "a.py").write_text("local value\n", encoding="utf-8")
    (remote / "src" / "a.py").write_text("remote value\n", encoding="utf-8")

    process = _RecordingProcess()
    env = CapabilityEnv(
        {
            "file": HostFSCapability(str(remote)),
            "process": process,
        }
    )
    context = {
        "env": env,
        "ops": {
            "file": env.get_ops("file"),
            "process": env.get_ops("process"),
        },
    }
    tools = CodingToolSet(workspace_root=str(local), include_notebook=False)

    read = await tools.read_file.execute(
        {"path": "src/a.py"}, runtime_context=context
    )
    glob = await tools.glob.execute(
        {"pattern": "*.py", "path": "src"}, runtime_context=context
    )
    grep = await tools.grep.execute(
        {"pattern": "remote", "path": "src"}, runtime_context=context
    )
    write = await tools.write_file.execute(
        {"path": "src/new.txt", "content": "created remotely"},
        runtime_context=context,
    )

    assert read["content"] == "remote value"
    assert read["content_sha256"] == hashlib.sha256(b"remote value\n").hexdigest()
    assert glob["files"] == ["src/a.py"]
    assert grep["matches"] == [
        {"path": "src/a.py", "line": 1, "text": "remote value"}
    ]
    assert write["status"] == "success"
    assert (remote / "src" / "new.txt").read_text(encoding="utf-8") == (
        "created remotely"
    )
    assert not (local / "src" / "new.txt").exists()
    assert all(call[0][0] == "rg" for call in process.calls)
    assert all(call[1] == "src" for call in process.calls)


@pytest.mark.asyncio
async def test_canonical_write_uses_read_revision_as_compare_and_swap(
    tmp_path: Path,
) -> None:
    target = tmp_path / "demo.txt"
    target.write_text("first\n", encoding="utf-8")
    process = _RecordingProcess()
    context = _tool_context(tmp_path, process)
    tools = CodingToolSet(workspace_root=str(tmp_path), include_notebook=False)

    read = await tools.read_file.execute(
        {"path": "demo.txt"},
        runtime_context=context,
    )
    replaced = await tools.write_file.execute(
        {
            "path": "demo.txt",
            "content": "second\n",
            "expected_sha256": read["content_sha256"],
        },
        runtime_context=context,
    )
    stale = await tools.write_file.execute(
        {
            "path": "demo.txt",
            "content": "stale\n",
            "expected_sha256": read["content_sha256"],
        },
        runtime_context=context,
    )

    assert replaced["status"] == "success"
    assert replaced["previous_sha256"] == read["content_sha256"]
    assert stale["status"] == "error"
    assert stale["error_category"] == "file_revision_conflict"
    assert stale["current_sha256"] == replaced["content_sha256"]
    assert target.read_text(encoding="utf-8") == "second\n"


@pytest.mark.asyncio
async def test_canonical_tools_preserve_offsets_replace_all_and_match_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "a.py"
    target.write_text("one\ntwo\nremote\nremote\nfive\n", encoding="utf-8")
    process = _RecordingProcess()
    env = CapabilityEnv(
        {
            "file": HostFSCapability(str(tmp_path)),
            "process": process,
        }
    )
    context = {
        "env": env,
        "ops": {
            "file": env.get_ops("file"),
            "process": env.get_ops("process"),
        },
    }
    tools = CodingToolSet(
        workspace_root=str(tmp_path),
        include_notebook=False,
    )

    read = await tools.read_file.execute(
        {"path": "src/a.py", "line_offset": 2, "line_count": 2},
        runtime_context=context,
    )
    edit = await tools.edit_file.execute(
        {
            "path": "src/a.py",
            "old_text": "remote",
            "new_text": "updated",
            "replace_all": True,
        },
        runtime_context=context,
    )
    grep = await tools.grep.execute(
        {"pattern": "remote", "path": "src"},
        runtime_context=context,
    )
    files = await tools.grep.execute(
        {
            "pattern": "remote",
            "path": "src",
            "files_with_matches": True,
        },
        runtime_context=context,
    )

    assert read["numbered_content"] == "3\tremote\n4\tremote"
    assert read["has_more"] is True
    assert edit["message"] == "Replaced 2 occurrences in src/a.py"
    assert target.read_text(encoding="utf-8").count("updated") == 2
    assert grep["matches"] == [
        {"path": "src/a.py", "line": 1, "text": "remote value"}
    ]
    assert files["matches"] == [{"path": "src/a.py"}]


@pytest.mark.asyncio
async def test_search_tools_use_nul_paths_stable_order_and_explicit_visibility(
    tmp_path: Path,
) -> None:
    process = _StaticProcess(
        {
            "status": "success",
            "returncode": 0,
            "stdout": "z.py\0line\nbreak.py\0a.py\0",
            "stderr": "",
        }
    )
    tools = CodingToolSet(
        workspace_root=str(tmp_path),
        include_notebook=False,
        allow_local_fallback=False,
    )

    result = await tools.glob.execute(
        {
            "pattern": "*.py",
            "include_hidden": True,
            "include_ignored": True,
        },
        runtime_context=_tool_context(tmp_path, process),
    )

    assert result["status"] == "success"
    assert result["files"] == ["a.py", "line\nbreak.py", "z.py"]
    assert result["total_count"] == 3
    assert result["returned_count"] == 3
    assert result["exit_code"] == 0
    argv = process.calls[0][0]
    assert argv[:4] == ["rg", "--files", "--sort=path", "--null"]
    assert "--hidden" in argv
    assert "--no-ignore" in argv
    assert argv[-1] == "."


@pytest.mark.asyncio
async def test_search_limits_are_strict_and_do_not_start_process(
    tmp_path: Path,
) -> None:
    process = _RecordingProcess()
    tools = CodingToolSet(
        workspace_root=str(tmp_path),
        include_notebook=False,
        allow_local_fallback=False,
    )
    context = _tool_context(tmp_path, process)

    low = await tools.glob.execute(
        {"pattern": "*.py", "limit": 0},
        runtime_context=context,
    )
    high = await tools.grep.execute(
        {"pattern": "value", "limit": 2001},
        runtime_context=context,
    )
    missing = await tools.grep.execute(
        {"pattern": "value", "path": "missing"},
        runtime_context=context,
    )

    assert low["status"] == "error"
    assert low["message"] == "limit must be between 1 and 2000"
    assert high["status"] == "error"
    assert high["message"] == "limit must be between 1 and 2000"
    assert missing["status"] == "error"
    assert missing["error_category"] == "search_path_not_found"
    assert process.calls == []


@pytest.mark.asyncio
async def test_grep_returns_sorted_matches_and_context_records(tmp_path: Path) -> None:
    def event(kind: str, path: str, line: int, text: str) -> str:
        return json.dumps(
            {
                "type": kind,
                "data": {
                    "path": {"text": path},
                    "line_number": line,
                    "lines": {"text": f"{text}\n"},
                },
            }
        )

    stdout = "\n".join(
        [
            event("match", "z.py", 8, "needle z"),
            event("context", "a.py", 1, "before"),
            event("match", "a.py", 2, "needle a"),
            event("context", "a.py", 3, "after"),
        ]
    )
    process = _StaticProcess(
        {
            "status": "success",
            "returncode": 0,
            "stdout": stdout,
            "stderr": "warning",
        }
    )
    tools = CodingToolSet(
        workspace_root=str(tmp_path),
        include_notebook=False,
        allow_local_fallback=False,
    )

    result = await tools.grep.execute(
        {
            "pattern": "needle",
            "context": 1,
            "limit": 1,
            "include_hidden": True,
            "include_ignored": True,
        },
        runtime_context=_tool_context(tmp_path, process),
    )

    assert result["matches"] == [
        {"path": "a.py", "line": 2, "text": "needle a"}
    ]
    assert result["records"] == [
        {"kind": "context", "path": "a.py", "line": 1, "text": "before"},
        {"kind": "match", "path": "a.py", "line": 2, "text": "needle a"},
        {"kind": "context", "path": "a.py", "line": 3, "text": "after"},
    ]
    assert result["total_count"] == 2
    assert result["returned_count"] == 1
    assert result["truncated"] is True
    assert result["stderr"] == "warning"
    argv = process.calls[0][0]
    assert argv[:4] == ["rg", "--color", "never", "--sort=path"]
    assert "--json" in argv
    assert "--context" in argv
    assert "--max-columns-preview" in argv
    assert "--hidden" in argv
    assert "--no-ignore" in argv
    assert argv[-3:] == ["--", "needle", "."]


@pytest.mark.asyncio
async def test_search_distinguishes_empty_errors_and_timeout(tmp_path: Path) -> None:
    tools = CodingToolSet(
        workspace_root=str(tmp_path),
        include_notebook=False,
        allow_local_fallback=False,
    )
    empty_process = _StaticProcess(
        {
            "status": "partial",
            "returncode": 1,
            "stdout": "",
            "stderr": "",
        }
    )
    failed_process = _StaticProcess(
        {
            "status": "partial",
            "returncode": 2,
            "stdout": "",
            "stderr": "invalid regex",
        }
    )
    timeout_process = _StaticProcess(
        error=subprocess.TimeoutExpired(["rg"], timeout=30)
    )

    empty = await tools.grep.execute(
        {"pattern": "missing"},
        runtime_context=_tool_context(tmp_path, empty_process),
    )
    failed = await tools.grep.execute(
        {"pattern": "["},
        runtime_context=_tool_context(tmp_path, failed_process),
    )
    timed_out = await tools.glob.execute(
        {"pattern": "*.py"},
        runtime_context=_tool_context(tmp_path, timeout_process),
    )

    assert empty["status"] == "success"
    assert empty["matches"] == []
    assert empty["exit_code"] == 1
    assert failed["status"] == "error"
    assert failed["error_category"] == "search_process_error"
    assert failed["exit_code"] == 2
    assert failed["stderr"] == "invalid regex"
    assert timed_out["status"] == "timed_out"
    assert timed_out["error_category"] == "process_timeout"


def test_canonical_read_returns_structured_tool_error(tmp_path: Path) -> None:
    tools = CodingToolSet(
        workspace_root=str(tmp_path),
        include_notebook=False,
    )

    result = tools.read_file(path="missing.txt")

    assert result == {
        "status": "error",
        "message": "File not found: missing.txt",
    }


@pytest.mark.asyncio
async def test_executor_promotes_structured_tool_failure_to_action_error(
    tmp_path: Path,
) -> None:
    env = CapabilityEnv(
        {
            "file": HostFSCapability(str(tmp_path)),
            "process": _RecordingProcess(),
        }
    )
    registry = ToolRegistry()
    registry.register_toolset(
        CodingToolSet(
            workspace_root=str(tmp_path),
            include_notebook=False,
            profile="workspace",
            auto_approve=True,
            allow_local_fallback=False,
        ),
        namespace="",
    )

    result = (
        await ActionExecutor(registry).execute(
            [Action(name="read_file", args={"path": "missing.txt"})],
            env=env,
        )
    )[0]

    assert result.status is ActionStatus.ERROR
    assert result.output == {"message": "File not found: missing.txt"}
    assert result.metadata["error_category"] == "tool_reported_error"
