from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

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
            stdout = "a.py\n"
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


def test_coding_tools_use_selected_environment_instead_of_local_fallback(
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

    read = tools.read_file.execute({"path": "src/a.py"}, runtime_context=context)
    glob = tools.glob_files.execute(
        {"pattern": "*.py", "path": "src"}, runtime_context=context
    )
    grep = tools.grep_files.execute(
        {"pattern": "remote", "path": "src"}, runtime_context=context
    )
    write = tools.write_file.execute(
        {"path": "src/new.txt", "content": "created remotely"},
        runtime_context=context,
    )

    assert read["content"] == "remote value"
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


def test_modern_aliases_preserve_offsets_replace_all_and_match_paths(
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
        expose_modern_names=True,
    )

    read = tools.Read.execute(
        {"file_path": "src/a.py", "offset": 2, "limit": 2},
        runtime_context=context,
    )
    edit = tools.Edit.execute(
        {
            "file_path": "src/a.py",
            "old_string": "remote",
            "new_string": "updated",
            "replace_all": True,
        },
        runtime_context=context,
    )
    grep = tools.Grep.execute(
        {"pattern": "remote", "path": "src", "output_mode": "content"},
        runtime_context=context,
    )
    files = tools.Grep.execute(
        {
            "pattern": "remote",
            "path": "src",
            "output_mode": "files_with_matches",
        },
        runtime_context=context,
    )

    assert read == "3\tremote\n4\tremote\n[truncated: use offset=4 to continue; total_lines=5]"
    assert edit == "Replaced 2 occurrences in src/a.py"
    assert target.read_text(encoding="utf-8").count("updated") == 2
    assert grep == "src/a.py:1:remote value"
    assert files == "src/a.py"


def test_modern_alias_returns_structured_tool_error(tmp_path: Path) -> None:
    tools = CodingToolSet(
        workspace_root=str(tmp_path),
        include_notebook=False,
        expose_modern_names=True,
    )

    result = tools.Read(file_path="missing.txt")

    assert result == {
        "status": "error",
        "message": "File not found: missing.txt",
    }


def test_executor_promotes_structured_tool_failure_to_action_error(
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
            expose_legacy_aliases=False,
            auto_approve=True,
            allow_local_fallback=False,
        ),
        namespace="",
    )

    result = ActionExecutor(registry).execute(
        [Action(name="read_file", args={"path": "missing.txt"})],
        env=env,
    )[0]

    assert result.status is ActionStatus.ERROR
    assert result.output == {
        "status": "error",
        "message": "File not found: missing.txt",
    }
    assert result.metadata["error_category"] == "tool_reported_error"
