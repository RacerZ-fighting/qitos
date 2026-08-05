from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pytest

from qitos import Action, AgentModule, Decision, Engine, StateSchema, ToolRegistry
from qitos.core.tool import BaseTool, ToolPermission, ToolSpec
from qitos.engine import RuntimeBudget
from qitos.kit.env import HostEnv
from qitos.kit.env.host_env import HostCommandCapability, HostFSCapability


class _OpsWriteFile(BaseTool):
    def __init__(self):
        super().__init__(
            ToolSpec(
                name="write_file",
                description="Write a file through env file ops.",
                parameters={
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                required=["filename", "content"],
                permissions=ToolPermission(filesystem_write=True),
                required_ops=["file"],
            )
        )

    def execute(
        self, args: Dict[str, Any], runtime_context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        ctx = runtime_context or {}
        ops = dict(ctx.get("ops") or {})
        file_ops = ops.get("file")
        if file_ops is None:
            return {"status": "error", "message": "Missing file ops"}
        filename = str(args.get("filename", ""))
        content = str(args.get("content", ""))
        file_ops.write_text(filename, content)
        return {"status": "success", "path": filename, "size": len(content)}


def test_host_env_replace_lines_and_command(tmp_path: Path):
    target = tmp_path / "m.py"
    target.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    env = HostEnv(workspace_root=str(tmp_path))

    out = env.execute_action(
        Action(
            name="replace_lines",
            args={
                "path": "m.py",
                "start_line": 2,
                "end_line": 2,
                "replacement": "    return a + b",
            },
        )
    )
    assert isinstance(out, dict) and out.get("status") == "success"
    assert "return a + b" in target.read_text(encoding="utf-8")

    run = env.execute_action(
        Action(name="run_command", args={"command": 'python -c "print(42)"'})
    )
    assert isinstance(run, dict)
    assert int(run.get("returncode", 1)) == 0


def test_host_file_capability_is_binary_safe_and_symlink_confined(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "data.bin").write_bytes(b"a\x00b")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    fs = HostFSCapability(str(root))

    assert fs.read_bytes("data.bin") == b"a\x00b"
    assert fs.read_bytes("data.bin", limit=2) == b"a\x00"
    assert fs.stat("data.bin").is_file is True
    assert [entry.path for entry in fs.list_entries()] == ["data.bin", "escape"]
    with pytest.raises(PermissionError, match="outside root"):
        fs.resolve_path("escape/secret.txt", allow_missing=True)
    with pytest.raises(PermissionError, match="parent traversal"):
        fs.resolve_path("../outside/secret.txt", allow_missing=True)


def test_host_file_capability_returns_bounded_text_chunks(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "sample.txt").write_text(
        "first\r\nsecond-line-is-long\r\nthird\r\n",
        encoding="utf-8",
        newline="",
    )
    fs = HostFSCapability(str(root))

    chunk = fs.read_text_chunk(
        "sample.txt",
        offset=1,
        limit=2,
        max_bytes=100,
        max_line_bytes=10,
    )

    assert chunk.content == "second-...\nthird"
    assert chunk.offset == 1
    assert chunk.line_count == 2
    assert chunk.total_lines == 3
    assert chunk.has_more is False
    assert chunk.truncated is True
    assert chunk.line_ending == "crlf"


def test_host_file_capability_rejects_binary_text_chunk(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "binary.bin").write_bytes(b"hello\x00world\n")
    fs = HostFSCapability(str(root))

    with pytest.raises(UnicodeError, match="NUL"):
        fs.read_text_chunk("binary.bin")


def test_host_command_capability_preserves_literal_argv(tmp_path: Path):
    command = HostCommandCapability(str(tmp_path))

    result = command.run_argv(
        ["python", "-c", "import sys; print(sys.argv[1])", "$(touch injected)"],
    )

    assert result["returncode"] == 0
    assert result["stdout"].strip() == "$(touch injected)"
    assert not (tmp_path / "injected").exists()


@dataclass
class _State(StateSchema):
    done: bool = False


class _EnvOnlyAgent(AgentModule[_State, Dict[str, Any], Action]):
    def __init__(self):
        super().__init__(tool_registry=None)

    def init_state(self, task: str, **kwargs: Any) -> _State:
        return _State(task=task, max_steps=2)

    def decide(self, state: _State, observation: Dict[str, Any]):
        if state.current_step == 0:
            return Decision.act(
                actions=[
                    Action(
                        name="write_file",
                        args={"filename": "x.txt", "content": "hello"},
                    )
                ]
            )
        return Decision.final("done")

    def reduce(
        self, state: _State, observation: Dict[str, Any], decision: Decision[Action]
    ) -> _State:
        if decision.mode == "final":
            state.done = True
        return state


def test_engine_executes_ops_aware_tool_with_env(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(_OpsWriteFile())

    class _EnvOpsAgent(_EnvOnlyAgent):
        def __init__(self):
            super().__init__()
            self.tool_registry = registry

    env = HostEnv(workspace_root=str(tmp_path))
    engine = Engine(agent=_EnvOpsAgent(), env=env, budget=RuntimeBudget(max_steps=3))
    result = engine.run("write file")
    assert result.state.final_result == "done"
    assert (tmp_path / "x.txt").exists()
    assert (tmp_path / "x.txt").read_text(encoding="utf-8") == "hello"


def test_engine_fails_when_required_ops_missing_env(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(_OpsWriteFile())

    class _NoEnvAgent(_EnvOnlyAgent):
        def __init__(self):
            super().__init__()
            self.tool_registry = registry

    result = Engine(agent=_NoEnvAgent(), budget=RuntimeBudget(max_steps=2)).run(
        "write file"
    )
    assert result.state.stop_reason == "env_capability_mismatch"
    assert result.step_count == 0
    assert result.events
    end_events = [e for e in result.events if e.phase.value == "END"]
    assert end_events
    issues = end_events[-1].payload.get("issues", [])
    assert issues and issues[0].get("code") == "ENV_REQUIRED_OPS_MISSING"
