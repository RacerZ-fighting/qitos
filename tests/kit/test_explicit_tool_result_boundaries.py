"""Regression tests for explicit ToolResult lifecycle ownership."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest

from qitos.core.agent_events import ToolExecutionEnd
from qitos.core.env import CommandCapability, Env, EnvObservation, EnvStepResult
from qitos.core.message import ToolCall
from qitos.core.tool import tool
from qitos.core.tool_executor import ToolBatchExecutor, ToolExecutionConfig
from qitos.core.tool_registry import ToolRegistry
from qitos.kit.tool.subagent import SubagentTool
from qitos.kit.tool.internal.coding_impl import CodingToolSet
from qitos.kit.tool.internal.delegating import DelegatingTool
from qitos.kit.tool.notebook import InsertNotebookCell


def _executor(
    tool_instance: Any,
    events: list[object],
    *,
    env: Env | None = None,
) -> ToolBatchExecutor:
    exposure = ToolRegistry().include_toolset(tool_instance).freeze()
    return ToolBatchExecutor(
        exposure,
        ToolExecutionConfig(env=env),
        emit=events.append,
    )


async def _execute(
    tool_instance: Any,
    args: dict[str, Any],
    *,
    env: Env | None = None,
) -> tuple[Any, ToolExecutionEnd]:
    events: list[object] = []
    result = (
        await _executor(tool_instance, events, env=env).execute_batch(
            [
                ToolCall(
                    id="call-1",
                    name=tool_instance.name,
                    arguments=args,
                )
            ]
        )
    )[0]
    end = events[-1]
    assert isinstance(end, ToolExecutionEnd)
    return result, end


@tool(name="domain_status")
def _domain_status() -> dict[str, str]:
    return {"status": "error", "state": "remote-domain-state"}


@pytest.mark.asyncio
async def test_plain_mapping_status_remains_success_through_delegating_tool() -> None:
    payload = {"status": "error", "state": "remote-domain-state"}

    result, end = await _execute(DelegatingTool(_domain_status), {})

    assert result.status == "success"
    assert result.output == payload
    assert result.error is None
    assert end.result is result
    assert end.is_error is False


def _unexpected_subagent_factory(*args: object, **kwargs: object) -> None:
    _ = args, kwargs
    raise AssertionError("invalid Agent input must not launch a Subagent")


@pytest.mark.asyncio
async def test_agent_expected_rejection_is_typed_error_and_error_event() -> None:
    agent = SubagentTool(invocation_factory=_unexpected_subagent_factory)

    result, end = await _execute(
        agent,
        {
            "description": "inspect target",
            "prompt": "  ",
            "success_criteria": ["Report the result"],
        },
    )

    assert result.status == "error"
    assert result.output == {"status": "error", "error": "prompt is required"}
    assert result.error == "prompt is required"
    assert end.result is result
    assert end.is_error is True


@pytest.mark.asyncio
async def test_regular_kit_tool_expected_error_is_typed_and_observable(
    tmp_path: Path,
) -> None:
    tool_instance = InsertNotebookCell(root_dir=str(tmp_path))

    result, end = await _execute(
        tool_instance,
        {
            "path": "notes.ipynb",
            "cell_type": "unsupported",
            "source": "text",
        },
    )

    assert result.status == "error"
    assert result.output == {
        "status": "error",
        "message": "Unsupported cell_type: unsupported",
    }
    assert result.error == "Unsupported cell_type: unsupported"
    assert end.result is result
    assert end.is_error is True


class _ErrorCommandCapability(CommandCapability):
    def __init__(self) -> None:
        self.payload = {
            "status": "error",
            "error": "command rejected by runtime",
            "exit_code": 126,
        }

    async def arun(self, command: str, timeout: float = 30) -> dict[str, Any]:
        _ = command, timeout
        return dict(self.payload)

    async def arun_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: float = 30,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> dict[str, Any]:
        _ = argv, timeout, cwd, stdin
        return dict(self.payload)

    def run(self, command: str, timeout: int = 30) -> dict[str, Any]:
        _ = command, timeout
        return dict(self.payload)

    def run_argv(
        self,
        argv: Sequence[str],
        *,
        timeout: int = 30,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> dict[str, Any]:
        _ = argv, timeout, cwd, stdin
        return dict(self.payload)


class _CommandEnv(Env):
    name = "boundary-command-env"

    def __init__(self, command: CommandCapability) -> None:
        self.command = command

    def reset(
        self,
        task: Any = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> EnvObservation:
        _ = task, workspace, kwargs
        return EnvObservation()

    def observe(self, state: Any = None) -> EnvObservation:
        _ = state
        return EnvObservation()

    def step(self, action: Any, state: Any = None) -> EnvStepResult:
        _ = action, state
        return EnvStepResult(observation=EnvObservation())

    def get_ops(self, group: str) -> Any:
        return self.command if group == "process" else None


@pytest.mark.asyncio
async def test_concrete_coding_handler_converts_capability_error_not_delegate() -> None:
    capability = _ErrorCommandCapability()
    original_payload = dict(capability.payload)
    coding = CodingToolSet(
        include_notebook=False,
        enable_lsp=False,
        enable_tasks=False,
        enable_web=False,
        profile="shell",
    )
    delegated = DelegatingTool(coding.run_command)

    result, end = await _execute(
        delegated,
        {"command": "restricted-command"},
        env=_CommandEnv(capability),
    )

    assert capability.payload == original_payload
    assert result.status == "error"
    assert result.output == original_payload
    assert result.error == "command rejected by runtime"
    assert end.result is result
    assert end.is_error is True
