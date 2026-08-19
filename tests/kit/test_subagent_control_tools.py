"""Behavior tests for model-facing Subagent control tools."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from qitos.core.subagent import SubagentInvocation, SubagentLaunchContext, SubagentLaunchRequest
from qitos.core.runtime_input import RuntimeInput
from qitos.kit.subagent import SubagentSupervisor
from qitos.kit.tool.subagent import (
    SubagentInterruptTool,
    SubagentMessageTool,
    SubagentStatusTool,
    SubagentWaitTool,
)


async def _ready_invocation(**kwargs: Any) -> SubagentInvocation:
    return SubagentInvocation(**kwargs)


class _ClosableEngine:
    async def aclose(self) -> None:
        return None


class _MailboxEngine(_ClosableEngine):
    active_run_id = "subagent-run"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cleaned = asyncio.Event()
        self.messages: list[RuntimeInput] = []

    async def arun(self, task: str, **kwargs: object) -> object:
        _ = task, kwargs
        self.started.set()
        try:
            await self.release.wait()
        finally:
            self.cleaned.set()
        return SimpleNamespace(
            state=SimpleNamespace(final_result="done", stop_reason="completed"),
            records=[],
            step_count=1,
            total_tokens=2,
            run_id=self.active_run_id,
        )

    async def apost_runtime_event(
        self,
        event: RuntimeInput,
        *,
        run_id: str,
    ) -> bool:
        assert run_id == self.active_run_id
        self.messages.append(event)
        return True

    def cancel(self, mode: str) -> None:
        assert mode == "immediate"
        self.release.set()


def _supervisor(engine: _MailboxEngine) -> SubagentSupervisor:
    return SubagentSupervisor(
        invocation_factory=lambda request, _context: _ready_invocation(
            engine=engine,
            task=request.task,
        )
    )


async def _launch(supervisor: SubagentSupervisor):
    return await supervisor.launch(
        SubagentLaunchRequest(task="inspect", description="inspect service"),
        SubagentLaunchContext(parent_run_id="parent-run"),
        background=True,
    )


@pytest.mark.asyncio
async def test_message_enters_active_subagent_mailbox() -> None:
    engine = _MailboxEngine()
    supervisor = _supervisor(engine)
    launched = await _launch(supervisor)
    await asyncio.wait_for(engine.started.wait(), timeout=1)
    tool = SubagentMessageTool(supervisor)

    result = await tool.execute(
        {"subagent_id": launched.handle.subagent_id, "content": "Check TLS too."},
        runtime_context={"parent_run_id": "parent-run"},
    )

    assert result["status"] == "success"
    assert result["subagent_status"] == "running"
    assert result["accepted"] is True
    assert len(engine.messages) == 1
    assert engine.messages[0].kind == "agent.parent.message"
    assert engine.messages[0].payload == {"content": "Check TLS too."}
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_wait_timeout_preserves_running_subagent() -> None:
    engine = _MailboxEngine()
    supervisor = _supervisor(engine)
    launched = await _launch(supervisor)
    await asyncio.wait_for(engine.started.wait(), timeout=1)

    result = await SubagentWaitTool(supervisor).execute(
        {"subagent_id": launched.handle.subagent_id, "timeout_seconds": 0},
        runtime_context={"parent_run_id": "parent-run"},
    )

    assert result["status"] == "success"
    assert result["subagent_status"] == "running"
    assert result["ready"] is False
    assert supervisor.active_count == 1
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_status_reports_live_subagent_progress() -> None:
    class _ProgressEngine(_MailboxEngine):
        step_count = 7
        token_usage = 1234
        cost_usage_usd = 0.0
        usage_complete = True
        cost_complete = True

    engine = _ProgressEngine()
    supervisor = _supervisor(engine)
    launched = await _launch(supervisor)
    await asyncio.wait_for(engine.started.wait(), timeout=1)

    result = await SubagentStatusTool(supervisor).execute(
        {"subagent_id": launched.handle.subagent_id},
        runtime_context={"parent_run_id": "parent-run"},
    )

    assert result["status"] == "success"
    assert result["subagent_status"] == "running"
    assert result["steps"] == 7
    assert result["elapsed_seconds"] >= 0
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_interrupt_reports_subagent_terminal_without_cancelling_tool() -> None:
    engine = _MailboxEngine()
    supervisor = _supervisor(engine)
    launched = await _launch(supervisor)
    await asyncio.wait_for(engine.started.wait(), timeout=1)

    result = await SubagentInterruptTool(supervisor).execute(
        {"subagent_id": launched.handle.subagent_id, "timeout_seconds": 1},
        runtime_context={"parent_run_id": "parent-run"},
    )

    assert result["status"] == "success"
    assert result["subagent_status"] == "cancelled"
    assert result["ready"] is True
    assert engine.cleaned.is_set()
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_unknown_or_foreign_handle_has_stable_status() -> None:
    engine = _MailboxEngine()
    supervisor = _supervisor(engine)
    launched = await _launch(supervisor)
    await asyncio.wait_for(engine.started.wait(), timeout=1)
    tool = SubagentStatusTool(supervisor)

    unknown = await tool.execute(
        {"subagent_id": "missing"},
        runtime_context={"parent_run_id": "parent-run"},
    )
    foreign = await tool.execute(
        {"subagent_id": launched.handle.subagent_id},
        runtime_context={"parent_run_id": "different-run"},
    )

    assert unknown["status"] == "success"
    assert unknown["subagent_status"] == "unknown"
    assert unknown["ready"] is True
    assert foreign["subagent_status"] == "unknown"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_control_tools_use_executor_owned_run_id() -> None:
    engine = _MailboxEngine()
    supervisor = _supervisor(engine)
    launched = await _launch(supervisor)
    await asyncio.wait_for(engine.started.wait(), timeout=1)
    tool = SubagentStatusTool(supervisor)

    owned = await tool.execute(
        {"subagent_id": launched.handle.subagent_id},
        runtime_context={"run_id": "parent-run"},
    )
    conflicting_legacy_parent = await tool.execute(
        {"subagent_id": launched.handle.subagent_id},
        runtime_context={
            "run_id": "different-run",
            "parent_run_id": "parent-run",
        },
    )

    assert owned["subagent_status"] == "running"
    assert conflicting_legacy_parent["subagent_status"] == "unknown"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_terminal_subagent_rejects_message_with_actionable_result() -> None:
    engine = _MailboxEngine()
    supervisor = _supervisor(engine)
    launched = await _launch(supervisor)
    await asyncio.wait_for(engine.started.wait(), timeout=1)
    engine.release.set()
    await supervisor.wait(launched.handle, timeout_seconds=1)

    result = await SubagentMessageTool(supervisor).execute(
        {"subagent_id": launched.handle.subagent_id, "content": "Continue."},
        runtime_context={"parent_run_id": "parent-run"},
    )

    assert result["status"] == "success"
    assert result["subagent_status"] == "completed"
    assert result["accepted"] is False
    assert "launch a new Subagent" in result["message"]
    await supervisor.aclose()
