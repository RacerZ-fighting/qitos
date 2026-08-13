"""Behavior tests for model-facing Child control tools."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from qitos.core.child import ChildInvocation, ChildLaunchRequest
from qitos.core.runtime_input import RuntimeInput
from qitos.kit.child import ChildSupervisor
from qitos.kit.tool.agent import (
    ChildInterruptTool,
    ChildMessageTool,
    ChildStatusTool,
    ChildWaitTool,
)


class _MailboxEngine:
    active_run_id = "child-run"

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


def _supervisor(engine: _MailboxEngine) -> ChildSupervisor:
    return ChildSupervisor(
        invocation_factory=lambda request, _context: ChildInvocation(
            engine=engine,
            task=request.task,
        )
    )


async def _launch(supervisor: ChildSupervisor):
    return await supervisor.launch(
        ChildLaunchRequest(task="inspect", description="inspect service"),
        {},
        parent_run_id="parent-run",
        background=True,
    )


@pytest.mark.asyncio
async def test_message_enters_active_child_mailbox() -> None:
    engine = _MailboxEngine()
    supervisor = _supervisor(engine)
    launched = await _launch(supervisor)
    await asyncio.wait_for(engine.started.wait(), timeout=1)
    tool = ChildMessageTool(supervisor)

    result = await tool.execute(
        {"child_id": launched.handle.child_id, "content": "Check TLS too."},
        runtime_context={"parent_run_id": "parent-run"},
    )

    assert result["status"] == "success"
    assert result["child_status"] == "running"
    assert result["accepted"] is True
    assert len(engine.messages) == 1
    assert engine.messages[0].kind == "agent.parent.message"
    assert engine.messages[0].payload == {"content": "Check TLS too."}
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_wait_timeout_preserves_running_child() -> None:
    engine = _MailboxEngine()
    supervisor = _supervisor(engine)
    launched = await _launch(supervisor)
    await asyncio.wait_for(engine.started.wait(), timeout=1)

    result = await ChildWaitTool(supervisor).execute(
        {"child_id": launched.handle.child_id, "timeout_seconds": 0},
        runtime_context={"parent_run_id": "parent-run"},
    )

    assert result["status"] == "success"
    assert result["child_status"] == "running"
    assert result["ready"] is False
    assert supervisor.active_count == 1
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_interrupt_reports_child_terminal_without_cancelling_tool() -> None:
    engine = _MailboxEngine()
    supervisor = _supervisor(engine)
    launched = await _launch(supervisor)
    await asyncio.wait_for(engine.started.wait(), timeout=1)

    result = await ChildInterruptTool(supervisor).execute(
        {"child_id": launched.handle.child_id, "timeout_seconds": 1},
        runtime_context={"parent_run_id": "parent-run"},
    )

    assert result["status"] == "success"
    assert result["child_status"] == "cancelled"
    assert result["ready"] is True
    assert engine.cleaned.is_set()
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_unknown_or_foreign_handle_has_stable_status() -> None:
    engine = _MailboxEngine()
    supervisor = _supervisor(engine)
    launched = await _launch(supervisor)
    await asyncio.wait_for(engine.started.wait(), timeout=1)
    tool = ChildStatusTool(supervisor)

    unknown = await tool.execute(
        {"child_id": "missing"},
        runtime_context={"parent_run_id": "parent-run"},
    )
    foreign = await tool.execute(
        {"child_id": launched.handle.child_id},
        runtime_context={"parent_run_id": "different-run"},
    )

    assert unknown["status"] == "success"
    assert unknown["child_status"] == "unknown"
    assert unknown["ready"] is True
    assert foreign["child_status"] == "unknown"
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_terminal_child_rejects_message_with_actionable_result() -> None:
    engine = _MailboxEngine()
    supervisor = _supervisor(engine)
    launched = await _launch(supervisor)
    await asyncio.wait_for(engine.started.wait(), timeout=1)
    engine.release.set()
    await supervisor.wait(launched.handle, timeout_seconds=1)

    result = await ChildMessageTool(supervisor).execute(
        {"child_id": launched.handle.child_id, "content": "Continue."},
        runtime_context={"parent_run_id": "parent-run"},
    )

    assert result["status"] == "success"
    assert result["child_status"] == "completed"
    assert result["accepted"] is False
    assert "launch a new Child" in result["message"]
    await supervisor.aclose()
