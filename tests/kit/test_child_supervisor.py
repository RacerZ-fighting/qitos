"""Structured-concurrency tests for the Run-owned child supervisor."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from qitos.core.child import (
    ChildInvocation,
    ChildLaunchRequest,
    ChildStatus,
)
from qitos.kit.child import ChildSupervisor


def _request(task: str = "inspect") -> ChildLaunchRequest:
    return ChildLaunchRequest(task=task, description=f"{task} task")


@pytest.mark.asyncio
async def test_wait_timeout_does_not_cancel_child() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class Engine:
        active_run_id = "child-run"

        async def arun(self, task: str, **kwargs: object) -> object:
            assert task == "inspect"
            assert kwargs == {}
            started.set()
            await release.wait()
            return SimpleNamespace(
                state=SimpleNamespace(final_result="done", stop_reason="completed"),
                records=[],
                step_count=1,
                total_tokens=2,
                run_id=self.active_run_id,
            )

        def cancel(self, mode: str) -> None:
            assert mode == "immediate"
            release.set()

    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: ChildInvocation(
            engine=Engine(),
            task=request.task,
        )
    )
    launched = await supervisor.launch(
        _request(),
        {},
        parent_run_id="parent-run",
        background=True,
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    waiting = await supervisor.wait(launched.handle, timeout_seconds=0)

    assert waiting is not None
    assert waiting.status is ChildStatus.RUNNING
    assert supervisor.active_count == 1

    release.set()
    terminal = await supervisor.wait(launched.handle, timeout_seconds=1)
    assert terminal is not None
    assert terminal.status is ChildStatus.COMPLETED
    assert await supervisor.aclose(wait_seconds=1) == 0


@pytest.mark.asyncio
async def test_terminal_delivery_remains_owned_until_close() -> None:
    delivery_started = asyncio.Event()
    delivery_cancelled = asyncio.Event()

    class Engine:
        active_run_id = "child-run"

        async def arun(self, task: str, **kwargs: object) -> object:
            _ = task, kwargs
            return SimpleNamespace(
                state=SimpleNamespace(final_result="done", stop_reason="completed"),
                records=[],
                step_count=1,
                total_tokens=2,
                run_id=self.active_run_id,
            )

        def cancel(self, mode: str) -> None:
            _ = mode

    async def post_runtime_event(_event: object) -> bool:
        delivery_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            delivery_cancelled.set()
            raise
        return True  # pragma: no cover

    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: ChildInvocation(
            engine=Engine(),
            task=request.task,
        )
    )
    launched = await supervisor.launch(
        _request(),
        {"post_runtime_event": post_runtime_event},
        parent_run_id="parent-run",
        background=True,
    )
    await asyncio.wait_for(delivery_started.wait(), timeout=1)

    terminal = supervisor.result(launched.handle)
    assert terminal is not None
    assert terminal.status is ChildStatus.COMPLETED
    assert supervisor.active_count == 0
    assert await supervisor.aclose(wait_seconds=1) == 0
    assert delivery_cancelled.is_set()
    assert supervisor.result(launched.handle) == terminal


@pytest.mark.asyncio
async def test_interrupt_waits_for_started_child_cleanup() -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()

    class Engine:
        active_run_id = "child-run"

        async def arun(self, task: str, **kwargs: object) -> object:
            _ = task, kwargs
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        def cancel(self, mode: str) -> None:
            assert mode == "immediate"

    supervisor = ChildSupervisor(
        invocation_factory=lambda request, _context: ChildInvocation(
            engine=Engine(),
            task=request.task,
        )
    )
    launched = await supervisor.launch(
        _request(),
        {},
        parent_run_id="parent-run",
        background=True,
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    terminal = await supervisor.interrupt(launched.handle, wait_seconds=1)

    assert terminal is not None
    assert terminal.status is ChildStatus.CANCELLED
    assert cleaned.is_set()
    assert supervisor.active_count == 0
    assert await supervisor.aclose(wait_seconds=0) == 0


@pytest.mark.asyncio
async def test_close_terminalizes_child_cancelled_before_task_start() -> None:
    factory_called = False

    def invocation_factory(request, _context):
        nonlocal factory_called
        factory_called = True
        return ChildInvocation(engine=object(), task=request.task)

    supervisor = ChildSupervisor(invocation_factory=invocation_factory)
    launched = await supervisor.launch(
        _request(),
        {},
        parent_run_id="parent-run",
        background=True,
    )

    assert await supervisor.aclose(wait_seconds=0) == 0
    terminal = supervisor.result(launched.handle)
    assert terminal is not None
    assert terminal.status is ChildStatus.CANCELLED
    assert factory_called is False
