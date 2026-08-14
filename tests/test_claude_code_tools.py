"""Tests for sub-agents, cron, and worktree management."""

import asyncio
import os
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
from qitos.core.child import (
    ChildHandle,
    ChildInvocation,
    ChildLaunchRequest,
    ChildRuntimeContext,
    ChildStatus,
)

from qitos.kit.tool.internal.coding_utils import (
    is_image_file,
    is_notebook_file,
    is_pdf_file,
    read_image_as_base64,
)


async def _ready_invocation(**kwargs: Any) -> ChildInvocation:
    return ChildInvocation(**kwargs)


class _ClosableEngine:
    async def aclose(self) -> None:
        return None


# ── File detection helpers ────────────────────────────────────────────────────


class TestFileDetection:
    def test_image_extensions(self):
        assert is_image_file("photo.png")
        assert is_image_file("photo.jpg")
        assert is_image_file("photo.jpeg")
        assert is_image_file("photo.gif")
        assert is_image_file("photo.webp")
        assert is_image_file("photo.svg")
        assert not is_image_file("photo.txt")
        assert not is_image_file("photo.py")

    def test_pdf_extensions(self):
        assert is_pdf_file("doc.pdf")
        assert not is_pdf_file("doc.txt")
        assert not is_pdf_file("doc.py")

    def test_notebook_extensions(self):
        assert is_notebook_file("analysis.ipynb")
        assert not is_notebook_file("analysis.py")
        assert not is_notebook_file("analysis.json")

    def test_case_insensitive(self):
        assert is_image_file("Photo.PNG")
        assert is_pdf_file("Doc.PDF")
        assert is_notebook_file("Analysis.IPYNB")


# ── Image reading ─────────────────────────────────────────────────────────────


class TestImageReading:
    def test_read_image_as_base64_nonexistent(self):
        result = read_image_as_base64("/nonexistent/path/image.png")
        assert result is None

    def test_read_image_as_base64_real_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Write a minimal PNG header
            f.write(b"\x89PNG\r\n\x1a\n")
            f.flush()
            path = f.name

        try:
            result = read_image_as_base64(path)
            assert result is not None
            assert result.startswith("data:image/png;base64,")
        finally:
            os.unlink(path)


# ── CronScheduler ─────────────────────────────────────────────────────────────


class TestCronScheduler:
    def test_create_job(self):
        from qitos.kit.tool.cron import CronScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = CronScheduler(workspace_root=tmpdir)
            job = scheduler.create_job(
                cron="*/5 * * * *",
                prompt="test prompt",
                recurring=True,
                durable=False,
            )
            assert job.id.startswith("cron-")
            assert job.cron == "*/5 * * * *"
            assert job.prompt == "test prompt"
            assert job.recurring is True

    def test_list_jobs(self):
        from qitos.kit.tool.cron import CronScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = CronScheduler(workspace_root=tmpdir)
            scheduler.create_job(cron="0 9 * * *", prompt="morning check")
            scheduler.create_job(cron="0 17 * * *", prompt="evening check")
            jobs = scheduler.list_jobs()
            assert len(jobs) == 2

    def test_delete_job(self):
        from qitos.kit.tool.cron import CronScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = CronScheduler(workspace_root=tmpdir)
            job = scheduler.create_job(cron="0 9 * * *", prompt="test")
            assert scheduler.delete_job(job.id) is True
            assert scheduler.delete_job(job.id) is False
            assert len(scheduler.list_jobs()) == 0

    def test_durable_job_persistence(self):
        from qitos.kit.tool.cron import CronScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and persist
            scheduler1 = CronScheduler(workspace_root=tmpdir)
            job = scheduler1.create_job(
                cron="0 9 * * *",
                prompt="persistent task",
                recurring=True,
                durable=True,
            )
            job_id = job.id

            # Load in new scheduler instance
            scheduler2 = CronScheduler(workspace_root=tmpdir)
            jobs = scheduler2.list_jobs()
            assert len(jobs) == 1
            assert jobs[0].id == job_id
            assert jobs[0].prompt == "persistent task"

    def test_fire_callback(self):
        from qitos.kit.tool.cron import CronScheduler

        fired_prompts = []

        def on_fire(prompt: str):
            fired_prompts.append(prompt)

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = CronScheduler(workspace_root=tmpdir, on_fire=on_fire)
            job = scheduler.create_job(cron="0 9 * * *", prompt="test prompt")
            scheduler._fire_job(job.id)
            assert len(fired_prompts) == 1
            assert fired_prompts[0] == "test prompt"


# ── CronCreateTool / CronDeleteTool / CronListTool ─────────────────────────────


class TestCronTools:
    @pytest.mark.asyncio
    async def test_create_tool(self):
        from qitos.kit.tool.cron import CronScheduler, CronCreateTool

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = CronScheduler(workspace_root=tmpdir)
            tool = CronCreateTool(scheduler)
            result = await tool.execute({"cron": "0 9 * * *", "prompt": "test"})
            assert result["status"] == "success"
            assert result["created"] is True

    @pytest.mark.asyncio
    async def test_create_tool_missing_params(self):
        from qitos.kit.tool.cron import CronScheduler, CronCreateTool

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = CronScheduler(workspace_root=tmpdir)
            tool = CronCreateTool(scheduler)
            result = await tool.execute({})
            assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_delete_tool(self):
        from qitos.kit.tool.cron import CronScheduler, CronCreateTool, CronDeleteTool

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = CronScheduler(workspace_root=tmpdir)
            create_tool = CronCreateTool(scheduler)
            delete_tool = CronDeleteTool(scheduler)

            result = await create_tool.execute({"cron": "0 9 * * *", "prompt": "test"})
            job_id = result["job"]["id"]

            del_result = await delete_tool.execute({"job_id": job_id})
            assert del_result["deleted"] is True

    @pytest.mark.asyncio
    async def test_list_tool(self):
        from qitos.kit.tool.cron import CronScheduler, CronCreateTool, CronListTool

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = CronScheduler(workspace_root=tmpdir)
            create_tool = CronCreateTool(scheduler)
            list_tool = CronListTool(scheduler)

            await create_tool.execute({"cron": "0 9 * * *", "prompt": "task1"})
            await create_tool.execute({"cron": "0 17 * * *", "prompt": "task2"})

            result = await list_tool.execute({})
            assert result["status"] == "success"
            assert result["count"] == 2


# ── WorktreeManager ───────────────────────────────────────────────────────────


class TestWorktreeManager:
    def test_list_empty(self):
        from qitos.kit.agent.worktree_manager import WorktreeManager

        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorktreeManager(workspace_root=tmpdir)
            assert wm.list_worktrees() == []

    def test_fallback_copy_creates_directory(self):
        from qitos.kit.agent.worktree_manager import WorktreeManager

        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorktreeManager(workspace_root=tmpdir)
            path = wm._fallback_copy("test-wt")
            assert os.path.isdir(path)
            assert "test-wt" in path

    def test_remove_worktree(self):
        from qitos.kit.agent.worktree_manager import WorktreeManager

        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorktreeManager(workspace_root=tmpdir)
            wm._fallback_copy("test-wt")
            assert wm.remove_worktree("test-wt") is True
            assert wm.list_worktrees() == []

    def test_remove_nonexistent(self):
        from qitos.kit.agent.worktree_manager import WorktreeManager

        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorktreeManager(workspace_root=tmpdir)
            assert wm.remove_worktree("nonexistent") is False


# ── AgentTool ─────────────────────────────────────────────────────────────────


class TestAgentTool:
    def test_import(self):
        from qitos.kit.tool.agent import AgentTool

        assert AgentTool is not None

    @pytest.mark.asyncio
    async def test_call_without_prompt(self):
        from qitos.kit.tool.agent import AgentTool

        tool = AgentTool(invocation_factory=lambda request, _context: None)
        result = await tool.execute({"description": "missing task"})
        assert result == {"status": "error", "error": "prompt is required"}

    @pytest.mark.asyncio
    async def test_run_scoped_factory_creates_fresh_invocation_per_call(self):
        from contextlib import contextmanager
        from types import SimpleNamespace

        from qitos.core.budget import BudgetLedger
        from qitos.core.tool_result import ToolResult
        from qitos.kit.tool.agent import AgentTool

        engines = []
        run_ids = []
        scopes = []
        budget_ledger = BudgetLedger()

        class FakeEngine(_ClosableEngine):
            active_run_id = "child-run"

            async def arun(self, task, **kwargs):
                assert task.startswith("seeded:")
                run_id = kwargs.pop("run_id")
                assert isinstance(run_id, str) and run_id.startswith("run_")
                run_ids.append(run_id)
                assert kwargs == {}
                return SimpleNamespace(
                    state=SimpleNamespace(
                        final_result="child result",
                        stop_reason="final",
                    ),
                    step_count=3,
                    total_tokens=42,
                )

        def build_invocation(
            request: ChildLaunchRequest,
            runtime_context: ChildRuntimeContext,
        ):
            assert runtime_context.delegate_depth == 0
            assert runtime_context.budget_ledger is budget_ledger
            assert request.budget.max_steps == 200
            assert request.profile == "restricted"
            assert request.allowed_tool_groups == ("files", "network")
            assert request.working_directory == "workspace"
            engine = FakeEngine()
            engines.append(engine)
            return _ready_invocation(engine=engine, task=f"seeded:{request.task}")

        @contextmanager
        def execution_scope(runtime_context: ChildRuntimeContext):
            assert runtime_context.delegate_depth == 0
            scopes.append("enter")
            try:
                yield
            finally:
                scopes.append("exit")

        tool = AgentTool(
            invocation_factory=build_invocation,
            execution_scope=execution_scope,
            execution_mode="foreground",
            child_profile="restricted",
            child_allowed_tool_groups=("files", "network"),
            child_working_directory="workspace",
        )
        first = await tool.execute(
            {"description": "first task", "prompt": "one"},
            runtime_context={
                "budget_ledger": budget_ledger,
                "delegate_depth": 0,
                "parent_run_id": "parent-run",
            },
        )
        second = await tool.execute(
            {"description": "second task", "prompt": "two"},
            runtime_context={
                "budget_ledger": budget_ledger,
                "delegate_depth": 0,
                "parent_run_id": "parent-run",
            },
        )

        assert first["status"] == "success"
        assert first["child_status"] == "completed"
        assert ToolResult.from_value(first).is_success
        assert first["steps"] == 3
        assert first["total_tokens"] == 42
        assert second["status"] == "success"
        assert second["child_status"] == "completed"
        assert len(engines) == 2
        assert engines[0] is not engines[1]
        assert len(set(run_ids)) == 2
        assert scopes == ["enter", "exit", "enter", "exit"]
        assert tool.spec.concurrency_safe is True
        assert "run_in_background" not in tool.spec.parameters

    @pytest.mark.asyncio
    async def test_run_scoped_factory_rejects_recursive_agent(self):
        from qitos.kit.tool.agent import AgentTool

        tool = AgentTool(
            invocation_factory=lambda request, context: None,
            execution_mode="foreground",
        )
        result = await tool.execute(
            {"description": "nested task", "prompt": "recurse"},
            runtime_context={"delegate_depth": 1, "parent_run_id": "parent-run"},
        )

        assert result["status"] == "error"
        assert "cannot launch another Agent" in result["error"]

    @pytest.mark.asyncio
    async def test_run_child_budget_rejects_calls_beyond_limit(self):
        from qitos.kit.tool.agent import AgentTool

        class FakeEngine(_ClosableEngine):
            active_run_id = "child-run"

            async def arun(self, task, **kwargs):
                _ = task, kwargs
                return SimpleNamespace(
                    state=SimpleNamespace(
                        final_result="done",
                        stop_reason="completed",
                    ),
                    records=[],
                    step_count=1,
                    total_tokens=1,
                )

        tool = AgentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="foreground",
        )
        context = {
            "delegate_depth": 0,
            "parent_run_id": "parent-run",
            "max_children": 1,
        }

        first = await tool.execute(
            {"description": "first", "prompt": "one"},
            runtime_context=context,
        )
        second = await tool.execute(
            {"description": "second", "prompt": "two"},
            runtime_context=context,
        )

        assert first["status"] == "success"
        assert first["child_status"] == "completed"
        assert second["status"] == "error"
        assert "max_children=1" in second["error"]

    @pytest.mark.asyncio
    async def test_forced_background_children_launch_concurrently_and_notify_parent(
        self,
    ):
        from qitos.core.runtime_input import RuntimeInput
        from qitos.kit.tool.agent import AgentTool

        both_started = asyncio.Event()
        release = asyncio.Event()
        completed = asyncio.Event()
        active = 0
        peak = 0
        events = []

        class FakeEngine(_ClosableEngine):
            active_run_id = "child-run"

            async def arun(self, task, **kwargs):
                nonlocal active, peak
                run_id = kwargs.pop("run_id")
                assert isinstance(run_id, str) and run_id.startswith("run_")
                assert kwargs == {}
                active += 1
                peak = max(peak, active)
                if active == 2:
                    both_started.set()
                await release.wait()
                active -= 1
                return SimpleNamespace(
                    state=SimpleNamespace(
                        final_result=f"validated:{task}",
                        stop_reason="final",
                    ),
                    records=[],
                    step_count=2,
                    total_tokens=5,
                )

            def cancel(self, mode):
                assert mode == "immediate"
                release.set()

        tool = AgentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="background",
            max_background_workers=2,
        )

        async def post_runtime_event(event):
            events.append(event)
            if len(events) == 2:
                completed.set()
            return True

        runtime_context = {
            "delegate_depth": 0,
            "parent_run_id": "parent-run",
            "post_runtime_event": post_runtime_event,
        }

        first = await tool.execute(
            {"description": "route one", "prompt": "one"},
            runtime_context=runtime_context,
        )
        second = await tool.execute(
            {"description": "route two", "prompt": "two"},
            runtime_context=runtime_context,
        )

        assert first["status"] == "running"
        assert second["status"] == "running"
        assert "run_in_background" not in tool.spec.parameters
        assert tool.spec.supports_background is True
        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert peak == 2
        assert tool.active_background_count == 2

        release.set()
        await asyncio.wait_for(completed.wait(), timeout=1)

        assert len(events) == 2
        assert all(isinstance(event, RuntimeInput) for event in events)
        assert {event.kind for event in events} == {"agent.child.completed"}
        assert {event.payload["output"] for event in events} == {
            "validated:one",
            "validated:two",
        }
        assert tool.active_background_count == 0
        assert await tool.aclose(wait_seconds=0) == 0

    @pytest.mark.asyncio
    async def test_background_invocation_is_created_after_execution_slot_opens(self):
        from contextlib import asynccontextmanager

        from qitos.kit.tool.agent import AgentTool

        waiting = asyncio.Event()
        open_slot = asyncio.Event()
        factory_called = asyncio.Event()
        before_launch = object()
        parent_messages = [before_launch]
        received_parent_history = ()
        received_parent_snapshot = ()

        @asynccontextmanager
        async def execution_scope(_runtime_context):
            waiting.set()
            await open_slot.wait()
            yield

        class FakeEngine(_ClosableEngine):
            active_run_id = "child-run"

            async def arun(self, task, **kwargs):
                _ = kwargs
                return SimpleNamespace(
                    state=SimpleNamespace(final_result=task, stop_reason="final"),
                    records=[],
                    step_count=1,
                    total_tokens=1,
                )

            def cancel(self, mode):
                _ = mode

        def invocation_factory(
            request: ChildLaunchRequest,
            runtime_context: ChildRuntimeContext,
        ):
            nonlocal received_parent_history, received_parent_snapshot
            factory_called.set()
            received_parent_history = runtime_context.parent_history
            received_parent_snapshot = runtime_context.parent_history_snapshot
            return _ready_invocation(engine=FakeEngine(), task=request.task)

        tool = AgentTool(
            invocation_factory=invocation_factory,
            execution_scope=execution_scope,
            execution_mode="background",
        )
        launched = await tool.execute(
            {
                "description": "queued route",
                "prompt": "inspect",
                "subagent_type": "fork",
            },
            runtime_context={
                "delegate_depth": 0,
                "parent_run_id": "parent-run",
                "agent": SimpleNamespace(
                    history=SimpleNamespace(
                        messages=parent_messages,
                        snapshot=lambda: tuple(parent_messages),
                    )
                ),
            },
        )

        assert launched["status"] == "running"
        await asyncio.wait_for(waiting.wait(), timeout=1)
        assert not factory_called.is_set()
        parent_messages.append(object())
        open_slot.set()
        handle = ChildHandle.from_dict(launched["handle"])
        result = tool.child_result(handle)
        async with asyncio.timeout(1):
            while result is not None and not result.ready:
                await asyncio.sleep(0)
                result = tool.child_result(handle)

        assert factory_called.is_set()
        assert received_parent_history == (before_launch,)
        assert received_parent_snapshot == (before_launch,)
        assert result is not None and result.status is ChildStatus.COMPLETED
        assert await tool.aclose(wait_seconds=0) == 0

    @pytest.mark.asyncio
    async def test_background_child_becomes_terminal_before_event_delivery(self):
        from qitos.kit.tool.agent import AgentTool

        delivery_started = asyncio.Event()

        class FakeEngine(_ClosableEngine):
            active_run_id = "child-run"

            async def arun(self, task, **kwargs):
                _ = kwargs
                return SimpleNamespace(
                    state=SimpleNamespace(final_result=task, stop_reason="final"),
                    records=[],
                    step_count=1,
                    total_tokens=1,
                )

            def cancel(self, mode):
                _ = mode

        async def post_runtime_event(_event):
            assert tool.active_background_count == 0
            delivery_started.set()
            return True

        tool = AgentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="background",
        )
        launched = await tool.execute(
            {"description": "fast route", "prompt": "inspect"},
            runtime_context={
                "delegate_depth": 0,
                "parent_run_id": "parent-run",
                "post_runtime_event": post_runtime_event,
            },
        )

        assert launched["status"] == "running"
        await asyncio.wait_for(delivery_started.wait(), timeout=1)
        assert tool.active_background_count == 0
        handle = ChildHandle.from_dict(launched["handle"])
        terminal = tool.child_result(handle)
        assert terminal is not None and terminal.status is ChildStatus.COMPLETED
        assert tool.cancel_child(handle) is False
        assert await tool.aclose(wait_seconds=1) == 0

    @pytest.mark.asyncio
    async def test_background_budget_stop_returns_partial_tool_evidence(self):
        from qitos.engine.states import StepRecord
        from qitos.kit.tool.agent import AgentTool

        events = []
        completed = asyncio.Event()

        class FakeEngine(_ClosableEngine):
            active_run_id = "child-run"

            async def arun(self, task, **kwargs):
                assert task == "validate"
                run_id = kwargs.pop("run_id")
                assert isinstance(run_id, str) and run_id.startswith("run_")
                assert kwargs == {}
                return SimpleNamespace(
                    state=SimpleNamespace(final_result="", stop_reason="budget_time"),
                    records=[
                        StepRecord(
                            step_id=4,
                            action_results=[
                                {"status": "success", "output": "uid=33(www-data)"}
                            ],
                            tool_invocations=[{"tool_name": "shell"}],
                        )
                    ],
                    step_count=5,
                    total_tokens=10,
                )

            def cancel(self, mode):
                _ = mode

        tool = AgentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="background",
        )

        async def post_runtime_event(event):
            events.append(event)
            completed.set()
            return True

        launched = await tool.execute(
            {"description": "validate route", "prompt": "validate"},
            runtime_context={
                "delegate_depth": 0,
                "parent_run_id": "parent-run",
                "post_runtime_event": post_runtime_event,
            },
        )

        await asyncio.wait_for(completed.wait(), timeout=1)

        assert launched["status"] == "running"
        assert len(events) == 1
        assert events[0].payload["status"] == "partial"
        assert events[0].payload["child_status"] == "budget_exhausted"
        assert events[0].payload["stop_reason"] == "budget_exhausted"
        assert "uid=33(www-data)" in events[0].payload["output"]
        assert await tool.aclose(wait_seconds=0) == 0

    @pytest.mark.asyncio
    async def test_running_background_child_exposes_bounded_tool_evidence_snapshot(
        self,
    ):
        from qitos.engine.states import StepRecord
        from qitos.kit.tool.agent import AgentTool

        started = asyncio.Event()
        cancelled = asyncio.Event()

        class FakeEngine(_ClosableEngine):
            active_run_id = "child-run"
            records = [
                StepRecord(
                    step_id=index,
                    model_response={"reasoning": "private child reasoning"},
                    action_results=[
                        {
                            "status": "success",
                            "output": f"evidence-{index}-" + ("x" * 2_000),
                        }
                    ],
                    tool_invocations=[{"tool_name": "shell"}],
                )
                for index in range(20)
            ]

            async def arun(self, task, **kwargs):
                _ = task, kwargs
                started.set()
                await cancelled.wait()
                return SimpleNamespace(
                    state=SimpleNamespace(
                        final_result="",
                        stop_reason="cancelled_immediate",
                    ),
                    records=self.records,
                    step_count=len(self.records),
                    total_tokens=321,
                    run_id=self.active_run_id,
                )

            def cancel(self, mode):
                assert mode == "immediate"
                cancelled.set()

        tool = AgentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="background",
        )
        launched = await tool.execute(
            {
                "description": "validate extension bypass",
                "name": "extension-bypass",
                "prompt": "validate",
            },
            runtime_context={"delegate_depth": 0, "parent_run_id": "parent-run"},
        )

        assert launched["status"] == "running"
        await asyncio.wait_for(started.wait(), timeout=1)

        snapshots = tool.snapshot_background_events()

        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert snapshot.kind == "agent.child.snapshot"
        assert snapshot.event_id == f"{launched['child_id']}:conclude-snapshot"
        assert snapshot.payload["child_id"] == launched["child_id"]
        assert snapshot.payload["handle"] == launched["handle"]
        assert snapshot.payload["status"] == "running"
        assert snapshot.payload["name"] == "extension-bypass"
        assert snapshot.payload["description"] == "validate extension bypass"
        assert snapshot.payload["steps"] == 20
        assert snapshot.payload["run_id"] == "child-run"
        assert "evidence-19" in snapshot.payload["output"]
        assert "evidence-0" not in snapshot.payload["output"]
        assert "private child reasoning" not in snapshot.payload["output"]
        assert len(snapshot.payload["output"]) <= 16_000
        assert await tool.aclose(wait_seconds=1) == 0

    @pytest.mark.asyncio
    async def test_completed_background_child_is_not_repeated_in_snapshot(self):
        from qitos.kit.tool.agent import AgentTool

        class FakeEngine(_ClosableEngine):
            active_run_id = "child-run"

            async def arun(self, task, **kwargs):
                _ = task, kwargs
                return SimpleNamespace(
                    state=SimpleNamespace(final_result="done", stop_reason="final"),
                    records=[],
                    step_count=1,
                    total_tokens=2,
                    run_id=self.active_run_id,
                )

            def cancel(self, mode):
                _ = mode

        tool = AgentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="background",
        )
        launched = await tool.execute(
            {"description": "done", "prompt": "done"},
            runtime_context={"delegate_depth": 0, "parent_run_id": "parent-run"},
        )

        handle = ChildHandle.from_dict(launched["handle"])
        async with asyncio.timeout(1):
            while not tool.child_result(handle).ready:
                await asyncio.sleep(0)

        assert tool.snapshot_background_events() == []
        assert await tool.aclose(wait_seconds=0) == 0

    @pytest.mark.asyncio
    async def test_close_cancels_a_running_background_child(self):
        from qitos.kit.tool.agent import AgentTool

        started = asyncio.Event()
        cancelled = asyncio.Event()

        class FakeEngine(_ClosableEngine):
            active_run_id = "child-run"

            async def arun(self, task, **kwargs):
                _ = task, kwargs
                started.set()
                await cancelled.wait()
                return SimpleNamespace(
                    state=SimpleNamespace(
                        final_result="",
                        stop_reason="cancelled_immediate",
                    ),
                    records=[],
                    step_count=0,
                    total_tokens=0,
                )

            def cancel(self, mode):
                assert mode == "immediate"
                cancelled.set()

        tool = AgentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="background",
        )
        result = await tool.execute(
            {"description": "long route", "prompt": "wait"},
            runtime_context={"delegate_depth": 0, "parent_run_id": "parent-run"},
        )

        assert result["status"] == "running"
        await asyncio.wait_for(started.wait(), timeout=1)
        assert await tool.aclose(wait_seconds=1) == 0
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_close_drains_an_already_cancelled_child(self):
        from qitos.kit.tool.agent import AgentTool

        started = asyncio.Event()
        cancelled = asyncio.Event()

        class FakeEngine(_ClosableEngine):
            active_run_id = "child-run"

            async def arun(self, task, **kwargs):
                _ = task, kwargs
                started.set()
                await cancelled.wait()
                return SimpleNamespace(
                    state=SimpleNamespace(
                        final_result="",
                        stop_reason="cancelled_immediate",
                    ),
                    records=[],
                    step_count=0,
                    total_tokens=0,
                )

            def cancel(self, mode):
                assert mode == "immediate"
                cancelled.set()

        tool = AgentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="background",
        )
        result = await tool.execute(
            {"description": "long route", "prompt": "wait"},
            runtime_context={"delegate_depth": 0, "parent_run_id": "parent-run"},
        )

        assert result["status"] == "running"
        await asyncio.wait_for(started.wait(), timeout=1)
        assert await tool.aclose(wait_seconds=0) == 0

    @pytest.mark.asyncio
    async def test_close_wakes_a_child_waiting_for_an_execution_slot(self):
        from contextlib import asynccontextmanager

        from qitos.kit.tool.agent import AgentTool

        waiting = asyncio.Event()
        child_started = asyncio.Event()
        factory_called = asyncio.Event()

        @asynccontextmanager
        async def blocked_scope(runtime_context: ChildRuntimeContext):
            waiting.set()
            cancelled = runtime_context.cancellation_requested
            while not cancelled():
                await asyncio.sleep(0)
            raise RuntimeError("cancelled before child slot opened")
            yield  # pragma: no cover

        class FakeEngine(_ClosableEngine):
            active_run_id = "child-run"

            async def arun(self, task, **kwargs):
                _ = task, kwargs
                child_started.set()
                raise AssertionError("cancelled child unexpectedly started")

            def cancel(self, mode):
                _ = mode

        def invocation_factory(request, _context):
            factory_called.set()
            return _ready_invocation(engine=FakeEngine(), task=request.task)

        tool = AgentTool(
            invocation_factory=invocation_factory,
            execution_scope=blocked_scope,
            execution_mode="background",
        )
        result = await tool.execute(
            {"description": "queued route", "prompt": "wait"},
            runtime_context={"delegate_depth": 0, "parent_run_id": "parent-run"},
        )

        assert result["status"] == "running"
        await asyncio.wait_for(waiting.wait(), timeout=1)
        assert await tool.aclose(wait_seconds=1) == 0
        handle = ChildHandle.from_dict(result["handle"])
        terminal = tool.child_result(handle)
        assert terminal is not None
        assert terminal.status is ChildStatus.CANCELLED
        assert not factory_called.is_set()
        assert not child_started.is_set()

    @pytest.mark.asyncio
    async def test_background_child_deadline_expires_before_execution_slot_opens(self):
        from contextlib import asynccontextmanager

        from qitos.kit.tool.agent import AgentTool

        factory_called = asyncio.Event()

        @asynccontextmanager
        async def expired_scope(_runtime_context):
            raise TimeoutError("deadline expired before child slot opened")
            yield  # pragma: no cover

        def invocation_factory(request, _context):
            factory_called.set()
            return _ready_invocation(engine=object(), task=request.task)

        tool = AgentTool(
            invocation_factory=invocation_factory,
            execution_scope=expired_scope,
            execution_mode="background",
        )
        launched = await tool.execute(
            {"description": "queued route", "prompt": "wait"},
            runtime_context={"delegate_depth": 0, "parent_run_id": "parent-run"},
        )

        handle = ChildHandle.from_dict(launched["handle"])
        terminal = tool.child_result(handle)
        async with asyncio.timeout(1):
            while terminal is not None and not terminal.ready:
                await asyncio.sleep(0)
                terminal = tool.child_result(handle)

        assert terminal is not None
        assert terminal.status is ChildStatus.BUDGET_EXHAUSTED
        assert not factory_called.is_set()
        assert tool.active_background_count == 0
        assert await tool.aclose(wait_seconds=0) == 0
