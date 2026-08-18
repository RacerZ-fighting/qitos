"""Tests for sub-agents, cron, and coding-tool helpers."""

import asyncio
import os
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Protocol

import pytest
from qitos.core.subagent import (
    SubagentHandle,
    SubagentInvocation,
    SubagentLaunchRequest,
    SubagentResult,
    SubagentRuntimeContext,
    SubagentStatus,
)
from qitos.core.tool_result import ToolResult

from qitos.kit.tool.internal.coding_utils import (
    is_image_file,
    is_notebook_file,
    is_pdf_file,
    read_image_as_base64,
)


async def _ready_invocation(**kwargs: Any) -> SubagentInvocation:
    return SubagentInvocation(**kwargs)


class _SubagentResultReader(Protocol):
    def subagent_result(self, handle: SubagentHandle) -> SubagentResult | None:
        ...


async def _wait_for_subagent_result(
    tool: _SubagentResultReader,
    handle: SubagentHandle,
) -> SubagentResult:
    while True:
        result = tool.subagent_result(handle)
        if result is not None and result.ready:
            return result
        await asyncio.sleep(0)


class _ClosableEngine:
    async def aclose(self) -> None:
        return None


def _agent_args(
    description: str,
    prompt: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "description": description,
        "prompt": prompt,
        "success_criteria": [f"Complete {description}"],
        **extra,
    }


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
            assert isinstance(result, ToolResult)
            assert result.status == "error"

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


# ── SubagentTool ─────────────────────────────────────────────────────────────────


class TestSubagentTool:
    def test_import(self):
        from qitos.kit.tool.subagent import SubagentTool

        assert SubagentTool is not None

    @pytest.mark.asyncio
    async def test_call_without_prompt(self):
        from qitos.kit.tool.subagent import SubagentTool

        tool = SubagentTool(invocation_factory=lambda request, _context: None)
        result = await tool.execute({"description": "missing task"})
        assert isinstance(result, ToolResult)
        assert result.status == "error"
        assert result.output == {"status": "error", "error": "prompt is required"}

    @pytest.mark.asyncio
    async def test_run_scoped_factory_creates_fresh_invocation_per_call(self):
        from contextlib import contextmanager
        from types import SimpleNamespace

        from qitos.core.budget import BudgetLedger
        from qitos.core.tool_result import ToolResult
        from qitos.kit.tool.subagent import SubagentTool

        engines = []
        run_ids = []
        scopes = []
        budget_ledger = BudgetLedger()

        class FakeEngine(_ClosableEngine):
            active_run_id = "subagent-run"

            async def arun(self, task, **kwargs):
                assert task.startswith("seeded:")
                run_id = kwargs.pop("run_id")
                assert isinstance(run_id, str) and run_id.startswith("run_")
                run_ids.append(run_id)
                assert kwargs == {}
                return SimpleNamespace(
                    state=SimpleNamespace(
                        final_result="subagent result",
                        stop_reason="final",
                    ),
                    step_count=3,
                    total_tokens=42,
                )

        def build_invocation(
            request: SubagentLaunchRequest,
            runtime_context: SubagentRuntimeContext,
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
        def execution_scope(runtime_context: SubagentRuntimeContext):
            assert runtime_context.delegate_depth == 0
            scopes.append("enter")
            try:
                yield
            finally:
                scopes.append("exit")

        tool = SubagentTool(
            invocation_factory=build_invocation,
            execution_scope=execution_scope,
            execution_mode="foreground",
            subagent_profile="restricted",
            subagent_allowed_tool_groups=("files", "network"),
            subagent_working_directory="workspace",
        )
        first = await tool.execute(
            _agent_args("first task", "one"),
            runtime_context={
                "budget_ledger": budget_ledger,
                "delegate_depth": 0,
                "parent_run_id": "parent-run",
            },
        )
        second = await tool.execute(
            _agent_args("second task", "two"),
            runtime_context={
                "budget_ledger": budget_ledger,
                "delegate_depth": 0,
                "parent_run_id": "parent-run",
            },
        )

        assert isinstance(first, ToolResult)
        assert first.output["status"] == "success"
        assert first.output["subagent_status"] == "completed"
        assert first.is_success
        assert first.output["steps"] == 3
        # Subagent token accounting moved to the typed usage carrier.
        assert "total_tokens" not in first.output
        assert first.usage is not None
        assert first.usage.total_tokens == 42
        assert second.output["status"] == "success"
        assert second.output["subagent_status"] == "completed"
        assert len(engines) == 2
        assert engines[0] is not engines[1]
        assert len(set(run_ids)) == 2
        assert scopes == ["enter", "exit", "enter", "exit"]
        assert tool.spec.concurrency_safe is True
        assert "run_in_background" not in tool.spec.parameters

    @pytest.mark.asyncio
    async def test_subagent_accounting_rides_the_typed_usage_carrier(self):
        from qitos.core.tool_result import ToolResult
        from qitos.kit.tool.subagent import SubagentTool

        class FakeEngine(_ClosableEngine):
            active_run_id = "subagent-run"

            async def arun(self, task, **kwargs):
                _ = task, kwargs
                return SimpleNamespace(
                    state=SimpleNamespace(
                        final_result="subagent result",
                        stop_reason="final",
                    ),
                    records=[],
                    step_count=3,
                    total_tokens=42,
                    total_cost_usd=0.003,
                    local_total_tokens=42,
                    local_total_cost_usd=0.003,
                    local_usage_complete=True,
                    local_cost_complete=True,
                )

        tool = SubagentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="foreground",
        )
        result = await tool.execute(
            _agent_args("scoped task", "one"),
            runtime_context={"delegate_depth": 0, "parent_run_id": "parent-run"},
        )

        assert isinstance(result, ToolResult)
        assert result.is_success
        # Token/cost accounting is typed Tool-boundary data, not payload text.
        assert result.usage is not None
        assert result.usage.total_tokens == 42
        assert result.usage["cost_usd"] == pytest.approx(0.003)
        assert "total_tokens" not in result.output
        assert "total_cost_usd" not in result.output
        # Domain conclusion facts, including completeness flags, stay in output.
        assert result.output["status"] == "success"
        assert result.output["subagent_status"] == "completed"
        assert result.output["steps"] == 3
        assert result.output["usage_complete"] is True
        assert result.output["cost_complete"] is True
        assert "elapsed_seconds" in result.output

    @pytest.mark.asyncio
    async def test_background_launch_receipt_carries_no_finished_usage(self):
        from qitos.core.tool_result import ToolResult
        from qitos.kit.tool.subagent import SubagentTool

        started = asyncio.Event()
        release = asyncio.Event()

        class FakeEngine(_ClosableEngine):
            active_run_id = "subagent-run"

            async def arun(self, task, **kwargs):
                _ = task, kwargs
                started.set()
                await release.wait()
                return SimpleNamespace(
                    state=SimpleNamespace(final_result="done", stop_reason="final"),
                    records=[],
                    step_count=1,
                    total_tokens=1,
                )

            def cancel(self, mode):
                _ = mode
                release.set()

        tool = SubagentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="background",
        )
        result = await tool.execute(
            _agent_args("long route", "wait"),
            runtime_context={"delegate_depth": 0, "parent_run_id": "parent-run"},
        )

        assert isinstance(result, ToolResult)
        assert result.is_success
        assert result.output["status"] == "running"
        # A launch receipt has no finished Subagent accounting yet.
        assert result.usage is None
        await asyncio.wait_for(started.wait(), timeout=1)
        release.set()
        assert await tool.aclose(wait_seconds=1) == 0

    @pytest.mark.asyncio
    async def test_run_scoped_factory_rejects_recursive_agent(self):
        from qitos.kit.tool.subagent import SubagentTool

        tool = SubagentTool(
            invocation_factory=lambda request, context: None,
            execution_mode="foreground",
        )
        result = await tool.execute(
            _agent_args("nested task", "recurse"),
            runtime_context={"delegate_depth": 1, "parent_run_id": "parent-run"},
        )

        assert isinstance(result, ToolResult)
        assert result.status == "error"
        assert result.error is not None
        assert "cannot launch another Subagent" in result.error

    @pytest.mark.asyncio
    async def test_run_subagent_budget_rejects_calls_beyond_limit(self):
        from qitos.kit.tool.subagent import SubagentTool

        class FakeEngine(_ClosableEngine):
            active_run_id = "subagent-run"

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

        tool = SubagentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="foreground",
        )
        context = {
            "delegate_depth": 0,
            "parent_run_id": "parent-run",
            "max_subagents": 1,
        }

        first = await tool.execute(
            _agent_args("first", "one"),
            runtime_context=context,
        )
        second = await tool.execute(
            _agent_args("second", "two"),
            runtime_context=context,
        )

        assert first.output["status"] == "success"
        assert first.output["subagent_status"] == "completed"
        assert isinstance(second, ToolResult)
        assert second.status == "error"
        assert second.error is not None
        assert "max_subagents=1" in second.error

    @pytest.mark.asyncio
    async def test_forced_background_subagents_launch_concurrently_and_notify_parent(
        self,
    ):
        from qitos.core.runtime_input import RuntimeInput
        from qitos.kit.tool.subagent import SubagentTool

        both_started = asyncio.Event()
        release = asyncio.Event()
        completed = asyncio.Event()
        active = 0
        peak = 0
        events = []

        class FakeEngine(_ClosableEngine):
            active_run_id = "subagent-run"

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

        tool = SubagentTool(
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
            _agent_args("route one", "one"),
            runtime_context=runtime_context,
        )
        second = await tool.execute(
            _agent_args("route two", "two"),
            runtime_context=runtime_context,
        )

        assert first.output["status"] == "running"
        assert second.output["status"] == "running"
        assert "run_in_background" not in tool.spec.parameters
        assert tool.spec.supports_background is True
        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert peak == 2
        assert tool.active_background_count == 2

        release.set()
        await asyncio.wait_for(completed.wait(), timeout=1)

        assert len(events) == 2
        assert all(isinstance(event, RuntimeInput) for event in events)
        assert {event.kind for event in events} == {"agent.subagent.completed"}
        assert {event.payload["output"] for event in events} == {
            "validated:one",
            "validated:two",
        }
        assert tool.active_background_count == 0
        assert await tool.aclose(wait_seconds=0) == 0

    @pytest.mark.asyncio
    async def test_background_invocation_is_created_after_execution_slot_opens(self):
        from contextlib import asynccontextmanager

        from qitos.kit.tool.subagent import SubagentTool

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
            active_run_id = "subagent-run"

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
            request: SubagentLaunchRequest,
            runtime_context: SubagentRuntimeContext,
        ):
            nonlocal received_parent_history, received_parent_snapshot
            factory_called.set()
            received_parent_history = runtime_context.parent_history
            received_parent_snapshot = runtime_context.parent_history_snapshot
            return _ready_invocation(engine=FakeEngine(), task=request.task)

        tool = SubagentTool(
            invocation_factory=invocation_factory,
            execution_scope=execution_scope,
            execution_mode="background",
        )
        launched = await tool.execute(
            _agent_args("queued route", "inspect", subagent_type="fork"),
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

        assert launched.output["status"] == "running"
        await asyncio.wait_for(waiting.wait(), timeout=1)
        assert not factory_called.is_set()
        parent_messages.append(object())
        open_slot.set()
        handle = SubagentHandle.from_dict(launched.output["handle"])
        result = await asyncio.wait_for(
            _wait_for_subagent_result(tool, handle),
            timeout=1,
        )

        assert factory_called.is_set()
        assert received_parent_history == (before_launch,)
        assert received_parent_snapshot == (before_launch,)
        assert result is not None and result.status is SubagentStatus.COMPLETED
        assert await tool.aclose(wait_seconds=0) == 0

    @pytest.mark.asyncio
    async def test_background_subagent_becomes_terminal_before_event_delivery(self):
        from qitos.kit.tool.subagent import SubagentTool

        delivery_started = asyncio.Event()

        class FakeEngine(_ClosableEngine):
            active_run_id = "subagent-run"

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

        tool = SubagentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="background",
        )
        launched = await tool.execute(
            _agent_args("fast route", "inspect"),
            runtime_context={
                "delegate_depth": 0,
                "parent_run_id": "parent-run",
                "post_runtime_event": post_runtime_event,
            },
        )

        assert launched.output["status"] == "running"
        await asyncio.wait_for(delivery_started.wait(), timeout=1)
        assert tool.active_background_count == 0
        handle = SubagentHandle.from_dict(launched.output["handle"])
        terminal = tool.subagent_result(handle)
        assert terminal is not None and terminal.status is SubagentStatus.COMPLETED
        assert tool.cancel_subagent(handle) is False
        assert await tool.aclose(wait_seconds=1) == 0

    @pytest.mark.asyncio
    async def test_background_budget_stop_does_not_fabricate_a_conclusion(self):
        from qitos.core.message import AssistantMessage, ToolResultMessage
        from qitos.core.tool_result import ToolResult
        from qitos.kit.tool.subagent import SubagentTool

        events = []
        completed = asyncio.Event()

        class FakeEngine(_ClosableEngine):
            active_run_id = "subagent-run"

            async def arun(self, task, **kwargs):
                assert task == "validate"
                run_id = kwargs.pop("run_id")
                assert isinstance(run_id, str) and run_id.startswith("run_")
                assert kwargs == {}
                return SimpleNamespace(
                    state=SimpleNamespace(final_result="", stop_reason="budget_time"),
                    messages=[
                        AssistantMessage(text="probing"),
                        ToolResultMessage(
                            tool_call_id="call-1",
                            tool_name="shell",
                            result=ToolResult(
                                status="success", output="uid=33(www-data)"
                            ),
                        ),
                    ],
                    step_count=5,
                    total_tokens=10,
                )

            def cancel(self, mode):
                _ = mode

        tool = SubagentTool(
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
            _agent_args("validate route", "validate"),
            runtime_context={
                "delegate_depth": 0,
                "parent_run_id": "parent-run",
                "post_runtime_event": post_runtime_event,
            },
        )

        await asyncio.wait_for(completed.wait(), timeout=1)

        assert launched.output["status"] == "running"
        assert len(events) == 1
        assert events[0].payload["status"] == "error"
        assert events[0].payload["subagent_status"] == "budget_exhausted"
        assert events[0].payload["stop_reason"] == "budget_exhausted"
        assert events[0].payload["output"] == ""
        assert events[0].payload["conclusion"]["summary"] == ""
        assert await tool.aclose(wait_seconds=0) == 0

    @pytest.mark.asyncio
    async def test_running_background_subagent_exposes_bounded_tool_evidence_snapshot(
        self,
    ):
        from qitos.core.message import AssistantMessage, ToolResultMessage
        from qitos.core.tool_result import ToolResult
        from qitos.kit.tool.subagent import SubagentTool

        started = asyncio.Event()
        cancelled = asyncio.Event()

        class FakeEngine(_ClosableEngine):
            active_run_id = "subagent-run"
            step_count = 20
            messages = [
                message
                for index in range(20)
                for message in (
                    AssistantMessage(text="private subagent reasoning"),
                    ToolResultMessage(
                        tool_call_id=f"call-{index}",
                        tool_name="shell",
                        result=ToolResult(
                            status="success",
                            output=f"evidence-{index}-" + ("x" * 2_000),
                        ),
                    ),
                )
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
                    messages=self.messages,
                    step_count=self.step_count,
                    total_tokens=321,
                    run_id=self.active_run_id,
                )

            def cancel(self, mode):
                assert mode == "immediate"
                cancelled.set()

        tool = SubagentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="background",
        )
        launched = await tool.execute(
            _agent_args(
                "validate extension bypass",
                "validate",
                name="extension-bypass",
            ),
            runtime_context={"delegate_depth": 0, "parent_run_id": "parent-run"},
        )

        assert launched.output["status"] == "running"
        await asyncio.wait_for(started.wait(), timeout=1)

        snapshots = tool.snapshot_background_events()

        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert snapshot.kind == "agent.subagent.snapshot"
        assert snapshot.event_id == f"{launched.output['subagent_id']}:conclude-snapshot"
        assert snapshot.payload["subagent_id"] == launched.output["subagent_id"]
        assert snapshot.payload["handle"] == launched.output["handle"]
        assert snapshot.payload["status"] == "running"
        assert snapshot.payload["name"] == "extension-bypass"
        assert snapshot.payload["description"] == "validate extension bypass"
        assert snapshot.payload["steps"] == 20
        assert snapshot.payload["run_id"] == "subagent-run"
        assert "evidence-19" in snapshot.payload["output"]
        assert "evidence-0" not in snapshot.payload["output"]
        assert "private subagent reasoning" not in snapshot.payload["output"]
        assert len(snapshot.payload["output"]) <= 16_000
        assert await tool.aclose(wait_seconds=1) == 0

    @pytest.mark.asyncio
    async def test_completed_background_subagent_is_not_repeated_in_snapshot(self):
        from qitos.kit.tool.subagent import SubagentTool

        class FakeEngine(_ClosableEngine):
            active_run_id = "subagent-run"

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

        tool = SubagentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="background",
        )
        launched = await tool.execute(
            _agent_args("done", "done"),
            runtime_context={"delegate_depth": 0, "parent_run_id": "parent-run"},
        )

        handle = SubagentHandle.from_dict(launched.output["handle"])
        await asyncio.wait_for(
            _wait_for_subagent_result(tool, handle),
            timeout=1,
        )

        assert tool.snapshot_background_events() == []
        assert await tool.aclose(wait_seconds=0) == 0

    @pytest.mark.asyncio
    async def test_close_cancels_a_running_background_subagent(self):
        from qitos.kit.tool.subagent import SubagentTool

        started = asyncio.Event()
        cancelled = asyncio.Event()

        class FakeEngine(_ClosableEngine):
            active_run_id = "subagent-run"

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

        tool = SubagentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="background",
        )
        result = await tool.execute(
            _agent_args("long route", "wait"),
            runtime_context={"delegate_depth": 0, "parent_run_id": "parent-run"},
        )

        assert result.output["status"] == "running"
        await asyncio.wait_for(started.wait(), timeout=1)
        assert await tool.aclose(wait_seconds=1) == 0
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_close_drains_an_already_cancelled_subagent(self):
        from qitos.kit.tool.subagent import SubagentTool

        started = asyncio.Event()
        cancelled = asyncio.Event()

        class FakeEngine(_ClosableEngine):
            active_run_id = "subagent-run"

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

        tool = SubagentTool(
            invocation_factory=lambda request, _context: _ready_invocation(
                engine=FakeEngine(),
                task=request.task,
            ),
            execution_mode="background",
        )
        result = await tool.execute(
            _agent_args("long route", "wait"),
            runtime_context={"delegate_depth": 0, "parent_run_id": "parent-run"},
        )

        assert result.output["status"] == "running"
        await asyncio.wait_for(started.wait(), timeout=1)
        assert await tool.aclose(wait_seconds=0) == 0

    @pytest.mark.asyncio
    async def test_close_wakes_a_subagent_waiting_for_an_execution_slot(self):
        from contextlib import asynccontextmanager

        from qitos.kit.tool.subagent import SubagentTool

        waiting = asyncio.Event()
        subagent_started = asyncio.Event()
        factory_called = asyncio.Event()

        @asynccontextmanager
        async def blocked_scope(runtime_context: SubagentRuntimeContext):
            waiting.set()
            cancelled = runtime_context.cancellation_requested
            while not cancelled():
                await asyncio.sleep(0)
            raise RuntimeError("cancelled before subagent slot opened")
            yield  # pragma: no cover

        class FakeEngine(_ClosableEngine):
            active_run_id = "subagent-run"

            async def arun(self, task, **kwargs):
                _ = task, kwargs
                subagent_started.set()
                raise AssertionError("cancelled subagent unexpectedly started")

            def cancel(self, mode):
                _ = mode

        def invocation_factory(request, _context):
            factory_called.set()
            return _ready_invocation(engine=FakeEngine(), task=request.task)

        tool = SubagentTool(
            invocation_factory=invocation_factory,
            execution_scope=blocked_scope,
            execution_mode="background",
        )
        result = await tool.execute(
            _agent_args("queued route", "wait"),
            runtime_context={"delegate_depth": 0, "parent_run_id": "parent-run"},
        )

        assert result.output["status"] == "running"
        await asyncio.wait_for(waiting.wait(), timeout=1)
        assert await tool.aclose(wait_seconds=1) == 0
        handle = SubagentHandle.from_dict(result.output["handle"])
        terminal = tool.subagent_result(handle)
        assert terminal is not None
        assert terminal.status is SubagentStatus.CANCELLED
        assert not factory_called.is_set()
        assert not subagent_started.is_set()

    @pytest.mark.asyncio
    async def test_background_subagent_deadline_expires_before_execution_slot_opens(self):
        from contextlib import asynccontextmanager

        from qitos.kit.tool.subagent import SubagentTool

        factory_called = asyncio.Event()
        never = asyncio.Event()

        @asynccontextmanager
        async def expired_scope(_runtime_context):
            await never.wait()
            yield

        def invocation_factory(request, _context):
            factory_called.set()
            return _ready_invocation(engine=object(), task=request.task)

        tool = SubagentTool(
            invocation_factory=invocation_factory,
            execution_scope=expired_scope,
            execution_mode="background",
        )
        launched = await tool.execute(
            _agent_args("queued route", "wait"),
            runtime_context={
                "deadline_monotonic": time.monotonic() + 0.02,
                "delegate_depth": 0,
                "parent_run_id": "parent-run",
            },
        )

        handle = SubagentHandle.from_dict(launched.output["handle"])
        terminal = await asyncio.wait_for(
            _wait_for_subagent_result(tool, handle),
            timeout=1,
        )

        assert terminal is not None
        assert terminal.status is SubagentStatus.BUDGET_EXHAUSTED
        assert not factory_called.is_set()
        assert tool.active_background_count == 0
        assert await tool.aclose(wait_seconds=0) == 0
