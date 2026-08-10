"""Tests for sub-agents, cron, and worktree management."""

import os
import tempfile
import threading
import time
from types import SimpleNamespace

from qitos.kit.tool.internal.coding_utils import (
    is_image_file,
    is_notebook_file,
    is_pdf_file,
    read_image_as_base64,
)


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
            scheduler = CronScheduler(
                workspace_root=tmpdir, on_fire=on_fire
            )
            job = scheduler.create_job(cron="0 9 * * *", prompt="test prompt")
            scheduler._fire_job(job.id)
            assert len(fired_prompts) == 1
            assert fired_prompts[0] == "test prompt"


# ── CronCreateTool / CronDeleteTool / CronListTool ─────────────────────────────


class TestCronTools:
    def test_create_tool(self):
        from qitos.kit.tool.cron import CronScheduler, CronCreateTool

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = CronScheduler(workspace_root=tmpdir)
            tool = CronCreateTool(scheduler)
            result = tool.execute({"cron": "0 9 * * *", "prompt": "test"})
            assert result["status"] == "success"
            assert result["created"] is True

    def test_create_tool_missing_params(self):
        from qitos.kit.tool.cron import CronScheduler, CronCreateTool

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = CronScheduler(workspace_root=tmpdir)
            tool = CronCreateTool(scheduler)
            result = tool.execute({})
            assert result["status"] == "error"

    def test_delete_tool(self):
        from qitos.kit.tool.cron import CronScheduler, CronCreateTool, CronDeleteTool

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = CronScheduler(workspace_root=tmpdir)
            create_tool = CronCreateTool(scheduler)
            delete_tool = CronDeleteTool(scheduler)

            result = create_tool.execute({"cron": "0 9 * * *", "prompt": "test"})
            job_id = result["job"]["id"]

            del_result = delete_tool.execute({"job_id": job_id})
            assert del_result["deleted"] is True

    def test_list_tool(self):
        from qitos.kit.tool.cron import CronScheduler, CronCreateTool, CronListTool

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = CronScheduler(workspace_root=tmpdir)
            create_tool = CronCreateTool(scheduler)
            list_tool = CronListTool(scheduler)

            create_tool.execute({"cron": "0 9 * * *", "prompt": "task1"})
            create_tool.execute({"cron": "0 17 * * *", "prompt": "task2"})

            result = list_tool.execute({})
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

    def test_call_without_prompt(self):
        from qitos.kit.tool.agent import AgentTool

        tool = AgentTool(invocation_factory=lambda request, _context: None)
        result = tool.execute({"description": "missing task"})
        assert result == {"status": "error", "error": "prompt is required"}

    def test_run_scoped_factory_creates_fresh_invocation_per_call(self):
        from contextlib import contextmanager
        from types import SimpleNamespace

        from qitos.kit.tool.agent import AgentInvocation, AgentTool

        engines = []
        scopes = []

        class FakeEngine:
            active_run_id = "child-run"

            def run(self, task, **kwargs):
                assert task.startswith("seeded:")
                assert kwargs == {}
                return SimpleNamespace(
                    state=SimpleNamespace(
                        final_result="child result",
                        stop_reason="final",
                    ),
                    step_count=3,
                    total_tokens=42,
                )

        def build_invocation(request, runtime_context):
            assert runtime_context["delegate_depth"] == 0
            assert request.max_turns == 200
            engine = FakeEngine()
            engines.append(engine)
            return AgentInvocation(engine=engine, task=f"seeded:{request.prompt}")

        @contextmanager
        def execution_scope(runtime_context):
            assert runtime_context["delegate_depth"] == 0
            scopes.append("enter")
            try:
                yield
            finally:
                scopes.append("exit")

        tool = AgentTool(
            invocation_factory=build_invocation,
            execution_scope=execution_scope,
            execution_mode="foreground",
        )
        first = tool.execute(
            {"description": "first task", "prompt": "one"},
            runtime_context={"delegate_depth": 0},
        )
        second = tool.execute(
            {"description": "second task", "prompt": "two"},
            runtime_context={"delegate_depth": 0},
        )

        assert first["status"] == "success"
        assert first["steps"] == 3
        assert first["total_tokens"] == 42
        assert second["status"] == "success"
        assert len(engines) == 2
        assert engines[0] is not engines[1]
        assert scopes == ["enter", "exit", "enter", "exit"]
        assert tool.spec.concurrency_safe is True
        assert "run_in_background" not in tool.spec.parameters

    def test_run_scoped_factory_rejects_recursive_agent(self):
        from qitos.kit.tool.agent import AgentTool

        tool = AgentTool(
            invocation_factory=lambda request, context: None,
            execution_mode="foreground",
        )
        result = tool.execute(
            {"description": "nested task", "prompt": "recurse"},
            runtime_context={"delegate_depth": 1},
        )

        assert result["status"] == "error"
        assert "cannot launch another Agent" in result["error"]

    def test_forced_background_children_launch_concurrently_and_notify_parent(self):
        from qitos.core.runtime_input import RuntimeInput
        from qitos.kit.tool.agent import AgentInvocation, AgentTool

        lock = threading.Lock()
        both_started = threading.Event()
        release = threading.Event()
        active = 0
        peak = 0
        events = []

        class FakeEngine:
            active_run_id = "child-run"

            def run(self, task, **kwargs):
                nonlocal active, peak
                assert kwargs == {}
                with lock:
                    active += 1
                    peak = max(peak, active)
                    if active == 2:
                        both_started.set()
                assert release.wait(timeout=1)
                with lock:
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
            invocation_factory=lambda request, _context: AgentInvocation(
                engine=FakeEngine(),
                task=request.prompt,
            ),
            execution_mode="background",
            max_background_workers=2,
        )
        runtime_context = {
            "delegate_depth": 0,
            "post_runtime_event": lambda event: events.append(event) or True,
        }

        first = tool.execute(
            {"description": "route one", "prompt": "one"},
            runtime_context=runtime_context,
        )
        second = tool.execute(
            {"description": "route two", "prompt": "two"},
            runtime_context=runtime_context,
        )

        assert first["status"] == "running"
        assert second["status"] == "running"
        assert "run_in_background" not in tool.spec.parameters
        assert tool.spec.supports_background is True
        assert both_started.wait(timeout=1)
        assert peak == 2
        assert tool.active_background_count == 2

        release.set()
        deadline = time.monotonic() + 1
        while len(events) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert len(events) == 2
        assert all(isinstance(event, RuntimeInput) for event in events)
        assert {event.kind for event in events} == {"agent.child.completed"}
        assert {event.payload["output"] for event in events} == {
            "validated:one",
            "validated:two",
        }
        assert tool.active_background_count == 0
        assert tool.close(wait_seconds=0) == 0

    def test_background_invocation_is_created_after_execution_slot_opens(self):
        from contextlib import contextmanager

        from qitos.kit.tool.agent import AgentInvocation, AgentTool

        waiting = threading.Event()
        open_slot = threading.Event()
        factory_called = threading.Event()
        before_launch = object()
        parent_messages = [before_launch]
        received_parent_history = ()
        received_parent_snapshot = ()

        @contextmanager
        def execution_scope(_runtime_context):
            waiting.set()
            assert open_slot.wait(timeout=1)
            yield

        class FakeEngine:
            active_run_id = "child-run"

            def run(self, task, **kwargs):
                _ = kwargs
                return SimpleNamespace(
                    state=SimpleNamespace(final_result=task, stop_reason="final"),
                    records=[],
                    step_count=1,
                    total_tokens=1,
                )

            def cancel(self, mode):
                _ = mode

        def invocation_factory(request, _runtime_context):
            nonlocal received_parent_history, received_parent_snapshot
            factory_called.set()
            received_parent_history = _runtime_context["parent_history"]
            received_parent_snapshot = _runtime_context["parent_history_snapshot"]
            return AgentInvocation(engine=FakeEngine(), task=request.prompt)

        tool = AgentTool(
            invocation_factory=invocation_factory,
            execution_scope=execution_scope,
            execution_mode="background",
        )
        launched = tool.execute(
            {
                "description": "queued route",
                "prompt": "inspect",
                "subagent_type": "fork",
            },
            runtime_context={
                "delegate_depth": 0,
                "agent": SimpleNamespace(
                    history=SimpleNamespace(
                        messages=parent_messages,
                        snapshot=lambda: tuple(parent_messages),
                    )
                ),
            },
        )

        assert launched["status"] == "running"
        assert waiting.wait(timeout=1)
        assert not factory_called.is_set()
        parent_messages.append(object())
        open_slot.set()
        deadline = time.monotonic() + 1
        result = tool.get_background_result(launched["task_id"])
        while result and result["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.01)
            result = tool.get_background_result(launched["task_id"])

        assert factory_called.is_set()
        assert received_parent_history == (before_launch,)
        assert received_parent_snapshot == (before_launch,)
        assert result is not None and result["status"] == "success"
        assert tool.close(wait_seconds=0) == 0

    def test_background_child_becomes_terminal_before_event_delivery(self):
        from qitos.kit.tool.agent import AgentInvocation, AgentTool

        delivery_started = threading.Event()
        release_delivery = threading.Event()

        class FakeEngine:
            active_run_id = "child-run"

            def run(self, task, **kwargs):
                _ = kwargs
                return SimpleNamespace(
                    state=SimpleNamespace(final_result=task, stop_reason="final"),
                    records=[],
                    step_count=1,
                    total_tokens=1,
                )

            def cancel(self, mode):
                _ = mode

        def post_runtime_event(_event):
            delivery_started.set()
            assert release_delivery.wait(timeout=1)

        tool = AgentTool(
            invocation_factory=lambda request, _context: AgentInvocation(
                engine=FakeEngine(),
                task=request.prompt,
            ),
            execution_mode="background",
        )
        launched = tool.execute(
            {"description": "fast route", "prompt": "inspect"},
            runtime_context={
                "delegate_depth": 0,
                "post_runtime_event": post_runtime_event,
            },
        )

        assert launched["status"] == "running"
        assert delivery_started.wait(timeout=1)
        assert tool.active_background_count == 0
        terminal = tool.get_background_result(launched["task_id"])
        assert terminal is not None and terminal["status"] == "success"
        assert tool.cancel_background(launched["task_id"]) is False
        release_delivery.set()
        assert tool.close(wait_seconds=1) == 0

    def test_background_budget_stop_returns_partial_tool_evidence(self):
        from qitos.engine.states import StepRecord
        from qitos.kit.tool.agent import AgentInvocation, AgentTool

        events = []

        class FakeEngine:
            active_run_id = "child-run"

            def run(self, task, **kwargs):
                assert task == "validate"
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
            invocation_factory=lambda request, _context: AgentInvocation(
                engine=FakeEngine(),
                task=request.prompt,
            ),
            execution_mode="background",
        )
        launched = tool.execute(
            {"description": "validate route", "prompt": "validate"},
            runtime_context={
                "delegate_depth": 0,
                "post_runtime_event": lambda event: events.append(event) or True,
            },
        )

        deadline = time.monotonic() + 1
        while not events and time.monotonic() < deadline:
            time.sleep(0.01)

        assert launched["status"] == "running"
        assert len(events) == 1
        assert events[0].payload["status"] == "partial"
        assert events[0].payload["stop_reason"] == "budget_time"
        assert "uid=33(www-data)" in events[0].payload["output"]
        assert tool.close(wait_seconds=0) == 0

    def test_running_background_child_exposes_bounded_tool_evidence_snapshot(self):
        from qitos.engine.states import StepRecord
        from qitos.kit.tool.agent import AgentInvocation, AgentTool

        started = threading.Event()
        cancelled = threading.Event()

        class FakeEngine:
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

            def run(self, task, **kwargs):
                _ = task, kwargs
                started.set()
                assert cancelled.wait(timeout=1)
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
            invocation_factory=lambda request, _context: AgentInvocation(
                engine=FakeEngine(),
                task=request.prompt,
            ),
            execution_mode="background",
        )
        launched = tool.execute(
            {
                "description": "validate extension bypass",
                "name": "extension-bypass",
                "prompt": "validate",
            },
            runtime_context={"delegate_depth": 0},
        )

        assert launched["status"] == "running"
        assert started.wait(timeout=1)

        snapshots = tool.snapshot_background_events()

        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert snapshot.kind == "agent.child.snapshot"
        assert snapshot.event_id == f"{launched['task_id']}:conclude-snapshot"
        assert snapshot.payload["task_id"] == launched["task_id"]
        assert snapshot.payload["status"] == "running"
        assert snapshot.payload["name"] == "extension-bypass"
        assert snapshot.payload["description"] == "validate extension bypass"
        assert snapshot.payload["steps"] == 20
        assert snapshot.payload["run_id"] == "child-run"
        assert "evidence-19" in snapshot.payload["output"]
        assert "evidence-0" not in snapshot.payload["output"]
        assert "private child reasoning" not in snapshot.payload["output"]
        assert len(snapshot.payload["output"]) <= 16_000
        assert tool.close(wait_seconds=1) == 0

    def test_completed_background_child_is_not_repeated_in_snapshot(self):
        from qitos.kit.tool.agent import AgentInvocation, AgentTool

        class FakeEngine:
            active_run_id = "child-run"

            def run(self, task, **kwargs):
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
            invocation_factory=lambda request, _context: AgentInvocation(
                engine=FakeEngine(),
                task=request.prompt,
            ),
            execution_mode="background",
        )
        launched = tool.execute(
            {"description": "done", "prompt": "done"},
            runtime_context={"delegate_depth": 0},
        )

        deadline = time.monotonic() + 1
        while (
            tool.get_background_result(launched["task_id"])["status"] == "running"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)

        assert tool.snapshot_background_events() == []
        assert tool.close(wait_seconds=0) == 0

    def test_close_cancels_a_running_background_child(self):
        from qitos.kit.tool.agent import AgentInvocation, AgentTool

        started = threading.Event()
        cancelled = threading.Event()

        class FakeEngine:
            active_run_id = "child-run"

            def run(self, task, **kwargs):
                _ = task, kwargs
                started.set()
                assert cancelled.wait(timeout=1)
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
            invocation_factory=lambda request, _context: AgentInvocation(
                engine=FakeEngine(),
                task=request.prompt,
            ),
            execution_mode="background",
        )
        result = tool.execute(
            {"description": "long route", "prompt": "wait"},
            runtime_context={"delegate_depth": 0},
        )

        assert result["status"] == "running"
        assert started.wait(timeout=1)
        assert tool.close(wait_seconds=1) == 0
        assert cancelled.is_set()

    def test_later_close_can_wait_for_an_already_cancelled_child(self):
        from qitos.kit.tool.agent import AgentInvocation, AgentTool

        started = threading.Event()
        cancelled = threading.Event()
        release = threading.Event()

        class FakeEngine:
            active_run_id = "child-run"

            def run(self, task, **kwargs):
                _ = task, kwargs
                started.set()
                assert cancelled.wait(timeout=1)
                assert release.wait(timeout=1)
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
            invocation_factory=lambda request, _context: AgentInvocation(
                engine=FakeEngine(),
                task=request.prompt,
            ),
            execution_mode="background",
        )
        result = tool.execute(
            {"description": "long route", "prompt": "wait"},
            runtime_context={"delegate_depth": 0},
        )

        assert result["status"] == "running"
        assert started.wait(timeout=1)
        assert tool.close(wait_seconds=0) == 1
        release.set()
        assert tool.close(wait_seconds=1) == 0

    def test_close_wakes_a_child_waiting_for_an_execution_slot(self):
        from contextlib import contextmanager

        from qitos.kit.tool.agent import AgentInvocation, AgentTool

        waiting = threading.Event()
        child_started = threading.Event()
        factory_called = threading.Event()

        @contextmanager
        def blocked_scope(runtime_context):
            waiting.set()
            cancelled = runtime_context["agent_cancelled"]
            while not cancelled():
                time.sleep(0.01)
            raise RuntimeError("cancelled before child slot opened")
            yield  # pragma: no cover

        class FakeEngine:
            active_run_id = "child-run"

            def run(self, task, **kwargs):
                _ = task, kwargs
                child_started.set()
                raise AssertionError("cancelled child unexpectedly started")

            def cancel(self, mode):
                _ = mode

        def invocation_factory(request, _context):
            factory_called.set()
            return AgentInvocation(engine=FakeEngine(), task=request.prompt)

        tool = AgentTool(
            invocation_factory=invocation_factory,
            execution_scope=blocked_scope,
            execution_mode="background",
        )
        result = tool.execute(
            {"description": "queued route", "prompt": "wait"},
            runtime_context={"delegate_depth": 0},
        )

        assert result["status"] == "running"
        assert waiting.wait(timeout=1)
        assert tool.close(wait_seconds=1) == 0
        terminal = tool.get_background_result(result["task_id"])
        assert terminal is not None
        assert terminal["stop_reason"] == "cancelled_immediate"
        assert not factory_called.is_set()
        assert not child_started.is_set()

    def test_background_child_deadline_expires_before_execution_slot_opens(self):
        from contextlib import contextmanager

        from qitos.kit.tool.agent import AgentInvocation, AgentTool

        factory_called = threading.Event()

        @contextmanager
        def expired_scope(_runtime_context):
            raise TimeoutError("deadline expired before child slot opened")
            yield  # pragma: no cover

        def invocation_factory(request, _context):
            factory_called.set()
            return AgentInvocation(engine=object(), task=request.prompt)

        tool = AgentTool(
            invocation_factory=invocation_factory,
            execution_scope=expired_scope,
            execution_mode="background",
        )
        launched = tool.execute(
            {"description": "queued route", "prompt": "wait"},
            runtime_context={"delegate_depth": 0},
        )

        deadline = time.monotonic() + 1
        terminal = tool.get_background_result(launched["task_id"])
        while (
            terminal is not None
            and terminal["status"] == "running"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
            terminal = tool.get_background_result(launched["task_id"])

        assert terminal is not None
        assert terminal["status"] == "error"
        assert terminal["stop_reason"] == "budget_time"
        assert not factory_called.is_set()
        assert tool.active_background_count == 0
        assert tool.close(wait_seconds=0) == 0
