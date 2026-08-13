"""FanOutTool: parallel delegation of multiple subtasks to sub-agents."""

from __future__ import annotations

import time
import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional

from ...core.agent_spec import AgentSpec, AgentRegistry, ContextStrategy
from ...core.tool import BaseTool, ToolSpec

if TYPE_CHECKING:
    from ...engine.engine import Engine
    from ...trace.writer import TraceWriter

MAX_DELEGATE_DEPTH = 3


class FanOutTool(BaseTool):
    """Parallel delegation: fan out multiple subtasks, fan in aggregated results.

    The parent agent calls this tool with a list of subtasks. Each subtask is
    dispatched to independently stateful child engines. All child runs
    run independently. Results are collected and aggregated before returning.
    """

    def __init__(self, agent_registry: AgentRegistry, max_workers: int = 4, per_task_timeout: float = 120.0):
        self.agent_registry = agent_registry
        self._max_workers = max_workers
        self._per_task_timeout = per_task_timeout
        tool_spec = ToolSpec(
            name="fanout",
            description=(
                "Delegate multiple subtasks to sub-agents in parallel. "
                "Each subtask runs independently. Returns aggregated results "
                "from all sub-agents."
            ),
            parameters={
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent": {
                                "type": "string",
                                "description": "Agent name from registry",
                            },
                            "task": {
                                "type": "string",
                                "description": "Subtask description",
                            },
                        },
                        "required": ["agent", "task"],
                    },
                    "description": "List of subtasks to delegate in parallel",
                },
            },
            required=["tasks"],
            timeout_s=300.0,
            concurrency_safe=True,
            supports_background=False,
        )
        super().__init__(tool_spec)
        # Override description after BaseTool.__init__
        self.spec.description = tool_spec.description

    async def execute(
        self, args: Dict[str, Any], runtime_context: Optional[Dict[str, Any]] = None
    ) -> Any:
        runtime_context = runtime_context or {}
        tasks = args.get("tasks", [])
        if not tasks:
            return {"status": "error", "message": "tasks is required and must be non-empty"}

        current_depth = int(runtime_context.get("delegate_depth", 0))
        if current_depth >= MAX_DELEGATE_DEPTH:
            return {
                "status": "error",
                "message": f"Maximum delegate depth ({MAX_DELEGATE_DEPTH}) exceeded",
            }

        trace_writer = runtime_context.get("trace_writer")

        self._emit_event(trace_writer, "FANOUT_START", {"task_count": len(tasks)})

        ordered: list[tuple[str, AgentSpec | None, str, dict[str, Any] | None]] = []
        max_children = max(1, int(runtime_context.get("max_children", len(tasks)) or 1))
        admitted_children = 0
        for i, task_spec in enumerate(tasks):
            agent_name = str(task_spec.get("agent", ""))
            task_text = str(task_spec.get("task", "")).strip()
            key = f"{agent_name}_{i}" if agent_name else f"invalid_{i}"
            if not agent_name or not task_text:
                ordered.append(
                    (
                        key,
                        None,
                        "",
                        {
                            "status": "error",
                            "message": "Each task requires 'agent' and 'task' fields",
                        },
                    )
                )
                continue
            try:
                spec = self.agent_registry.resolve(agent_name)
            except KeyError:
                ordered.append(
                    (
                        key,
                        None,
                        "",
                        {
                            "status": "error",
                            "message": f"Agent '{agent_name}' not found in registry",
                        },
                    )
                )
                continue
            if admitted_children >= max_children:
                ordered.append(
                    (
                        key,
                        None,
                        "",
                        {
                            "status": "error",
                            "message": f"Run child-agent budget exhausted: max_children={max_children}",
                        },
                    )
                )
                continue
            admitted_children += 1
            ordered.append(
                (
                    key,
                    spec,
                    self._prepare_task(spec, task_text, runtime_context),
                    None,
                )
            )

        semaphore = asyncio.Semaphore(min(self._max_workers, max_children))

        async def _run_one(spec: AgentSpec, task_text: str, idx: int) -> Dict[str, Any]:
            async with semaphore:
                return await self._run_sub_agent(
                    spec,
                    task_text,
                    runtime_context,
                    current_depth,
                    idx,
                )

        running = [
            asyncio.create_task(_run_one(spec, task_text, idx))
            for idx, (_, spec, task_text, error) in enumerate(ordered)
            if spec is not None and error is None
        ]
        try:
            completed = await asyncio.gather(*running, return_exceptions=True)
        finally:
            pending = [task for task in running if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        outcomes = iter(completed)
        results: Dict[str, Any] = {}
        for key, spec, _, error in ordered:
            if error is not None:
                results[key] = error
                continue
            assert spec is not None
            outcome = next(outcomes)
            results[key] = (
                {"status": "error", "agent": spec.name, "message": str(outcome)}
                if isinstance(outcome, Exception)
                else outcome
            )

        self._emit_event(trace_writer, "FANOUT_END", {
            "total": len(results),
            "succeeded": sum(1 for r in results.values() if r.get("status") == "success"),
        })

        return {
            "status": "success" if any(r.get("status") == "success" for r in results.values()) else "error",
            "results": results,
            "summary": self._aggregate(results),
        }

    async def _run_sub_agent(
        self,
        spec: AgentSpec,
        task: str,
        runtime_context: Dict[str, Any],
        depth: int,
        idx: int,
    ) -> Dict[str, Any]:
        """Await one child under the parent and per-task deadlines."""
        try:
            sub_engine = self._build_sub_engine(spec, runtime_context, depth, idx)
            parent_deadline = runtime_context.get("deadline_monotonic")
            task_deadline = time.monotonic() + self._per_task_timeout
            if isinstance(parent_deadline, (int, float)):
                task_deadline = min(task_deadline, float(parent_deadline))
            remaining = max(0.0, task_deadline - time.monotonic())
            result = await asyncio.wait_for(sub_engine.arun(task), timeout=remaining)
            return {
                "status": (
                    "success"
                    if result.state.stop_reason in {"completed", "final", "success"}
                    else "partial"
                ),
                "agent": spec.name,
                "final_result": result.state.final_result or "",
                "steps": result.step_count,
                "stop_reason": str(result.state.stop_reason or ""),
            }
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "agent": spec.name,
                "message": f"Task timed out after {self._per_task_timeout}s",
            }
        except Exception as exc:
            return {"status": "error", "agent": spec.name, "message": str(exc)}

    def _build_sub_engine(
        self,
        spec: AgentSpec,
        runtime_context: Dict[str, Any],
        depth: int,
        idx: int,
    ) -> Engine:
        from ...engine.engine import Engine
        from ...engine.states import RuntimeBudget

        sub_agent = spec.agent
        env = runtime_context.get("env") if spec.shared_env else None
        budget = RuntimeBudget(
            max_steps=spec.max_steps_override or 10,
            deadline_monotonic=runtime_context.get("deadline_monotonic"),
            max_children=max(1, int(runtime_context.get("max_children", 1) or 1)),
        )
        trace_writer = self._build_sub_trace_writer(
            runtime_context.get("trace_writer"), spec.name, idx
        )
        return Engine(
            agent=sub_agent,
            budget=budget,
            env=env,
            trace_writer=trace_writer,
            delegate_depth=depth + 1,
            shared_memory=spec.shared_memory,
        )

    def _build_sub_trace_writer(
        self,
        parent_trace_writer: Optional[TraceWriter],
        agent_name: str,
        idx: int,
    ) -> Optional[TraceWriter]:
        if parent_trace_writer is None:
            return None

        from ...trace.writer import TraceWriter

        parent_run_id = getattr(parent_trace_writer, "run_id", "")
        output_dir = getattr(parent_trace_writer, "output_dir", "runs")
        sub_run_id = f"{parent_run_id}__fanout_{agent_name}_{idx}"
        metadata = dict(getattr(parent_trace_writer, "metadata", {}) or {})
        metadata["parent_run_id"] = parent_run_id
        metadata["agent_name"] = agent_name
        metadata["fanout_index"] = idx
        return TraceWriter(
            output_dir=output_dir,
            run_id=sub_run_id,
            metadata=metadata,
        )

    def _prepare_task(
        self,
        spec: AgentSpec,
        task: str,
        runtime_context: Dict[str, Any],
    ) -> str:
        """Apply ContextStrategy to the task before passing to sub-agent."""
        if spec.context_strategy == ContextStrategy.ISOLATED:
            return task
        state = runtime_context.get("state")
        if state is None:
            return task
        scratchpad = getattr(state, "scratchpad", None)
        if not scratchpad:
            return task
        if spec.context_strategy == ContextStrategy.FULL:
            prefix = "Parent agent context:\n" + "\n".join(scratchpad[-16:]) + "\n\nYour task:\n"
            return prefix + task
        if spec.context_strategy == ContextStrategy.SUMMARY:
            prefix = "Parent agent summary:\n" + "\n".join(scratchpad[-4:]) + "\n\nYour task:\n"
            return prefix + task
        return task

    def _aggregate(self, results: Dict[str, Any]) -> str:
        """Reduce sub-agent results into a summary string."""
        successful = [r for r in results.values() if r.get("status") == "success"]
        partial = [r for r in results.values() if r.get("status") == "partial"]
        errors = [r for r in results.values() if r.get("status") == "error"]
        lines = [
            f"Total: {len(results)} tasks, "
            f"{len(successful)} succeeded, "
            f"{len(partial)} partial, "
            f"{len(errors)} failed."
        ]
        for key, r in results.items():
            result_text = r.get("final_result", r.get("message", ""))
            lines.append(f"- {key}: {str(result_text)[:200]}")
        return "\n".join(lines)

    def _emit_event(
        self,
        trace_writer: Optional[TraceWriter],
        phase: str,
        payload: Dict[str, Any],
    ) -> None:
        if trace_writer is None:
            return
        from ...trace.events import TraceEvent
        from datetime import datetime, timezone

        event = TraceEvent(
            run_id=getattr(trace_writer, "run_id", ""),
            step_id=0,
            phase=phase,
            ok=True,
            payload=payload,
            error=None,
            ts=datetime.now(timezone.utc).isoformat(),
        )
        trace_writer.write_event(event)
