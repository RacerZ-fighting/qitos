"""DelegateTool: wraps an AgentSpec as one awaited child-agent call."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Dict, Optional

from ...core.agent_spec import AgentSpec, AgentRegistry, ContextStrategy
from ...core.tool import BaseTool, ToolSpec

if TYPE_CHECKING:
    from ...engine.engine import Engine
    from ...trace.writer import TraceWriter

MAX_DELEGATE_DEPTH = 3


class DelegateTool(BaseTool):
    """Wrap an AgentSpec as an async callable tool.

    When the parent agent calls this tool, a nested ``Engine.arun()`` is awaited
    for the sub-agent. The parent agent's reduce() receives the sub-agent's
    final_result as a normal tool result.
    """

    def __init__(self, spec: AgentSpec, agent_registry: AgentRegistry):
        self.agent_spec = spec
        self.agent_registry = agent_registry
        self._execution_context: str = ""
        public_tool_name = str(spec.tool_name or "").strip() or f"delegate_to_{spec.name}"
        tool_spec = ToolSpec(
            name=public_tool_name,
            description=spec.description,
            parameters={
                "task": {
                    "type": "string",
                    "description": "The subtask to delegate to the agent",
                },
                "context": {
                    "type": "object",
                    "description": "Optional structured context to pass to the sub-agent",
                },
            },
            required=["task"],
            timeout_s=120.0,
        )
        super().__init__(tool_spec)
        # Override description after BaseTool.__init__ which may replace it
        # with the class docstring. We want the agent spec's description.
        self.spec.description = spec.description

    async def execute(
        self, args: Dict[str, Any], runtime_context: Optional[Dict[str, Any]] = None
    ) -> Any:
        runtime_context = runtime_context or {}
        task = str(args.get("task", "")).strip()
        if not task:
            return {"status": "error", "message": "task is required"}

        current_depth = int(runtime_context.get("delegate_depth", 0))
        if current_depth >= MAX_DELEGATE_DEPTH:
            return {
                "status": "error",
                "message": f"Maximum delegate depth ({MAX_DELEGATE_DEPTH}) exceeded",
            }

        trace_writer = runtime_context.get("trace_writer")
        record_runtime_event = runtime_context.get("record_runtime_event")
        parent_run_id = runtime_context.get("parent_run_id", "") or ""

        self._emit_delegate_event(
            trace_writer,
            parent_run_id,
            "DELEGATE_START",
            args,
            record_runtime_event,
        )

        try:
            sub_engine = self._build_sub_engine(runtime_context, current_depth)
            # Inject execution context into the sub-agent before running
            sub_agent = self.agent_spec.agent
            if sub_agent is not None and self._execution_context:
                setattr(sub_agent, '_execution_context', self._execution_context)
            prepared_task = self._prepare_task(task, runtime_context)
            run_kwargs: Dict[str, Any] = {}
            context_arg = args.get("context")
            if isinstance(context_arg, dict):
                run_kwargs["context"] = dict(context_arg)
            result = await sub_engine.arun(prepared_task, **run_kwargs)
        except Exception as exc:
            self._emit_delegate_event(
                trace_writer,
                parent_run_id,
                "DELEGATE_END",
                {"agent": self.agent_spec.name, "status": "error", "error": str(exc)},
                record_runtime_event,
            )
            return {
                "status": "error",
                "agent": self.agent_spec.name,
                "message": str(exc),
            }

        final_result = result.state.final_result or ""
        stop_reason = str(result.state.stop_reason or "")

        self._emit_delegate_event(
            trace_writer,
            parent_run_id,
            "DELEGATE_END",
            {
                "agent": self.agent_spec.name,
                "status": (
                    "success"
                    if stop_reason in {"completed", "final", "success"}
                    else "partial"
                ),
                "steps": result.step_count,
                "stop_reason": stop_reason,
            },
            record_runtime_event,
        )

        return {
            "status": (
                "success"
                if stop_reason in {"completed", "final", "success"}
                else "partial"
            ),
            "agent": self.agent_spec.name,
            "final_result": final_result,
            "steps": result.step_count,
            "stop_reason": stop_reason,
        }

    def _build_sub_engine(
        self, runtime_context: Dict[str, Any], current_depth: int
    ) -> Engine:
        from ...engine.engine import Engine
        from ...engine.states import RuntimeBudget
        from ...engine.stop_criteria import FinalResultCriteria

        sub_agent = self.agent_spec.agent
        env = runtime_context.get("env") if self.agent_spec.shared_env else None
        budget = RuntimeBudget(
            max_steps=self.agent_spec.max_steps_override or 10,
            deadline_monotonic=runtime_context.get("deadline_monotonic"),
            max_children=max(1, int(runtime_context.get("max_children", 1) or 1)),
        )

        trace_writer = runtime_context.get("trace_writer")
        sub_trace_writer = self._build_sub_trace_writer(trace_writer, current_depth)

        # Build tool registry: use override if provided, else keep agent's default
        if self.agent_spec.tools_override is not None:
            sub_agent.tool_registry = self.agent_spec.tools_override

        # Resolve model override if provided
        llm = None
        if self.agent_spec.model_override:
            try:
                from ...models.profile_registry import resolve_model
                llm = resolve_model(self.agent_spec.model_override)
            except Exception:
                pass
        if llm is None:
            llm = getattr(sub_agent, "llm", None)
        if llm is not None and hasattr(sub_agent, "llm"):
            sub_agent.llm = llm

        return Engine(
            agent=sub_agent,
            budget=budget,
            env=env,
            trace_writer=sub_trace_writer,
            delegate_depth=current_depth + 1,
            shared_memory=self.agent_spec.shared_memory,
            agent_registry=self.agent_registry,
            stop_criteria=[FinalResultCriteria()],
        )

    def _build_sub_trace_writer(
        self, parent_trace_writer: Optional[TraceWriter], current_depth: int = 0
    ) -> Optional[TraceWriter]:
        if parent_trace_writer is None:
            return None

        from ...trace.writer import TraceWriter

        parent_run_id = getattr(parent_trace_writer, "run_id", "")
        output_dir = getattr(parent_trace_writer, "output_dir", "runs")

        sub_run_id = f"{parent_run_id}__delegate_{self.agent_spec.name}_depth{current_depth}"
        metadata = dict(getattr(parent_trace_writer, "metadata", {}) or {})
        metadata["parent_run_id"] = parent_run_id
        metadata["agent_name"] = self.agent_spec.name

        return TraceWriter(
            output_dir=output_dir,
            run_id=sub_run_id,
            metadata=metadata,
        )

    def _emit_delegate_event(
        self,
        trace_writer: Optional[TraceWriter],
        parent_run_id: str,
        phase: str,
        payload: Dict[str, Any],
        record_runtime_event: Optional[
            Callable[[str, Dict[str, Any]], None]
        ] = None,
    ) -> None:
        if record_runtime_event is not None:
            record_runtime_event(phase, payload)
        if trace_writer is None:
            return

        from ...trace.events import TraceEvent
        from datetime import datetime, timezone

        event = TraceEvent(
            run_id=getattr(trace_writer, "run_id", parent_run_id),
            step_id=0,
            phase=phase,
            ok=True,
            payload=payload,
            error=None,
            ts=datetime.now(timezone.utc).isoformat(),
        )
        trace_writer.write_event(event)

    def _prepare_task(
        self,
        task: str,
        runtime_context: Dict[str, Any],
    ) -> str:
        """Apply ContextStrategy to the task before passing to sub-agent."""
        if self.agent_spec.context_strategy == ContextStrategy.ISOLATED:
            return task
        state = runtime_context.get("state")
        if state is None:
            return task
        scratchpad = getattr(state, "scratchpad", None)
        if not scratchpad:
            return task
        if self.agent_spec.context_strategy == ContextStrategy.FULL:
            prefix = "Parent agent context:\n" + "\n".join(scratchpad[-16:]) + "\n\nYour task:\n"
            return prefix + task
        if self.agent_spec.context_strategy == ContextStrategy.SUMMARY:
            prefix = "Parent agent summary:\n" + "\n".join(scratchpad[-4:]) + "\n\nYour task:\n"
            return prefix + task
        return task
