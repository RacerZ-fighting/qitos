"""Generic tool for launching one independently stateful child agent."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

from ....core.agent_module import AgentModule
from ....core.tool import BaseTool, ToolPermission, ToolSpec

DEFAULT_SUBAGENT_MAX_TURNS = 200


@dataclass(frozen=True)
class AgentRequest:
    """One parent-authored child-agent assignment."""

    prompt: str
    description: str = ""
    name: str = ""
    subagent_type: str = "general-purpose"
    max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS


@dataclass(frozen=True)
class AgentInvocation:
    """A fresh child engine plus the exact task it should run."""

    engine: Any
    task: str
    run_kwargs: dict[str, Any] = field(default_factory=dict)


AgentInvocationFactory = Callable[[AgentRequest, dict[str, Any]], AgentInvocation]
AgentExecutionScope = Callable[[dict[str, Any]], AbstractContextManager[Any]]


@dataclass
class AgentResult:
    """Normalized result from one child-agent run."""

    agent_type: str
    task: str
    success: bool
    output: Any = None
    error: str | None = None
    workspace_root: str | None = None
    run_id: str | None = None
    name: str = ""
    description: str = ""
    steps: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    stop_reason: str = ""


class AgentTool(BaseTool):
    """Launch a fresh child agent for a parent-authored task.

    Applications with run-scoped resources should inject ``invocation_factory``.
    The factory is called once per invocation and must return a fresh Engine; this
    prevents concurrent children from sharing an AgentModule history or model client.
    The class registry remains as a compatibility path for QitOS's built-in agents.
    """

    _agent_types: dict[str, type[AgentModule]] = {}

    def __init__(
        self,
        workspace_root: str = ".",
        model_factory: Callable[..., Any] | None = None,
        max_background_workers: int = 4,
        *,
        invocation_factory: AgentInvocationFactory | None = None,
        execution_scope: AgentExecutionScope | None = None,
        allow_background: bool = True,
        max_delegate_depth: int = 1,
        max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS,
    ) -> None:
        if max_background_workers <= 0:
            raise ValueError("max_background_workers must be positive")
        if max_delegate_depth <= 0:
            raise ValueError("max_delegate_depth must be positive")
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.workspace_root = workspace_root
        self.model_factory = model_factory
        self._invocation_factory = invocation_factory
        self._execution_scope = execution_scope
        self._allow_background = allow_background
        self._max_delegate_depth = max_delegate_depth
        self._max_turns = max_turns
        self._executor = ThreadPoolExecutor(max_workers=max_background_workers)
        self._background_tasks: dict[str, Future[AgentResult]] = {}
        self._background_results: dict[str, AgentResult] = {}

        parameters: dict[str, dict[str, Any]] = {
            "description": {
                "type": "string",
                "description": "A short description of the delegated task.",
            },
            "prompt": {
                "type": "string",
                "description": "The task for the child agent to perform.",
            },
            "name": {
                "type": "string",
                "description": "Optional short name used to identify this child run.",
            },
            "subagent_type": {
                "type": "string",
                "description": (
                    "Optional specialized agent type. Omit it to use the runtime's "
                    "general-purpose child."
                ),
            },
        }
        if allow_background:
            parameters["run_in_background"] = {
                "type": "boolean",
                "description": "Run asynchronously and return a task id.",
            }
        description = (
            "Launch an independent child agent for one clearly scoped multi-step task. "
            "For two or more independent multi-step tasks, make one Agent call per task "
            "in the same response; same-response calls run concurrently under the "
            "concurrent action policy. Do not repeat delegated work in the parent. Keep "
            "dependent steps in the parent until their prerequisites are available. Use "
            "ordinary tools, preferably one bounded batch, for cheap mechanical variants "
            "instead of delegating them."
        )
        tool_spec = ToolSpec(
            name="Agent",
            description=description,
            parameters=parameters,
            required=["description", "prompt"],
            max_retries=0,
            permissions=ToolPermission(),
            concurrency_safe=True,
            supports_background=allow_background,
        )
        super().__init__(spec=tool_spec)
        self.spec.description = description

    @classmethod
    def register_agent_type(cls, name: str, agent_class: type[AgentModule]) -> None:
        """Register one legacy class-constructed agent type."""

        cls._agent_types[name] = agent_class

    def execute(
        self, args: dict[str, Any], runtime_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run one child synchronously, or start a configured background run."""

        context = dict(runtime_context or {})
        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return {"status": "error", "error": "prompt is required"}
        # The model schema requires a description, while this fallback preserves
        # compatibility for older programmatic callers that passed only a prompt.
        description = str(args.get("description", "")).strip() or prompt

        current_depth = int(context.get("delegate_depth", 0))
        if current_depth >= self._max_delegate_depth:
            return {
                "status": "error",
                "error": (
                    "Child agents cannot launch another Agent; maximum delegation "
                    f"depth is {self._max_delegate_depth}."
                ),
            }

        request = AgentRequest(
            prompt=prompt,
            description=description,
            name=str(args.get("name", "")).strip(),
            subagent_type=(
                str(args.get("subagent_type", "general-purpose")).strip()
                or "general-purpose"
            ),
            max_turns=self._max_turns,
        )
        workspace = self._resolve_workspace(args.get("isolation"))
        if isinstance(workspace, dict):
            return workspace

        run_in_background = bool(args.get("run_in_background", False))
        if run_in_background and not self._allow_background:
            return {
                "status": "error",
                "error": "Background child agents are disabled in this runtime.",
            }
        if run_in_background:
            task_id = f"agent-{uuid.uuid4().hex[:8]}"
            future = self._executor.submit(
                self._run_request, request, context, workspace
            )
            self._background_tasks[task_id] = future

            def _on_done(fut: Future[AgentResult], tid: str = task_id) -> None:
                try:
                    self._background_results[tid] = fut.result()
                except Exception as exc:  # pragma: no cover - defensive callback
                    self._background_results[tid] = AgentResult(
                        agent_type=request.subagent_type,
                        task=request.prompt,
                        success=False,
                        error=str(exc),
                        name=request.name,
                        description=request.description,
                    )

            future.add_done_callback(_on_done)
            return {
                "status": "running",
                "task_id": task_id,
                "agent_type": request.subagent_type,
                "description": request.description,
                "workspace": workspace if args.get("isolation") == "worktree" else None,
            }

        return self._result_payload(self._run_request(request, context, workspace))

    def _resolve_workspace(self, isolation: Any) -> str | dict[str, Any]:
        workspace = self.workspace_root
        if isolation != "worktree":
            return workspace
        try:
            from ....kit.agent.worktree_manager import WorktreeManager

            manager = WorktreeManager(workspace_root=workspace)
            return manager.create_worktree(f"agent-{uuid.uuid4().hex[:8]}")
        except Exception as exc:
            return {"status": "error", "error": f"Failed to create worktree: {exc}"}

    def _run_request(
        self,
        request: AgentRequest,
        runtime_context: dict[str, Any],
        workspace_root: str,
    ) -> AgentResult:
        started = time.monotonic()
        scope = (
            self._execution_scope(runtime_context)
            if self._execution_scope is not None
            else nullcontext()
        )
        try:
            with scope:
                if self._invocation_factory is not None:
                    result = self._run_invocation(request, runtime_context)
                else:
                    result = self._run_legacy_agent(request, workspace_root)
        except Exception as exc:
            result = AgentResult(
                agent_type=request.subagent_type,
                task=request.prompt,
                success=False,
                error=str(exc),
                name=request.name,
                description=request.description,
            )
        result.elapsed_seconds = max(0.0, time.monotonic() - started)
        return result

    def _run_invocation(
        self, request: AgentRequest, runtime_context: dict[str, Any]
    ) -> AgentResult:
        factory = self._invocation_factory
        if factory is None:  # pragma: no cover - guarded by _run_request
            raise RuntimeError("run-scoped invocation factory is unavailable")
        invocation = factory(request, runtime_context)
        if not isinstance(invocation, AgentInvocation):
            raise TypeError("invocation_factory must return AgentInvocation")
        engine_result = invocation.engine.run(
            invocation.task,
            **dict(invocation.run_kwargs),
        )
        state = getattr(engine_result, "state", None)
        stop_reason = str(getattr(state, "stop_reason", "") or "")
        final_result = getattr(state, "final_result", "") or ""
        return AgentResult(
            agent_type=request.subagent_type,
            task=request.prompt,
            success=stop_reason == "final",
            output=final_result,
            run_id=str(getattr(invocation.engine, "active_run_id", "") or ""),
            name=request.name,
            description=request.description,
            steps=int(getattr(engine_result, "step_count", 0) or 0),
            total_tokens=int(getattr(engine_result, "total_tokens", 0) or 0),
            stop_reason=stop_reason,
        )

    def _run_legacy_agent(
        self, request: AgentRequest, workspace_root: str
    ) -> AgentResult:
        agent_class = self._agent_types.get(request.subagent_type)
        if agent_class is None:
            return AgentResult(
                agent_type=request.subagent_type,
                task=request.prompt,
                success=False,
                error=(
                    f"Unknown agent type: {request.subagent_type}. "
                    f"Available: {list(self._agent_types.keys())}"
                ),
                name=request.name,
                description=request.description,
            )
        try:
            kwargs: dict[str, Any] = {"workspace_root": workspace_root}
            if self.model_factory:
                kwargs["llm"] = self.model_factory()
            agent = agent_class(**kwargs)
            result = agent.run(task=request.prompt, max_steps=request.max_turns)
            output = None
            if hasattr(result, "final_answer"):
                output = result.final_answer
            elif hasattr(result, "output"):
                output = result.output
            elif hasattr(result, "state") and hasattr(result.state, "final_result"):
                output = result.state.final_result
            return AgentResult(
                agent_type=request.subagent_type,
                task=request.prompt,
                success=True,
                output=output,
                workspace_root=workspace_root,
                name=request.name,
                description=request.description,
                steps=int(getattr(result, "step_count", 0) or 0),
                total_tokens=int(getattr(result, "total_tokens", 0) or 0),
            )
        except Exception as exc:
            return AgentResult(
                agent_type=request.subagent_type,
                task=request.prompt,
                success=False,
                error=str(exc),
                name=request.name,
                description=request.description,
            )

    @staticmethod
    def _result_payload(result: AgentResult) -> dict[str, Any]:
        status = (
            "success" if result.success else "partial" if result.output else "error"
        )
        return {
            "status": status,
            "agent_type": result.agent_type,
            "name": result.name,
            "description": result.description,
            "output": result.output,
            "error": result.error,
            "steps": result.steps,
            "total_tokens": result.total_tokens,
            "elapsed_seconds": result.elapsed_seconds,
            "stop_reason": result.stop_reason,
            "run_id": result.run_id,
        }

    def get_background_result(self, task_id: str) -> dict[str, Any] | None:
        """Return one completed background result, or its running status."""

        result = self._background_results.get(task_id)
        if result is None:
            future = self._background_tasks.get(task_id)
            if future is not None and not future.done():
                return {"status": "running", "task_id": task_id}
            return None
        return self._result_payload(result)
