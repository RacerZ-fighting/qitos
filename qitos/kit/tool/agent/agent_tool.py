"""Generic tool for launching one independently stateful child agent."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, wait
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal

from ....core.runtime_input import RuntimeInput
from ....core.tool import BaseTool, ToolPermission, ToolSpec
from ....core.tool_result import ToolResult
from ....engine._daemon_pool import DaemonTaskPool

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
AgentExecutionMode = Literal["foreground", "optional_background", "background"]

_PARTIAL_RESULT_MAX_ITEMS = 12
_PARTIAL_RESULT_MAX_CHARS = 16_000


@dataclass
class AgentResult:
    """Normalized result from one child-agent run."""

    agent_type: str
    task: str
    success: bool
    output: Any = None
    error: str | None = None
    run_id: str | None = None
    name: str = ""
    description: str = ""
    steps: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    stop_reason: str = ""


class AgentTool(BaseTool):
    """Launch a fresh child agent for a parent-authored task.

    The factory is called once per invocation and must return a fresh Engine; this
    prevents concurrent children from sharing an AgentModule history or model client.
    """

    def __init__(
        self,
        *,
        invocation_factory: AgentInvocationFactory,
        execution_scope: AgentExecutionScope | None = None,
        execution_mode: AgentExecutionMode = "foreground",
        max_background_workers: int = 4,
        max_delegate_depth: int = 1,
        max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS,
    ) -> None:
        if max_background_workers <= 0:
            raise ValueError("max_background_workers must be positive")
        if max_delegate_depth <= 0:
            raise ValueError("max_delegate_depth must be positive")
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if execution_mode not in {
            "foreground",
            "optional_background",
            "background",
        }:
            raise ValueError(f"unsupported Agent execution_mode: {execution_mode}")
        self._invocation_factory = invocation_factory
        self._execution_scope = execution_scope
        self._execution_mode = execution_mode
        self._max_delegate_depth = max_delegate_depth
        self._max_turns = max_turns
        self._executor = DaemonTaskPool(
            max_workers=max_background_workers,
            thread_name_prefix="qitos-agent",
        )
        self._lock = threading.RLock()
        self._closed = False
        self._background_tasks: dict[str, Future[AgentResult]] = {}
        self._background_results: dict[str, AgentResult] = {}
        self._background_engines: dict[str, Any] = {}
        self._background_requests: dict[str, AgentRequest] = {}
        self._background_cancel: dict[str, threading.Event] = {}

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
        if execution_mode == "optional_background":
            parameters["run_in_background"] = {
                "type": "boolean",
                "description": "Run asynchronously and return a task id.",
            }
        execution_description = {
            "foreground": "This runtime waits for the child result before continuing. ",
            "optional_background": (
                "Set run_in_background=true for an asynchronous child; its completion "
                "is delivered as a later runtime event. "
            ),
            "background": (
                "This runtime always starts the child asynchronously and immediately "
                "returns a task id. Completion is delivered as a later runtime event. "
            ),
        }[execution_mode]
        description = execution_description + (
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
            permissions=ToolPermission(),
            concurrency_safe=True,
            supports_background=execution_mode != "foreground",
        )
        super().__init__(spec=tool_spec)
        self.spec.description = description

    def execute(
        self, args: dict[str, Any], runtime_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run one child synchronously, or start a configured background run."""

        context = dict(runtime_context or {})
        parent_agent = context.get("agent")
        parent_history = getattr(parent_agent, "history", None)
        parent_messages = getattr(parent_history, "messages", None)
        if parent_messages is not None and "parent_history" not in context:
            context["parent_history"] = tuple(parent_messages)
        subagent_type = (
            str(args.get("subagent_type", "general-purpose")).strip()
            or "general-purpose"
        )
        if subagent_type == "fork" and "parent_history_snapshot" not in context:
            snapshot = getattr(parent_history, "snapshot", None)
            if callable(snapshot):
                context["parent_history_snapshot"] = snapshot()
        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return {"status": "error", "error": "prompt is required"}
        description = str(args.get("description", "")).strip()
        if not description:
            return {"status": "error", "error": "description is required"}

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
            subagent_type=subagent_type,
            max_turns=self._max_turns,
        )
        requested_background = bool(args.get("run_in_background", False))
        if requested_background and self._execution_mode == "foreground":
            return {
                "status": "error",
                "error": "Background child agents are disabled in this runtime.",
            }
        run_in_background = self._execution_mode == "background" or (
            self._execution_mode == "optional_background" and requested_background
        )
        if run_in_background:
            task_id = f"agent-{uuid.uuid4().hex[:8]}"
            cancel_event = threading.Event()
            with self._lock:
                if self._closed:
                    return {
                        "status": "error",
                        "error": "This Agent runtime has already closed.",
                    }
                future = self._executor.submit(
                    self._run_request,
                    request,
                    context,
                    task_id,
                    cancel_event,
                )
                self._background_tasks[task_id] = future
                self._background_cancel[task_id] = cancel_event
                self._background_requests[task_id] = request

            def _on_done(fut: Future[AgentResult], tid: str = task_id) -> None:
                try:
                    result = fut.result()
                except CancelledError:
                    result = AgentResult(
                        agent_type=request.subagent_type,
                        task=request.prompt,
                        success=False,
                        error="Child agent was cancelled.",
                        name=request.name,
                        description=request.description,
                        stop_reason="cancelled_immediate",
                    )
                except Exception as exc:  # pragma: no cover - defensive callback
                    result = AgentResult(
                        agent_type=request.subagent_type,
                        task=request.prompt,
                        success=False,
                        error=str(exc),
                        name=request.name,
                        description=request.description,
                        stop_reason="error",
                    )
                with self._lock:
                    self._background_results[tid] = result
                    self._background_tasks.pop(tid, None)
                    self._background_engines.pop(tid, None)
                    self._background_requests.pop(tid, None)
                    self._background_cancel.pop(tid, None)
                self._post_completion_event(tid, result, context)

            future.add_done_callback(_on_done)
            return {
                "status": "running",
                "task_id": task_id,
                "agent_type": request.subagent_type,
                "description": request.description,
            }

        return self._result_payload(
            self._run_request(
                request,
                context,
                task_id=None,
                cancel_event=None,
            )
        )

    def _run_request(
        self,
        request: AgentRequest,
        runtime_context: dict[str, Any],
        task_id: str | None,
        cancel_event: threading.Event | None,
    ) -> AgentResult:
        started = time.monotonic()
        scoped_context = dict(runtime_context)
        if cancel_event is not None:
            scoped_context["agent_cancelled"] = cancel_event.is_set
        scope = (
            self._execution_scope(scoped_context)
            if self._execution_scope is not None
            else nullcontext()
        )
        try:
            with scope:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Child agent was cancelled before it started")
                result = self._run_invocation(
                    request,
                    scoped_context,
                    task_id=task_id,
                )
        except Exception as exc:
            result = AgentResult(
                agent_type=request.subagent_type,
                task=request.prompt,
                success=False,
                error=str(exc),
                name=request.name,
                description=request.description,
                stop_reason=(
                    "cancelled_immediate"
                    if cancel_event is not None and cancel_event.is_set()
                    else "budget_time"
                    if isinstance(exc, TimeoutError)
                    else "error"
                ),
            )
        result.elapsed_seconds = max(0.0, time.monotonic() - started)
        return result

    def _run_invocation(
        self,
        request: AgentRequest,
        runtime_context: dict[str, Any],
        *,
        task_id: str | None,
    ) -> AgentResult:
        invocation = self._invocation_factory(request, runtime_context)
        if not isinstance(invocation, AgentInvocation):
            raise TypeError("invocation_factory must return AgentInvocation")
        if task_id is not None:
            with self._lock:
                if self._closed:
                    invocation.engine.cancel("immediate")
                    raise RuntimeError("Agent runtime closed before child start")
                self._background_engines[task_id] = invocation.engine
        engine_result = invocation.engine.run(
            invocation.task,
            **dict(invocation.run_kwargs),
        )
        state = getattr(engine_result, "state", None)
        raw_stop_reason = getattr(state, "stop_reason", "") or ""
        stop_reason = str(getattr(raw_stop_reason, "value", raw_stop_reason))
        final_result = getattr(state, "final_result", "") or ""
        output = final_result or self._partial_result(engine_result)
        return AgentResult(
            agent_type=request.subagent_type,
            task=request.prompt,
            success=stop_reason == "final",
            output=output,
            run_id=str(
                getattr(engine_result, "run_id", "")
                or getattr(invocation.engine, "active_run_id", "")
                or ""
            ),
            name=request.name,
            description=request.description,
            steps=int(getattr(engine_result, "step_count", 0) or 0),
            total_tokens=int(getattr(engine_result, "total_tokens", 0) or 0),
            stop_reason=stop_reason,
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
            "output": AgentTool._json_safe(result.output),
            "error": result.error,
            "steps": result.steps,
            "total_tokens": result.total_tokens,
            "elapsed_seconds": result.elapsed_seconds,
            "stop_reason": result.stop_reason,
            "run_id": result.run_id,
        }

    def get_background_result(self, task_id: str) -> dict[str, Any] | None:
        """Return one completed background result, or its running status."""

        with self._lock:
            result = self._background_results.get(task_id)
            future = self._background_tasks.get(task_id)
        if result is None:
            if future is not None:
                return {"status": "running", "task_id": task_id}
            return None
        return self._result_payload(result)

    @property
    def active_background_count(self) -> int:
        """Return the number of submitted children that have not reached terminal state."""

        with self._lock:
            return len(self._background_tasks)

    def snapshot_background_events(self) -> list[RuntimeInput]:
        """Project active child tool evidence into bounded runtime events.

        The snapshot intentionally excludes model responses and reasoning. It is
        designed for an owning parent that must conclude before its children have
        emitted terminal completion events.
        """

        with self._lock:
            active = [
                (task_id, self._background_requests[task_id], engine)
                for task_id, engine in self._background_engines.items()
                if not self._background_tasks[task_id].done()
            ]

        events: list[RuntimeInput] = []
        for task_id, request, engine in active:
            records = list(getattr(engine, "records", []) or [])
            events.append(
                RuntimeInput(
                    event_id=f"{task_id}:conclude-snapshot",
                    kind="agent.child.snapshot",
                    correlation_id=task_id,
                    source="qitos.agent",
                    payload={
                        "task_id": task_id,
                        "status": "running",
                        "agent_type": request.subagent_type,
                        "name": request.name,
                        "description": request.description,
                        "output": self._partial_result(engine),
                        "steps": len(records),
                        "total_tokens": int(getattr(engine, "_token_usage", 0) or 0),
                        "run_id": str(getattr(engine, "active_run_id", "") or ""),
                    },
                )
            )
        return events

    def cancel_background(self, task_id: str) -> bool:
        """Cooperatively cancel one background child if it is still active."""

        with self._lock:
            future = self._background_tasks.get(task_id)
            engine = self._background_engines.get(task_id)
            cancel_event = self._background_cancel.get(task_id)
        if future is None or future.done():
            return False
        if cancel_event is not None:
            cancel_event.set()
        if engine is not None:
            engine.cancel("immediate")
        future.cancel()
        return True

    def close(self, *, wait_seconds: float = 5.0) -> int:
        """Cancel remaining children and return how many did not stop in time."""

        if wait_seconds < 0:
            raise ValueError("wait_seconds must be non-negative")
        with self._lock:
            self._closed = True
            futures = list(self._background_tasks.values())
            engines = list(self._background_engines.values())
            cancel_events = list(self._background_cancel.values())
            for cancel_event in cancel_events:
                cancel_event.set()
            for engine in engines:
                engine.cancel("immediate")
            for future in futures:
                future.cancel()
        _, pending = wait(futures, timeout=wait_seconds) if futures else (set(), set())
        self._executor.shutdown(
            wait_for_workers=False,
            cancel_futures=True,
        )
        return len(pending)

    @staticmethod
    def _partial_result(engine_result: Any) -> str:
        items: list[str] = []
        for record in list(getattr(engine_result, "records", []) or []):
            step_id = int(getattr(record, "step_id", 0) or 0)
            invocations = list(getattr(record, "tool_invocations", []) or [])
            results = list(getattr(record, "action_results", []) or [])
            for index, raw_result in enumerate(results):
                invocation = invocations[index] if index < len(invocations) else {}
                tool_name = (
                    str(invocation.get("tool_name", "") or "tool")
                    if isinstance(invocation, dict)
                    else "tool"
                )
                result = ToolResult.from_value(raw_result)
                text = (
                    str(result.error or "")
                    if result.output is None and result.error
                    else result.text
                ).strip()
                if text:
                    items.append(f"[step {step_id} {tool_name}] {text}")
        if not items:
            return ""
        rendered = "\n".join(items[-_PARTIAL_RESULT_MAX_ITEMS:])
        if len(rendered) <= _PARTIAL_RESULT_MAX_CHARS:
            return rendered
        marker = "\n...[earlier child evidence clipped]...\n"
        tail_size = _PARTIAL_RESULT_MAX_CHARS - len(marker)
        return marker + rendered[-tail_size:]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError):
            return str(value)

    def _post_completion_event(
        self,
        task_id: str,
        result: AgentResult,
        runtime_context: dict[str, Any],
    ) -> None:
        post_runtime_event = runtime_context.get("post_runtime_event")
        if not callable(post_runtime_event):
            return
        payload = self._result_payload(result)
        payload["task_id"] = task_id
        try:
            post_runtime_event(
                RuntimeInput(
                    event_id=f"{task_id}:terminal",
                    kind="agent.child.completed",
                    correlation_id=task_id,
                    source="qitos.agent",
                    payload=payload,
                )
            )
        except Exception:
            # Completion delivery is best-effort when the parent run has already
            # closed; the terminal result remains queryable from this tool.
            return
