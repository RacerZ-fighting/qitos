"""Generic tool for launching one independently stateful child agent."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager, nullcontext
from dataclasses import replace
from typing import Any, Literal

from ....core.child import (
    DEFAULT_CHILD_MAX_STEPS,
    AgentConclusion,
    ChildHandle,
    ChildInvocation,
    ChildLaunchRequest,
    ChildResult,
    ChildStatus,
)
from ....core.runtime_input import RuntimeInput
from ....core.task import TaskBudget
from ....core.tool import BaseTool, ToolPermission, ToolSpec
from ....core.tool_result import ToolResult

ChildInvocationFactory = Callable[
    [ChildLaunchRequest, dict[str, Any]], ChildInvocation
]
AgentExecutionScope = Callable[
    [dict[str, Any]],
    AbstractContextManager[Any] | AbstractAsyncContextManager[Any],
]
AgentExecutionMode = Literal["foreground", "optional_background", "background"]

_PARTIAL_RESULT_MAX_ITEMS = 12
_PARTIAL_RESULT_MAX_CHARS = 16_000


class AgentTool(BaseTool):
    """Launch a fresh child agent for a parent-authored task.

    The factory is called once per invocation and must return a fresh Engine; this
    prevents concurrent children from sharing an AgentModule history or model client.
    """

    def __init__(
        self,
        *,
        invocation_factory: ChildInvocationFactory,
        execution_scope: AgentExecutionScope | None = None,
        execution_mode: AgentExecutionMode = "foreground",
        max_background_workers: int = 4,
        max_delegate_depth: int = 1,
        child_budget: TaskBudget | None = None,
    ) -> None:
        if max_background_workers <= 0:
            raise ValueError("max_background_workers must be positive")
        if max_delegate_depth <= 0:
            raise ValueError("max_delegate_depth must be positive")
        resolved_budget = child_budget or TaskBudget(
            max_steps=DEFAULT_CHILD_MAX_STEPS
        )
        if not isinstance(resolved_budget, TaskBudget):
            raise TypeError("child_budget must be a TaskBudget")
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
        self._child_budget = resolved_budget
        self._max_background_workers = max_background_workers
        self._background_limit = asyncio.Semaphore(max_background_workers)
        self._child_limit_lock = asyncio.Lock()
        self._children_started = 0
        self._closed = False
        self._background_tasks: dict[ChildHandle, asyncio.Task[ChildResult]] = {}
        self._background_results: dict[ChildHandle, ChildResult] = {}
        self._background_engines: dict[ChildHandle, Any] = {}
        self._background_requests: dict[ChildHandle, ChildLaunchRequest] = {}
        self._background_cancel: dict[ChildHandle, asyncio.Event] = {}

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
                "is delivered as a later runtime event with a stable child handle. "
            ),
            "background": (
                "This runtime always starts the child asynchronously and immediately "
                "returns a child handle. Completion is delivered as a later runtime "
                "event. "
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

    async def execute(
        self, args: dict[str, Any], runtime_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Await one child, or register an event-loop-owned background child."""

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
        parent_run_id = str(context.get("parent_run_id") or "").strip()
        if not parent_run_id:
            return {
                "status": "error",
                "error": "parent_run_id is required for child ownership",
            }

        current_depth = int(context.get("delegate_depth", 0))
        if current_depth >= self._max_delegate_depth:
            return {
                "status": "error",
                "error": (
                    "Child agents cannot launch another Agent; maximum delegation "
                    f"depth is {self._max_delegate_depth}."
                ),
            }

        request = ChildLaunchRequest(
            task=prompt,
            description=description,
            name=str(args.get("name", "")).strip(),
            agent_type=subagent_type,
            budget=self._child_budget,
        )
        requested_background = bool(args.get("run_in_background", False))
        if requested_background and self._execution_mode == "foreground":
            return {
                "status": "error",
                "error": "Background child agents are disabled in this runtime.",
            }
        if self._closed:
            return {
                "status": "error",
                "error": "This Agent runtime has already closed.",
            }
        max_children = int(context.get("max_children", 0) or 0)
        async with self._child_limit_lock:
            if max_children > 0 and self._children_started >= max_children:
                return {
                    "status": "error",
                    "error": (
                        "Run child-agent budget exhausted: "
                        f"max_children={max_children}."
                    ),
                }
            self._children_started += 1
        run_in_background = self._execution_mode == "background" or (
            self._execution_mode == "optional_background" and requested_background
        )
        handle = ChildHandle(
            child_id=f"child-{uuid.uuid4().hex[:12]}",
            parent_run_id=parent_run_id,
        )
        if run_in_background:
            cancel_event = asyncio.Event()
            task = asyncio.create_task(
                self._supervise_background_request(
                    request,
                    context,
                    handle,
                    cancel_event,
                ),
                name=f"qitos-{handle.child_id}",
            )
            self._background_tasks[handle] = task
            self._background_cancel[handle] = cancel_event
            self._background_requests[handle] = request
            return {
                "status": "running",
                "child_status": ChildStatus.RUNNING.value,
                "handle": handle.to_dict(),
                "child_id": handle.child_id,
                "agent_type": request.agent_type,
                "description": request.description,
            }

        return self._result_payload(
            await self._run_request(
                request,
                context,
                handle=handle,
                cancel_event=None,
            )
        )

    async def _supervise_background_request(
        self,
        request: ChildLaunchRequest,
        runtime_context: dict[str, Any],
        handle: ChildHandle,
        cancel_event: asyncio.Event,
    ) -> ChildResult:
        """Own one child through terminal state and reliable parent delivery."""

        try:
            result = await self._run_background_request(
                request,
                runtime_context,
                handle,
                cancel_event,
            )
        except asyncio.CancelledError:
            result = ChildResult(
                handle=handle,
                request=request,
                status=ChildStatus.CANCELLED,
                conclusion=AgentConclusion(
                    failure_paths=("Child agent was cancelled.",),
                ),
                error="Child agent was cancelled.",
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            result = ChildResult(
                handle=handle,
                request=request,
                status=ChildStatus.FAILED,
                conclusion=AgentConclusion(failure_paths=(str(exc),)),
                error=str(exc),
            )
        self._background_results[handle] = result
        self._background_tasks.pop(handle, None)
        self._background_engines.pop(handle, None)
        self._background_requests.pop(handle, None)
        self._background_cancel.pop(handle, None)
        await self._post_completion_event(handle, result, runtime_context)
        return result

    async def _run_background_request(
        self,
        request: ChildLaunchRequest,
        runtime_context: dict[str, Any],
        handle: ChildHandle,
        cancel_event: asyncio.Event,
    ) -> ChildResult:
        async with self._background_limit:
            return await self._run_request(
                request,
                runtime_context,
                handle=handle,
                cancel_event=cancel_event,
            )

    async def _run_request(
        self,
        request: ChildLaunchRequest,
        runtime_context: dict[str, Any],
        handle: ChildHandle,
        cancel_event: asyncio.Event | None,
    ) -> ChildResult:
        started = time.monotonic()
        scoped_context = dict(runtime_context)
        if cancel_event is not None:
            scoped_context["agent_cancelled"] = cancel_event.is_set
        scope = (
            self._execution_scope(scoped_context)
            if self._execution_scope is not None
            else nullcontext()
        )

        async def _run_in_scope() -> ChildResult:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Child agent was cancelled before it started")
            return await self._run_invocation(
                request,
                scoped_context,
                handle=handle,
            )

        try:
            if isinstance(scope, AbstractAsyncContextManager):
                async with scope:
                    result = await _run_in_scope()
            else:
                with scope:
                    result = await _run_in_scope()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            status = (
                ChildStatus.CANCELLED
                if cancel_event is not None and cancel_event.is_set()
                else ChildStatus.BUDGET_EXHAUSTED
                if isinstance(exc, TimeoutError)
                else ChildStatus.FAILED
            )
            result = ChildResult(
                handle=handle,
                request=request,
                status=status,
                conclusion=AgentConclusion(failure_paths=(str(exc),)),
                error=str(exc),
            )
        return replace(
            result,
            elapsed_seconds=max(0.0, time.monotonic() - started),
        )

    async def _run_invocation(
        self,
        request: ChildLaunchRequest,
        runtime_context: dict[str, Any],
        *,
        handle: ChildHandle,
    ) -> ChildResult:
        invocation = self._invocation_factory(request, runtime_context)
        if not isinstance(invocation, ChildInvocation):
            raise TypeError("invocation_factory must return ChildInvocation")
        if handle in self._background_tasks:
            if self._closed:
                invocation.engine.cancel("immediate")
                raise RuntimeError("Agent runtime closed before child start")
            self._background_engines[handle] = invocation.engine
        engine_result = await invocation.engine.arun(
            invocation.task,
            **dict(invocation.run_kwargs),
        )
        state = getattr(engine_result, "state", None)
        raw_stop_reason = getattr(state, "stop_reason", "") or ""
        stop_reason = str(getattr(raw_stop_reason, "value", raw_stop_reason))
        summary = str(getattr(state, "final_result", "") or "")
        summary = summary or self._partial_result(engine_result)
        status = self._child_status(stop_reason)
        return ChildResult(
            handle=handle,
            request=request,
            status=status,
            conclusion=AgentConclusion(
                summary=summary,
                failure_paths=(stop_reason,) if status is ChildStatus.FAILED else (),
                unknowns=(stop_reason,)
                if status is ChildStatus.BUDGET_EXHAUSTED
                else (),
            ),
            child_run_id=str(
                getattr(engine_result, "run_id", "")
                or getattr(invocation.engine, "active_run_id", "")
                or ""
            ),
            steps=int(getattr(engine_result, "step_count", 0) or 0),
            total_tokens=int(getattr(engine_result, "total_tokens", 0) or 0),
        )

    @staticmethod
    def _result_payload(result: ChildResult) -> dict[str, Any]:
        execution_status = AgentTool._execution_status(result)
        return {
            "status": execution_status,
            "child_status": result.status.value,
            "ready": result.ready,
            "handle": result.handle.to_dict(),
            "child_id": result.handle.child_id,
            "agent_type": result.request.agent_type,
            "name": result.request.name,
            "description": result.request.description,
            "output": AgentTool._json_safe(result.conclusion.summary),
            "conclusion": result.conclusion.to_dict(),
            "error": result.error,
            "steps": result.steps,
            "total_tokens": result.total_tokens,
            "elapsed_seconds": result.elapsed_seconds,
            "stop_reason": result.status.value,
            "run_id": result.child_run_id,
        }

    def child_result(self, handle: ChildHandle) -> ChildResult | None:
        """Return one owned child's current immutable result."""

        if not isinstance(handle, ChildHandle):
            raise TypeError("handle must be a ChildHandle")
        result = self._background_results.get(handle)
        future = self._background_tasks.get(handle)
        if result is None:
            if future is not None:
                request = self._background_requests[handle]
                return ChildResult(
                    handle=handle,
                    request=request,
                    status=ChildStatus.RUNNING,
                    child_run_id=str(
                        getattr(
                            self._background_engines.get(handle),
                            "active_run_id",
                            "",
                        )
                        or ""
                    ),
                )
            return None
        return result

    @property
    def active_background_count(self) -> int:
        """Return the number of submitted children that have not reached terminal state."""

        return len(self._background_tasks)

    def snapshot_background_events(self) -> list[RuntimeInput]:
        """Project active child tool evidence into bounded runtime events.

        The snapshot intentionally excludes model responses and reasoning. It is
        designed for an owning parent that must conclude before its children have
        emitted terminal completion events.
        """

        active = [
            (handle, self._background_requests[handle], engine)
            for handle, engine in self._background_engines.items()
            if handle in self._background_tasks
            and not self._background_tasks[handle].done()
        ]

        events: list[RuntimeInput] = []
        for handle, request, engine in active:
            records = list(getattr(engine, "records", []) or [])
            events.append(
                RuntimeInput(
                    event_id=f"{handle.child_id}:conclude-snapshot",
                    kind="agent.child.snapshot",
                    correlation_id=handle.child_id,
                    source="qitos.agent",
                    payload={
                        "handle": handle.to_dict(),
                        "child_id": handle.child_id,
                        "status": "running",
                        "agent_type": request.agent_type,
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

    def cancel_child(self, handle: ChildHandle) -> bool:
        """Cooperatively cancel one background child if it is still active."""

        if not isinstance(handle, ChildHandle):
            raise TypeError("handle must be a ChildHandle")
        future = self._background_tasks.get(handle)
        engine = self._background_engines.get(handle)
        cancel_event = self._background_cancel.get(handle)
        if future is None or future.done():
            return False
        if cancel_event is not None:
            cancel_event.set()
        if engine is not None:
            engine.cancel("immediate")
        future.cancel()
        return True

    def setup(self, context: dict[str, Any] | None = None) -> None:
        """Open this run-scoped child registry for a fresh parent run."""

        _ = context
        if self._background_tasks:
            raise RuntimeError("cannot reopen AgentTool with active children")
        self._background_limit = asyncio.Semaphore(self._max_background_workers)
        self._child_limit_lock = asyncio.Lock()
        self._closed = False
        self._children_started = 0

    async def asetup(self, context: dict[str, Any] | None = None) -> None:
        self.setup(context)

    async def aclose(self, *, wait_seconds: float = 5.0) -> int:
        """Cancel children, await cleanup, and return the remaining task count."""

        if wait_seconds < 0:
            raise ValueError("wait_seconds must be non-negative")
        self._closed = True
        tasks = list(self._background_tasks.values())
        for cancel_event in self._background_cancel.values():
            cancel_event.set()
        for engine in self._background_engines.values():
            engine.cancel("immediate")
        for task in tasks:
            task.cancel()
        if not tasks:
            return 0
        done, pending = await asyncio.wait(tasks, timeout=wait_seconds)
        if done:
            await asyncio.gather(*done, return_exceptions=True)
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        return sum(1 for task in tasks if not task.done())

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

    async def _post_completion_event(
        self,
        handle: ChildHandle,
        result: ChildResult,
        runtime_context: dict[str, Any],
    ) -> None:
        post_runtime_event = runtime_context.get("post_runtime_event")
        if not callable(post_runtime_event):
            return
        payload = self._result_payload(result)
        try:
            accepted = await post_runtime_event(
                RuntimeInput(
                    event_id=f"{handle.child_id}:terminal",
                    kind="agent.child.completed",
                    correlation_id=handle.child_id,
                    source="qitos.agent",
                    payload=payload,
                )
            )
            if accepted is not True:
                return
        except Exception:
            # The terminal result remains queryable if its parent has closed or
            # durable mailbox acceptance fails.
            return

    @staticmethod
    def _child_status(stop_reason: str) -> ChildStatus:
        if stop_reason in {"completed", "final", "success"}:
            return ChildStatus.COMPLETED
        if stop_reason == "blocked":
            return ChildStatus.BLOCKED
        if stop_reason.startswith("budget_"):
            return ChildStatus.BUDGET_EXHAUSTED
        if stop_reason.startswith("cancelled"):
            return ChildStatus.CANCELLED
        if stop_reason.startswith("interrupt"):
            return ChildStatus.INTERRUPTED
        return ChildStatus.FAILED

    @staticmethod
    def _execution_status(result: ChildResult) -> str:
        if result.status is ChildStatus.COMPLETED:
            return "success"
        if result.status in {ChildStatus.PENDING, ChildStatus.RUNNING}:
            return "running"
        if result.status in {ChildStatus.CANCELLED, ChildStatus.INTERRUPTED}:
            return "cancelled"
        if result.conclusion.summary:
            return "partial"
        return "error"
