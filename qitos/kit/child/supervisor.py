"""Structured async supervision for independently stateful child Agent runs."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import Any

from ...core.child import (
    AgentConclusion,
    ChildEngine,
    ChildHandle,
    ChildInvocation,
    ChildLaunchRequest,
    ChildResult,
    ChildStatus,
)
from ...core.runtime_input import RuntimeInput
from ...core.tool_result import ToolResult

ChildInvocationFactory = Callable[
    [ChildLaunchRequest, dict[str, Any]], ChildInvocation
]
ChildExecutionScope = Callable[
    [dict[str, Any]],
    AbstractContextManager[Any] | AbstractAsyncContextManager[Any],
]

_PARTIAL_RESULT_MAX_ITEMS = 12
_PARTIAL_RESULT_MAX_CHARS = 16_000


@dataclass(slots=True)
class _OwnedChild:
    handle: ChildHandle
    request: ChildLaunchRequest
    runtime_context: dict[str, Any]
    cancel_event: asyncio.Event
    terminal_event: asyncio.Event
    task: asyncio.Task[ChildResult] | None = None
    engine: ChildEngine | None = None
    result: ChildResult | None = None


class ChildSupervisor:
    """Own child Engines, Tasks, terminal results, and parent delivery for one Run."""

    def __init__(
        self,
        *,
        invocation_factory: ChildInvocationFactory,
        execution_scope: ChildExecutionScope | None = None,
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._invocation_factory = invocation_factory
        self._execution_scope = execution_scope
        self._max_concurrency = max_concurrency
        self._limit = asyncio.Semaphore(max_concurrency)
        self._admission_lock = asyncio.Lock()
        self._children_started = 0
        self._closed = False
        self._children: dict[ChildHandle, _OwnedChild] = {}

    async def launch(
        self,
        request: ChildLaunchRequest,
        runtime_context: dict[str, Any],
        *,
        parent_run_id: str,
        background: bool,
        max_children: int = 0,
    ) -> ChildResult:
        """Launch one fresh child, returning a running or terminal projection."""

        if not isinstance(request, ChildLaunchRequest):
            raise TypeError("request must be a ChildLaunchRequest")
        normalized_parent = str(parent_run_id or "").strip()
        if not normalized_parent:
            raise ValueError("parent_run_id must be a non-empty string")
        if isinstance(max_children, bool) or not isinstance(max_children, int):
            raise TypeError("max_children must be an integer")
        if max_children < 0:
            raise ValueError("max_children must be non-negative")
        async with self._admission_lock:
            if self._closed:
                raise RuntimeError("child supervisor is closed")
            if max_children > 0 and self._children_started >= max_children:
                raise RuntimeError(
                    f"Run child-agent budget exhausted: max_children={max_children}."
                )
            self._children_started += 1
            handle = ChildHandle(
                child_id=f"child-{uuid.uuid4().hex[:12]}",
                parent_run_id=normalized_parent,
            )
            owned = _OwnedChild(
                handle=handle,
                request=request,
                runtime_context=dict(runtime_context),
                cancel_event=asyncio.Event(),
                terminal_event=asyncio.Event(),
            )
            self._children[handle] = owned
            if background:
                owned.task = asyncio.create_task(
                    self._supervise_background(owned),
                    name=f"qitos-{handle.child_id}",
                )
                return self._current_result(owned)
            owned.task = asyncio.current_task()

        try:
            result = await self._run_request(owned)
        except asyncio.CancelledError:
            self._cancel_engine(owned)
            self._set_terminal(owned, self._cancelled_result(owned))
            raise
        finally:
            owned.task = None
            owned.engine = None
            owned.runtime_context.clear()
        self._set_terminal(owned, result)
        return result

    def result(self, handle: ChildHandle) -> ChildResult | None:
        """Return one owned child's immutable current state."""

        owned = self._owned(handle)
        return None if owned is None else self._current_result(owned)

    async def wait(
        self,
        handle: ChildHandle,
        *,
        timeout_seconds: float | None = None,
    ) -> ChildResult | None:
        """Wait for terminal state without cancelling the child on timeout."""

        owned = self._owned(handle)
        if owned is None:
            return None
        current = self._current_result(owned)
        if current.ready:
            return current
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative or None")
        try:
            if timeout_seconds is None:
                await owned.terminal_event.wait()
            else:
                async with asyncio.timeout(timeout_seconds):
                    await owned.terminal_event.wait()
        except TimeoutError:
            return self._current_result(owned)
        return self._current_result(owned)

    async def interrupt(
        self,
        handle: ChildHandle,
        *,
        wait_seconds: float = 5.0,
    ) -> ChildResult | None:
        """Cancel one active child and wait a bounded time for its cleanup."""

        if wait_seconds < 0:
            raise ValueError("wait_seconds must be non-negative")
        owned = self._owned(handle)
        if owned is None:
            return None
        current = self._current_result(owned)
        if current.ready:
            return current
        self.request_interrupt(handle)
        task = owned.task
        if task is not None:
            if wait_seconds > 0:
                done, _ = await asyncio.wait((task,), timeout=wait_seconds)
                if done:
                    await asyncio.gather(*done, return_exceptions=True)
            if task.done() and not owned.terminal_event.is_set():
                self._set_terminal(owned, self._cancelled_result(owned))
        return self._current_result(owned)

    def request_interrupt(self, handle: ChildHandle) -> bool:
        """Signal one active child without waiting for terminal cleanup."""

        owned = self._owned(handle)
        if owned is None or self._current_result(owned).ready:
            return False
        owned.result = ChildResult(
            handle=owned.handle,
            request=owned.request,
            status=ChildStatus.CANCEL_REQUESTED,
            child_run_id=self._child_run_id(owned),
        )
        owned.cancel_event.set()
        self._cancel_engine(owned)
        if owned.task is not None:
            owned.task.cancel()
        return True

    @property
    def active_count(self) -> int:
        """Return children whose execution has not reached terminal state."""

        return sum(
            1
            for owned in self._children.values()
            if not self._current_result(owned).ready
        )

    def snapshot_events(self) -> list[RuntimeInput]:
        """Project bounded completed Tool evidence from active child Engines."""

        events: list[RuntimeInput] = []
        for owned in self._children.values():
            if self._current_result(owned).ready or owned.engine is None:
                continue
            records = list(getattr(owned.engine, "records", []) or [])
            events.append(
                RuntimeInput(
                    event_id=f"{owned.handle.child_id}:conclude-snapshot",
                    kind="agent.child.snapshot",
                    correlation_id=owned.handle.child_id,
                    source="qitos.agent",
                    payload={
                        "handle": owned.handle.to_dict(),
                        "child_id": owned.handle.child_id,
                        "status": "running",
                        "child_status": ChildStatus.RUNNING.value,
                        "agent_type": owned.request.agent_type,
                        "name": owned.request.name,
                        "description": owned.request.description,
                        "output": self._partial_result(owned.engine),
                        "steps": len(records),
                        "total_tokens": int(
                            getattr(owned.engine, "_token_usage", 0) or 0
                        ),
                        "run_id": self._child_run_id(owned),
                    },
                )
            )
        return events

    def setup(self) -> None:
        """Open this supervisor for a fresh owner Run."""

        if any(
            owned.task is not None and not owned.task.done()
            for owned in self._children.values()
        ):
            raise RuntimeError("cannot reopen a child supervisor with owned tasks")
        self._limit = asyncio.Semaphore(self._max_concurrency)
        self._admission_lock = asyncio.Lock()
        self._children_started = 0
        self._closed = False
        self._children.clear()

    async def aclose(self, *, wait_seconds: float = 5.0) -> int:
        """Cancel and drain every owned Task, including terminal event delivery."""

        if wait_seconds < 0:
            raise ValueError("wait_seconds must be non-negative")
        async with self._admission_lock:
            self._closed = True
            owned_tasks = [
                owned
                for owned in self._children.values()
                if owned.task is not None and not owned.task.done()
            ]
        for owned in owned_tasks:
            if not self._current_result(owned).ready:
                owned.result = ChildResult(
                    handle=owned.handle,
                    request=owned.request,
                    status=ChildStatus.CANCEL_REQUESTED,
                    child_run_id=self._child_run_id(owned),
                )
                owned.cancel_event.set()
                self._cancel_engine(owned)
            if owned.task is not None:
                owned.task.cancel()

        tasks = [owned.task for owned in owned_tasks if owned.task is not None]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=wait_seconds)
            if done:
                await asyncio.gather(*done, return_exceptions=True)
            for task in pending:
                task.cancel()
            await asyncio.sleep(0)
        for owned in owned_tasks:
            if owned.task is not None and owned.task.done():
                if not owned.terminal_event.is_set():
                    self._set_terminal(owned, self._cancelled_result(owned))
        return sum(1 for task in tasks if not task.done())

    async def _supervise_background(self, owned: _OwnedChild) -> ChildResult:
        try:
            async with self._limit:
                result = await self._run_request(owned)
        except asyncio.CancelledError:
            result = self._cancelled_result(owned)
        except Exception as exc:  # pragma: no cover - defensive boundary
            result = self._failed_result(owned, exc)
        self._set_terminal(owned, result)
        try:
            await self._post_completion_event(owned, result)
        except asyncio.CancelledError:
            raise
        finally:
            owned.task = None
            owned.engine = None
            owned.runtime_context.clear()
        return result

    async def _run_request(self, owned: _OwnedChild) -> ChildResult:
        started = time.monotonic()
        scoped_context = dict(owned.runtime_context)
        scoped_context["agent_cancelled"] = owned.cancel_event.is_set
        scope = (
            self._execution_scope(scoped_context)
            if self._execution_scope is not None
            else nullcontext()
        )

        async def _run_in_scope() -> ChildResult:
            if owned.cancel_event.is_set():
                raise RuntimeError("Child agent was cancelled before it started")
            return await self._run_invocation(owned, scoped_context)

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
            if owned.cancel_event.is_set():
                result = self._cancelled_result(owned, error=str(exc))
            elif isinstance(exc, TimeoutError):
                error = str(exc) or "Child deadline expired."
                result = ChildResult(
                    handle=owned.handle,
                    request=owned.request,
                    status=ChildStatus.BUDGET_EXHAUSTED,
                    conclusion=AgentConclusion(unknowns=(error,)),
                    error=error,
                    child_run_id=self._child_run_id(owned),
                )
            else:
                result = self._failed_result(owned, exc)
        return replace(
            result,
            elapsed_seconds=max(0.0, time.monotonic() - started),
        )

    async def _run_invocation(
        self,
        owned: _OwnedChild,
        runtime_context: dict[str, Any],
    ) -> ChildResult:
        invocation = self._invocation_factory(owned.request, runtime_context)
        if not isinstance(invocation, ChildInvocation):
            raise TypeError("invocation_factory must return ChildInvocation")
        if self._closed:
            invocation.engine.cancel("immediate")
            raise RuntimeError("child supervisor closed before child start")
        owned.engine = invocation.engine
        engine_result = await invocation.engine.arun(
            invocation.task,
            **dict(invocation.run_kwargs),
        )
        state = engine_result.state
        raw_stop_reason = state.stop_reason or ""
        stop_reason = str(getattr(raw_stop_reason, "value", raw_stop_reason))
        summary = str(state.final_result or "") or self._partial_result(engine_result)
        status = self._child_status(stop_reason)
        status_detail = stop_reason or "unknown stop reason"
        return ChildResult(
            handle=owned.handle,
            request=owned.request,
            status=status,
            conclusion=AgentConclusion(
                summary=summary,
                failure_paths=(status_detail,)
                if status is ChildStatus.FAILED
                else (),
                unknowns=(status_detail,)
                if status is ChildStatus.BUDGET_EXHAUSTED
                else (),
            ),
            child_run_id=str(
                getattr(engine_result, "run_id", "")
                or invocation.engine.active_run_id
                or ""
            ),
            steps=int(engine_result.step_count or 0),
            total_tokens=int(engine_result.total_tokens or 0),
        )

    def _set_terminal(self, owned: _OwnedChild, result: ChildResult) -> None:
        owned.result = result
        owned.terminal_event.set()

    def _current_result(self, owned: _OwnedChild) -> ChildResult:
        if owned.result is not None:
            return owned.result
        return ChildResult(
            handle=owned.handle,
            request=owned.request,
            status=(
                ChildStatus.CANCEL_REQUESTED
                if owned.cancel_event.is_set()
                else ChildStatus.RUNNING
                if owned.task is not None
                else ChildStatus.PENDING
            ),
            child_run_id=self._child_run_id(owned),
        )

    @staticmethod
    def result_payload(result: ChildResult) -> dict[str, Any]:
        """Project typed child state without overloading Tool execution status."""

        return {
            "status": ChildSupervisor._execution_status(result),
            "child_status": result.status.value,
            "ready": result.ready,
            "handle": result.handle.to_dict(),
            "child_id": result.handle.child_id,
            "agent_type": result.request.agent_type,
            "name": result.request.name,
            "description": result.request.description,
            "output": ChildSupervisor._json_safe(result.conclusion.summary),
            "conclusion": result.conclusion.to_dict(),
            "error": result.error,
            "steps": result.steps,
            "total_tokens": result.total_tokens,
            "elapsed_seconds": result.elapsed_seconds,
            "stop_reason": result.status.value,
            "run_id": result.child_run_id,
        }

    async def _post_completion_event(
        self,
        owned: _OwnedChild,
        result: ChildResult,
    ) -> None:
        post_runtime_event = owned.runtime_context.get("post_runtime_event")
        if not callable(post_runtime_event):
            return
        try:
            await post_runtime_event(
                RuntimeInput(
                    event_id=f"{owned.handle.child_id}:terminal",
                    kind="agent.child.completed",
                    correlation_id=owned.handle.child_id,
                    source="qitos.agent",
                    payload=self.result_payload(result),
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # The terminal result remains queryable when the parent has closed or
            # durable mailbox acceptance fails.
            return

    def _owned(self, handle: ChildHandle) -> _OwnedChild | None:
        if not isinstance(handle, ChildHandle):
            raise TypeError("handle must be a ChildHandle")
        return self._children.get(handle)

    @staticmethod
    def _cancel_engine(owned: _OwnedChild) -> None:
        if owned.engine is not None:
            owned.engine.cancel("immediate")

    def _cancelled_result(
        self,
        owned: _OwnedChild,
        *,
        error: str = "Child agent was cancelled.",
    ) -> ChildResult:
        error = error or "Child agent was cancelled."
        return ChildResult(
            handle=owned.handle,
            request=owned.request,
            status=ChildStatus.CANCELLED,
            conclusion=AgentConclusion(failure_paths=(error,)),
            child_run_id=self._child_run_id(owned),
            error=error,
        )

    def _failed_result(self, owned: _OwnedChild, exc: Exception) -> ChildResult:
        error = str(exc) or type(exc).__name__
        return ChildResult(
            handle=owned.handle,
            request=owned.request,
            status=ChildStatus.FAILED,
            conclusion=AgentConclusion(failure_paths=(error,)),
            child_run_id=self._child_run_id(owned),
            error=error,
        )

    @staticmethod
    def _child_run_id(owned: _OwnedChild) -> str:
        return str(getattr(owned.engine, "active_run_id", "") or "")

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

    @staticmethod
    def _child_status(stop_reason: str) -> ChildStatus:
        if stop_reason in {"completed", "final", "success"}:
            return ChildStatus.COMPLETED
        if stop_reason == "blocked":
            return ChildStatus.BLOCKED
        if stop_reason == "max_steps" or stop_reason == "context_overflow":
            return ChildStatus.BUDGET_EXHAUSTED
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
