"""Batched Tool execution for the minimal agent loop.

Owns the proven execution invariants previously held by the Engine action
executor, now expressed over typed ``ToolCall`` / ``ToolResult``:

- every admitted call receives exactly one terminal ``ToolResult`` — unknown
  Tool, admission rejection, denial, timeout, cancellation and failure alike;
- Tools run serially by default; only calls whose ``ToolSpec.concurrency_safe``
  is explicitly true share a bounded parallel segment, exclusive calls act as
  barriers, and results always commit in input order;
- each call gets one absolute monotonic deadline — the earlier of the run
  deadline and the Tool's ``timeout_s`` — propagated down to invocation and
  retry backoff;
- cancellation stops new admissions, lets started calls settle to a real
  terminal state, and records not-started calls as cancelled;
- a per-Tool ``RetryPolicy`` is the single retry owner;
- ``before_tool_call`` may block admission and ``after_tool_call`` may replace
  the executed result; both are configuration hooks that must not throw.

The executor never raises for Tool-level failures. Persistence faults from the
transaction boundary (including ``JournalAppendCancelled``) and caller
cancellation propagate.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Mapping, Optional, Protocol, Sequence, Union

from .agent_events import (
    EventSink,
    ToolExecutionEnd,
    ToolExecutionStart,
    emit_to,
)
from .cancellation import CancelToken
from .env import Env
from .journal import JournalAppendCancelled
from .message import ToolCall
from .tool import BaseTool, ToolValidationResult
from .tool_registry import ToolExposure
from .tool_result import ToolResult
from .tool_schema import tool_input_schema_errors

_FAILED_STATUSES = frozenset({"error", "timed_out"})


class ToolTransactionBoundary(Protocol):
    """Durable barriers around Tool admission and finalization."""

    async def tool_started(self, turn: int, call: ToolCall) -> None:
        """Record that the loop took responsibility for one Tool call."""
        ...

    async def tool_terminal(self, turn: int, call: ToolCall, result: ToolResult) -> None:
        """Record the unique terminal result of one Tool call."""
        ...


@dataclass(frozen=True, slots=True)
class BeforeToolCallContext:
    tool_call: ToolCall
    args: Mapping[str, Any]
    turn: int
    run_id: str


@dataclass(frozen=True, slots=True)
class BeforeToolCallDecision:
    """Admission override; ``block`` rejects the call before execution."""

    block: bool = False
    reason: str = ""
    terminate: bool = False


@dataclass(frozen=True, slots=True)
class AfterToolCallContext:
    tool_call: ToolCall
    args: Mapping[str, Any]
    result: ToolResult
    turn: int
    run_id: str


BeforeToolCallHook = Callable[
    [BeforeToolCallContext],
    Union[BeforeToolCallDecision, None, Awaitable[Optional[BeforeToolCallDecision]]],
]
AfterToolCallHook = Callable[
    [AfterToolCallContext],
    Union[ToolResult, None, Awaitable[Optional[ToolResult]]],
]


@dataclass(frozen=True, slots=True)
class ToolExecutionConfig:
    """Immutable execution policy for one Tool batch."""

    mode: Literal["sequential", "parallel"] = "sequential"
    max_concurrency: int = 8
    fail_fast: bool = False
    deadline_monotonic: float | None = None
    cancel_token: CancelToken | None = None
    run_id: str = ""
    turn: int = 0
    env: Env | None = None
    before_tool_call: BeforeToolCallHook | None = None
    after_tool_call: AfterToolCallHook | None = None
    extra_runtime_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("sequential", "parallel"):
            raise ValueError("mode must be 'sequential' or 'parallel'")
        if isinstance(self.max_concurrency, bool) or self.max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class ToolBatchExecutor:
    """Execute one assistant message's Tool calls against a frozen exposure."""

    def __init__(
        self,
        exposure: ToolExposure,
        config: ToolExecutionConfig,
        *,
        emit: EventSink | None = None,
        transaction: ToolTransactionBoundary | None = None,
    ) -> None:
        if not isinstance(exposure, ToolExposure):
            raise TypeError("ToolBatchExecutor requires a frozen ToolExposure")
        self._exposure = exposure
        self._config = config
        self._emit = emit
        self._transaction = transaction

    async def execute_batch(self, calls: Sequence[ToolCall]) -> List[ToolResult]:
        """Execute one batch and return terminal results in input order."""

        if not calls:
            return []
        if self._is_cancelled():
            return [
                await self._prevented(call, "cancel_token") for call in calls
            ]
        if len(calls) == 1:
            return [await self._execute_call(calls[0])]
        if self._config.mode == "sequential":
            return await self._execute_serial(calls)
        return await self._execute_segmented(calls)

    # ── batch strategies ────────────────────────────────────────────────

    async def _execute_serial(self, calls: Sequence[ToolCall]) -> List[ToolResult]:
        results: List[ToolResult] = []
        aborted: Optional[str] = None
        for call in calls:
            if aborted is not None:
                results.append(await self._prevented(call, aborted))
                continue
            if self._is_cancelled():
                aborted = "cancel_token"
                results.append(await self._prevented(call, aborted))
                continue
            result = await self._execute_call(call)
            results.append(result)
            cancel_source = result.metadata.get("cancel_source")
            if isinstance(cancel_source, str) and cancel_source:
                aborted = cancel_source
            elif self._config.fail_fast and result.status in _FAILED_STATUSES:
                aborted = "fail_fast"
        return results

    def _segment_calls(self, calls: Sequence[ToolCall]) -> List[List[int]]:
        """Split into contiguous safe runs separated by exclusive barriers."""

        segments: List[List[int]] = []
        current: List[int] = []
        for index, call in enumerate(calls):
            if self._is_concurrency_safe(call.name):
                current.append(index)
                continue
            if current:
                segments.append(current)
                current = []
            segments.append([index])
        if current:
            segments.append(current)
        return segments

    def _is_concurrency_safe(self, tool_name: str) -> bool:
        tool = self._exposure.get(tool_name)
        if tool is None or tool.spec.needs_approval:
            return False
        return tool.spec.concurrency_safe is True

    async def _execute_segmented(
        self, calls: Sequence[ToolCall]
    ) -> List[ToolResult]:
        segments = self._segment_calls(calls)
        results: List[Optional[ToolResult]] = [None] * len(calls)
        aborted: Optional[str] = None
        for segment in segments:
            if aborted is None and self._is_cancelled():
                aborted = "cancel_token"
            if aborted is not None:
                for index in segment:
                    results[index] = await self._prevented(calls[index], aborted)
                continue
            if len(segment) == 1:
                index = segment[0]
                result = await self._execute_call(calls[index])
                results[index] = result
                cancel_source = result.metadata.get("cancel_source")
                if isinstance(cancel_source, str) and cancel_source:
                    aborted = cancel_source
            else:
                aborted = await self._execute_segment_concurrently(
                    calls, segment, results
                )
            if aborted is None and self._config.fail_fast:
                for index in segment:
                    item = results[index]
                    if item is not None and item.status in _FAILED_STATUSES:
                        aborted = "fail_fast"
                        break
        return [
            item
            if item is not None
            else self._missing_result(call, "concurrent_execution_failed")
            for item, call in zip(results, calls)
        ]

    async def _execute_segment_concurrently(
        self,
        calls: Sequence[ToolCall],
        segment: List[int],
        results: List[Optional[ToolResult]],
    ) -> Optional[str]:
        """Run one safe segment in parallel, draining started calls to terminal."""

        max_concurrency = min(self._config.max_concurrency, len(segment))
        abort_reason: Optional[str] = None
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_index(index: int) -> ToolResult:
            try:
                async with semaphore:
                    if self._is_cancelled():
                        return await self._prevented(calls[index], "cancel_token")
                    return await self._execute_call(calls[index])
            except JournalAppendCancelled:
                raise
            except asyncio.CancelledError:
                return await self._prevented(
                    calls[index], abort_reason or "parent_cancelled"
                )

        tasks = {
            asyncio.create_task(
                _run_index(index), name=f"qitos-tool-{calls[index].name}"
            ): index
            for index in segment
        }
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    index = tasks[task]
                    try:
                        results[index] = task.result()
                    except JournalAppendCancelled:
                        raise
                    except asyncio.CancelledError:  # pragma: no cover - defensive
                        results[index] = await self._prevented(
                            calls[index], abort_reason or "parent_cancelled"
                        )
                    except Exception as exc:  # pragma: no cover - defensive path
                        results[index] = self._missing_result(calls[index], str(exc))
                    item = results[index]
                    if (
                        abort_reason is None
                        and self._config.fail_fast
                        and item is not None
                        and item.status in _FAILED_STATUSES
                    ):
                        abort_reason = "fail_fast"
                if abort_reason is None and self._is_cancelled():
                    abort_reason = "cancel_token"
                if abort_reason is not None and pending:
                    pending_tasks = list(pending)
                    for task in pending_tasks:
                        task.cancel()
                    drained = await asyncio.gather(*pending_tasks, return_exceptions=True)
                    for pending_task, outcome in zip(pending_tasks, drained):
                        index = tasks[pending_task]
                        if isinstance(outcome, ToolResult):
                            results[index] = outcome
                        elif isinstance(outcome, JournalAppendCancelled):
                            raise outcome
                        else:
                            results[index] = self._missing_result(
                                calls[index], str(outcome)
                            )
                    pending.clear()
        except asyncio.CancelledError:
            abort_reason = (
                "cancel_token" if self._is_cancelled() else "caller_cancelled"
            )
            pending_tasks = [task for task in tasks if not task.done()]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                drained = await asyncio.gather(*pending_tasks, return_exceptions=True)
                for pending_task, outcome in zip(pending_tasks, drained):
                    index = tasks[pending_task]
                    if isinstance(outcome, ToolResult):
                        results[index] = outcome
                    elif isinstance(outcome, JournalAppendCancelled):
                        raise outcome
                    else:
                        results[index] = await self._prevented(
                            calls[index], abort_reason
                        )
            for task, index in tasks.items():
                if results[index] is None and task.done() and not task.cancelled():
                    results[index] = task.result()
        finally:
            unfinished = [task for task in tasks if not task.done()]
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)
        return abort_reason

    # ── single call ─────────────────────────────────────────────────────

    async def _execute_call(self, call: ToolCall) -> ToolResult:
        await emit_to(
            self._emit,
            ToolExecutionStart(
                tool_call_id=call.id, tool_name=call.name, args=call.arguments
            ),
        )
        if self._transaction is not None:
            await self._transaction.tool_started(self._config.turn, call)
        try:
            result = await self._execute_call_inner(call)
        except (asyncio.CancelledError, JournalAppendCancelled):
            raise
        except Exception as exc:  # admission helpers must not break the invariant
            result = self._missing_result(call, str(exc))
        result = dataclasses.replace(result, call_id=call.id)
        if self._transaction is not None:
            await self._transaction.tool_terminal(self._config.turn, call, result)
        await emit_to(
            self._emit,
            ToolExecutionEnd(
                tool_call_id=call.id,
                tool_name=call.name,
                result=result,
                is_error=result.status != "success",
            ),
        )
        return result

    async def _execute_call_inner(self, call: ToolCall) -> ToolResult:
        start = time.monotonic()
        deadline, timeout_source = self._resolve_call_deadline(call, start)

        stopped = self._stop_result(call, start, attempts=0, deadline=deadline)
        if stopped is not None:
            return stopped

        if call.parse_error is not None:
            return self._finish_result(
                call,
                status="error",
                start=start,
                attempts=0,
                error=call.parse_error,
                extra_metadata={
                    "error_category": "invalid_tool_arguments",
                    "recoverable": True,
                    "started": False,
                },
            )

        tool = self._exposure.get(call.name)
        if tool is None:
            return self._finish_result(
                call,
                status="error",
                start=start,
                attempts=0,
                error=f"Unknown tool: {call.name}",
                extra_metadata={
                    "error_category": "tool_not_found",
                    "available_tools": self._exposure.list_tools(),
                    "recoverable": True,
                    "started": False,
                },
            )

        runtime_context = self._build_runtime_context(
            call, tool, deadline_monotonic=deadline
        )
        ordering_meta: Dict[str, Any] = {"started": False}
        if timeout_source != "none":
            ordering_meta["timeout_source"] = timeout_source

        stopped = self._stop_result(call, start, attempts=0, deadline=deadline)
        if stopped is not None:
            return stopped

        permission = tool.check_permissions(dict(call.arguments), runtime_context)
        if permission.decision == "deny":
            return self._finish_result(
                call,
                status="denied",
                start=start,
                attempts=0,
                error=permission.message or "Tool permission denied",
                output={"message": permission.message, "scope": permission.scope},
                extra_metadata={
                    **ordering_meta,
                    "error_category": "permission_denied",
                    "permission_scope": permission.scope,
                },
            )
        if permission.decision == "ask":
            return self._finish_result(
                call,
                status="needs_approval",
                start=start,
                attempts=0,
                output={"message": permission.message, "scope": permission.scope},
                extra_metadata={
                    **ordering_meta,
                    "error_category": "permission_ask",
                    "permission_scope": permission.scope,
                },
            )

        effective_args = dict(
            call.arguments
            if permission.updated_args is None
            else permission.updated_args
        )
        validation = self._validate(tool, effective_args, runtime_context)
        if not validation.valid:
            return self._finish_result(
                call,
                status="error",
                start=start,
                attempts=0,
                error=validation.message or "tool input validation failed",
                extra_metadata={
                    **ordering_meta,
                    "error_category": validation.code or "validation_error",
                    "started": False,
                },
            )

        stopped = self._stop_result(call, start, attempts=0, deadline=deadline)
        if stopped is not None:
            return stopped

        if self._config.before_tool_call is not None:
            try:
                decision = await _maybe_await(
                    self._config.before_tool_call(
                        BeforeToolCallContext(
                            tool_call=call,
                            args=dict(effective_args),
                            turn=self._config.turn,
                            run_id=self._config.run_id,
                        )
                    )
                )
            except (asyncio.CancelledError, JournalAppendCancelled):
                raise
            except Exception as exc:
                return self._finish_result(
                    call,
                    status="error",
                    start=start,
                    attempts=0,
                    error=str(exc),
                    extra_metadata={
                        **ordering_meta,
                        "error_category": "before_tool_call",
                        "started": False,
                    },
                )
            if decision is not None and decision.block:
                extra: Dict[str, Any] = {
                    **ordering_meta,
                    "error_category": "before_tool_call_blocked",
                    "started": False,
                }
                if decision.terminate:
                    extra["terminate"] = True
                return self._finish_result(
                    call,
                    status="denied",
                    start=start,
                    attempts=0,
                    error=decision.reason or "Tool execution was blocked",
                    extra_metadata=extra,
                )

        result = await self._invoke_with_retries(
            tool,
            call,
            effective_args,
            runtime_context=runtime_context,
            start=start,
            deadline=deadline,
            timeout_source=timeout_source,
            ordering_meta=ordering_meta,
        )

        if self._config.after_tool_call is not None:
            try:
                override = await _maybe_await(
                    self._config.after_tool_call(
                        AfterToolCallContext(
                            tool_call=call,
                            args=dict(effective_args),
                            result=result,
                            turn=self._config.turn,
                            run_id=self._config.run_id,
                        )
                    )
                )
            except (asyncio.CancelledError, JournalAppendCancelled):
                raise
            except Exception as exc:
                return self._finish_result(
                    call,
                    status="error",
                    start=start,
                    attempts=0,
                    error=str(exc),
                    extra_metadata={
                        **ordering_meta,
                        "error_category": "after_tool_call",
                    },
                )
            if override is not None:
                if not isinstance(override, ToolResult):
                    raise TypeError(
                        "after_tool_call must return a ToolResult or None"
                    )
                result = override
        return result

    async def _invoke_with_retries(
        self,
        tool: BaseTool,
        call: ToolCall,
        effective_args: Dict[str, Any],
        *,
        runtime_context: Dict[str, Any],
        start: float,
        deadline: Optional[float],
        timeout_source: str,
        ordering_meta: Dict[str, Any],
    ) -> ToolResult:
        retry_policy = tool.spec.retry_policy
        max_attempts = retry_policy.max_attempts if retry_policy is not None else 1
        retryable_exceptions = (
            retry_policy.retryable_exceptions if retry_policy is not None else ()
        )
        attempts = 0
        last_error: Optional[str] = None
        while attempts < max_attempts:
            stopped = self._stop_result(
                call, start, attempts=attempts, deadline=deadline
            )
            if stopped is not None:
                return stopped
            attempts += 1
            ordering_meta["started"] = True
            try:
                output = await self._invoke_tool(
                    tool,
                    effective_args,
                    runtime_context=runtime_context,
                    timeout_s=self._remaining_seconds(deadline),
                )
                if self._is_cancelled():
                    return self._finish_result(
                        call,
                        status="cancelled",
                        start=start,
                        attempts=attempts,
                        error="tool call cancelled",
                        extra_metadata={
                            **ordering_meta,
                            "error_category": "cancelled",
                            "cancel_source": "cancel_token",
                        },
                    )
                if output is None:
                    return self._finish_result(
                        call,
                        status="error",
                        start=start,
                        attempts=attempts,
                        error="Tool returned no result",
                        extra_metadata={
                            "error_category": "tool_result_missing",
                            "error_code": "TOOL_RESULT_MISSING",
                            "recoverable": True,
                            "started": True,
                        },
                    )
                reported = ToolResult.from_value(output)
                latency_ms = (time.monotonic() - start) * 1000
                reported_category = reported.metadata.get("error_category")
                metadata = {
                    **reported.metadata,
                    **ordering_meta,
                    "error_category": (
                        None
                        if reported.is_success
                        else reported_category
                        if isinstance(reported_category, str)
                        else f"tool_reported_{reported.status}"
                    ),
                    "latency_ms": latency_ms,
                    "attempts": attempts,
                    "progress_count": len(runtime_context["progress_events"]),
                    "artifacts": list(runtime_context["artifacts"]),
                }
                return ToolResult(
                    status=reported.status,
                    output=reported.output,
                    error=reported.error,
                    metadata=metadata,
                    artifacts=reported.artifacts,
                    model_output=reported.model_output,
                )
            except JournalAppendCancelled:
                raise
            except asyncio.CancelledError:
                return self._finish_result(
                    call,
                    status="cancelled",
                    start=start,
                    attempts=attempts,
                    error="tool call cancelled",
                    extra_metadata={
                        **ordering_meta,
                        "error_category": "cancelled",
                        "cancel_source": (
                            "cancel_token"
                            if self._is_cancelled()
                            else "caller_cancelled"
                        ),
                    },
                )
            except asyncio.TimeoutError as exc:
                return self._finish_result(
                    call,
                    status="timed_out",
                    start=start,
                    attempts=attempts,
                    error=str(exc) or "tool call deadline expired",
                    extra_metadata={
                        **ordering_meta,
                        "error_category": "timeout",
                        "timeout_source": timeout_source,
                    },
                )
            except Exception as exc:
                last_error = str(exc)
                if not isinstance(exc, retryable_exceptions):
                    break
                if (
                    retry_policy is not None
                    and attempts < max_attempts
                    and retry_policy.backoff_factor > 0
                ):
                    delay = min(
                        retry_policy.backoff_factor * (2 ** (attempts - 1)),
                        retry_policy.max_backoff,
                    )
                    if retry_policy.jitter:
                        delay = delay * (0.5 + random.random())
                    remaining = self._remaining_seconds(deadline)
                    if remaining is not None and delay >= remaining:
                        return self._finish_result(
                            call,
                            status="timed_out",
                            start=start,
                            attempts=attempts,
                            error="tool call deadline expired before retry",
                            extra_metadata={
                                **ordering_meta,
                                "error_category": "timeout",
                                "timeout_source": timeout_source,
                            },
                        )
                    retry_cancel = await self._wait_for_retry(
                        delay, deadline_monotonic=deadline
                    )
                    if retry_cancel is not None:
                        return self._finish_result(
                            call,
                            status="cancelled",
                            start=start,
                            attempts=attempts,
                            error="tool call cancelled during retry backoff",
                            extra_metadata={
                                **ordering_meta,
                                "error_category": "cancelled",
                                "cancel_source": retry_cancel,
                            },
                        )

        if tool.spec.on_failure is not None:
            try:
                tool.spec.on_failure(
                    action={"name": call.name, "args": effective_args},
                    error=last_error,
                    attempts=attempts,
                )
            except Exception:
                pass  # on_failure must not raise
        return self._finish_result(
            call,
            status="error",
            start=start,
            attempts=attempts,
            error=last_error or "unknown tool execution error",
            extra_metadata={
                **ordering_meta,
                "error_category": "runtime_error",
                "progress_count": len(runtime_context["progress_events"]),
                "artifacts": list(runtime_context["artifacts"]),
            },
        )

    # ── admission helpers ───────────────────────────────────────────────

    def _validate(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> ToolValidationResult:
        schema_errors = tool_input_schema_errors(
            tool.spec.input_schema or {}, dict(args)
        )
        if schema_errors:
            return ToolValidationResult.fail(
                "Tool arguments do not match input_schema:\n"
                + "\n".join(f"- {error}" for error in schema_errors),
                code="invalid_tool_arguments",
            )
        return tool.validate_input(dict(args), runtime_context=runtime_context)

    def _build_runtime_context(
        self,
        call: ToolCall,
        tool: BaseTool,
        *,
        deadline_monotonic: Optional[float],
    ) -> Dict[str, Any]:
        env = self._config.env
        progress_events: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, Any]] = []

        def _emit_progress(payload: Dict[str, Any]) -> None:
            progress_events.append(dict(payload))

        def _record_artifact(payload: Dict[str, Any]) -> None:
            artifacts.append(dict(payload))

        context: Dict[str, Any] = {
            "env": env,
            "ops": self._resolve_ops(tool, env),
            "tool_registry": self._exposure,
            "progress_events": progress_events,
            "artifacts": artifacts,
            "emit_progress": _emit_progress,
            "record_artifact": _record_artifact,
            "run_id": self._config.run_id,
            "deadline_monotonic": deadline_monotonic,
            "remaining_seconds": lambda: self._remaining_seconds(deadline_monotonic),
            "agent_cancelled": self._is_cancelled,
        }
        context.update(dict(self._config.extra_runtime_context))
        return context

    def _resolve_ops(self, tool: BaseTool, env: Optional[Env]) -> Dict[str, Any]:
        required = list(tool.spec.required_ops) + list(tool.spec.environment_ops)
        if not required:
            return {}
        if env is None:
            raise ValueError(
                f"Tool '{tool.name}' requires ops {required} but no env was provided"
            )
        out: Dict[str, Any] = {}
        for group in required:
            ops = env.get_ops(group)
            if ops is None:
                raise ValueError(
                    f"Env '{getattr(env, 'name', 'env')}' missing required ops group: {group}"
                )
            out[group] = ops
        return out

    # ── deadlines, cancellation, results ────────────────────────────────

    def _is_cancelled(self) -> bool:
        token = self._config.cancel_token
        return token is not None and token.is_cancel_requested

    def _resolve_call_deadline(
        self, call: ToolCall, started: float
    ) -> tuple[Optional[float], str]:
        tool = self._exposure.get(call.name)
        tool_timeout = (
            float(tool.spec.timeout_s)
            if tool is not None and tool.spec.timeout_s is not None
            else None
        )
        tool_deadline = started + tool_timeout if tool_timeout is not None else None
        run_deadline = self._config.deadline_monotonic
        if run_deadline is not None and (
            tool_deadline is None or run_deadline <= tool_deadline
        ):
            return float(run_deadline), "run_deadline"
        if tool_deadline is not None:
            return tool_deadline, "tool_spec"
        return None, "none"

    @staticmethod
    def _remaining_seconds(deadline_monotonic: Optional[float]) -> Optional[float]:
        if deadline_monotonic is None:
            return None
        return max(0.0, deadline_monotonic - time.monotonic())

    async def _invoke_tool(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        runtime_context: Optional[Dict[str, Any]],
        timeout_s: Optional[float],
    ) -> Any:
        invocation = tool.execute(args, runtime_context=runtime_context)
        if timeout_s is None:
            return await invocation
        if timeout_s <= 0:
            raise asyncio.TimeoutError("tool call deadline expired")
        return await asyncio.wait_for(invocation, timeout=timeout_s)

    async def _wait_for_retry(
        self,
        delay: float,
        *,
        deadline_monotonic: Optional[float],
    ) -> str | None:
        wake_at = time.monotonic() + max(0.0, delay)
        while True:
            if self._is_cancelled():
                return "cancel_token"
            now = time.monotonic()
            if now >= wake_at:
                return None
            if deadline_monotonic is not None and now >= deadline_monotonic:
                return None
            remaining_delay = wake_at - now
            remaining_deadline = (
                None
                if deadline_monotonic is None
                else max(0.0, deadline_monotonic - now)
            )
            sleep_for = min(0.05, remaining_delay)
            if remaining_deadline is not None:
                sleep_for = min(sleep_for, remaining_deadline)
            if sleep_for <= 0:
                return None
            try:
                await asyncio.sleep(sleep_for)
            except asyncio.CancelledError:
                return "caller_cancelled"

    def _stop_result(
        self,
        call: ToolCall,
        start: float,
        *,
        attempts: int,
        deadline: Optional[float],
    ) -> Optional[ToolResult]:
        if self._is_cancelled():
            return self._finish_result(
                call,
                status="cancelled",
                start=start,
                attempts=attempts,
                error="tool call cancelled",
                extra_metadata={
                    "error_category": "cancelled",
                    "cancel_source": "cancel_token",
                    "started": attempts > 0,
                },
            )
        remaining = self._remaining_seconds(deadline)
        if remaining is None or remaining > 0:
            return None
        return self._finish_result(
            call,
            status="timed_out",
            start=start,
            attempts=attempts,
            error="tool call deadline expired before admission",
            extra_metadata={
                "error_category": "timeout",
                "started": attempts > 0,
            },
        )

    async def _prevented(self, call: ToolCall, cancel_source: str) -> ToolResult:
        """Commit a not-started call's cancelled terminal through all barriers."""

        await emit_to(
            self._emit,
            ToolExecutionStart(
                tool_call_id=call.id, tool_name=call.name, args=call.arguments
            ),
        )
        if self._transaction is not None:
            await self._transaction.tool_started(self._config.turn, call)
        result = dataclasses.replace(
            self._prevented_result(call, cancel_source), call_id=call.id
        )
        if self._transaction is not None:
            await self._transaction.tool_terminal(self._config.turn, call, result)
        await emit_to(
            self._emit,
            ToolExecutionEnd(
                tool_call_id=call.id,
                tool_name=call.name,
                result=result,
                is_error=True,
            ),
        )
        return result

    def _prevented_result(self, call: ToolCall, cancel_source: str) -> ToolResult:
        """Terminal result for a call that was prevented from starting."""

        return ToolResult(
            status="cancelled",
            output=None,
            error=f"tool call cancelled: {cancel_source}",
            metadata={
                "tool_name": call.name,
                "error_category": "cancelled",
                "cancel_source": cancel_source,
                "started": False,
            },
        )

    def _missing_result(self, call: ToolCall, message: str) -> ToolResult:
        return ToolResult(
            status="error",
            output=(
                "[TOOL_RESULT_MISSING]\n\n"
                f"Tool: `{call.name}`\n"
                "Code: `TOOL_RESULT_MISSING`\n\n"
                "The executor did not produce a result. No success was inferred."
            ),
            error=message,
            metadata={
                "tool_name": call.name,
                "error_category": "executor_error",
                "error_code": "TOOL_RESULT_MISSING",
            },
        )

    def _finish_result(
        self,
        call: ToolCall,
        *,
        status: Any,
        start: float,
        attempts: int,
        output: Any = None,
        error: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        if output is None and status not in ("success", "partial"):
            code = str(
                (extra_metadata or {}).get("error_code") or "TOOL_EXECUTION_ERROR"
            )
            output = "\n".join(
                [
                    "[TOOL:error]",
                    "",
                    f"Tool: `{call.name}`",
                    f"Code: `{code}`",
                    "",
                    str(error or "The tool did not produce a result."),
                    "No success was inferred.",
                ]
            )
        metadata = {"tool_name": call.name, "attempts": attempts}
        metadata.update(extra_metadata or {})
        metadata.setdefault("latency_ms", (time.monotonic() - start) * 1000)
        return ToolResult(
            status=status,
            output=output,
            error=error,
            metadata=metadata,
        )


__all__ = [
    "AfterToolCallContext",
    "AfterToolCallHook",
    "BeforeToolCallContext",
    "BeforeToolCallDecision",
    "BeforeToolCallHook",
    "ToolBatchExecutor",
    "ToolExecutionConfig",
    "ToolTransactionBoundary",
]
