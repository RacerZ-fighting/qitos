"""Action executor for QitOS."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    wait,
)
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from ..core.action import Action, ActionExecutionPolicy, ActionResult, ActionStatus
from ..core.env import Env
from ..core.interceptor import InterceptorChain, InterceptorContext
from ..core.tool_result import ToolResult
from ..core.tool_registry import ToolRegistry
from ..core.tool import (
    BaseTool,
    ToolPermissionContext,
    ToolPermissionDecision,
    ToolValidationResult,
)
from .states import RuntimePhase

if TYPE_CHECKING:
    from ._protocol import _EngineProtocol
    from .cancellation import CancelToken


# Terminal states that count as a failure for fail_fast purposes.
_FAILED_STATUSES = frozenset({
    ActionStatus.ERROR,
    ActionStatus.TIMED_OUT,
})


class _ConcurrencyTracker:
    """Thread-safe peak-concurrency counter for a single execute() batch."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            if self._active > self.peak:
                self.peak = self._active

    def exit(self) -> None:
        with self._lock:
            self._active -= 1


class ActionExecutor:
    """Executes normalized actions against a tool registry."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        policy: Optional[ActionExecutionPolicy] = None,
        trace_writer: Any = None,
        delegate_depth: int = 0,
        shared_memory: Any = None,
        engine: Optional[_EngineProtocol] = None,
        permission_pipeline: Any = None,
        read_before_write_enforcer: Any = None,
        permission_interaction_callback: Optional[Any] = None,
        interceptor_chain: Optional[InterceptorChain] = None,
        auto_approve: bool = False,
        cancel_token: Optional[CancelToken] = None,
    ):
        self.tool_registry = tool_registry
        self.policy = policy or ActionExecutionPolicy()
        self.trace_writer = trace_writer
        self.delegate_depth = delegate_depth
        self.shared_memory = shared_memory
        self._engine = engine
        self._pipeline = permission_pipeline
        self._rbw_enforcer = read_before_write_enforcer
        self._permission_interaction_callback = permission_interaction_callback
        self._interceptor_chain = interceptor_chain
        self.auto_approve = auto_approve
        self._cancel_token = cancel_token
        # Populated by execute(); consumed by the trace layer.
        self.last_execution_stats: Dict[str, Any] = {}

    # ── Cancellation ───────────────────────────────────────────────────────────

    def _resolve_cancel_token(self) -> Optional[CancelToken]:
        """Prefer an explicit token, else fall back to the owning Engine's."""
        if self._cancel_token is not None:
            return self._cancel_token
        if self._engine is not None:
            return getattr(self._engine, "_cancel_token", None)
        return None

    def _is_cancelled(self) -> bool:
        token = self._resolve_cancel_token()
        if token is None:
            return False
        return bool(getattr(token, "is_cancel_requested", False))

    def _remaining_runtime_seconds(self) -> Optional[float]:
        if self._engine is None:
            return None
        remaining = getattr(self._engine, "remaining_runtime_seconds", None)
        if callable(remaining):
            value = remaining()
            return None if value is None else max(0.0, float(value))
        deadline = getattr(self._engine, "runtime_deadline_monotonic", None)
        if deadline is None:
            return None
        return max(0.0, float(deadline) - time.monotonic())

    def execute(
        self, actions: Sequence[Action], env: Optional[Env] = None, state: Any = None
    ) -> List[ActionResult]:
        self.last_execution_stats = {
            "policy": {
                "mode": self.policy.mode,
                "fail_fast": bool(self.policy.fail_fast),
                "max_concurrency": int(self.policy.max_concurrency),
            },
            "concurrency_peak": 0,
            "segments": 0,
            "cancel_source": None,
        }
        if not actions:
            return []

        tracker = _ConcurrencyTracker()

        # A cancellation already requested before the batch starts prevents
        # every action from running.
        if self._is_cancelled():
            self.last_execution_stats["cancel_source"] = "cancel_token"
            return [
                self._terminal_result(
                    action, ActionStatus.CANCELLED, "cancel_token", segment_index=0
                )
                for action in actions
            ]

        # Single action: execute directly
        if len(actions) == 1:
            result = self._execute_one(
                actions[0], env=env, state=state, tracker=tracker, segment_index=0
            )
            self.last_execution_stats["concurrency_peak"] = tracker.peak
            self.last_execution_stats["segments"] = 1
            return [result]

        # Respect ActionExecutionPolicy.mode
        if self.policy.mode == "serial":
            return self._execute_serial(actions, env=env, state=state, tracker=tracker)

        return self._execute_segmented(actions, env=env, state=state, tracker=tracker)

    def _execute_serial(
        self,
        actions: Sequence[Action],
        env: Optional[Env],
        state: Any,
        tracker: _ConcurrencyTracker,
    ) -> List[ActionResult]:
        """Run every action in order, honouring fail_fast and cancellation."""
        results: List[ActionResult] = []
        aborted: Optional[str] = None
        for idx, action in enumerate(actions):
            if aborted is not None:
                results.append(
                    self._terminal_result(
                        action, ActionStatus.CANCELLED, aborted, segment_index=idx
                    )
                )
                continue
            if self._is_cancelled():
                aborted = "cancel_token"
                self.last_execution_stats["cancel_source"] = aborted
                results.append(
                    self._terminal_result(
                        action, ActionStatus.CANCELLED, aborted, segment_index=idx
                    )
                )
                continue
            result = self._execute_one(
                action, env=env, state=state, tracker=tracker, segment_index=idx
            )
            results.append(result)
            if self.policy.fail_fast and result.status in _FAILED_STATUSES:
                aborted = "fail_fast"
                self.last_execution_stats["cancel_source"] = aborted
        self.last_execution_stats["concurrency_peak"] = tracker.peak
        self.last_execution_stats["segments"] = len(actions)
        return results

    def _is_concurrency_safe(self, tool_name: str) -> bool:
        """Return whether the registered tool explicitly permits concurrency."""
        allowed = self.policy.parallel_tool_names
        if allowed is not None and tool_name not in allowed:
            return False
        tool = self._resolve_tool(tool_name)
        if tool is None or tool.spec.needs_approval:
            return False
        return tool.spec.concurrency_safe is True

    def _segment_actions(self, actions: Sequence[Action]) -> List[List[int]]:
        """Split actions into ordered runs separated by exclusive barriers.

        Each returned segment is either a contiguous run of concurrency-safe
        action indices (which may execute in parallel) or a single exclusive
        action index (which acts as a barrier). Segment order always matches
        the model's original call order.
        """
        segments: List[List[int]] = []
        current: List[int] = []
        for idx, action in enumerate(actions):
            if self._is_concurrency_safe(action.name):
                current.append(idx)
                continue
            if current:
                segments.append(current)
                current = []
            segments.append([idx])
        if current:
            segments.append(current)
        return segments

    def _execute_segmented(
        self,
        actions: Sequence[Action],
        env: Optional[Env],
        state: Any,
        tracker: _ConcurrencyTracker,
    ) -> List[ActionResult]:
        """Execute actions segment by segment, preserving call-order semantics.

        Contiguous concurrency-safe actions run in parallel; every exclusive
        action is a barrier that must complete before later actions start.
        Results are returned in the model's original call order.
        """
        segments = self._segment_actions(actions)
        results: List[Optional[ActionResult]] = [None] * len(actions)
        aborted: Optional[str] = None

        for segment_index, segment in enumerate(segments):
            if aborted is None and self._is_cancelled():
                aborted = "cancel_token"
                self.last_execution_stats["cancel_source"] = aborted

            if aborted is not None:
                for idx in segment:
                    results[idx] = self._terminal_result(
                        actions[idx],
                        ActionStatus.CANCELLED,
                        aborted,
                        segment_index=segment_index,
                    )
                continue

            if len(segment) == 1:
                idx = segment[0]
                results[idx] = self._execute_one(
                    actions[idx],
                    env=env,
                    state=state,
                    tracker=tracker,
                    segment_index=segment_index,
                )
            else:
                aborted = self._execute_segment_concurrently(
                    actions,
                    segment,
                    results,
                    env=env,
                    state=state,
                    tracker=tracker,
                    segment_index=segment_index,
                )
                if aborted is not None:
                    self.last_execution_stats["cancel_source"] = aborted
                    continue

            if self.policy.fail_fast:
                for idx in segment:
                    item = results[idx]
                    if item is not None and item.status in _FAILED_STATUSES:
                        aborted = "fail_fast"
                        self.last_execution_stats["cancel_source"] = aborted
                        break

        self.last_execution_stats["concurrency_peak"] = tracker.peak
        self.last_execution_stats["segments"] = len(segments)

        return [
            r
            if r is not None
            else self._error_result(actions[i], "concurrent_execution_failed")
            for i, r in enumerate(results)
        ]

    def _execute_segment_concurrently(
        self,
        actions: Sequence[Action],
        segment: List[int],
        results: List[Optional[ActionResult]],
        *,
        env: Optional[Env],
        state: Any,
        tracker: _ConcurrencyTracker,
        segment_index: int,
    ) -> Optional[str]:
        """Run one contiguous safe segment in parallel.

        Returns the abort reason (``"fail_fast"`` / ``"cancel_token"``) if the
        segment was cut short, else ``None``. Actions that already started are
        always drained to their real terminal state — never relabelled as
        errors — while actions that never started are recorded as cancelled.
        """
        max_workers = min(max(1, self.policy.max_concurrency), len(segment))
        abort_reason: Optional[str] = None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures: Dict[Any, int] = {}
            for idx in segment:
                future = pool.submit(
                    self._execute_one,
                    actions[idx],
                    env=env,
                    state=state,
                    tracker=tracker,
                    segment_index=segment_index,
                )
                futures[future] = idx

            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:  # pragma: no cover - defensive path
                        results[idx] = self._error_result(actions[idx], str(exc))
                    item = results[idx]
                    if (
                        abort_reason is None
                        and self.policy.fail_fast
                        and item is not None
                        and item.status in _FAILED_STATUSES
                    ):
                        abort_reason = "fail_fast"
                if abort_reason is None and self._is_cancelled():
                    abort_reason = "cancel_token"
                if abort_reason is not None and pending:
                    # Cancel only what has not started. Anything already
                    # running is drained below so its true result is kept.
                    still_pending = set()
                    for future in pending:
                        if future.cancel():
                            idx = futures[future]
                            results[idx] = self._terminal_result(
                                actions[idx],
                                ActionStatus.CANCELLED,
                                abort_reason,
                                segment_index=segment_index,
                            )
                        else:
                            still_pending.add(future)
                    pending = still_pending

        return abort_reason

    def _terminal_result(
        self,
        action: Action,
        status: ActionStatus,
        cancel_source: str,
        *,
        segment_index: int = 0,
    ) -> ActionResult:
        """Build a result for an action that was prevented from starting."""
        return ActionResult(
            name=action.name,
            status=status,
            output=None,
            error=f"action {status.value}: {cancel_source}",
            action_id=action.action_id,
            attempts=0,
            latency_ms=0.0,
            metadata={
                **self._tool_meta(action.name),
                "error_category": status.value,
                "cancel_source": cancel_source,
                "segment_index": segment_index,
                "started": False,
            },
        )

    def _error_result(self, action: Action, message: str) -> ActionResult:
        """Create an error ActionResult for a failed concurrent execution slot."""
        card = "\n".join(
            [
                "[TOOL_RESULT_MISSING]",
                "",
                f"Tool: `{action.name}`",
                "Code: `TOOL_RESULT_MISSING`",
                "",
                "The executor did not produce a result. No success was inferred.",
                "Retry the call or choose another distinguishable action.",
            ]
        )
        return ActionResult(
            name=action.name,
            status=ActionStatus.ERROR,
            output=card,
            error=message,
            action_id=action.action_id,
            attempts=1,
            latency_ms=0.0,
            metadata={"error_category": "concurrent_execution_error"},
        )

    # ── Timeout resolution ─────────────────────────────────────────────────────

    def _resolve_timeout(
        self, action: Action, tool: BaseTool
    ) -> Tuple[Optional[float], str]:
        """Resolve the effective timeout for an action.

        Precedence: ``Action.timeout_s`` override > ``ToolSpec.timeout_s``
        default > no timeout. Returns ``(timeout_s, source)``.
        """
        timeout: Optional[float] = None
        source = "none"
        action_timeout = action.timeout_s
        if action_timeout is not None and action_timeout > 0:
            timeout = float(action_timeout)
            source = "action"
        elif tool.spec.timeout_s is not None and tool.spec.timeout_s > 0:
            timeout = float(tool.spec.timeout_s)
            source = "tool_spec"
        remaining = self._remaining_runtime_seconds()
        if remaining is not None and (timeout is None or remaining < timeout):
            return remaining, "runtime_deadline"
        return timeout, source

    def _resolve_awaitable(self, value: Any, timeout_s: Optional[float]) -> Any:
        """Drive a coroutine/awaitable returned by a tool to completion.

        Async handlers are awaited rather than being handed back as an
        un-awaited coroutine. Raises ``TimeoutError`` when ``timeout_s``
        elapses first.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        else:
            running_loop = True

        if running_loop is not None:
            # We are inside a live event loop on this thread, so we cannot
            # drive the coroutine here without deadlocking. Close it rather
            # than leaking an un-awaited coroutine, and report honestly.
            close = getattr(value, "close", None)
            if callable(close):
                close()
            raise RuntimeError(
                "async tool handler cannot be awaited from a synchronous "
                "executor running inside an active event loop"
            )

        async def _driver() -> Any:
            if timeout_s is not None:
                return await asyncio.wait_for(value, timeout=timeout_s)
            return await value

        try:
            return asyncio.run(_driver())
        except asyncio.TimeoutError as exc:
            raise TimeoutError("async action timed out") from exc

    def _call_tool_with_timeout(
        self,
        tool: BaseTool,
        name: str,
        args: Dict[str, Any],
        runtime_context: Optional[Dict[str, Any]],
        timeout_s: Optional[float],
    ) -> Any:
        """Call a tool, enforcing ``timeout_s`` and awaiting async handlers.

        Sync handlers run on a bounded worker thread so the timeout can be
        observed. Python cannot forcibly kill that thread, so a timeout is
        reported as such without claiming the worker was terminated.
        """
        def _invoke() -> Any:
            output = self._call_tool(tool, args, runtime_context=runtime_context)
            if inspect.isawaitable(output):
                output = self._resolve_awaitable(output, timeout_s)
            return output

        if timeout_s is None:
            return _invoke()

        # A ThreadPoolExecutor registers non-daemon workers for interpreter
        # shutdown, so shutdown(wait=False) still lets timed-out tools block
        # process exit. A dedicated daemon worker preserves honest timeout
        # reporting without turning an uncooperative tool into a CLI hang.
        future: Future[Any] = Future()

        def _run() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(_invoke())
            except BaseException as exc:
                future.set_exception(exc)

        threading.Thread(
            target=_run,
            name=f"qitos-tool-{name}",
            daemon=True,
        ).start()
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError as exc:
            raise TimeoutError(f"action exceeded timeout of {timeout_s}s") from exc

    def _execute_one(
        self,
        action: Action,
        env: Optional[Env] = None,
        state: Any = None,
        tracker: Optional[_ConcurrencyTracker] = None,
        segment_index: int = 0,
    ) -> ActionResult:
        if tracker is not None:
            tracker.enter()
        try:
            return self._execute_one_inner(
                action, env=env, state=state, segment_index=segment_index
            )
        finally:
            if tracker is not None:
                tracker.exit()

    def _execute_one_inner(
        self,
        action: Action,
        env: Optional[Env] = None,
        state: Any = None,
        segment_index: int = 0,
    ) -> ActionResult:
        start = time.monotonic()
        started_at = time.time()
        attempts = 0
        last_error = None
        tool_meta = self._tool_meta(action.name)
        remaining = self._remaining_runtime_seconds()
        if remaining is not None and remaining <= 0:
            return self._deadline_result(
                action=action,
                start=start,
                attempts=0,
                tool_meta=tool_meta,
                segment_index=segment_index,
                started=False,
            )
        protocol_error = str(action.metadata.get("protocol_error") or "").strip()
        if protocol_error:
            error_code = protocol_error.upper()
            card = "\n".join(
                [
                    "[TOOL:invalid_call]",
                    "",
                    f"Tool: `{action.name}`",
                    f"Code: `{error_code}`",
                    "",
                    "No tool was executed.",
                    "Retry the call with an exact tool name and a valid JSON object for arguments.",
                ]
            )
            return self._finish_result(
                action=action,
                status=ActionStatus.ERROR,
                start=start,
                attempts=0,
                tool_meta=tool_meta,
                output=card,
                error=protocol_error,
                extra_metadata={
                    "error_category": protocol_error,
                    "raw_arguments": action.metadata.get("raw_arguments"),
                    "recoverable": True,
                    "executed": False,
                    "segment_index": segment_index,
                    "started": False,
                },
            )
        available = self.tool_registry.list_tools()
        tool = self._resolve_tool(action.name)
        if tool is None:
            card = "\n".join(
                [
                    "[TOOL:unknown]",
                    "",
                    f"Unknown tool: `{action.name}`",
                    "",
                    "No tool was executed.",
                    "",
                    "Available tools:",
                    ", ".join(f"`{item}`" for item in available),
                    "",
                    "Retry using an exact tool name and its declared schema.",
                ]
            )
            return self._finish_result(
                action=action,
                status=ActionStatus.ERROR,
                start=start,
                attempts=1,
                tool_meta=tool_meta,
                output=card,
                error=f"Unknown tool: {action.name}",
                extra_metadata={
                    "error_category": "tool_not_found",
                    "raw_tool_name": action.name,
                    "raw_arguments": dict(action.args),
                    "available_tools": available,
                    "recoverable": True,
                    "executed": False,
                },
            )
        runtime_context = self._build_runtime_context(tool, env=env, state=state)
        ordering_meta: Dict[str, Any] = {
            "segment_index": segment_index,
            "started_at": started_at,
            "started": True,
        }

        _retry_policy = tool.spec.retry_policy
        _on_failure = tool.spec.on_failure

        # Unified timeout: Action.timeout_s override > ToolSpec.timeout_s default
        _timeout_s, _timeout_source = self._resolve_timeout(action, tool)
        if _timeout_s is not None:
            ordering_meta["timeout_s"] = _timeout_s
            ordering_meta["timeout_source"] = _timeout_source

        # 1. Interceptor before_execute — can modify action args
        interceptor_context = InterceptorContext(
            tool_name=action.name,
            tool_args=dict(action.args),
            step_id=getattr(state, "current_step", 0) if state else 0,
            state=self._engine,
            run_id=getattr(self._engine, "_active_run_id", "") if self._engine else "",
        )
        if self._interceptor_chain is not None:
            action = self._interceptor_chain.before_execute(action, interceptor_context)

        # 2. Check needs_approval — triggers interrupt() for human approval
        _auto_approved = False
        if tool.spec.needs_approval:
            if self.auto_approve:
                _auto_approved = True
            else:
                from ..engine.interrupt import interrupt
                from ..engine.approval import ToolApprovalItem

                approval_item = ToolApprovalItem(
                    tool_name=action.name,
                    tool_args=action.args,
                    message=f"Tool '{action.name}' requires approval before execution.",
                )
                approval = interrupt(approval_item)
                if approval == "deny":
                    return self._finish_result(
                        action=action,
                        status=ActionStatus.DENIED,
                        start=start,
                        attempts=1,
                        tool_meta=tool_meta,
                        output={"message": "User denied approval"},
                        error="User denied approval",
                        extra_metadata={"error_category": "approval_denied"},
                    )

        # Compute effective max attempts from retry_policy or fallback to max_retries
        if _retry_policy is not None:
            _max_attempts = _retry_policy.max_attempts
            _backoff_factor = _retry_policy.backoff_factor
            _max_backoff = _retry_policy.max_backoff
            _jitter = _retry_policy.jitter
            _retryable_exceptions = _retry_policy.retryable_exceptions
        else:
            _max_attempts = action.max_retries + 1  # existing behavior
            _backoff_factor = 0
            _max_backoff = 0
            _jitter = False
            _retryable_exceptions = (Exception,)

        while attempts < _max_attempts:
            remaining = self._remaining_runtime_seconds()
            if remaining is not None and remaining <= 0:
                return self._deadline_result(
                    action=action,
                    start=start,
                    attempts=attempts,
                    tool_meta=tool_meta,
                    segment_index=segment_index,
                    started=attempts > 0,
                )
            attempts += 1
            try:
                validation = self._validate(tool, action.args, runtime_context)
                if not validation.valid:
                    return self._finish_result(
                        action=action,
                        status=ActionStatus.ERROR,
                        start=start,
                        attempts=attempts,
                        tool_meta=tool_meta,
                        error=validation.message or "tool input validation failed",
                        extra_metadata={
                            "error_category": validation.code or "validation_error",
                            "validation": {
                                "valid": validation.valid,
                                "message": validation.message,
                                "code": validation.code,
                                "suggested_args": validation.suggested_args,
                            },
                        },
                    )

                # Read-before-write check for file editing tools
                rbw_blocked = self._check_read_before_write(action)
                if rbw_blocked is not None:
                    return rbw_blocked

                permission = self._check_permissions(tool, action.args, runtime_context)
                if permission.decision == "deny":
                    self._dispatch_tool_hook(
                        "on_permission_denied", action.name, action.args,
                        tool_result=None, permission_decision="deny",
                    )
                    return self._finish_result(
                        action=action,
                        status=ActionStatus.DENIED,
                        start=start,
                        attempts=attempts,
                        tool_meta=tool_meta,
                        output={
                            "message": permission.message,
                            "scope": permission.scope,
                        },
                        error=permission.message or "Tool permission denied",
                        extra_metadata={
                            "error_category": "permission_denied",
                            "permission": self._permission_payload(permission),
                        },
                    )
                if permission.decision == "ask":
                    # Try interactive resolution if callback is set
                    if self._permission_interaction_callback is not None:
                        try:
                            user_decision = self._permission_interaction_callback(
                                tool_name=action.name,
                                args=action.args,
                                permission=permission,
                            )
                            if user_decision == "allow":
                                permission = ToolPermissionDecision.allow()
                            elif user_decision == "deny":
                                self._dispatch_tool_hook(
                                    "on_permission_denied", action.name, action.args,
                                    tool_result=None, permission_decision="deny",
                                )
                                return self._finish_result(
                                    action=action,
                                    status=ActionStatus.DENIED,
                                    start=start,
                                    attempts=attempts,
                                    tool_meta=tool_meta,
                                    output={
                                        "message": "User denied permission",
                                        "scope": permission.scope,
                                    },
                                    error="User denied permission",
                                    extra_metadata={
                                        "error_category": "permission_denied",
                                        "permission": self._permission_payload(permission),
                                    },
                                )
                            # else: fall through to SKIPPED
                        except Exception:
                            pass  # Callback failed, fall through to SKIPPED

                    self._dispatch_tool_hook(
                        "on_permission_denied", action.name, action.args,
                        tool_result=None, permission_decision="ask",
                    )
                    return self._finish_result(
                        action=action,
                        status=ActionStatus.NEEDS_APPROVAL,
                        start=start,
                        attempts=attempts,
                        tool_meta=tool_meta,
                        output={
                            "message": permission.message,
                            "scope": permission.scope,
                        },
                        extra_metadata={
                            "error_category": "permission_ask",
                            "permission": self._permission_payload(permission),
                        },
                    )

                effective_args = dict(permission.updated_args or action.args)
                self._dispatch_tool_hook(
                    "on_before_tool_use", action.name, effective_args,
                    tool_result=None, permission_decision=permission.decision,
                )
                output = self._call_tool_with_timeout(
                    tool,
                    action.name,
                    effective_args,
                    runtime_context=runtime_context,
                    timeout_s=_timeout_s,
                )
                if output is None:
                    card = "\n".join(
                        [
                            "[TOOL_RESULT_MISSING]",
                            "",
                            f"Tool: `{action.name}`",
                            "Code: `TOOL_RESULT_MISSING`",
                            "",
                            "The tool returned no result. No success was inferred.",
                            "Retry the same call or choose another distinguishable action.",
                        ]
                    )
                    return self._finish_result(
                        action=action,
                        status=ActionStatus.ERROR,
                        start=start,
                        attempts=attempts,
                        tool_meta=tool_meta,
                        output=card,
                        error="Tool returned no result",
                        extra_metadata={
                            "error_category": "tool_result_missing",
                            "error_code": "TOOL_RESULT_MISSING",
                            "raw_tool_name": action.name,
                            "raw_arguments": dict(effective_args),
                            "recoverable": True,
                            "executed": True,
                        },
                    )
                normalized_output = self._normalize_output(tool, output)
                reported_result = ToolResult.from_value(normalized_output)
                self._dispatch_tool_hook(
                    "on_after_tool_use",
                    action.name,
                    effective_args,
                    tool_result=reported_result.to_dict(),
                    permission_decision=permission.decision,
                )
                if reported_result.is_success:
                    self._track_file_access(
                        action.name, effective_args, reported_result.output
                    )
                latency = (time.monotonic() - start) * 1000
                result_metadata = {
                    **tool_meta,
                    **ordering_meta,
                    "error_category": (
                        None
                        if reported_result.is_success
                        else f"tool_reported_{reported_result.status}"
                    ),
                    "permission": self._permission_payload(permission),
                    "progress_count": len(runtime_context["progress_events"]),
                    "artifacts": list(runtime_context["artifacts"]),
                    "ended_at": time.time(),
                }
                if _auto_approved:
                    result_metadata["auto_approved"] = True
                    result_metadata["approval_required"] = True
                result = ActionResult(
                    name=action.name,
                    status=ActionStatus(reported_result.status),
                    output=reported_result.output,
                    error=reported_result.error,
                    action_id=action.action_id,
                    attempts=attempts,
                    latency_ms=latency,
                    metadata=result_metadata,
                )
                # 6. Interceptor after_execute — can modify result
                if self._interceptor_chain is not None:
                    result = self._interceptor_chain.after_execute(action, result, interceptor_context)
                return result
            except TimeoutError as exc:
                # A timeout is a distinct terminal state, never retried: the
                # worker thread may still be running and we must not claim
                # otherwise.
                timed_out_result = self._finish_result(
                    action=action,
                    status=ActionStatus.TIMED_OUT,
                    start=start,
                    attempts=attempts,
                    tool_meta=tool_meta,
                    error=str(exc),
                    extra_metadata={
                        **ordering_meta,
                        "error_category": "timeout",
                        "worker_still_running": True,
                        "ended_at": time.time(),
                    },
                )
                if self._interceptor_chain is not None:
                    timed_out_result = self._interceptor_chain.after_execute(
                        action, timed_out_result, interceptor_context
                    )
                return timed_out_result
            except Exception as exc:  # pragma: no cover - defensive path
                last_error = str(exc)
                # Check if this exception type is retryable
                if not isinstance(exc, _retryable_exceptions):
                    break
                # Exponential backoff with optional jitter
                if attempts < _max_attempts and _backoff_factor > 0:
                    import random
                    delay = min(_backoff_factor * (2 ** (attempts - 1)), _max_backoff)
                    if _jitter:
                        delay = delay * (0.5 + random.random())
                    remaining = self._remaining_runtime_seconds()
                    if remaining is not None and delay >= remaining:
                        return self._deadline_result(
                            action=action,
                            start=start,
                            attempts=attempts,
                            tool_meta=tool_meta,
                            segment_index=segment_index,
                            started=True,
                        )
                    token = self._resolve_cancel_token()
                    wait_for_cancel = getattr(token, "wait", None)
                    if callable(wait_for_cancel) and wait_for_cancel(timeout=delay):
                        return self._finish_result(
                            action=action,
                            status=ActionStatus.CANCELLED,
                            start=start,
                            attempts=attempts,
                            tool_meta=tool_meta,
                            error="action cancelled during retry backoff",
                            extra_metadata={
                                **ordering_meta,
                                "error_category": "cancelled",
                                "cancel_source": "cancel_token",
                                "ended_at": time.time(),
                            },
                        )
                    if not callable(wait_for_cancel):
                        time.sleep(delay)

        error_category = "runtime_error"
        if last_error and "not found" in last_error.lower():
            error_category = "tool_not_found"

        # Call on_failure callback if registered
        if _on_failure is not None:
            try:
                _on_failure(action=action, error=last_error, attempts=attempts)
            except Exception:
                pass  # on_failure must not raise

        error_result = self._finish_result(
            action=action,
            status=ActionStatus.ERROR,
            start=start,
            attempts=attempts,
            tool_meta=tool_meta,
            error=last_error or "unknown action execution error",
            extra_metadata={
                **ordering_meta,
                "error_category": error_category,
                "progress_count": len(runtime_context["progress_events"]),
                "artifacts": list(runtime_context["artifacts"]),
                "ended_at": time.time(),
            },
        )
        # Interceptor after_execute on error path too
        if self._interceptor_chain is not None:
            error_result = self._interceptor_chain.after_execute(action, error_result, interceptor_context)
        return error_result

    def _deadline_result(
        self,
        *,
        action: Action,
        start: float,
        attempts: int,
        tool_meta: Dict[str, Any],
        segment_index: int,
        started: bool,
    ) -> ActionResult:
        return self._finish_result(
            action=action,
            status=ActionStatus.TIMED_OUT,
            start=start,
            attempts=attempts,
            tool_meta=tool_meta,
            error="runtime deadline expired before tool completion",
            extra_metadata={
                "error_category": "timeout",
                "timeout_source": "runtime_deadline",
                "timeout_s": 0.0,
                "segment_index": segment_index,
                "started": started,
                "worker_still_running": False,
                "ended_at": time.time(),
            },
        )

    def _finish_result(
        self,
        *,
        action: Action,
        status: ActionStatus,
        start: float,
        attempts: int,
        tool_meta: Dict[str, Any],
        output: Any = None,
        error: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        if output is None:
            code = str((extra_metadata or {}).get("error_code") or "TOOL_EXECUTION_ERROR")
            output = "\n".join(
                [
                    "[TOOL:error]",
                    "",
                    f"Tool: `{action.name}`",
                    f"Code: `{code}`",
                    "",
                    str(error or "The tool did not produce a result."),
                    "No success was inferred.",
                ]
            )
        latency = (time.monotonic() - start) * 1000
        metadata = dict(tool_meta)
        metadata.update(extra_metadata or {})
        return ActionResult(
            name=action.name,
            status=status,
            output=output,
            error=error,
            action_id=action.action_id,
            attempts=attempts,
            latency_ms=latency,
            metadata=metadata,
        )

    def _build_runtime_context(
        self, tool: BaseTool, env: Optional[Env], state: Any
    ) -> Dict[str, Any]:
        required_ops = list(tool.spec.required_ops)
        environment_ops = list(tool.spec.environment_ops)
        permission_context = self._resolve_permission_context(env=env, state=state)
        progress_events: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, Any]] = []

        def _emit_progress(payload: Dict[str, Any]) -> None:
            progress_events.append(dict(payload))

        def _record_artifact(payload: Dict[str, Any]) -> None:
            artifacts.append(dict(payload))

        active_run_id = (
            str(getattr(self._engine, "active_run_id", "") or "")
            if self._engine is not None
            else ""
        )

        def _post_runtime_event(event: Any) -> bool:
            if self._engine is None or not active_run_id:
                return False
            return bool(self._engine.post_runtime_event(event, run_id=active_run_id))

        runtime_deadline = (
            getattr(self._engine, "runtime_deadline_monotonic", None)
            if self._engine is not None
            else None
        )

        return {
            "env": env,
            "environment_attestation": dict(
                getattr(env, "attestation", {}) or {}
            ) if env is not None else {},
            "state": state,
            "ops": self._resolve_ops(required_ops, env)
            | self._resolve_environment_ops(environment_ops, env),
            "tool_registry": self.tool_registry,
            "permission_context": permission_context,
            "progress_events": progress_events,
            "artifacts": artifacts,
            "emit_progress": _emit_progress,
            "record_artifact": _record_artifact,
            "delegate_depth": self.delegate_depth,
            "parent_run_id": "",
            "post_runtime_event": _post_runtime_event,
            "deadline_monotonic": runtime_deadline,
            "remaining_seconds": self._remaining_runtime_seconds,
            "agent_cancelled": self._is_cancelled,
            "trace_writer": self.trace_writer,
            "shared_memory": self.shared_memory,
            "agent": getattr(self._engine, "agent", None) if self._engine is not None else None,
        }

    def _resolve_tool(self, name: str) -> Optional[BaseTool]:
        return self.tool_registry.get(name)

    def _validate(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> ToolValidationResult:
        return tool.validate_input(dict(args), runtime_context=runtime_context)

    def _check_permissions(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> ToolPermissionDecision:
        # Use permission pipeline if available
        if self._pipeline is not None:
            return self._pipeline.evaluate(
                tool_name=tool.name,
                args=dict(args),
                tool_spec=tool.spec,
                runtime_context=runtime_context,
            )
        return tool.check_permissions(dict(args), runtime_context=runtime_context)

    def _call_tool(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return tool.execute(args, runtime_context=runtime_context)

    def _normalize_output(self, tool: BaseTool, output: Any) -> Any:
        if isinstance(output, ToolResult):
            output = output.to_dict()
        max_chars = tool.spec.result_max_chars
        if not max_chars or max_chars <= 0:
            return output
        if isinstance(output, str):
            return self._truncate_text(output, max_chars)
        if isinstance(output, dict):
            normalized = dict(output)
            for key in ("content", "stdout", "stderr", "result", "summary", "message"):
                value = normalized.get(key)
                if isinstance(value, str):
                    normalized[key] = self._truncate_text(value, max_chars)
            return normalized
        return output

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... [truncated]"

    def _resolve_permission_context(
        self, env: Optional[Env], state: Any
    ) -> ToolPermissionContext:
        candidate = None
        if state is not None:
            metadata = getattr(state, "metadata", None)
            if isinstance(metadata, dict):
                candidate = metadata.get("tool_permission_context")
        if candidate is None and env is not None:
            candidate = getattr(env, "tool_permission_context", None)
        if isinstance(candidate, ToolPermissionContext):
            return candidate
        if isinstance(candidate, dict):
            return ToolPermissionContext.from_dict(candidate)
        return ToolPermissionContext()

    def _permission_payload(self, decision: ToolPermissionDecision) -> Dict[str, Any]:
        return {
            "decision": decision.decision,
            "message": decision.message,
            "scope": decision.scope,
            "matched_rule": (
                {
                    "effect": decision.matched_rule.effect,
                    "tool_name": decision.matched_rule.tool_name,
                    "tool_family": decision.matched_rule.tool_family,
                    "scope": decision.matched_rule.scope,
                    "message": decision.matched_rule.message,
                }
                if decision.matched_rule is not None
                else None
            ),
        }

    def _resolve_ops(
        self, required_ops: List[str], env: Optional[Env]
    ) -> Dict[str, Any]:
        if not required_ops:
            return {}
        if env is None:
            raise ValueError(
                f"Tool requires ops {required_ops} but no env was provided"
            )
        out: Dict[str, Any] = {}
        for group in required_ops:
            ops = env.get_ops(group)
            if ops is None:
                raise ValueError(
                    f"Env '{getattr(env, 'name', 'env')}' missing required ops group: {group}"
                )
            out[group] = ops
        return out

    def _resolve_environment_ops(
        self, environment_ops: List[str], env: Optional[Env]
    ) -> Dict[str, Any]:
        if env is None or not environment_ops:
            return {}
        return self._resolve_ops(environment_ops, env)

    def _tool_meta(self, name: str) -> dict[str, Any]:
        if self.tool_registry.get(name) is not None:
            desc = self.tool_registry.describe_tool(name)
            origin = desc.get("origin", {})
            return {
                "tool_name": desc.get("name", name),
                "toolset_name": origin.get("toolset_name"),
                "toolset_version": origin.get("toolset_version"),
                "source": origin.get("source", "function"),
            }
        return {
            "tool_name": name,
            "toolset_name": None,
            "toolset_version": None,
            "source": "unknown",
        }

    def _dispatch_tool_hook(
        self,
        hook_method: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Any = None,
        permission_decision: Optional[str] = None,
    ) -> None:
        """Dispatch a tool-level hook to all registered engine hooks."""
        if self._engine is None:
            return
        hooks = getattr(self._engine, "hooks", None)
        if not hooks:
            return
        from .hooks import ToolHookContext
        ctx = ToolHookContext(
            task="",
            step_id=0,
            phase=RuntimePhase.ACT,
            state=None,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            permission_decision=permission_decision,
        )
        for hook in hooks:
            method = getattr(hook, hook_method, None)
            if method is not None:
                try:
                    method(ctx, self._engine)
                except Exception:
                    pass

    # ── Read-before-write support ──────────────────────────────────────────────

    _WRITE_TOOL_NAMES = frozenset({
        "file_edit_v2", "write_file", "Edit", "Write",
        "str_replace", "insert", "replace_lines", "append_file",
    })

    _READ_TOOL_NAMES = frozenset({
        "file_read_v2", "read_file", "Read", "view",
    })

    def _check_read_before_write(self, action: Action) -> Optional[ActionResult]:
        """Check read-before-write enforcement for file editing tools.

        Returns an ActionResult if the action should be blocked, None otherwise.
        """
        if self._rbw_enforcer is None:
            return None
        if action.name not in self._WRITE_TOOL_NAMES:
            return None

        path = action.args.get("path") or action.args.get("file_path", "")
        if not path:
            return None

        allowed, reason = self._rbw_enforcer.check_write(path)
        if allowed:
            return None

        start = time.monotonic()
        return self._finish_result(
            action=action,
            status=ActionStatus.DENIED,
            start=start,
            attempts=1,
            tool_meta=self._tool_meta(action.name),
            output={
                "message": reason,
                "error_category": "read_before_write",
            },
            error=reason,
            extra_metadata={
                "error_category": "read_before_write",
            },
        )

    def _track_file_access(
        self, tool_name: str, args: Dict[str, Any], output: Any
    ) -> None:
        """Track file reads and invalidate cache on writes for RBW enforcement."""
        if self._rbw_enforcer is None:
            return

        # Record successful file reads
        if tool_name in self._READ_TOOL_NAMES:
            path = args.get("path") or args.get("file_path", "")
            if path and isinstance(output, dict):
                content = output.get("content", "")
                if content:
                    self._rbw_enforcer.record_read(path, content)
                elif isinstance(output, str) and output:
                    self._rbw_enforcer.record_read(path, output)

        # Invalidate cache after successful writes
        if tool_name in self._WRITE_TOOL_NAMES:
            path = args.get("path") or args.get("file_path", "")
            if path:
                self._rbw_enforcer.invalidate(path)
