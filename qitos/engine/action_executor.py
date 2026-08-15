"""Action executor for QitOS."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..core.action import Action, ActionExecutionPolicy, ActionResult, ActionStatus
from ..core.artifact import ArtifactRef
from ..core.budget import BudgetLedger
from ..core.env import Env, RuntimeCapabilitySnapshot
from ..core.journal import SessionJournal
from ..core.runtime_input import RuntimeInput
from ..core.tool_result import ToolResult
from ..core.turn import TurnBudgetSnapshot
from ..core.tool_registry import ToolRegistry
from ..core.tool import (
    BaseTool,
    ToolPermissionContext,
    ToolPermissionDecision,
    ToolValidationResult,
)
from ..core.tool_schema import tool_input_schema_errors
from .interrupt import EngineInterrupt
from .cancellation import CancelToken
from .states import RuntimeBudget, RuntimeEvent, RuntimePhase


class _ActionRuntimeOwner(Protocol):
    """Engine surface required while projecting one Tool runtime context."""

    agent: Any
    budget: RuntimeBudget
    events: List[RuntimeEvent]
    journal: SessionJournal | None

    @property
    def active_run_id(self) -> str:
        raise NotImplementedError

    @property
    def budget_ledger(self) -> BudgetLedger:
        raise NotImplementedError

    @property
    def runtime_deadline_monotonic(self) -> float | None:
        raise NotImplementedError

    def remaining_runtime_seconds(self) -> float | None:
        raise NotImplementedError

    async def apost_runtime_event(
        self,
        event: RuntimeInput,
        *,
        run_id: str,
    ) -> bool:
        raise NotImplementedError


# Terminal states that count as a failure for fail_fast purposes.
_FAILED_STATUSES = frozenset(
    {
        ActionStatus.ERROR,
        ActionStatus.TIMED_OUT,
    }
)


class _ConcurrencyTracker:
    """Peak-concurrency counter owned by one event loop."""

    def __init__(self) -> None:
        self._active = 0
        self.peak = 0

    def enter(self) -> None:
        self._active += 1
        if self._active > self.peak:
            self.peak = self._active

    def exit(self) -> None:
        self._active -= 1


class _ActionProgress:
    """Invocation count for one event-loop-owned action task."""

    def __init__(self) -> None:
        self._attempts = 0

    def begin_attempt(self) -> int:
        self._attempts += 1
        return self._attempts

    @property
    def attempts(self) -> int:
        return self._attempts


class ActionExecutor:
    """Executes normalized actions against a tool registry."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        policy: Optional[ActionExecutionPolicy] = None,
        trace_writer: Any = None,
        delegate_depth: int = 0,
        shared_memory: Any = None,
        engine: _ActionRuntimeOwner | None = None,
        permission_pipeline: Any = None,
        read_before_write_enforcer: Any = None,
        permission_interaction_callback: Optional[Any] = None,
        auto_approve: bool = False,
        cancel_token: Optional[CancelToken] = None,
        turn_budget: TurnBudgetSnapshot | None = None,
        runtime_capabilities: RuntimeCapabilitySnapshot | None = None,
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
        self.auto_approve = auto_approve
        self._cancel_token = cancel_token
        self._turn_budget = turn_budget
        self._runtime_capabilities = runtime_capabilities
        # Populated by execute(); consumed by the trace layer.
        self.last_execution_stats: Dict[str, Any] = {}

    # ── Cancellation ───────────────────────────────────────────────────────────

    def _resolve_cancel_token(self) -> Optional[CancelToken]:
        """Return the explicitly owned cancellation dependency, if configured."""
        return self._cancel_token

    def _is_cancelled(self) -> bool:
        token = self._resolve_cancel_token()
        if token is None:
            return False
        return token.is_cancel_requested

    def _remaining_runtime_seconds(self) -> Optional[float]:
        if self._turn_budget is not None:
            deadline = self._turn_budget.deadline_monotonic
            return (
                None
                if deadline is None
                else max(0.0, float(deadline) - time.monotonic())
            )
        if self._engine is None:
            return None
        remaining = self._engine.remaining_runtime_seconds()
        return None if remaining is None else max(0.0, float(remaining))

    async def execute(
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
            result = await self._execute_one(
                actions[0], env=env, state=state, tracker=tracker, segment_index=0
            )
            self.last_execution_stats["concurrency_peak"] = tracker.peak
            self.last_execution_stats["segments"] = 1
            cancel_source = result.metadata.get("cancel_source")
            if isinstance(cancel_source, str) and cancel_source:
                self.last_execution_stats["cancel_source"] = cancel_source
            return [result]

        # Respect ActionExecutionPolicy.mode
        if self.policy.mode == "serial":
            return await self._execute_serial(
                actions, env=env, state=state, tracker=tracker
            )

        return await self._execute_segmented(
            actions, env=env, state=state, tracker=tracker
        )

    async def _execute_serial(
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
            result = await self._execute_one(
                action, env=env, state=state, tracker=tracker, segment_index=idx
            )
            results.append(result)
            cancel_source = result.metadata.get("cancel_source")
            if isinstance(cancel_source, str) and cancel_source:
                aborted = cancel_source
                self.last_execution_stats["cancel_source"] = aborted
            elif self.policy.fail_fast and result.status in _FAILED_STATUSES:
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

    async def _execute_segmented(
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
                segment_result = await self._execute_one(
                    actions[idx],
                    env=env,
                    state=state,
                    tracker=tracker,
                    segment_index=segment_index,
                )
                results[idx] = segment_result
                cancel_source = segment_result.metadata.get("cancel_source")
                if isinstance(cancel_source, str) and cancel_source:
                    aborted = cancel_source
                    self.last_execution_stats["cancel_source"] = aborted
            else:
                aborted = await self._execute_segment_concurrently(
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
            (
                r
                if r is not None
                else self._error_result(actions[i], "concurrent_execution_failed")
            )
            for i, r in enumerate(results)
        ]

    async def _execute_segment_concurrently(
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

        Returns the abort reason if the segment was cut short, else ``None``.
        Actions that already started are always drained to their real terminal
        state, while actions that never started are recorded as cancelled.
        """
        max_concurrency = min(max(1, self.policy.max_concurrency), len(segment))
        abort_reason: Optional[str] = None
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_index(index: int) -> ActionResult:
            try:
                async with semaphore:
                    if self._is_cancelled():
                        return self._terminal_result(
                            actions[index],
                            ActionStatus.CANCELLED,
                            "cancel_token",
                            segment_index=segment_index,
                        )
                    return await self._execute_one(
                        actions[index],
                        env=env,
                        state=state,
                        tracker=tracker,
                        segment_index=segment_index,
                    )
            except asyncio.CancelledError:
                return self._terminal_result(
                    actions[index],
                    ActionStatus.CANCELLED,
                    abort_reason or "parent_cancelled",
                    segment_index=segment_index,
                )

        tasks = {
            asyncio.create_task(
                _run_index(index), name=f"qitos-tool-{actions[index].name}"
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
                    idx = tasks[task]
                    try:
                        results[idx] = task.result()
                    except EngineInterrupt:
                        raise
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
                    pending_tasks = list(pending)
                    for task in pending_tasks:
                        task.cancel()
                    drained_outcomes = await asyncio.gather(
                        *pending_tasks, return_exceptions=True
                    )
                    for pending_task, drained_outcome in zip(
                        pending_tasks, drained_outcomes
                    ):
                        idx = tasks[pending_task]
                        results[idx] = (
                            drained_outcome
                            if isinstance(drained_outcome, ActionResult)
                            else self._error_result(actions[idx], str(drained_outcome))
                        )
                    pending.clear()
        except asyncio.CancelledError:
            abort_reason = (
                "cancel_token" if self._is_cancelled() else "caller_cancelled"
            )
            self.last_execution_stats["cancel_source"] = abort_reason
            pending = {task for task in tasks if not task.done()}
            pending_tasks = list(pending)
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                drained = await asyncio.gather(*pending_tasks, return_exceptions=True)
                for pending_task, drained_outcome in zip(pending_tasks, drained):
                    idx = tasks[pending_task]
                    results[idx] = (
                        drained_outcome
                        if isinstance(drained_outcome, ActionResult)
                        else self._terminal_result(
                            actions[idx],
                            ActionStatus.CANCELLED,
                            abort_reason,
                            segment_index=segment_index,
                        )
                    )
            for task, idx in tasks.items():
                if results[idx] is None and task.done() and not task.cancelled():
                    completed_outcome = task.result()
                    results[idx] = completed_outcome
        finally:
            unfinished = [task for task in tasks if not task.done()]
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)
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

    # ── Deadline resolution ────────────────────────────────────────────────────

    def _resolve_action_deadline(
        self, action: Action, started: float
    ) -> tuple[Optional[float], str, Optional[float]]:
        """Return one absolute deadline for admission, retries, and invocation."""

        tool = self._resolve_tool(action.name)
        tool_timeout = (
            float(tool.spec.timeout_s)
            if tool is not None and tool.spec.timeout_s is not None
            else None
        )
        tool_deadline = started + tool_timeout if tool_timeout is not None else None
        runtime_deadline = (
            self._turn_budget.deadline_monotonic
            if self._turn_budget is not None
            else (
                getattr(self._engine, "runtime_deadline_monotonic", None)
                if self._engine is not None
                else None
            )
        )
        if runtime_deadline is not None and (
            tool_deadline is None or runtime_deadline <= tool_deadline
        ):
            return (
                float(runtime_deadline),
                "runtime_deadline",
                max(0.0, float(runtime_deadline) - started),
            )
        if tool_deadline is not None:
            return tool_deadline, "tool_spec", tool_timeout
        return None, "none", None

    @staticmethod
    def _remaining_action_seconds(
        deadline_monotonic: Optional[float],
    ) -> Optional[float]:
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
        """Await a tool with one absolute, downward-propagated deadline."""

        invocation = self._call_tool(tool, args, runtime_context=runtime_context)
        if timeout_s is None:
            return await invocation
        if timeout_s <= 0:
            raise asyncio.TimeoutError("action deadline expired")
        return await asyncio.wait_for(invocation, timeout=timeout_s)

    async def _execute_one(
        self,
        action: Action,
        env: Optional[Env] = None,
        state: Any = None,
        tracker: Optional[_ConcurrencyTracker] = None,
        segment_index: int = 0,
    ) -> ActionResult:
        start = time.monotonic()
        deadline, timeout_source, timeout_s = self._resolve_action_deadline(
            action, start
        )
        progress = _ActionProgress()
        if tracker is not None:
            tracker.enter()
        try:
            return await self._execute_one_inner(
                action,
                env=env,
                state=state,
                segment_index=segment_index,
                action_deadline_monotonic=deadline,
                timeout_source=timeout_source,
                timeout_s=timeout_s,
                progress=progress,
            )
        finally:
            if tracker is not None:
                tracker.exit()

    async def _execute_one_inner(
        self,
        action: Action,
        env: Optional[Env] = None,
        state: Any = None,
        segment_index: int = 0,
        action_deadline_monotonic: Optional[float] = None,
        timeout_source: str = "none",
        timeout_s: Optional[float] = None,
        progress: Optional[_ActionProgress] = None,
    ) -> ActionResult:
        start = time.monotonic()
        progress = progress or _ActionProgress()
        started_at = time.time()
        attempts = 0
        last_error = None
        tool_meta = self._tool_meta(action.name)
        stop_result = self._action_stop_result(
            action=action,
            start=start,
            attempts=0,
            tool_meta=tool_meta,
            segment_index=segment_index,
            action_deadline_monotonic=action_deadline_monotonic,
            timeout_source=timeout_source,
            timeout_s=timeout_s,
        )
        if stop_result is not None:
            return stop_result
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
                attempts=0,
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
        runtime_context = self._build_runtime_context(
            tool,
            env=env,
            state=state,
            deadline_monotonic=action_deadline_monotonic,
        )
        ordering_meta: Dict[str, Any] = {
            "segment_index": segment_index,
            "started_at": started_at,
            "started": False,
        }
        if timeout_s is not None:
            ordering_meta["timeout_s"] = timeout_s
            ordering_meta["timeout_source"] = timeout_source

        stop_result = self._action_stop_result(
            action=action,
            start=start,
            attempts=0,
            tool_meta=tool_meta,
            segment_index=segment_index,
            action_deadline_monotonic=action_deadline_monotonic,
            timeout_source=timeout_source,
            timeout_s=timeout_s,
        )
        if stop_result is not None:
            return stop_result

        # A configured permission pipeline owns parameter-level allow/deny/ask.
        # Static needs_approval remains the fallback for simpler callers.
        _auto_approved = False
        if tool.spec.needs_approval and self._pipeline is None:
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
                        attempts=0,
                        tool_meta=tool_meta,
                        output={"message": "User denied approval"},
                        error="User denied approval",
                        extra_metadata={
                            "error_category": "approval_denied",
                            "executed": False,
                        },
                    )

        stop_result = self._action_stop_result(
            action=action,
            start=start,
            attempts=0,
            tool_meta=tool_meta,
            segment_index=segment_index,
            action_deadline_monotonic=action_deadline_monotonic,
            timeout_source=timeout_source,
            timeout_s=timeout_s,
        )
        if stop_result is not None:
            return stop_result

        try:
            permission = self._check_permissions(tool, action.args, runtime_context)
        except EngineInterrupt:
            raise
        except Exception as exc:
            return self._finish_result(
                action=action,
                status=ActionStatus.ERROR,
                start=start,
                attempts=0,
                tool_meta=tool_meta,
                error=str(exc),
                extra_metadata={
                    **ordering_meta,
                    "error_category": "permission_error",
                    "executed": False,
                },
            )
        ordering_meta["permission"] = self._permission_payload(permission)
        if permission.decision == "deny":
            self._dispatch_tool_hook(
                "on_permission_denied",
                action.name,
                action.args,
                tool_result=None,
                permission_decision="deny",
            )
            return self._finish_result(
                action=action,
                status=ActionStatus.DENIED,
                start=start,
                attempts=0,
                tool_meta=tool_meta,
                output={"message": permission.message, "scope": permission.scope},
                error=permission.message or "Tool permission denied",
                extra_metadata={
                    **ordering_meta,
                    "error_category": "permission_denied",
                    "executed": False,
                    "permission": self._permission_payload(permission),
                },
            )
        if permission.decision == "ask":
            if self._permission_interaction_callback is not None:
                try:
                    user_decision = self._permission_interaction_callback(
                        tool_name=action.name,
                        args=action.args,
                        permission=permission,
                    )
                except EngineInterrupt:
                    raise
                except Exception as exc:
                    return self._finish_result(
                        action=action,
                        status=ActionStatus.ERROR,
                        start=start,
                        attempts=0,
                        tool_meta=tool_meta,
                        error=str(exc),
                        extra_metadata={
                            **ordering_meta,
                            "error_category": "permission_interaction_error",
                            "executed": False,
                        },
                    )
                if user_decision == "allow":
                    permission = ToolPermissionDecision.allow(
                        scope=permission.scope,
                        updated_args=permission.updated_args,
                    )
                elif user_decision == "deny":
                    self._dispatch_tool_hook(
                        "on_permission_denied",
                        action.name,
                        action.args,
                        tool_result=None,
                        permission_decision="deny",
                    )
                    return self._finish_result(
                        action=action,
                        status=ActionStatus.DENIED,
                        start=start,
                        attempts=0,
                        tool_meta=tool_meta,
                        output={
                            "message": "User denied permission",
                            "scope": permission.scope,
                        },
                        error="User denied permission",
                        extra_metadata={
                            **ordering_meta,
                            "error_category": "permission_denied",
                            "executed": False,
                            "permission": self._permission_payload(permission),
                        },
                    )
            if permission.decision == "ask":
                self._dispatch_tool_hook(
                    "on_permission_denied",
                    action.name,
                    action.args,
                    tool_result=None,
                    permission_decision="ask",
                )
                return self._finish_result(
                    action=action,
                    status=ActionStatus.NEEDS_APPROVAL,
                    start=start,
                    attempts=0,
                    tool_meta=tool_meta,
                    output={"message": permission.message, "scope": permission.scope},
                    extra_metadata={
                        **ordering_meta,
                        "error_category": "permission_ask",
                        "executed": False,
                        "permission": self._permission_payload(permission),
                    },
                )

        effective_args = dict(
            action.args if permission.updated_args is None else permission.updated_args
        )
        try:
            validation = self._validate(tool, effective_args, runtime_context)
        except EngineInterrupt:
            raise
        except Exception as exc:
            return self._finish_result(
                action=action,
                status=ActionStatus.ERROR,
                start=start,
                attempts=0,
                tool_meta=tool_meta,
                error=str(exc),
                extra_metadata={
                    **ordering_meta,
                    "error_category": "validation_error",
                    "executed": False,
                },
            )
        if not validation.valid:
            return self._finish_result(
                action=action,
                status=ActionStatus.ERROR,
                start=start,
                attempts=0,
                tool_meta=tool_meta,
                error=validation.message or "tool input validation failed",
                extra_metadata={
                    **ordering_meta,
                    "error_category": validation.code or "validation_error",
                    "executed": False,
                    "validation": {
                        "valid": validation.valid,
                        "message": validation.message,
                        "code": validation.code,
                        "suggested_args": validation.suggested_args,
                    },
                },
            )

        admitted_action = Action(
            name=action.name,
            args=effective_args,
            action_id=action.action_id,
            metadata=dict(action.metadata),
        )
        rbw_blocked = self._check_read_before_write(admitted_action)
        if rbw_blocked is not None:
            return rbw_blocked

        stop_result = self._action_stop_result(
            action=action,
            start=start,
            attempts=0,
            tool_meta=tool_meta,
            segment_index=segment_index,
            action_deadline_monotonic=action_deadline_monotonic,
            timeout_source=timeout_source,
            timeout_s=timeout_s,
        )
        if stop_result is not None:
            return stop_result

        self._dispatch_tool_hook(
            "on_before_tool_use",
            action.name,
            effective_args,
            tool_result=None,
            permission_decision=permission.decision,
        )

        retry_policy = tool.spec.retry_policy
        max_attempts = retry_policy.max_attempts if retry_policy is not None else 1
        retryable_exceptions = (
            retry_policy.retryable_exceptions if retry_policy is not None else ()
        )

        while attempts < max_attempts:
            stop_result = self._action_stop_result(
                action=action,
                start=start,
                attempts=attempts,
                tool_meta=tool_meta,
                segment_index=segment_index,
                action_deadline_monotonic=action_deadline_monotonic,
                timeout_source=timeout_source,
                timeout_s=timeout_s,
            )
            if stop_result is not None:
                return stop_result
            attempts = progress.begin_attempt()
            ordering_meta["started"] = True
            try:
                output = await self._invoke_tool(
                    tool,
                    effective_args,
                    runtime_context=runtime_context,
                    timeout_s=self._remaining_action_seconds(action_deadline_monotonic),
                )
                if self._is_cancelled():
                    interruption = self._interruption_result(runtime_context)
                    return self._finish_result(
                        action=action,
                        status=ActionStatus.CANCELLED,
                        start=start,
                        attempts=attempts,
                        tool_meta=tool_meta,
                        output=(None if interruption is None else interruption.output),
                        error="action cancelled",
                        extra_metadata={
                            **ordering_meta,
                            "error_category": "cancelled",
                            "cancel_source": "cancel_token",
                            "worker_still_running": self._worker_still_running(
                                interruption
                            ),
                            "ended_at": time.time(),
                        },
                        artifacts=(
                            () if interruption is None else interruption.artifacts
                        ),
                        model_output=self._interruption_model_output(interruption),
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
                reported_result = ToolResult.from_value(output)
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
                reported_error_category = reported_result.metadata.get("error_category")
                if not isinstance(reported_error_category, str):
                    reported_error_category = None
                result_metadata = {
                    **reported_result.metadata,
                    **tool_meta,
                    **ordering_meta,
                    "error_category": (
                        None
                        if reported_result.is_success
                        else reported_error_category
                        or f"tool_reported_{reported_result.status}"
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
                    artifacts=reported_result.artifacts,
                    model_output=reported_result.model_output,
                )
                return result
            except EngineInterrupt:
                raise
            except asyncio.CancelledError:
                cancel_source = (
                    "cancel_token" if self._is_cancelled() else "caller_cancelled"
                )
                interruption = self._interruption_result(runtime_context)
                return self._finish_result(
                    action=action,
                    status=ActionStatus.CANCELLED,
                    start=start,
                    attempts=attempts,
                    tool_meta=tool_meta,
                    output=(None if interruption is None else interruption.output),
                    error="action cancelled",
                    extra_metadata={
                        **ordering_meta,
                        "error_category": "cancelled",
                        "cancel_source": cancel_source,
                        "worker_still_running": self._worker_still_running(
                            interruption
                        ),
                        "ended_at": time.time(),
                    },
                    artifacts=(
                        () if interruption is None else interruption.artifacts
                    ),
                    model_output=self._interruption_model_output(interruption),
                )
            except asyncio.TimeoutError as exc:
                interruption = self._interruption_result(runtime_context)
                timed_out_result = self._finish_result(
                    action=action,
                    status=ActionStatus.TIMED_OUT,
                    start=start,
                    attempts=attempts,
                    tool_meta=tool_meta,
                    output=(None if interruption is None else interruption.output),
                    error=str(exc),
                    extra_metadata={
                        **ordering_meta,
                        "error_category": "timeout",
                        "worker_still_running": self._worker_still_running(
                            interruption
                        ),
                        "ended_at": time.time(),
                    },
                    artifacts=(
                        () if interruption is None else interruption.artifacts
                    ),
                    model_output=self._interruption_model_output(interruption),
                )
                return timed_out_result
            except Exception as exc:  # pragma: no cover - defensive path
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
                    remaining = self._remaining_action_seconds(
                        action_deadline_monotonic
                    )
                    if remaining is not None and delay >= remaining:
                        return self._deadline_result(
                            action=action,
                            start=start,
                            attempts=attempts,
                            tool_meta=tool_meta,
                            segment_index=segment_index,
                            started=True,
                            timeout_source=timeout_source,
                            timeout_s=timeout_s,
                        )
                    retry_cancel_source = await self._wait_for_retry(
                        delay,
                        deadline_monotonic=action_deadline_monotonic,
                    )
                    if retry_cancel_source is not None:
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
                                "cancel_source": retry_cancel_source,
                                "ended_at": time.time(),
                            },
                        )

        error_category = "runtime_error"
        if last_error and "not found" in last_error.lower():
            error_category = "tool_not_found"

        # Call on_failure callback if registered
        if tool.spec.on_failure is not None:
            try:
                tool.spec.on_failure(
                    action=action,
                    error=last_error,
                    attempts=attempts,
                )
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
        return error_result

    async def _wait_for_retry(
        self,
        delay: float,
        *,
        deadline_monotonic: Optional[float],
    ) -> str | None:
        """Wait for retry backoff and identify the cancellation source."""

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

    def _action_stop_result(
        self,
        *,
        action: Action,
        start: float,
        attempts: int,
        tool_meta: Dict[str, Any],
        segment_index: int,
        action_deadline_monotonic: Optional[float],
        timeout_source: str,
        timeout_s: Optional[float],
    ) -> Optional[ActionResult]:
        if self._is_cancelled():
            return self._finish_result(
                action=action,
                status=ActionStatus.CANCELLED,
                start=start,
                attempts=attempts,
                tool_meta=tool_meta,
                error="action cancelled",
                extra_metadata={
                    "error_category": "cancelled",
                    "cancel_source": "cancel_token",
                    "segment_index": segment_index,
                    "started": attempts > 0,
                    "worker_still_running": False,
                    "ended_at": time.time(),
                },
            )
        remaining = self._remaining_action_seconds(action_deadline_monotonic)
        if remaining is None or remaining > 0:
            return None
        return self._deadline_result(
            action=action,
            start=start,
            attempts=attempts,
            tool_meta=tool_meta,
            segment_index=segment_index,
            started=attempts > 0,
            timeout_source=timeout_source,
            timeout_s=timeout_s,
        )

    def _deadline_result(
        self,
        *,
        action: Action,
        start: float,
        attempts: int,
        tool_meta: Dict[str, Any],
        segment_index: int,
        started: bool,
        timeout_source: str = "runtime_deadline",
        timeout_s: Optional[float] = 0.0,
        worker_still_running: bool = False,
    ) -> ActionResult:
        label = (
            "runtime deadline"
            if timeout_source == "runtime_deadline"
            else "tool deadline"
        )
        return self._finish_result(
            action=action,
            status=ActionStatus.TIMED_OUT,
            start=start,
            attempts=attempts,
            tool_meta=tool_meta,
            error=f"{label} expired before action completion",
            extra_metadata={
                "error_category": "timeout",
                "timeout_source": timeout_source,
                "timeout_s": timeout_s,
                "segment_index": segment_index,
                "started": started,
                "worker_still_running": worker_still_running,
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
        artifacts: tuple[ArtifactRef, ...] = (),
        model_output: str | None = None,
    ) -> ActionResult:
        if output is None:
            code = str(
                (extra_metadata or {}).get("error_code") or "TOOL_EXECUTION_ERROR"
            )
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
            artifacts=artifacts,
            model_output=model_output,
        )

    @staticmethod
    def _interruption_result(
        runtime_context: Dict[str, Any],
    ) -> ToolResult | None:
        result = runtime_context.get("interruption_result")
        return result if isinstance(result, ToolResult) else None

    @staticmethod
    def _worker_still_running(result: ToolResult | None) -> bool:
        if result is None or not isinstance(result.output, dict):
            return False
        return result.output.get("process_status") == "running"

    @staticmethod
    def _interruption_model_output(result: ToolResult | None) -> str | None:
        if result is None:
            return None
        output = result.model_visible_output
        return output if isinstance(output, str) else None

    def _build_runtime_context(
        self,
        tool: BaseTool,
        env: Optional[Env],
        state: Any,
        *,
        deadline_monotonic: Optional[float] = None,
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

        active_run_id = self._engine.active_run_id if self._engine is not None else ""

        async def _post_runtime_event(event: Any) -> bool:
            if self._engine is None or not active_run_id:
                return False
            return bool(
                await self._engine.apost_runtime_event(
                    event,
                    run_id=active_run_id,
                )
            )

        def _record_runtime_event(phase: str, payload: Dict[str, Any]) -> None:
            if self._engine is None:
                return
            try:
                runtime_phase = RuntimePhase(phase)
            except ValueError:
                return
            self._engine.events.append(
                RuntimeEvent(
                    step_id=int(getattr(state, "current_step", 0) or 0),
                    phase=runtime_phase,
                    payload=dict(payload),
                )
            )

        runtime_deadline = deadline_monotonic
        max_children = (
            self._turn_budget.max_children
            if self._turn_budget is not None
            else int(
                self._engine.budget.max_children if self._engine is not None else 0
            )
        )

        def _remaining_seconds() -> Optional[float]:
            return self._remaining_action_seconds(runtime_deadline)

        return {
            "env": env,
            "runtime_capabilities": (
                self._runtime_capabilities
                if self._runtime_capabilities is not None
                else (env.capability_snapshot() if env is not None else None)
            ),
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
            "run_id": active_run_id,
            "parent_run_id": active_run_id,
            "journal": self._engine.journal if self._engine is not None else None,
            "budget_ledger": (
                self._engine.budget_ledger if self._engine is not None else None
            ),
            "max_children": max_children,
            "post_runtime_event": _post_runtime_event,
            "record_runtime_event": _record_runtime_event,
            "deadline_monotonic": runtime_deadline,
            "remaining_seconds": _remaining_seconds,
            "agent_cancelled": self._is_cancelled,
            "trace_writer": self.trace_writer,
            "shared_memory": self.shared_memory,
            "agent": self._engine.agent if self._engine is not None else None,
        }

    def _resolve_tool(self, name: str) -> Optional[BaseTool]:
        return self.tool_registry.get(name)

    def _validate(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> ToolValidationResult:
        schema_errors = tool_input_schema_errors(
            tool.spec.input_schema or {},
            dict(args),
        )
        if schema_errors:
            return ToolValidationResult.fail(
                "Tool arguments do not match input_schema:\n"
                + "\n".join(f"- {error}" for error in schema_errors),
                code="invalid_tool_arguments",
            )
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

    async def _call_tool(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return await tool.execute(args, runtime_context=runtime_context)

    def _resolve_permission_context(
        self, env: Optional[Env], state: Any
    ) -> ToolPermissionContext | None:
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
        return None

    def _permission_payload(self, decision: ToolPermissionDecision) -> Dict[str, Any]:
        mode = getattr(self._pipeline, "mode", None)
        mode_value = getattr(mode, "value", mode)
        return {
            "mode": str(mode_value or "tool"),
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

    _WRITE_TOOL_NAMES = frozenset({"edit_file", "write_file"})

    _READ_TOOL_NAMES = frozenset({"read_file"})

    def _check_read_before_write(self, action: Action) -> Optional[ActionResult]:
        """Check read-before-write enforcement for file editing tools.

        Returns an ActionResult if the action should be blocked, None otherwise.
        """
        if self._rbw_enforcer is None or self._uses_autonomous_permission():
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
            attempts=0,
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
        if self._rbw_enforcer is None or self._uses_autonomous_permission():
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

    def _uses_autonomous_permission(self) -> bool:
        mode = getattr(self._pipeline, "mode", None)
        return getattr(mode, "value", mode) == "autonomous"
