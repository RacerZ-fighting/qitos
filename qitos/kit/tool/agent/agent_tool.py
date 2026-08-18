"""Model-facing tool projection for child Agent supervision."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from ....core.budget import BudgetLedger
from ....core.child import (
    DEFAULT_CHILD_MAX_STEPS,
    ChildHandle,
    ChildLaunchContext,
    ChildLaunchRequest,
    ChildResult,
)
from ....core.model_response import ModelUsage
from ....core.runtime_input import RuntimeInput
from ....core.task import Task, TaskBudget
from ....core.tool import (
    BaseTool,
    ToolPermission,
    ToolPermissionContext,
    ToolSpec,
)
from ....core.tool_registry import ToolExposure
from ....core.tool_result import ToolResult, ToolResultStatus
from ...child import (
    ChildExecutionScope,
    ChildInvocationFactory,
    ChildJournalFactory,
    ChildRunLimiter,
    ChildSupervisor,
)
from ..internal.results import tool_result

AgentExecutionMode = Literal["foreground", "optional_background", "background"]


def _child_result_usage(result: ChildResult) -> ModelUsage | None:
    """Project one terminal Child's accounting into typed Tool usage.

    ``ChildResult`` tracks cumulative totals only, so ``total_tokens`` is the
    available token fact and ``cost_usd`` rides as a lossless detail key.
    A non-terminal (background launch) result has no finished accounting
    yet and carries no usage; completeness flags stay in the output
    payload either way.
    """

    if not result.ready:
        return None
    return ModelUsage.from_mapping(
        {
            "total_tokens": result.total_tokens,
            "cost_usd": float(result.total_cost_usd),
        }
    )


class AgentTool(BaseTool):
    """Launch a fresh child Agent through one Run-owned async supervisor."""

    def __init__(
        self,
        *,
        invocation_factory: ChildInvocationFactory | None = None,
        execution_scope: ChildExecutionScope | None = None,
        execution_mode: AgentExecutionMode = "foreground",
        max_background_workers: int = 4,
        run_limiter: ChildRunLimiter | None = None,
        owns_run_limiter: bool = True,
        child_journal_factory: ChildJournalFactory | None = None,
        max_delegate_depth: int = 1,
        child_budget: TaskBudget | None = None,
        child_profile: str = "default",
        child_allowed_tool_groups: tuple[str, ...] = (),
        child_working_directory: str | None = None,
        supervisor: ChildSupervisor | None = None,
    ) -> None:
        if max_delegate_depth <= 0:
            raise ValueError("max_delegate_depth must be positive")
        if not isinstance(owns_run_limiter, bool):
            raise TypeError("owns_run_limiter must be a boolean")
        resolved_budget = child_budget or TaskBudget(max_steps=DEFAULT_CHILD_MAX_STEPS)
        if not isinstance(resolved_budget, TaskBudget):
            raise TypeError("child_budget must be a TaskBudget")
        if not isinstance(child_profile, str) or not child_profile.strip():
            raise ValueError("child_profile must be a non-empty string")
        if not isinstance(child_allowed_tool_groups, tuple) or any(
            not isinstance(group, str) or not group.strip()
            for group in child_allowed_tool_groups
        ):
            raise TypeError("child_allowed_tool_groups must contain non-empty strings")
        if child_working_directory is not None and (
            not isinstance(child_working_directory, str)
            or not child_working_directory.strip()
        ):
            raise ValueError(
                "child_working_directory must be a non-empty string or None"
            )
        if execution_mode not in {
            "foreground",
            "optional_background",
            "background",
        }:
            raise ValueError(f"unsupported Agent execution_mode: {execution_mode}")
        if supervisor is not None and (
            invocation_factory is not None
            or execution_scope is not None
            or run_limiter is not None
            or child_journal_factory is not None
            or max_background_workers != 4
        ):
            raise ValueError(
                "invocation_factory, execution_scope, run_limiter, "
                "child_journal_factory, and max_background_workers "
                "belong to the supplied supervisor"
            )
        if supervisor is None:
            if invocation_factory is None:
                raise TypeError("invocation_factory is required without a supervisor")
            supervisor = ChildSupervisor(
                invocation_factory=invocation_factory,
                execution_scope=execution_scope,
                max_concurrency=max_background_workers,
                run_limiter=run_limiter,
                child_journal_factory=child_journal_factory,
            )
        self._supervisor = supervisor
        self._owns_run_limiter = owns_run_limiter
        self._execution_mode = execution_mode
        self._max_delegate_depth = max_delegate_depth
        self._child_budget = resolved_budget
        self._child_profile = child_profile.strip()
        self._child_allowed_tool_groups = tuple(
            dict.fromkeys(group.strip() for group in child_allowed_tool_groups)
        )
        self._child_working_directory = (
            child_working_directory.strip()
            if child_working_directory is not None
            else None
        )

        parameters: dict[str, dict[str, Any]] = {
            "description": {
                "type": "string",
                "description": "A short description of the delegated task.",
            },
            "prompt": {
                "type": "string",
                "description": "The task for the child agent to perform.",
            },
            "success_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": (
                    "Concrete conditions the child must satisfy before it can "
                    "report completion."
                ),
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
            "plan_assignment": {
                "type": "string",
                "description": (
                    "Optional ready parent Plan node id assigned to this child. "
                    "The runtime durably reserves it before the child starts."
                ),
            },
        }
        if execution_mode == "optional_background":
            parameters["run_in_background"] = {
                "type": "boolean",
                "description": "Run asynchronously and return a child handle.",
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
        super().__init__(
            spec=ToolSpec(
                name="Agent",
                description=description,
                parameters=parameters,
                required=["description", "prompt", "success_criteria"],
                permissions=ToolPermission(),
                concurrency_safe=True,
                supports_background=execution_mode != "foreground",
            )
        )
        self.spec.description = description

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | ToolResult:
        """Launch one child and project its typed lifecycle as a Tool result."""

        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return tool_result(
                {"status": "error", "error": "prompt is required"},
                status="error",
            )
        description = str(args.get("description", "")).strip()
        if not description:
            return tool_result(
                {"status": "error", "error": "description is required"},
                status="error",
            )
        raw_criteria = args.get("success_criteria")
        if not isinstance(raw_criteria, list) or not raw_criteria or any(
            not isinstance(item, str) or not item.strip() for item in raw_criteria
        ):
            return tool_result(
                {
                    "status": "error",
                    "error": (
                        "success_criteria must be a non-empty array of "
                        "non-empty strings"
                    ),
                },
                status="error",
            )
        success_criteria = tuple(item.strip() for item in raw_criteria)
        agent_type = (
            str(args.get("subagent_type", "general-purpose")).strip()
            or "general-purpose"
        )
        try:
            context = self._snapshot_parent_context(
                runtime_context,
                agent_type=agent_type,
            )
        except (TypeError, ValueError) as exc:
            return tool_result(
                {"status": "error", "error": str(exc)}, status="error"
            )
        current_depth = context.delegate_depth
        if current_depth >= self._max_delegate_depth:
            payload = {
                "status": "error",
                "error": (
                    "Child agents cannot launch another Agent; maximum delegation "
                    f"depth is {self._max_delegate_depth}."
                ),
            }
            return tool_result(payload, status="error")

        requested_background = bool(args.get("run_in_background", False))
        if requested_background and self._execution_mode == "foreground":
            payload = {
                "status": "error",
                "error": "Background child agents are disabled in this runtime.",
            }
            return tool_result(payload, status="error")
        background = self._execution_mode == "background" or (
            self._execution_mode == "optional_background" and requested_background
        )
        context_values = runtime_context or {}
        parent_task_id = context_values.get("task_id")
        if parent_task_id is not None and (
            not isinstance(parent_task_id, str) or not parent_task_id.strip()
        ):
            return tool_result(
                {
                    "status": "error",
                    "error": (
                        "runtime context task_id must be a non-empty string or None"
                    ),
                },
                status="error",
            )
        parent_task = context_values.get("task")
        if parent_task is not None and not isinstance(parent_task, Task):
            return tool_result(
                {
                    "status": "error",
                    "error": "runtime context task must be a Task or None",
                },
                status="error",
            )
        if (
            isinstance(parent_task, Task)
            and parent_task_id is not None
            and parent_task.task_id != parent_task_id
        ):
            return tool_result(
                {
                    "status": "error",
                    "error": "runtime context task does not match task_id",
                },
                status="error",
            )
        raw_assignment = args.get("plan_assignment")
        if raw_assignment is not None and (
            not isinstance(raw_assignment, str) or not raw_assignment.strip()
        ):
            return tool_result(
                {
                    "status": "error",
                    "error": "plan_assignment must be a non-empty string or omitted",
                },
                status="error",
            )
        request = ChildLaunchRequest(
            task=prompt,
            description=description,
            name=str(args.get("name", "")).strip(),
            agent_type=agent_type,
            success_criteria=success_criteria,
            constraints=(parent_task.constraints if parent_task is not None else {}),
            references=(parent_task.references if parent_task is not None else ()),
            permission_context=context.parent_permission_context,
            profile=self._child_profile,
            allowed_tool_groups=self._child_allowed_tool_groups,
            working_directory=self._child_working_directory,
            budget=self._child_budget,
            parent_task_id=parent_task_id,
            plan_assignment=(
                raw_assignment.strip()
                if isinstance(raw_assignment, str)
                else None
            ),
        )
        try:
            result = await self._supervisor.launch(
                request,
                context,
                background=background,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            return tool_result(
                {"status": "error", "error": str(exc)}, status="error"
            )
        payload = self._supervisor.result_payload(result)
        usage = _child_result_usage(result)
        lifecycle = str(payload.get("status") or "success")
        if lifecycle in {"error", "cancelled", "partial"}:
            return tool_result(
                payload,
                status=cast(ToolResultStatus, lifecycle),
                usage=usage,
            )
        # A background Child's ``running`` value describes the Child domain;
        # launching it was a successful, terminal Tool call.
        return tool_result(payload, status="success", usage=usage)

    def child_result(self, handle: ChildHandle) -> ChildResult | None:
        """Return one owned child's current immutable result."""

        return self._supervisor.result(handle)

    async def wait_child(
        self,
        handle: ChildHandle,
        *,
        timeout_seconds: float | None = None,
    ) -> ChildResult | None:
        """Wait for one child without cancelling it when the wait times out."""

        return await self._supervisor.wait(
            handle,
            timeout_seconds=timeout_seconds,
        )

    async def interrupt_child(
        self,
        handle: ChildHandle,
        *,
        wait_seconds: float = 5.0,
    ) -> ChildResult | None:
        """Cancel one child and wait a bounded time for terminal cleanup."""

        return await self._supervisor.interrupt(handle, wait_seconds=wait_seconds)

    def cancel_child(self, handle: ChildHandle) -> bool:
        """Signal cancellation without blocking the current Tool call."""

        return self._supervisor.request_interrupt(handle)

    @property
    def active_background_count(self) -> int:
        return self._supervisor.active_count

    @property
    def supervisor(self) -> ChildSupervisor:
        """Return the shared Run-owned lifecycle capability for control tools."""

        return self._supervisor

    def snapshot_background_events(self) -> list[RuntimeInput]:
        return self._supervisor.snapshot_events()

    def setup(self, context: dict[str, Any] | None = None) -> None:
        _ = context
        self._supervisor.setup(reset_run_limiter=self._owns_run_limiter)

    async def asetup(self, context: dict[str, Any] | None = None) -> None:
        self.setup(context)
        payload = context or {}
        journal = payload.get("journal")
        parent_run_id = str(payload.get("run_id") or "").strip()
        if payload.get("resume_journal") is True and journal is not None:
            await self._supervisor.recover(
                parent_run_id=parent_run_id,
                journal=journal,
                budget_ledger=payload.get("budget_ledger"),
            )

    async def aclose(self, *, wait_seconds: float = 5.0) -> int:
        return await self._supervisor.aclose(wait_seconds=wait_seconds)

    @staticmethod
    def _snapshot_parent_context(
        runtime_context: dict[str, Any] | None,
        *,
        agent_type: str,
    ) -> ChildLaunchContext:
        context = runtime_context or {}
        parent_agent = context.get("agent")
        parent_history = getattr(parent_agent, "history", None)
        parent_messages = getattr(parent_history, "messages", None)
        history = context.get("parent_history")
        if history is None and parent_messages is not None:
            history = tuple(parent_messages)
        history_snapshot = context.get("parent_history_snapshot")
        if agent_type == "fork" and history_snapshot is None:
            snapshot = getattr(parent_history, "snapshot", None)
            if callable(snapshot):
                history_snapshot = snapshot()
        budget_ledger = context.get("budget_ledger")
        if budget_ledger is not None and not isinstance(budget_ledger, BudgetLedger):
            raise TypeError("budget_ledger must be a BudgetLedger or None")
        parent_tool_authority = context.get("tool_registry")
        if parent_tool_authority is not None and not isinstance(
            parent_tool_authority, ToolExposure
        ):
            raise TypeError("tool_registry must be a frozen ToolExposure or None")
        parent_permission_context = context.get("permission_context")
        if isinstance(parent_permission_context, Mapping):
            parent_permission_context = ToolPermissionContext.from_dict(
                dict(parent_permission_context)
            )
        if parent_permission_context is not None and not isinstance(
            parent_permission_context, ToolPermissionContext
        ):
            raise TypeError(
                "permission_context must be a ToolPermissionContext, mapping, or None"
            )
        return ChildLaunchContext(
            parent_run_id=str(
                context.get("run_id") or context.get("parent_run_id") or ""
            ).strip(),
            delegate_depth=context.get("delegate_depth", 0),
            max_children=context.get("max_children", 0) or 0,
            deadline_monotonic=context.get("deadline_monotonic"),
            budget_ledger=budget_ledger,
            journal=context.get("journal"),
            post_runtime_event=context.get("post_runtime_event"),
            parent_tool_authority=parent_tool_authority,
            parent_permission_context=parent_permission_context,
            parent_history=tuple(history or ()),
            parent_history_snapshot=history_snapshot,
        )
