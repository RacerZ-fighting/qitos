"""Model-facing tool projection for child Agent supervision."""

from __future__ import annotations

from typing import Any, Literal

from ....core.child import (
    DEFAULT_CHILD_MAX_STEPS,
    ChildHandle,
    ChildLaunchRequest,
    ChildResult,
)
from ....core.runtime_input import RuntimeInput
from ....core.task import TaskBudget
from ....core.tool import BaseTool, ToolPermission, ToolSpec
from ...child import (
    ChildExecutionScope,
    ChildInvocationFactory,
    ChildSupervisor,
)

AgentExecutionMode = Literal["foreground", "optional_background", "background"]


class AgentTool(BaseTool):
    """Launch a fresh child Agent through one Run-owned async supervisor."""

    def __init__(
        self,
        *,
        invocation_factory: ChildInvocationFactory | None = None,
        execution_scope: ChildExecutionScope | None = None,
        execution_mode: AgentExecutionMode = "foreground",
        max_background_workers: int = 4,
        max_delegate_depth: int = 1,
        child_budget: TaskBudget | None = None,
        supervisor: ChildSupervisor | None = None,
    ) -> None:
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
        if supervisor is not None and (
            invocation_factory is not None
            or execution_scope is not None
            or max_background_workers != 4
        ):
            raise ValueError(
                "invocation_factory, execution_scope, and max_background_workers "
                "belong to the supplied supervisor"
            )
        if supervisor is None:
            if invocation_factory is None:
                raise TypeError("invocation_factory is required without a supervisor")
            supervisor = ChildSupervisor(
                invocation_factory=invocation_factory,
                execution_scope=execution_scope,
                max_concurrency=max_background_workers,
            )
        self._supervisor = supervisor
        self._execution_mode = execution_mode
        self._max_delegate_depth = max_delegate_depth
        self._child_budget = resolved_budget

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
                required=["description", "prompt"],
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
    ) -> dict[str, Any]:
        """Launch one child and project its typed lifecycle as a Tool result."""

        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return {"status": "error", "error": "prompt is required"}
        description = str(args.get("description", "")).strip()
        if not description:
            return {"status": "error", "error": "description is required"}
        agent_type = (
            str(args.get("subagent_type", "general-purpose")).strip()
            or "general-purpose"
        )
        context = self._snapshot_parent_context(runtime_context, agent_type=agent_type)
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

        requested_background = bool(args.get("run_in_background", False))
        if requested_background and self._execution_mode == "foreground":
            return {
                "status": "error",
                "error": "Background child agents are disabled in this runtime.",
            }
        background = self._execution_mode == "background" or (
            self._execution_mode == "optional_background" and requested_background
        )
        request = ChildLaunchRequest(
            task=prompt,
            description=description,
            name=str(args.get("name", "")).strip(),
            agent_type=agent_type,
            budget=self._child_budget,
        )
        try:
            result = await self._supervisor.launch(
                request,
                context,
                parent_run_id=parent_run_id,
                background=background,
                max_children=int(context.get("max_children", 0) or 0),
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            return {"status": "error", "error": str(exc)}
        return self._supervisor.result_payload(result)

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
        self._supervisor.setup()

    async def asetup(self, context: dict[str, Any] | None = None) -> None:
        self.setup(context)
        payload = context or {}
        journal = payload.get("journal")
        parent_run_id = str(payload.get("run_id") or "").strip()
        if payload.get("resume_journal") is True and journal is not None:
            await self._supervisor.recover(
                parent_run_id=parent_run_id,
                journal=journal,
            )

    async def aclose(self, *, wait_seconds: float = 5.0) -> int:
        return await self._supervisor.aclose(wait_seconds=wait_seconds)

    @staticmethod
    def _snapshot_parent_context(
        runtime_context: dict[str, Any] | None,
        *,
        agent_type: str,
    ) -> dict[str, Any]:
        context = dict(runtime_context or {})
        parent_agent = context.get("agent")
        parent_history = getattr(parent_agent, "history", None)
        parent_messages = getattr(parent_history, "messages", None)
        if parent_messages is not None and "parent_history" not in context:
            context["parent_history"] = tuple(parent_messages)
        if agent_type == "fork" and "parent_history_snapshot" not in context:
            snapshot = getattr(parent_history, "snapshot", None)
            if callable(snapshot):
                context["parent_history_snapshot"] = snapshot()
        return context
