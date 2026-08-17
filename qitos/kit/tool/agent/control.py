"""Model-facing control tools for one Run-owned ChildSupervisor."""

from __future__ import annotations

import math
from typing import Any

from ....core.child import ChildHandle, ChildResult, ChildStatus
from ....core.tool import BaseTool, ToolPermission, ToolSpec
from ....core.tool_result import ToolResult
from ...child import ChildSupervisor
from ..internal.results import tool_result


class _ChildControlTool(BaseTool):
    def __init__(
        self,
        *,
        supervisor: ChildSupervisor,
        name: str,
        description: str,
        parameters: dict[str, dict[str, Any]],
        required: list[str],
    ) -> None:
        if not isinstance(supervisor, ChildSupervisor):
            raise TypeError("supervisor must be a ChildSupervisor")
        self._supervisor = supervisor
        super().__init__(
            ToolSpec(
                name=name,
                description=description,
                parameters={
                    "child_id": {
                        "type": "string",
                        "description": "The child id returned by Agent.",
                    },
                    **parameters,
                },
                required=["child_id", *required],
                permissions=ToolPermission(),
                concurrency_safe=True,
                group="agent",
            )
        )
        self.spec.description = description

    @staticmethod
    def _handle(
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None,
    ) -> ChildHandle:
        child_id = str(args.get("child_id") or "").strip()
        if not child_id:
            raise ValueError("child_id is required")
        context = runtime_context or {}
        # ToolBatchExecutor owns ``run_id`` for the frozen turn. Keep the
        # explicit parent id for direct application callers, but never let it
        # replace the executor-owned identity when both are present.
        parent_run_id = str(
            context.get("run_id") or context.get("parent_run_id") or ""
        ).strip()
        if not parent_run_id:
            raise ValueError("parent_run_id is required for child ownership")
        return ChildHandle(child_id=child_id, parent_run_id=parent_run_id)

    @staticmethod
    def _timeout(
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None,
        *,
        default: float,
    ) -> float:
        raw = args.get("timeout_seconds", default)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError("timeout_seconds must be a number")
        timeout = float(raw)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout_seconds must be finite and non-negative")
        remaining = (runtime_context or {}).get("remaining_seconds")
        if callable(remaining):
            available = remaining()
            if available is not None:
                timeout = min(timeout, max(0.0, float(available)))
        return timeout

    @staticmethod
    def _projection(result: ChildResult) -> dict[str, Any]:
        payload = ChildSupervisor.result_payload(result)
        payload["status"] = "success"
        return payload

    @staticmethod
    def _unknown(handle: ChildHandle) -> dict[str, Any]:
        return {
            "status": "success",
            "child_status": ChildStatus.UNKNOWN.value,
            "ready": True,
            "handle": handle.to_dict(),
            "child_id": handle.child_id,
            "output": "No child with this handle belongs to the current Run.",
        }


class ChildStatusTool(_ChildControlTool):
    """Query one Child without changing its lifecycle."""

    def __init__(self, supervisor: ChildSupervisor) -> None:
        super().__init__(
            supervisor=supervisor,
            name="child_status",
            description="Return the current typed status and conclusion of one Child.",
            parameters={},
            required=[],
        )

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | ToolResult:
        try:
            handle = self._handle(args, runtime_context)
        except (TypeError, ValueError) as exc:
            return tool_result(
                {"status": "error", "error": str(exc)}, status="error"
            )
        result = self._supervisor.result(handle)
        return self._unknown(handle) if result is None else self._projection(result)


class ChildWaitTool(_ChildControlTool):
    """Wait a bounded time for one Child without cancelling it on timeout."""

    def __init__(self, supervisor: ChildSupervisor) -> None:
        super().__init__(
            supervisor=supervisor,
            name="child_wait",
            description=(
                "Wait for one Child to reach terminal state. A wait timeout leaves the "
                "Child running and returns its current status."
            ),
            parameters={
                "timeout_seconds": {
                    "type": "number",
                    "description": "Maximum wait, from 0 to 60 seconds.",
                    "minimum": 0,
                    "maximum": 60,
                }
            },
            required=[],
        )

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | ToolResult:
        try:
            handle = self._handle(args, runtime_context)
            timeout = min(
                60.0,
                self._timeout(args, runtime_context, default=30.0),
            )
            result = await self._supervisor.wait(
                handle,
                timeout_seconds=timeout,
            )
        except (TypeError, ValueError) as exc:
            return tool_result(
                {"status": "error", "error": str(exc)}, status="error"
            )
        return self._unknown(handle) if result is None else self._projection(result)


class ChildMessageTool(_ChildControlTool):
    """Deliver a parent message through an active Child Engine mailbox."""

    def __init__(self, supervisor: ChildSupervisor) -> None:
        super().__init__(
            supervisor=supervisor,
            name="child_message",
            description=(
                "Send context or follow-up instructions to an active Child. The Child "
                "accepts the message at its next turn safe point."
            ),
            parameters={
                "content": {
                    "type": "string",
                    "description": "The new context or instruction for the Child.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Maximum time to wait for mailbox acceptance.",
                    "minimum": 0,
                    "maximum": 30,
                },
            },
            required=["content"],
        )

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | ToolResult:
        try:
            handle = self._handle(args, runtime_context)
            timeout = min(
                30.0,
                self._timeout(args, runtime_context, default=5.0),
            )
            accepted, result = await self._supervisor.message(
                handle,
                str(args.get("content") or ""),
                timeout_seconds=timeout,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            return tool_result(
                {"status": "error", "error": str(exc)}, status="error"
            )
        if result is None:
            payload = self._unknown(handle)
        else:
            payload = self._projection(result)
        payload["accepted"] = accepted
        if not accepted and result is not None and result.ready:
            payload["message"] = (
                "Terminal children cannot accept messages; launch a new Child for "
                "follow-up work."
            )
        return payload


class ChildInterruptTool(_ChildControlTool):
    """Interrupt one Child and wait for bounded terminal cleanup."""

    def __init__(self, supervisor: ChildSupervisor) -> None:
        super().__init__(
            supervisor=supervisor,
            name="child_interrupt",
            description=(
                "Request immediate cancellation of one active Child and wait for its "
                "cleanup. Terminal and unknown handles are stable no-ops."
            ),
            parameters={
                "timeout_seconds": {
                    "type": "number",
                    "description": "Maximum cleanup wait, from 0 to 30 seconds.",
                    "minimum": 0,
                    "maximum": 30,
                }
            },
            required=[],
        )

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | ToolResult:
        try:
            handle = self._handle(args, runtime_context)
            timeout = min(
                30.0,
                self._timeout(args, runtime_context, default=5.0),
            )
            result = await self._supervisor.interrupt(
                handle,
                wait_seconds=timeout,
            )
        except (TypeError, ValueError) as exc:
            return tool_result(
                {"status": "error", "error": str(exc)}, status="error"
            )
        return self._unknown(handle) if result is None else self._projection(result)


class ChildControlToolSet:
    """Expose status, wait, message, and interrupt over one ChildSupervisor."""

    name = "child"
    version = "1"

    def __init__(self, supervisor: ChildSupervisor) -> None:
        self._tools = [
            ChildStatusTool(supervisor),
            ChildWaitTool(supervisor),
            ChildMessageTool(supervisor),
            ChildInterruptTool(supervisor),
        ]

    def tools(self) -> list[BaseTool]:
        return list(self._tools)


__all__ = [
    "ChildControlToolSet",
    "ChildInterruptTool",
    "ChildMessageTool",
    "ChildStatusTool",
    "ChildWaitTool",
]
