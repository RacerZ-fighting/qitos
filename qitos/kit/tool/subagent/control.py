"""Model-facing control tools for one Run-owned SubagentSupervisor."""

from __future__ import annotations

import math
from typing import Any

from ....core.subagent import SubagentHandle, SubagentResult, SubagentStatus
from ....core.tool import BaseTool, ToolPermission, ToolSpec
from ....core.tool_result import ToolResult
from ...subagent import SubagentSupervisor
from ..internal.results import tool_result


class _SubagentControlTool(BaseTool):
    def __init__(
        self,
        *,
        supervisor: SubagentSupervisor,
        name: str,
        description: str,
        parameters: dict[str, dict[str, Any]],
        required: list[str],
    ) -> None:
        if not isinstance(supervisor, SubagentSupervisor):
            raise TypeError("supervisor must be a SubagentSupervisor")
        self._supervisor = supervisor
        super().__init__(
            ToolSpec(
                name=name,
                description=description,
                parameters={
                    "subagent_id": {
                        "type": "string",
                        "description": "The subagent id returned by `subagent`.",
                    },
                    **parameters,
                },
                required=["subagent_id", *required],
                permissions=ToolPermission(),
                concurrency_safe=True,
                group="subagent",
            )
        )
        self.spec.description = description

    @staticmethod
    def _handle(
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None,
    ) -> SubagentHandle:
        subagent_id = str(args.get("subagent_id") or "").strip()
        if not subagent_id:
            raise ValueError("subagent_id is required")
        context = runtime_context or {}
        # ToolBatchExecutor owns ``run_id`` for the frozen turn. Keep the
        # explicit parent id for direct application callers, but never let it
        # replace the executor-owned identity when both are present.
        parent_run_id = str(
            context.get("run_id") or context.get("parent_run_id") or ""
        ).strip()
        if not parent_run_id:
            raise ValueError("parent_run_id is required for subagent ownership")
        return SubagentHandle(subagent_id=subagent_id, parent_run_id=parent_run_id)

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
    def _projection(result: SubagentResult) -> dict[str, Any]:
        payload = SubagentSupervisor.result_payload(result)
        payload["status"] = "success"
        return payload

    @staticmethod
    def _unknown(handle: SubagentHandle) -> dict[str, Any]:
        return {
            "status": "success",
            "subagent_status": SubagentStatus.UNKNOWN.value,
            "ready": True,
            "handle": handle.to_dict(),
            "subagent_id": handle.subagent_id,
            "output": "No subagent with this handle belongs to the current Run.",
        }


class SubagentStatusTool(_SubagentControlTool):
    """Query one Subagent without changing its lifecycle."""

    def __init__(self, supervisor: SubagentSupervisor) -> None:
        super().__init__(
            supervisor=supervisor,
            name="subagent_status",
            description="Return the current typed status and conclusion of one Subagent.",
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


class SubagentWaitTool(_SubagentControlTool):
    """Wait a bounded time for one Subagent without cancelling it on timeout."""

    def __init__(self, supervisor: SubagentSupervisor) -> None:
        super().__init__(
            supervisor=supervisor,
            name="subagent_wait",
            description=(
                "Wait for one Subagent to reach terminal state. A wait timeout leaves the "
                "Subagent running and returns its current status."
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


class SubagentMessageTool(_SubagentControlTool):
    """Deliver a parent message through an active Subagent Engine mailbox."""

    def __init__(self, supervisor: SubagentSupervisor) -> None:
        super().__init__(
            supervisor=supervisor,
            name="subagent_message",
            description=(
                "Send context or follow-up instructions to an active Subagent. The Subagent "
                "accepts the message at its next turn safe point."
            ),
            parameters={
                "content": {
                    "type": "string",
                    "description": "The new context or instruction for the Subagent.",
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
                "Terminal subagents cannot accept messages; launch a new Subagent for "
                "follow-up work."
            )
        return payload


class SubagentInterruptTool(_SubagentControlTool):
    """Interrupt one Subagent and wait for bounded terminal cleanup."""

    def __init__(self, supervisor: SubagentSupervisor) -> None:
        super().__init__(
            supervisor=supervisor,
            name="subagent_interrupt",
            description=(
                "Request immediate cancellation of one active Subagent and wait for its "
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


class SubagentControlToolSet:
    """Expose status, wait, message, and interrupt over one SubagentSupervisor."""

    name = "subagent"
    version = "1"

    def __init__(self, supervisor: SubagentSupervisor) -> None:
        self._tools = [
            SubagentStatusTool(supervisor),
            SubagentWaitTool(supervisor),
            SubagentMessageTool(supervisor),
            SubagentInterruptTool(supervisor),
        ]

    def tools(self) -> list[BaseTool]:
        return list(self._tools)


__all__ = [
    "SubagentControlToolSet",
    "SubagentInterruptTool",
    "SubagentMessageTool",
    "SubagentStatusTool",
    "SubagentWaitTool",
]
