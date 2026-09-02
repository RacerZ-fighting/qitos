"""Model-facing control tools for one Run-owned SubagentSupervisor."""

from __future__ import annotations

import math
from typing import Any

from ....core.subagent import (
    SubagentHandle,
    SubagentMessageRequest,
    SubagentResult,
    SubagentStatus,
)
from ....core.tool import BaseTool, ToolPermission, ToolSpec
from ....core.tool_result import ToolResult
from ...subagent import SubagentSupervisor
from ..internal.results import tool_result

# A wait a caller did not size for itself: long enough to catch a Subagent that
# is about to finish, short enough that the caller keeps its own turn.
WAIT_DEFAULT_TIMEOUT_SECONDS = 30.0
# The ceiling a caller may ask for. Waiting is always a caller's own cost, so
# the cap only stops one call from eating a Run.
WAIT_MAX_TIMEOUT_SECONDS = 600.0


class _SubagentControlTool(BaseTool):
    def __init__(
        self,
        *,
        supervisor: SubagentSupervisor,
        name: str,
        description: str,
        parameters: dict[str, dict[str, Any]],
        required: list[str],
        subagent_id_required: bool = True,
    ) -> None:
        if not isinstance(supervisor, SubagentSupervisor):
            raise TypeError("supervisor must be a SubagentSupervisor")
        self._supervisor = supervisor
        subagent_id_description = "The subagent id returned by `subagent`."
        if not subagent_id_required:
            subagent_id_description += (
                " Omit it to apply this call to all Subagents of the current Run."
            )
        super().__init__(
            ToolSpec(
                name=name,
                description=description,
                parameters={
                    "subagent_id": {
                        "type": "string",
                        "description": subagent_id_description,
                    },
                    **parameters,
                },
                required=(
                    ["subagent_id", *required]
                    if subagent_id_required
                    else list(required)
                ),
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

    def _capacity(
        self,
        runtime_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Project admission capacity from the ledger this turn exposes."""

        return self._supervisor.capacity_payload(
            (runtime_context or {}).get("budget_ledger")
        )

    def _projection(
        self,
        result: SubagentResult,
        runtime_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = SubagentSupervisor.result_payload(result)
        payload["status"] = "success"
        # The parent sizes its next front from whatever this call just changed,
        # so admission capacity travels with the answer instead of waiting for
        # a separate projection to be re-rendered into the prompt.
        payload["capacity"] = self._capacity(runtime_context)
        return payload

    def _unknown(
        self,
        handle: SubagentHandle,
        runtime_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "status": "success",
            "subagent_status": SubagentStatus.UNKNOWN.value,
            "ready": True,
            "handle": handle.to_dict(),
            "subagent_id": handle.subagent_id,
            "output": "No subagent with this handle belongs to the current Run.",
            "capacity": self._capacity(runtime_context),
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
        return (
            self._unknown(handle, runtime_context)
            if result is None
            else self._projection(result, runtime_context)
        )


class SubagentWaitTool(_SubagentControlTool):
    """Wait a bounded time for Subagent terminal state without cancelling it."""

    def __init__(
        self,
        supervisor: SubagentSupervisor,
        *,
        max_timeout_seconds: float = WAIT_MAX_TIMEOUT_SECONDS,
    ) -> None:
        if isinstance(max_timeout_seconds, bool) or not isinstance(
            max_timeout_seconds, (int, float)
        ):
            raise TypeError("max_timeout_seconds must be a number")
        resolved_max_timeout = float(max_timeout_seconds)
        if not math.isfinite(resolved_max_timeout) or resolved_max_timeout <= 0:
            raise ValueError("max_timeout_seconds must be finite and positive")
        self._max_timeout_seconds = resolved_max_timeout
        self._default_timeout_seconds = min(
            WAIT_DEFAULT_TIMEOUT_SECONDS,
            resolved_max_timeout,
        )
        super().__init__(
            supervisor=supervisor,
            name="subagent_wait",
            description=(
                "Wait for one Subagent to reach terminal state, or for the next "
                "Subagent terminal state in this Run when subagent_id is omitted. "
                "A wait timeout leaves every Subagent running and returns its "
                "current status. A background Subagent delivers its own terminal "
                "as a message, so a wait never causes a result to arrive; it only "
                "trades the caller's own turns for hearing about it sooner. Wait "
                "when the next thing you would do depends on that result, and for "
                "no longer than it takes to have something else worth doing."
            ),
            parameters={
                "timeout_seconds": {
                    "type": "number",
                    "description": (
                        "Maximum wait, from 0 to "
                        f"{resolved_max_timeout:g} seconds; "
                        f"{self._default_timeout_seconds:g} when omitted."
                    ),
                    "minimum": 0,
                    "maximum": resolved_max_timeout,
                }
            },
            required=[],
            subagent_id_required=False,
        )

    async def execute(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | ToolResult:
        subagent_id = str(args.get("subagent_id") or "").strip()
        try:
            timeout = min(
                self._max_timeout_seconds,
                self._timeout(
                    args,
                    runtime_context,
                    default=self._default_timeout_seconds,
                ),
            )
            if subagent_id:
                handle = self._handle(args, runtime_context)
                result = await self._supervisor.wait(
                    handle,
                    timeout_seconds=timeout,
                )
            else:
                result = await self._supervisor.wait_any(timeout_seconds=timeout)
        except (TypeError, ValueError) as exc:
            return tool_result(
                {"status": "error", "error": str(exc)}, status="error"
            )
        if subagent_id:
            return (
                self._unknown(handle, runtime_context)
                if result is None
                else self._projection(result, runtime_context)
            )
        if result is not None:
            return self._projection(result, runtime_context)
        return self._wait_any_pending(timeout, runtime_context)

    def _wait_any_pending(
        self,
        timeout: float,
        runtime_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        running = [
            result.handle.subagent_id
            for result in self._supervisor.active_results()
        ]
        if not running:
            output = "No Subagent belongs to the current Run."
        else:
            output = (
                f"No Subagent reached terminal state within {timeout:g} seconds; "
                f"still running: {', '.join(running)}."
            )
        return {
            "status": "success",
            "ready": False,
            "output": output,
            "subagent_ids": running,
            "capacity": self._capacity(runtime_context),
        }


class SubagentMessageTool(_SubagentControlTool):
    """Deliver a parent message through an active Subagent Engine mailbox."""

    def __init__(self, supervisor: SubagentSupervisor) -> None:
        super().__init__(
            supervisor=supervisor,
            name="subagent_message",
            description=(
                "Send context or follow-up instructions to an active Subagent. The message "
                "is queued to the Subagent's mailbox and delivered at its next turn "
                "safe point."
            ),
            parameters={
                "content": {
                    "type": "string",
                    "description": "The new context or instruction for the Subagent.",
                },
                "resource_refs": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                    "description": (
                        "Stable references to resources the Subagent should receive "
                        "with this message."
                    ),
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
            raw_resource_refs = args.get("resource_refs", [])
            if not isinstance(raw_resource_refs, list):
                raise TypeError("resource_refs must be an array")
            request = SubagentMessageRequest(
                content=args.get("content"),
                resource_refs=tuple(raw_resource_refs),
            )
            timeout = min(
                30.0,
                self._timeout(args, runtime_context, default=5.0),
            )
            accepted, result = await self._supervisor.message(
                handle,
                request.content,
                resource_refs=request.resource_refs,
                timeout_seconds=timeout,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            return tool_result(
                {"status": "error", "error": str(exc)}, status="error"
            )
        if result is None:
            payload = self._unknown(handle, runtime_context)
        else:
            payload = self._projection(result, runtime_context)
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
        return (
            self._unknown(handle, runtime_context)
            if result is None
            else self._projection(result, runtime_context)
        )


class SubagentControlToolSet:
    """Expose status, wait, message, and interrupt over one SubagentSupervisor."""

    name = "subagent"
    version = "1"

    def __init__(
        self,
        supervisor: SubagentSupervisor,
        *,
        wait_max_timeout_seconds: float = WAIT_MAX_TIMEOUT_SECONDS,
    ) -> None:
        self._tools = [
            SubagentStatusTool(supervisor),
            SubagentWaitTool(
                supervisor,
                max_timeout_seconds=wait_max_timeout_seconds,
            ),
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
