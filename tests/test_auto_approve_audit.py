"""Tests for auto_approve audit trail in ActionExecutor."""
from __future__ import annotations

from typing import Any, Dict

import pytest

from qitos.core.action import Action, ActionStatus
from qitos.core.function_tool_decorator import function_tool
from qitos.core.tool_registry import ToolRegistry
from qitos.engine.action_executor import ActionExecutor
from qitos.engine.interrupt import EngineInterrupt


# --- Test tools ---


class _AuditToolSet:
    """Toolset with needs_approval and regular tools for audit testing."""

    name = "audit_test"
    version = "1"

    def setup(self, context):
        pass

    def teardown(self, context):
        pass

    def tools(self):
        return [
            self.approval_tool,
            self.normal_tool,
        ]

    @function_tool(
        name="dangerous_action",
        description="An action requiring approval",
        needs_approval=True,
    )
    def approval_tool(self, command: str) -> Dict[str, Any]:
        return {
            "status": "success",
            "domain_outcome": "executed",
            "command": command,
        }

    @function_tool(
        name="safe_action",
        description="A safe tool without approval",
    )
    def normal_tool(self, value: str) -> Dict[str, Any]:
        return {"status": "success", "value": value}


# --- Tests ---


class TestAutoApproveAudit:
    """Audit trail tests for auto_approve bypass in ActionExecutor."""

    def test_auto_approve_adds_audit_metadata(self):
        """When auto_approve=True and tool has needs_approval=True,
        the result metadata contains auto_approved=True and approval_required=True."""
        ts = _AuditToolSet()
        registry = ToolRegistry().register_toolset(ts, namespace="")
        executor = ActionExecutor(tool_registry=registry, auto_approve=True)

        action = Action(name="dangerous_action", args={"command": "echo hello"}, kind="tool")
        result = executor._execute_one(action)

        assert result.status == ActionStatus.SUCCESS
        assert result.metadata.get("auto_approved") is True
        assert result.metadata.get("approval_required") is True

    def test_auto_approve_no_metadata_for_non_approval_tools(self):
        """When auto_approve=True but tool does NOT need approval,
        the result metadata does NOT contain auto_approved or approval_required."""
        ts = _AuditToolSet()
        registry = ToolRegistry().register_toolset(ts, namespace="")
        executor = ActionExecutor(tool_registry=registry, auto_approve=True)

        action = Action(name="safe_action", args={"value": "test"}, kind="tool")
        result = executor._execute_one(action)

        assert result.status == ActionStatus.SUCCESS
        assert "auto_approved" not in result.metadata
        assert "approval_required" not in result.metadata

    def test_no_auto_approve_raises_interrupt(self):
        """When auto_approve=False, needs_approval tools still raise EngineInterrupt
        (the normal flow is unchanged)."""
        ts = _AuditToolSet()
        registry = ToolRegistry().register_toolset(ts, namespace="")
        executor = ActionExecutor(tool_registry=registry, auto_approve=False)

        action = Action(name="dangerous_action", args={"command": "rm -rf /"}, kind="tool")
        with pytest.raises(EngineInterrupt):
            executor._execute_one(action)

    def test_no_auto_approve_no_audit_for_non_approval_tools(self):
        """When auto_approve=False and tool does not need approval,
        no audit metadata is added (tool executes normally)."""
        ts = _AuditToolSet()
        registry = ToolRegistry().register_toolset(ts, namespace="")
        executor = ActionExecutor(tool_registry=registry, auto_approve=False)

        action = Action(name="safe_action", args={"value": "test"}, kind="tool")
        result = executor._execute_one(action)

        assert result.status == ActionStatus.SUCCESS
        assert "auto_approved" not in result.metadata
        assert "approval_required" not in result.metadata

    def test_execute_batch_auto_approve_audit(self):
        """Batch execute with auto_approve=True includes audit metadata
        only on tools that require approval."""
        ts = _AuditToolSet()
        registry = ToolRegistry().register_toolset(ts, namespace="")
        executor = ActionExecutor(
            tool_registry=registry,
            auto_approve=True,
        )

        actions = [
            Action(name="safe_action", args={"value": "first"}, kind="tool"),
            Action(name="dangerous_action", args={"command": "echo hello"}, kind="tool"),
        ]
        results = executor.execute(actions)

        assert len(results) == 2

        # safe_action: no audit metadata
        assert results[0].status == ActionStatus.SUCCESS
        assert "auto_approved" not in results[0].metadata
        assert "approval_required" not in results[0].metadata

        # dangerous_action: audit metadata present
        assert results[1].status == ActionStatus.SUCCESS
        assert results[1].metadata.get("auto_approved") is True
        assert results[1].metadata.get("approval_required") is True
