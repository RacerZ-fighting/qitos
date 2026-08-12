"""Tests for needs_approval feature (Task 2.5)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from qitos.core.action import Action, ActionStatus
from qitos.core.tool import ToolMeta, ToolSpec, FunctionTool, tool
from qitos.core.tool_registry import ToolRegistry
from qitos.engine.approval import ToolApprovalItem
from qitos.engine.action_executor import ActionExecutor


# ---------------------------------------------------------------------------
# 2.5.3: ToolSpec / ToolMeta / @tool propagation
# ---------------------------------------------------------------------------


class TestNeedsApprovalField:
    """Test that needs_approval is properly stored in ToolSpec and ToolMeta."""

    def test_toolspec_default_is_false(self):
        spec = ToolSpec(name="t", description="d")
        assert spec.needs_approval is False

    def test_toolspec_needs_approval_true(self):
        spec = ToolSpec(name="t", description="d", needs_approval=True)
        assert spec.needs_approval is True

    def test_toolmeta_default_is_false(self):
        meta = ToolMeta()
        assert meta.needs_approval is False

    def test_toolmeta_needs_approval_true(self):
        meta = ToolMeta(needs_approval=True)
        assert meta.needs_approval is True


class TestToolDecoratorNeedsApproval:
    """Test that @tool(needs_approval=True) propagates to ToolSpec."""

    def test_tool_decorator_needs_approval_true(self):
        @tool(needs_approval=True)
        def dangerous_op(target: str) -> str:
            """Perform a dangerous operation."""
            return f"done: {target}"

        meta = getattr(dangerous_op, "__qitos_tool_meta__", None)
        assert meta is not None
        assert meta.needs_approval is True

    def test_tool_decorator_needs_approval_default(self):
        @tool()
        def safe_op(target: str) -> str:
            """Perform a safe operation."""
            return f"done: {target}"

        meta = getattr(safe_op, "__qitos_tool_meta__", None)
        assert meta is not None
        assert meta.needs_approval is False

    def test_needs_approval_propagates_to_function_tool_spec(self):
        @tool(needs_approval=True)
        def delete_db(confirm: bool) -> str:
            """Delete the database."""
            return "deleted"

        ft = FunctionTool(delete_db)
        assert ft.spec.needs_approval is True

    def test_build_tool_spec_propagates_needs_approval(self):
        @tool(needs_approval=True)
        def deploy_app(env: str) -> str:
            """Deploy the app."""
            return f"deployed to {env}"

        from qitos.core.tool import build_tool_spec, get_tool_meta

        meta = get_tool_meta(deploy_app)
        spec = build_tool_spec(deploy_app, meta)
        assert spec.needs_approval is True


# ---------------------------------------------------------------------------
# 2.5.3: ToolApprovalItem dataclass
# ---------------------------------------------------------------------------


class TestToolApprovalItem:
    """Test ToolApprovalItem creation and fields."""

    def test_creation_with_required_fields(self):
        item = ToolApprovalItem(tool_name="delete_db")
        assert item.tool_name == "delete_db"
        assert item.tool_args == {}
        assert item.message == ""
        assert item.tool_spec is None

    def test_creation_with_all_fields(self):
        spec = ToolSpec(name="delete_db", description="Delete DB")
        item = ToolApprovalItem(
            tool_name="delete_db",
            tool_args={"confirm": True},
            message="This will delete the database!",
            tool_spec=spec,
        )
        assert item.tool_name == "delete_db"
        assert item.tool_args == {"confirm": True}
        assert item.message == "This will delete the database!"
        assert item.tool_spec is spec

    def test_default_factory_for_tool_args(self):
        item1 = ToolApprovalItem(tool_name="a")
        item2 = ToolApprovalItem(tool_name="b")
        # Each instance should get its own dict
        item1.tool_args["key"] = "val"
        assert "key" not in item2.tool_args


# ---------------------------------------------------------------------------
# 2.5.2: ActionExecutor integration
# ---------------------------------------------------------------------------


def _make_registry(tools_dict):
    """Create a real registry so executor tests exercise the canonical boundary."""
    registry = ToolRegistry()
    for item in tools_dict.values():
        registry.register(item)
    return registry


class TestActionExecutorNeedsApproval:
    """Test ActionExecutor integration with needs_approval."""

    def test_needs_approval_false_does_not_interrupt(self):
        """Tool without needs_approval should execute without interruption."""
        @tool()
        def safe_read(path: str) -> str:
            """Read a file."""
            return f"content of {path}"

        ft = FunctionTool(safe_read)
        registry = _make_registry({"safe_read": ft})
        executor = ActionExecutor(tool_registry=registry)

        action = Action(name="safe_read", args={"path": "/tmp/test.txt"})
        result = executor._execute_one(action)
        assert result.status == ActionStatus.SUCCESS

    @patch("qitos.engine.interrupt.interrupt")
    def test_needs_approval_true_triggers_interrupt(self, mock_interrupt):
        """Tool with needs_approval=True should call interrupt()."""
        mock_interrupt.return_value = "allow"

        @tool(needs_approval=True)
        def dangerous_delete(target: str) -> str:
            """Delete something."""
            return f"deleted {target}"

        ft = FunctionTool(dangerous_delete)
        registry = _make_registry({"dangerous_delete": ft})
        executor = ActionExecutor(tool_registry=registry)

        action = Action(name="dangerous_delete", args={"target": "db"})
        result = executor._execute_one(action)

        # interrupt should have been called
        assert mock_interrupt.called
        # With "allow", the tool should execute successfully
        assert result.status == ActionStatus.SUCCESS

    @patch("qitos.engine.interrupt.interrupt")
    def test_approval_deny_returns_denied(self, mock_interrupt):
        """When interrupt returns 'deny', the action should be DENIED."""
        mock_interrupt.return_value = "deny"

        @tool(needs_approval=True)
        def dangerous_delete(target: str) -> str:
            """Delete something."""
            return f"deleted {target}"

        ft = FunctionTool(dangerous_delete)
        registry = _make_registry({"dangerous_delete": ft})
        executor = ActionExecutor(tool_registry=registry)

        action = Action(name="dangerous_delete", args={"target": "db"})
        result = executor._execute_one(action)

        assert result.status == ActionStatus.DENIED
        assert result.output["message"] == "User denied approval"
        assert result.metadata.get("error_category") == "approval_denied"

    @patch("qitos.engine.interrupt.interrupt")
    def test_approval_allow_continues_execution(self, mock_interrupt):
        """When interrupt returns 'allow', the tool should execute normally."""
        mock_interrupt.return_value = "allow"

        @tool(needs_approval=True)
        def deploy_app(target_env: str) -> str:
            """Deploy the app."""
            return f"deployed to {target_env}"

        ft = FunctionTool(deploy_app)
        registry = _make_registry({"deploy_app": ft})
        executor = ActionExecutor(tool_registry=registry)

        action = Action(name="deploy_app", args={"target_env": "production"})
        result = executor._execute_one(action)

        assert result.status == ActionStatus.SUCCESS
        assert "deployed to production" in result.output

    @patch("qitos.engine.interrupt.interrupt")
    def test_interrupt_receives_tool_approval_item(self, mock_interrupt):
        """interrupt() should receive a ToolApprovalItem as its value."""
        mock_interrupt.return_value = "allow"

        @tool(needs_approval=True)
        def risky_op(x: int) -> int:
            """Risky operation."""
            return x * 2

        ft = FunctionTool(risky_op)
        registry = _make_registry({"risky_op": ft})
        executor = ActionExecutor(tool_registry=registry)

        action = Action(name="risky_op", args={"x": 5})
        executor._execute_one(action)

        # Check that interrupt was called with a ToolApprovalItem
        call_args = mock_interrupt.call_args
        approval_item = call_args[0][0]
        assert isinstance(approval_item, ToolApprovalItem)
        assert approval_item.tool_name == "risky_op"
        assert approval_item.tool_args == {"x": 5}
        assert "requires approval" in approval_item.message

    def test_callable_needs_approval_is_rejected(self):
        """Approval policy is a runtime concern, not an overloaded spec field."""

        def conditional_approval(runtime_context, args):
            # Only require approval for production environment
            return args.get("target_env") == "production"

        @tool(needs_approval=conditional_approval)
        def deploy(target_env: str) -> str:
            """Deploy."""
            return f"deployed to {target_env}"

        with pytest.raises(TypeError, match="needs_approval must be a boolean"):
            FunctionTool(deploy)

    def test_unknown_tool_no_approval_check(self):
        """If tool is not in registry, no approval check should be attempted."""
        registry = _make_registry({})
        executor = ActionExecutor(tool_registry=registry)

        action = Action(name="nonexistent", args={})
        with patch("qitos.engine.interrupt.interrupt") as mock_interrupt:
            result = executor._execute_one(action)
            mock_interrupt.assert_not_called()
            assert result.status is ActionStatus.ERROR
            assert result.metadata["error_category"] == "tool_not_found"
