"""Tests for ToolInterceptor protocol, InterceptorChain, and built-in interceptors."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from qitos.core.action import Action, ActionResult, ActionKind, ActionStatus
from qitos.core.interceptor import InterceptorChain, InterceptorContext, ToolInterceptor
from qitos.core.tool import BaseTool, ToolSpec
from qitos.core.tool_registry import ToolRegistry
from qitos.engine.action_executor import ActionExecutor
from qitos.kit.interceptor.cache import CacheInterceptor
from qitos.kit.interceptor.logging import LoggingInterceptor
from qitos.kit.interceptor.retry import RetryInterceptor


# ── Helpers ──────────────────────────────────────────────────────────────────


class _TracingInterceptor(ToolInterceptor):
    """Records calls for testing ordering."""

    def __init__(self, name: str):
        self.name = name
        self.calls: List[str] = []

    def before_execute(self, action: Action, context: InterceptorContext) -> Action:
        self.calls.append(f"before:{self.name}")
        return action

    def after_execute(
        self, action: Action, result: ActionResult, context: InterceptorContext
    ) -> ActionResult:
        self.calls.append(f"after:{self.name}")
        return result


class _MutatingInterceptor(ToolInterceptor):
    """Modifies action args and result for testing."""

    def before_execute(self, action: Action, context: InterceptorContext) -> Action:
        new_args = dict(action.args)
        new_args["injected_by"] = "mutating"
        return Action(
            name=action.name,
            args=new_args,
            kind=action.kind,
            action_id=action.action_id,
            timeout_s=action.timeout_s,
            max_retries=action.max_retries,
            idempotent=action.idempotent,
            classification=action.classification,
            metadata=action.metadata,
        )

    def after_execute(
        self, action: Action, result: ActionResult, context: InterceptorContext
    ) -> ActionResult:
        new_meta = dict(result.metadata)
        new_meta["modified_by"] = "mutating"
        return ActionResult(
            name=result.name,
            status=result.status,
            output=result.output,
            error=result.error,
            action_id=result.action_id,
            attempts=result.attempts,
            latency_ms=result.latency_ms,
            metadata=new_meta,
        )


def _make_action(name: str = "test_tool", args: Optional[Dict] = None) -> Action:
    return Action(name=name, args=args or {})


def _make_result(
    name: str = "test_tool",
    status: ActionStatus = ActionStatus.SUCCESS,
    output: Any = None,
) -> ActionResult:
    return ActionResult(name=name, status=status, output=output)


def _make_context(
    tool_name: str = "test_tool",
    tool_args: Optional[Dict] = None,
    step_id: int = 0,
) -> InterceptorContext:
    return InterceptorContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        step_id=step_id,
    )


def _make_tool(
    name: str = "test_tool",
    read_only: bool = False,
    func: Optional[Any] = None,
) -> BaseTool:
    spec = ToolSpec(
        name=name,
        description=f"Test tool {name}",
        read_only=read_only,
    )
    if func is not None:

        class _FuncTool(BaseTool):
            def execute(self, args, runtime_context=None):
                return func(args)

        tool = _FuncTool(spec)
        return tool

    class _NoopTool(BaseTool):
        def execute(self, args, runtime_context=None):
            _ = args, runtime_context
            return None

    return _NoopTool(spec)


# ── ToolInterceptor Protocol Tests ──────────────────────────────────────────


class TestToolInterceptorProtocol:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            ToolInterceptor()

    def test_subclass_must_implement_both_methods(self):
        class _Incomplete(ToolInterceptor):
            def before_execute(self, action, context):
                return action

        with pytest.raises(TypeError):
            _Incomplete()

    def test_complete_subclass_instantiates(self):
        class _Complete(ToolInterceptor):
            def before_execute(self, action, context):
                return action

            def after_execute(self, action, result, context):
                return result

        obj = _Complete()
        assert isinstance(obj, ToolInterceptor)


# ── InterceptorContext Tests ────────────────────────────────────────────────


class TestInterceptorContext:
    def test_fields(self):
        ctx = InterceptorContext(
            tool_name="my_tool",
            tool_args={"x": 1},
            step_id=5,
            state="some_state",
            run_id="run_abc",
        )
        assert ctx.tool_name == "my_tool"
        assert ctx.tool_args == {"x": 1}
        assert ctx.step_id == 5
        assert ctx.state == "some_state"
        assert ctx.run_id == "run_abc"

    def test_defaults(self):
        ctx = InterceptorContext(tool_name="t", tool_args={}, step_id=0)
        assert ctx.state is None
        assert ctx.run_id == ""


# ── InterceptorChain Tests ─────────────────────────────────────────────────


class TestInterceptorChain:
    def test_empty_chain_passes_through(self):
        chain = InterceptorChain()
        action = _make_action()
        context = _make_context()
        result = chain.before_execute(action, context)
        assert result is action

    def test_before_execute_runs_in_order(self):
        a = _TracingInterceptor("A")
        b = _TracingInterceptor("B")
        chain = InterceptorChain([a, b])
        chain.before_execute(_make_action(), _make_context())
        assert a.calls == ["before:A"]
        assert b.calls == ["before:B"]

    def test_after_execute_runs_in_reverse_order(self):
        a = _TracingInterceptor("A")
        b = _TracingInterceptor("B")
        chain = InterceptorChain([a, b])
        chain.after_execute(_make_action(), _make_result(), _make_context())
        # Reverse order: B first, then A
        assert a.calls == ["after:A"]
        assert b.calls == ["after:B"]
        # But B should be called before A in time
        # We can verify by checking both calls in one list
        a2 = _TracingInterceptor("A")
        b2 = _TracingInterceptor("B")
        shared_calls: List[str] = []
        a2.calls = shared_calls
        b2.calls = shared_calls
        chain2 = InterceptorChain([a2, b2])
        chain2.after_execute(_make_action(), _make_result(), _make_context())
        assert shared_calls == ["after:B", "after:A"]

    def test_add_appends_interceptor(self):
        chain = InterceptorChain()
        t = _TracingInterceptor("X")
        chain.add(t)
        assert len(chain.interceptors) == 1
        chain.before_execute(_make_action(), _make_context())
        assert t.calls == ["before:X"]

    def test_mutation_in_before(self):
        chain = InterceptorChain([_MutatingInterceptor()])
        action = _make_action(args={"original": True})
        result = chain.before_execute(action, _make_context())
        assert result.args.get("injected_by") == "mutating"
        assert result.args.get("original") is True

    def test_mutation_in_after(self):
        chain = InterceptorChain([_MutatingInterceptor()])
        result = chain.after_execute(_make_action(), _make_result(), _make_context())
        assert result.metadata.get("modified_by") == "mutating"


# ── RetryInterceptor Tests ─────────────────────────────────────────────────


class TestRetryInterceptor:
    def test_before_execute_increases_max_retries(self):
        ri = RetryInterceptor(max_retries=5)
        action = _make_action()
        assert action.max_retries == 0
        modified = ri.before_execute(action, _make_context())
        assert modified.max_retries == 5

    def test_before_execute_preserves_higher_existing_retries(self):
        ri = RetryInterceptor(max_retries=2)
        action = _make_action()
        action = Action(
            name="test_tool",
            args={},
            max_retries=10,
        )
        modified = ri.before_execute(action, _make_context())
        assert modified.max_retries == 10

    def test_before_execute_copies_action_fields(self):
        ri = RetryInterceptor(max_retries=3)
        action = Action(
            name="my_tool",
            args={"x": 1},
            kind=ActionKind.TOOL,
            action_id="abc",
            timeout_s=5.0,
            max_retries=0,
            idempotent=False,
            classification="high",
            metadata={"key": "val"},
        )
        modified = ri.before_execute(action, _make_context())
        assert modified.name == "my_tool"
        assert modified.args == {"x": 1}
        assert modified.action_id == "abc"
        assert modified.timeout_s == 5.0
        assert modified.idempotent is False
        assert modified.classification == "high"

    def test_after_execute_adds_metadata(self):
        ri = RetryInterceptor(max_retries=3)
        action = _make_action()
        # First run before to set metadata
        action = ri.before_execute(action, _make_context())
        result = _make_result()
        result = ri.after_execute(action, result, _make_context())
        assert result.metadata.get("retry_interceptor_max") == 3

    def test_default_parameters(self):
        ri = RetryInterceptor()
        assert ri.max_retries == 2
        assert ri.retry_on_exception is True
        assert ri.backoff_factor == 1.0


# ── CacheInterceptor Tests ─────────────────────────────────────────────────


class TestCacheInterceptor:
    def test_cache_key_deterministic(self):
        key1 = CacheInterceptor._cache_key("tool", {"a": 1, "b": 2})
        key2 = CacheInterceptor._cache_key("tool", {"b": 2, "a": 1})
        assert key1 == key2  # order-independent

    def test_different_keys_for_different_tools(self):
        key1 = CacheInterceptor._cache_key("tool_a", {"x": 1})
        key2 = CacheInterceptor._cache_key("tool_b", {"x": 1})
        assert key1 != key2

    def test_caches_read_only_successful_result(self):
        cache = CacheInterceptor()
        # Simulate a read-only tool via engine context
        engine_mock = MagicMock()
        tool_mock = MagicMock()
        tool_mock.spec.read_only = True
        engine_mock.tool_registry = MagicMock()
        engine_mock.tool_registry.get.return_value = tool_mock

        ctx = InterceptorContext(
            tool_name="read_tool",
            tool_args={"q": "hello"},
            step_id=0,
            state=engine_mock,
        )
        action = _make_action("read_tool", {"q": "hello"})
        action = cache.before_execute(action, ctx)
        # Not a cache hit yet
        assert action.metadata.get("_cache_hit") is False

        # Now simulate a successful result
        result = _make_result("read_tool", ActionStatus.SUCCESS, output="cached_value")
        result = cache.after_execute(action, result, ctx)
        assert len(cache._cache) == 1

        # Second call should be a cache hit
        action2 = _make_action("read_tool", {"q": "hello"})
        action2 = cache.before_execute(action2, ctx)
        assert action2.metadata.get("_cache_hit") is True

        result2 = ActionResult(
            name="read_tool",
            status=ActionStatus.SUCCESS,
            output="should_be_replaced",
        )
        result2 = cache.after_execute(action2, result2, ctx)
        assert result2.output == "cached_value"

    def test_does_not_cache_non_read_only(self):
        cache = CacheInterceptor()
        engine_mock = MagicMock()
        tool_mock = MagicMock()
        tool_mock.spec.read_only = False
        engine_mock.tool_registry = MagicMock()
        engine_mock.tool_registry.get.return_value = tool_mock

        ctx = InterceptorContext(
            tool_name="write_tool",
            tool_args={},
            step_id=0,
            state=engine_mock,
        )
        action = _make_action("write_tool")
        action = cache.before_execute(action, ctx)
        assert action.metadata.get("_cache_read_only") is False

        result = _make_result("write_tool", ActionStatus.SUCCESS, output="data")
        result = cache.after_execute(action, result, ctx)
        assert len(cache._cache) == 0

    def test_does_not_cache_error_result(self):
        cache = CacheInterceptor()
        engine_mock = MagicMock()
        tool_mock = MagicMock()
        tool_mock.spec.read_only = True
        engine_mock.tool_registry = MagicMock()
        engine_mock.tool_registry.get.return_value = tool_mock

        ctx = InterceptorContext(
            tool_name="read_tool",
            tool_args={},
            step_id=0,
            state=engine_mock,
        )
        action = _make_action("read_tool")
        action = cache.before_execute(action, ctx)

        result = _make_result("read_tool", ActionStatus.ERROR, output=None)
        result.error = "something failed"
        result = cache.after_execute(action, result, ctx)
        assert len(cache._cache) == 0

    def test_max_size_eviction(self):
        cache = CacheInterceptor(max_size=2)
        engine_mock = MagicMock()
        tool_mock = MagicMock()
        tool_mock.spec.read_only = True
        engine_mock.tool_registry = MagicMock()
        engine_mock.tool_registry.get.return_value = tool_mock

        for i in range(3):
            ctx = InterceptorContext(
                tool_name=f"tool_{i}",
                tool_args={"i": i},
                step_id=i,
                state=engine_mock,
            )
            action = _make_action(f"tool_{i}", {"i": i})
            action = cache.before_execute(action, ctx)
            result = _make_result(f"tool_{i}", ActionStatus.SUCCESS, output=i)
            cache.after_execute(action, result, ctx)

        assert len(cache._cache) == 2
        # First entry should have been evicted
        assert CacheInterceptor._cache_key("tool_0", {"i": 0}) not in cache._cache

    def test_clear(self):
        cache = CacheInterceptor()
        cache._cache["key"] = _make_result()
        cache._insertion_order.append("key")
        cache.clear()
        assert len(cache._cache) == 0
        assert len(cache._insertion_order) == 0


# ── LoggingInterceptor Tests ───────────────────────────────────────────────


class TestLoggingInterceptor:
    def test_before_execute_logs(self, caplog):
        li = LoggingInterceptor(log_args=True)
        action = _make_action("my_tool", {"x": 1})
        ctx = _make_context(tool_name="my_tool", tool_args={"x": 1}, step_id=3)
        with caplog.at_level(logging.INFO, logger="qitos.interceptor.logging"):
            li.before_execute(action, ctx)
        assert "before" in caplog.text
        assert "my_tool" in caplog.text

    def test_after_execute_logs(self, caplog):
        li = LoggingInterceptor(log_args=True)
        action = _make_action("my_tool")
        result = _make_result("my_tool", ActionStatus.SUCCESS, output="ok")
        ctx = _make_context(tool_name="my_tool")
        with caplog.at_level(logging.INFO, logger="qitos.interceptor.logging"):
            li.after_execute(action, result, ctx)
        assert "after" in caplog.text
        assert "success" in caplog.text

    def test_log_args_false_hides_args(self, caplog):
        li = LoggingInterceptor(log_args=False)
        action = _make_action("my_tool", {"secret": "val"})
        ctx = _make_context(tool_name="my_tool", tool_args={"secret": "val"})
        with caplog.at_level(logging.INFO, logger="qitos.interceptor.logging"):
            li.before_execute(action, ctx)
        # The args dict itself should not appear
        assert "secret" not in caplog.text

    def test_log_result_includes_output(self, caplog):
        li = LoggingInterceptor(log_result=True)
        action = _make_action("my_tool")
        result = _make_result("my_tool", ActionStatus.SUCCESS, output="special_output")
        ctx = _make_context(tool_name="my_tool")
        with caplog.at_level(logging.INFO, logger="qitos.interceptor.logging"):
            li.after_execute(action, result, ctx)
        assert "special_output" in caplog.text

    def test_log_result_false_hides_output(self, caplog):
        li = LoggingInterceptor(log_result=False)
        action = _make_action("my_tool")
        result = _make_result("my_tool", ActionStatus.SUCCESS, output="big_output_data")
        ctx = _make_context(tool_name="my_tool")
        with caplog.at_level(logging.INFO, logger="qitos.interceptor.logging"):
            li.after_execute(action, result, ctx)
        assert "big_output_data" not in caplog.text

    def test_callback_receives_events(self):
        events: List[Dict] = []

        def cb(event: str, **kwargs):
            events.append({"event": event, **kwargs})

        li = LoggingInterceptor(callback=cb, log_args=True, log_result=True)
        action = _make_action("tool1", {"a": 1})
        ctx = _make_context(tool_name="tool1", step_id=2, tool_args={"a": 1})
        li.before_execute(action, ctx)
        result = _make_result("tool1", ActionStatus.SUCCESS, output="out")
        li.after_execute(action, result, ctx)

        assert len(events) == 2
        assert events[0]["event"] == "before"
        assert events[0]["tool_name"] == "tool1"
        assert events[1]["event"] == "after"
        assert events[1]["tool_name"] == "tool1"
        assert events[1]["result"] == "out"

    def test_callback_error_does_not_propagate(self):
        def bad_cb(event, **kwargs):
            raise RuntimeError("callback blew up")

        li = LoggingInterceptor(callback=bad_cb)
        action = _make_action()
        ctx = _make_context()
        # Should not raise
        li.before_execute(action, ctx)
        result = _make_result()
        li.after_execute(action, result, ctx)

    def test_custom_logger(self, caplog):
        custom_logger = logging.getLogger("test.custom.logger")
        li = LoggingInterceptor(logger=custom_logger, log_args=True)
        action = _make_action("xtool")
        ctx = _make_context(tool_name="xtool")
        with caplog.at_level(logging.INFO, logger="test.custom.logger"):
            li.before_execute(action, ctx)
        assert "xtool" in caplog.text


# ── ActionExecutor Integration Tests ───────────────────────────────────────


class _FakeTool(BaseTool):
    """A simple test tool that returns a fixed result."""

    def __init__(self, name: str, result: Any = "ok", read_only: bool = False, side_effect=None):
        spec = ToolSpec(name=name, description=f"Test {name}", read_only=read_only)
        super().__init__(spec)
        self._result = result
        self._side_effect = side_effect
        self.call_args_list: List[Dict] = []

    def execute(self, args, runtime_context=None):
        self.call_args_list.append(dict(args))
        if self._side_effect is not None:
            raise self._side_effect
        return self._result


def _make_executor(
    tools: Optional[Dict[str, BaseTool]] = None,
    interceptor_chain: Optional[InterceptorChain] = None,
) -> ActionExecutor:
    registry = ToolRegistry()
    for tool in (tools or {}).values():
        registry.register(tool)
    return ActionExecutor(
        tool_registry=registry,
        interceptor_chain=interceptor_chain,
    )


class TestActionExecutorIntegration:
    def test_interceptor_chain_called_on_success(self):
        """Verify before_execute and after_execute are called during tool execution."""
        trace = _TracingInterceptor("integration")
        chain = InterceptorChain([trace])

        tool = _FakeTool("mock_tool", result="result_data")
        executor = _make_executor({"mock_tool": tool}, interceptor_chain=chain)

        action = Action(name="mock_tool", args={"key": "value"})
        results = executor.execute([action])
        assert len(results) == 1
        assert results[0].status == ActionStatus.SUCCESS
        assert results[0].output == "result_data"
        assert "before:integration" in trace.calls
        assert "after:integration" in trace.calls

    def test_interceptor_can_modify_action_args(self):
        """Verify before_execute can modify action args before the tool is called."""

        class _ArgInjector(ToolInterceptor):
            def before_execute(self, action, context):
                new_args = dict(action.args)
                new_args["injected"] = True
                return Action(
                    name=action.name,
                    args=new_args,
                    kind=action.kind,
                    action_id=action.action_id,
                    timeout_s=action.timeout_s,
                    max_retries=action.max_retries,
                    idempotent=action.idempotent,
                    classification=action.classification,
                    metadata=action.metadata,
                )

            def after_execute(self, action, result, context):
                return result

        chain = InterceptorChain([_ArgInjector()])
        tool = _FakeTool("inject_tool", result="ok")
        executor = _make_executor({"inject_tool": tool}, interceptor_chain=chain)

        action = Action(name="inject_tool", args={"original": 1})
        results = executor.execute([action])
        assert len(results) == 1
        assert results[0].status == ActionStatus.SUCCESS
        # The tool should have received the injected arg
        assert tool.call_args_list[0].get("injected") is True
        assert tool.call_args_list[0].get("original") == 1

    def test_interceptor_can_modify_result(self):
        """Verify after_execute can modify the ActionResult."""

        class _ResultModifier(ToolInterceptor):
            def before_execute(self, action, context):
                return action

            def after_execute(self, action, result, context):
                new_meta = dict(result.metadata)
                new_meta["modified"] = True
                return ActionResult(
                    name=result.name,
                    status=result.status,
                    output="modified_output",
                    error=result.error,
                    action_id=result.action_id,
                    attempts=result.attempts,
                    latency_ms=result.latency_ms,
                    metadata=new_meta,
                )

        chain = InterceptorChain([_ResultModifier()])
        tool = _FakeTool("mod_tool", result="original_output")
        executor = _make_executor({"mod_tool": tool}, interceptor_chain=chain)

        action = Action(name="mod_tool", args={})
        results = executor.execute([action])
        assert len(results) == 1
        assert results[0].output == "modified_output"
        assert results[0].metadata.get("modified") is True

    def test_no_interceptor_chain_works_as_before(self):
        """Verify executor works without an interceptor chain (backward compat)."""
        tool = _FakeTool("plain_tool", result="plain_result")
        executor = _make_executor({"plain_tool": tool})

        action = Action(name="plain_tool", args={})
        results = executor.execute([action])
        assert len(results) == 1
        assert results[0].status == ActionStatus.SUCCESS
        assert results[0].output == "plain_result"

    def test_after_execute_called_on_error(self):
        """Verify after_execute is also called on error results."""

        class _ErrorTracker(ToolInterceptor):
            def __init__(self):
                self.after_called = False
                self.after_status: Optional[ActionStatus] = None

            def before_execute(self, action, context):
                return action

            def after_execute(self, action, result, context):
                self.after_called = True
                self.after_status = result.status
                return result

        tracker = _ErrorTracker()
        chain = InterceptorChain([tracker])

        # Use a tool that raises an exception
        tool = _FakeTool("failing_tool", side_effect=RuntimeError("boom"))
        executor = _make_executor({"failing_tool": tool}, interceptor_chain=chain)

        action = Action(name="failing_tool", args={})
        results = executor.execute([action])
        assert len(results) == 1
        assert results[0].status == ActionStatus.ERROR
        assert tracker.after_called is True
        assert tracker.after_status == ActionStatus.ERROR

    def test_multiple_interceptors_order(self):
        """Verify multiple interceptors run in correct order."""
        a = _TracingInterceptor("A")
        b = _TracingInterceptor("B")
        chain = InterceptorChain([a, b])

        tool = _FakeTool("order_tool", result="ok")
        executor = _make_executor({"order_tool": tool}, interceptor_chain=chain)

        action = Action(name="order_tool", args={})
        executor.execute([action])

        # before: A then B; after: B then A
        shared: List[str] = []
        a.calls = shared
        b.calls = shared
        executor.execute([Action(name="order_tool", args={})])
        assert shared == [
            "before:A", "before:B",
            "after:B", "after:A",
        ]
