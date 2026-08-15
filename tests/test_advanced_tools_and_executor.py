from __future__ import annotations

import asyncio
from dataclasses import dataclass
import threading
import time
from uuid import uuid4

import pytest

from qitos import Action, StateSchema, ToolPermissionContext, ToolPermissionRule, ToolRegistry
from qitos.core.action import ActionExecutionPolicy, ActionStatus
from qitos.core.tool import BaseTool, ToolPermission, ToolSpec, ToolValidationResult
from qitos.engine.action_executor import ActionExecutor
from qitos.kit.toolset import advanced_coding_tools
from qitos.kit.tool import (
    AskUserChoiceTool,
    CodingToolSet,
    LSPQueryTool,
    MCPListResourcesTool,
    MCPReadResourceTool,
    UpdateWorkPlanTool,
    ToolSearchTool,
)
from qitos.kit.tool.file import EditFile, ReadFile
from qitos.kit.tool.shell import RunCommand


pytestmark = pytest.mark.asyncio


class _EchoTool(BaseTool):
    def __init__(self, name: str = "echo_tool"):
        super().__init__(
            ToolSpec(
                name=name,
                description="demo tool",
                parameters={"value": {"type": "string"}},
                required=["value"],
                permissions=ToolPermission(),
            )
        )

    def validate_input(self, args, runtime_context=None):
        _ = runtime_context
        if str(args.get("value", "")) == "bad":
            return ToolValidationResult.fail("bad input", code="bad_input")
        return ToolValidationResult.ok()

    async def execute(self, args, runtime_context=None):
        _ = runtime_context
        return {"result": args["value"]}


class _SleepReadTool(BaseTool):
    def __init__(self, name: str = "sleep_read_tool", delay: float = 0.15):
        self.delay = delay
        self.starts: list[float] = []
        self._lock = threading.Lock()
        super().__init__(
            ToolSpec(
                name=name,
                description="sleepy read-only tool",
                parameters={"value": {"type": "string"}},
                required=["value"],
                permissions=ToolPermission(filesystem_read=True),
                read_only=True,
                concurrency_safe=True,
            )
        )

    async def execute(self, args, runtime_context=None):
        _ = runtime_context
        with self._lock:
            self.starts.append(time.perf_counter())
        await asyncio.sleep(self.delay)
        return {"value": args["value"]}


class _UnsafeSleepTool(BaseTool):
    def __init__(self, name: str = "unsafe_sleep_tool", delay: float = 0.05):
        self.delay = delay
        self.starts: list[float] = []
        self._lock = threading.Lock()
        super().__init__(
            ToolSpec(
                name=name,
                description="sleepy non-concurrency-safe tool",
                parameters={"value": {"type": "string"}},
                required=["value"],
                permissions=ToolPermission(filesystem_read=True),
                read_only=True,
                concurrency_safe=False,
            )
        )

    async def execute(self, args, runtime_context=None):
        _ = runtime_context
        with self._lock:
            self.starts.append(time.perf_counter())
        await asyncio.sleep(self.delay)
        return {"value": args["value"]}


class _MissingResultTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(ToolSpec(name="MISSING", description="returns nothing"))

    async def execute(self, args, runtime_context=None):
        _ = args, runtime_context
        return None


@dataclass
class _ExecutorState(StateSchema):
    pass


@dataclass
class _CandidateReadyState(StateSchema):
    poc_path: str = ""
    candidate_ready_for_submit: bool = False
    workspace_root: str = ""


async def test_action_executor_preserves_output_before_model_projection():
    registry = ToolRegistry().register(_EchoTool())
    executor = ActionExecutor(registry)
    state = _ExecutorState(task="demo")

    ok = (await executor.execute(
        [Action(name="echo_tool", args={"value": "1234567890"})], state=state
    ))[0]
    assert ok.status == ActionStatus.SUCCESS
    assert ok.output["result"] == "1234567890"

    invalid = (await executor.execute(
        [Action(name="echo_tool", args={"value": "bad"})], state=state
    ))[0]
    assert invalid.status == ActionStatus.ERROR
    assert invalid.metadata["error_category"] == "bad_input"

    state.metadata["tool_permission_context"] = ToolPermissionContext(
        deny_rules=[
            ToolPermissionRule(effect="deny", tool_name="echo_tool", message="blocked")
        ]
    )
    denied = (await executor.execute(
        [Action(name="echo_tool", args={"value": "ok"})], state=state
    ))[0]
    assert denied.status == ActionStatus.DENIED
    assert denied.output["message"] == "blocked"

    state.metadata["tool_permission_context"] = ToolPermissionContext(
        ask_rules=[
            ToolPermissionRule(
                effect="ask", tool_name="echo_tool", message="need approval"
            )
        ]
    )
    ask = (await executor.execute(
        [Action(name="echo_tool", args={"value": "ok"})], state=state
    ))[0]
    assert ask.status == ActionStatus.NEEDS_APPROVAL
    assert ask.output["message"] == "need approval"


async def test_action_executor_reports_unknown_tool_without_name_repair():
    registry = ToolRegistry().register(_EchoTool(name="GREP"))
    executor = ActionExecutor(registry)

    result = (await executor.execute([Action(name="Grep", args={"value": "x"})]))[0]

    assert result.status == ActionStatus.ERROR
    assert result.metadata["error_category"] == "tool_not_found"
    assert result.metadata["executed"] is False
    assert "Unknown tool: `Grep`" in result.output
    assert "`GREP`" in result.output


async def test_action_executor_invokes_only_execute():
    tool = _EchoTool()

    def forbidden_entry(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("legacy tool entry was invoked")

    setattr(tool, "call", forbidden_entry)
    setattr(tool, "run", forbidden_entry)
    registry = ToolRegistry().register(tool)

    result = (
        await ActionExecutor(registry).execute(
            [Action(name="echo_tool", args={"value": "ok"})]
        )
    )[0]

    assert result.status == ActionStatus.SUCCESS
    assert result.output == {"result": "ok"}


async def test_action_executor_reports_missing_result_instead_of_none():
    executor = ActionExecutor(ToolRegistry().register(_MissingResultTool()))

    result = (await executor.execute([Action(name="MISSING", args={})]))[0]

    assert result.status == ActionStatus.ERROR
    assert result.output is not None
    assert "TOOL_RESULT_MISSING" in result.output
    assert result.metadata["error_category"] == "tool_result_missing"


async def test_action_executor_does_not_embed_agent_specific_candidate_policy(tmp_path):
    (tmp_path / "poc.bin").write_bytes(b"candidate")
    registry = ToolRegistry().register(_EchoTool()).register(_EchoTool(name="submit_poc"))
    executor = ActionExecutor(registry)
    state = _CandidateReadyState(
        task="demo",
        workspace_root=str(tmp_path),
        poc_path="poc.bin",
        candidate_ready_for_submit=True,
    )

    blocked = (await executor.execute(
        [Action(name="echo_tool", args={"value": "ignored"})],
        state=state,
    ))[0]
    allowed = (await executor.execute(
        [Action(name="submit_poc", args={"value": "poc.bin"})],
        state=state,
    ))[0]

    assert blocked.status == ActionStatus.SUCCESS
    assert allowed.status == ActionStatus.SUCCESS


async def test_action_executor_allows_regeneration_when_ready_candidate_file_missing(tmp_path):
    registry = ToolRegistry().register(_EchoTool())
    executor = ActionExecutor(registry)
    state = _CandidateReadyState(
        task="demo",
        workspace_root=str(tmp_path),
        poc_path="missing.bin",
        candidate_ready_for_submit=True,
    )

    result = (await executor.execute(
        [Action(name="echo_tool", args={"value": "regenerate"})],
        state=state,
    ))[0]

    assert result.status == ActionStatus.SUCCESS


async def test_action_executor_runs_concurrency_safe_read_only_tools_in_parallel():
    tool = _SleepReadTool()
    registry = ToolRegistry().register(tool)
    executor = ActionExecutor(
        registry,
        policy=ActionExecutionPolicy(mode="parallel", max_concurrency=4),
    )

    started = time.perf_counter()
    results = await executor.execute(
        [
            Action(name="sleep_read_tool", args={"value": "a"}),
            Action(name="sleep_read_tool", args={"value": "b"}),
            Action(name="sleep_read_tool", args={"value": "c"}),
        ]
    )
    elapsed = time.perf_counter() - started

    assert [item.status for item in results] == [ActionStatus.SUCCESS] * 3
    assert elapsed < 0.35
    assert len(tool.starts) == 3
    assert max(tool.starts) - min(tool.starts) < 0.08


async def test_action_executor_keeps_non_concurrency_safe_tools_serial_even_in_parallel_mode():
    tool = _UnsafeSleepTool()
    registry = ToolRegistry().register(tool)
    executor = ActionExecutor(
        registry,
        policy=ActionExecutionPolicy(mode="parallel", max_concurrency=4),
    )

    started = time.perf_counter()
    results = await executor.execute(
        [
            Action(name="unsafe_sleep_tool", args={"value": "a"}),
            Action(name="unsafe_sleep_tool", args={"value": "b"}),
            Action(name="unsafe_sleep_tool", args={"value": "c"}),
        ]
    )
    elapsed = time.perf_counter() - started

    assert [item.status for item in results] == [ActionStatus.SUCCESS] * 3
    assert elapsed >= 0.14
    assert len(tool.starts) == 3
    assert tool.starts[1] - tool.starts[0] >= 0.04


async def test_run_command_executes_in_workspace(tmp_path):
    tool = RunCommand(workspace_root=str(tmp_path))
    result = await tool.execute({"command": "pwd"})
    assert result["status"] == "success"
    assert str(tmp_path) in result["stdout"]


async def test_run_command_executes_compound_shell_syntax_after_admission(tmp_path):
    first = uuid4().hex
    second = uuid4().hex
    tool = RunCommand(workspace_root=str(tmp_path))

    result = await tool.execute({"command": f"printf {first} && printf {second}"})

    assert result["status"] == "success"
    assert result["stdout"] == first + second


async def test_run_command_does_not_repeat_permission_checks(tmp_path):
    name = uuid4().hex
    target = tmp_path / name
    target.write_text(uuid4().hex, encoding="utf-8")
    tool = RunCommand(workspace_root=str(tmp_path))

    result = await tool.execute({"command": f"rm -rf {name}"})

    assert result["status"] == "success"
    assert not target.exists()


async def test_read_and_edit_file_preserve_line_endings(tmp_path):
    path = tmp_path / "demo.txt"
    path.write_bytes(b"hello\r\nworld\r\n")

    reader = ReadFile(workspace_root=str(tmp_path))
    read_out = await reader.execute({"path": "demo.txt"})
    assert read_out["status"] == "success"
    assert "hello" in read_out["content"]

    editor = EditFile(workspace_root=str(tmp_path))
    edit_out = await editor.execute(
        {"path": "demo.txt", "old_text": "world", "new_text": "qitos"}
    )
    assert edit_out["status"] == "success"
    assert b"\r\n" in path.read_bytes()

    replaced = await editor.execute(
        {"path": "demo.txt", "old_text": "qitos", "new_text": "done"}
    )
    assert replaced["status"] == "success"
    assert "done" in path.read_text(encoding="utf-8")


async def test_web_fetch_handles_redirect_and_text_extraction(monkeypatch):
    tool = CodingToolSet()

    async def _redirect(
        url: str,
        params=None,
        headers=None,
        timeout=None,
        verify_tls=True,
        allow_redirects: bool = False,
    ):
        _ = params
        _ = headers
        _ = timeout
        _ = verify_tls
        _ = allow_redirects
        return {
            "status": "success",
            "url": "https://redirected.example.com/doc",
            "status_code": 302,
            "content": "",
            "headers": {"Location": "https://redirected.example.com/doc"},
        }

    monkeypatch.setattr(tool, "http_get", _redirect)
    redirect = await tool.web_fetch(url="https://example.com/doc")
    assert redirect["redirect_url"] == "https://redirected.example.com/doc"

    async def _content(
        url: str,
        params=None,
        headers=None,
        timeout=None,
        verify_tls=True,
        allow_redirects: bool = False,
    ):
        _ = url
        _ = params
        _ = headers
        _ = timeout
        _ = verify_tls
        _ = allow_redirects
        return {
            "status": "success",
            "url": "https://github.com/openai/example",
            "status_code": 200,
            "content": "<html><body><p>QitOS adds advanced coding tools.</p><p>Advanced tools include bash, file edit, and tool search.</p></body></html>",
            "headers": {},
        }

    monkeypatch.setattr(tool, "http_get", _content)
    out = await tool.web_fetch(url="https://github.com/openai/example")
    assert out["status"] == "success"
    assert "tool search" in out["content"].lower()
    assert out["auth_hint"]


async def test_session_tools_and_tool_search(tmp_path):
    registry = advanced_coding_tools(str(tmp_path), enable_lsp=False, enable_web=False)
    state = _ExecutorState(task="advanced")
    ctx = {"state": state, "tool_registry": registry}

    todo = await UpdateWorkPlanTool().execute(
        {"plan": [{"step": "ship", "status": "pending"}]},
        runtime_context=ctx,
    )
    assert len(todo["plan"]) == 1

    plan_enter = await registry.get("enter_plan_mode").execute(
        {"reason": "decompose"}, runtime_context=ctx
    )
    assert plan_enter["current_mode"] == "plan"

    create = await registry.get("task_create").execute(
        {"subject": "Implement", "description": "Do the work"}, runtime_context=ctx
    )
    listed = await registry.get("task_list").execute({}, runtime_context=ctx)
    assert create["status"] == "success"
    assert listed["count"] == 1

    search = await ToolSearchTool().execute({"query": "plan"}, runtime_context=ctx)
    assert search["count"] >= 1


async def test_lsp_query_and_mcp_resource_tools():
    class _FakeLSP:
        def query(self, **kwargs):
            return {"status": "success", "kwargs": kwargs}

    lsp = LSPQueryTool()
    out = await lsp.execute(
        {"operation": "definition", "symbol": "demo"},
        runtime_context={"ops": {"lsp": _FakeLSP()}},
    )
    assert out["status"] == "success"
    assert out["kwargs"]["operation"] == "definition"

    resources = {
        "docs": [
            {"uri": "memo://one", "text": "alpha"},
            {"uri": "memo://two", "text": "beta"},
        ]
    }
    listed = await MCPListResourcesTool().execute(
        {}, runtime_context={"mcp_resources": resources}
    )
    assert "docs" in listed["resources"]

    read = await MCPReadResourceTool().execute(
        {"server": "docs", "uri": "memo://two"},
        runtime_context={"mcp_resources": resources},
    )
    assert read["resource"]["text"] == "beta"


async def test_ask_user_choice_returns_needs_input_without_answers():
    tool = AskUserChoiceTool()
    out = await tool.execute(
        {
            "questions": [
                {
                    "header": "Mode",
                    "question": "Which mode?",
                    "options": [{"label": "A"}, {"label": "B"}],
                }
            ]
        }
    )
    assert out["status"] == "needs_input"
