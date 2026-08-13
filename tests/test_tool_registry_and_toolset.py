from typing import Any

import pytest

from qitos import Action, AgentModule, Decision, Engine, StateSchema, ToolRegistry, tool
from qitos.core.tool import BaseTool, ToolMeta, ToolSpec, build_tool_spec
from qitos.engine import RuntimeBudget
from qitos.kit import tool as tool_pkg
from qitos.kit.tool import (
    CodingToolSet,
    EpubToolSet,
    NotebookToolSet,
    ReportToolSet,
    TaskToolSet,
    ThinkingToolSet,
)
from qitos.kit.tool.gui import Click
from qitos.kit.tool.codebase import Glob
from qitos.kit.tool.file import ReadFile
from qitos.kit.tool.shell import RunCommand
from qitos.kit.tool.toolset import toolset_from_tools
from qitos.kit.toolset import (
    ComputerUseToolSet,
    CodebaseToolSet,
    StaticToolSet,
    coding_tools as coding_tools_builder,
    computer_use_tools as computer_use_tools_builder,
)
from qitos.kit.toolset import editor_tools as editor_tools_builder
from qitos.kit.toolset import report_tools as report_tools_builder
from qitos.kit.toolset import security_audit_tools as security_audit_tools_builder
from qitos.kit.tool.experimental.security_research import SecurityAuditToolSet


class _ToolState(StateSchema):
    pass


class _Agent(AgentModule[_ToolState, dict[str, Any], Action]):
    def __init__(self, tool_registry: ToolRegistry):
        super().__init__(tool_registry=tool_registry)

    def init_state(self, task: str, **kwargs: Any) -> _ToolState:
        return _ToolState(task=task, max_steps=2)

    def decide(
        self, state: _ToolState, observation: dict[str, Any]
    ) -> Decision[Action]:
        if state.current_step == 0:
            return Decision.act([Action(name="math.add", args={"a": 1, "b": 2})])
        return Decision.final("done")

    def reduce(
        self, state: _ToolState, observation: dict[str, Any], decision: Decision[Action]
    ) -> _ToolState:
        return state


def test_tool_registry_include_and_toolset_lifecycle(tmp_path):
    events: list[str] = []

    class MathToolSet:
        name = "math"
        version = "1"

        def setup(self, context: dict[str, Any]) -> None:
            events.append("setup")

        def teardown(self, context: dict[str, Any]) -> None:
            events.append("teardown")

        @tool(name="add")
        def add(self, a: int, b: int) -> int:
            return a + b

        def tools(self):
            return [self.add]

    registry = ToolRegistry()
    registry.register_toolset(MathToolSet())
    result = Engine(
        agent=_Agent(tool_registry=registry), budget=RuntimeBudget(max_steps=2)
    ).run("x")
    assert result.state.stop_reason == "final"
    assert events == ["setup", "teardown"]

    editor = ToolRegistry()
    editor.include(
        CodingToolSet(
            workspace_root=str(tmp_path),
            include_notebook=False,
            enable_lsp=False,
            enable_tasks=False,
            enable_web=False,
            profile="editor",
        )
    )
    assert "read_file" in editor.list_tools()
    assert "edit_file" in editor.list_tools()


def test_register_name_override_does_not_mutate_source_tool(tmp_path):
    coding = CodingToolSet(
        workspace_root=str(tmp_path),
        include_notebook=False,
        enable_lsp=False,
        enable_tasks=False,
        enable_web=False,
        profile="editor",
    )

    registry = ToolRegistry()
    registry.register(coding.edit_file, name="EDIT_FILE")

    assert "EDIT_FILE" in registry.list_tools()
    assert coding.edit_file.spec.name == "edit_file"


def test_unregister_removes_exact_dynamic_tool():
    @tool(name="temporary")
    def temporary() -> str:
        return "ok"

    registry = ToolRegistry()
    registry.register(temporary)

    removed = registry.unregister("temporary")

    assert removed.name == "temporary"
    assert registry.get("temporary") is None
    with pytest.raises(ValueError, match="not found"):
        registry.unregister("temporary")


def test_curated_toolsets_register_cleanly(tmp_path):
    toolsets = [
        NotebookToolSet(workspace_root=str(tmp_path)),
        ReportToolSet(workspace_root=str(tmp_path)),
        SecurityAuditToolSet(workspace_root=str(tmp_path)),
        TaskToolSet(workspace_root=str(tmp_path)),
        EpubToolSet(workspace_root=str(tmp_path)),
        ThinkingToolSet(),
        CodingToolSet(
            workspace_root=str(tmp_path),
            include_notebook=False,
            enable_lsp=False,
            enable_tasks=False,
            enable_web=False,
            profile="editor",
        ),
        CodingToolSet(
            workspace_root=str(tmp_path),
            include_notebook=False,
            enable_lsp=False,
            enable_tasks=False,
            enable_web=False,
            profile="codebase",
        ),
        CodingToolSet(workspace_root=str(tmp_path)),
        ComputerUseToolSet(),
    ]
    for toolset in toolsets:
        registry = ToolRegistry()
        registry.register_toolset(toolset, namespace="")
        assert (
            registry.list_tools()
        ), f"{toolset.__class__.__name__} registered no tools"


def test_tool_schemas_resolve_future_annotations_to_valid_json_types(tmp_path):
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError

    def _future_annotated(path):
        return {"path": path}

    _future_annotated.__annotations__ = {"path": "str"}
    synthetic_spec = build_tool_spec(_future_annotated, ToolMeta(name="synthetic"))

    registry = ToolRegistry()
    registry.register_toolset(
        SecurityAuditToolSet(workspace_root=str(tmp_path)), namespace=""
    )

    specs = {spec["function"]["name"]: spec for spec in registry.get_all_specs()}

    assert (
        synthetic_spec.input_schema["properties"]["path"]["type"]
        == "string"
    )
    findings_schema = specs["audit_hotspots"]["function"]["parameters"][
        "properties"
    ]["findings"]
    Draft202012Validator.check_schema(findings_schema)
    validator = Draft202012Validator(findings_schema)
    validator.validate(None)
    validator.validate([{"file": "src/example.py"}])
    with pytest.raises(ValidationError):
        validator.validate("not-a-finding-list")


def test_tool_package_does_not_export_uncurated_cyber_toolsets():
    exported = set(getattr(tool_pkg, "__all__", []))
    assert "ReportToolSet" in exported
    assert "CodingToolSet" in exported
    assert "SecurityAuditToolSet" not in exported
    assert "security_audit_tools" not in exported
    assert "EditorToolSet" not in exported
    assert "CodebaseToolSet" not in exported
    assert "RunCommand" not in exported
    assert "WriteFile" not in exported
    assert "ReadFile" not in exported
    assert "AdvancedCodingToolSet" not in exported
    assert "advanced_coding_tools" not in exported
    assert "ReconToolSet" not in exported
    assert "NetworkToolSet" not in exported
    assert "VulnScanToolSet" not in exported
    assert "WebTestToolSet" not in exported
    assert "ExploitToolSet" not in exported
    assert "PasswordToolSet" not in exported
    assert "ExploitToolSet" not in exported
    assert "PasswordToolSet" not in exported


def test_new_tool_domains_and_toolset_surface_are_importable(tmp_path):
    sample = tmp_path / "demo.txt"
    sample.write_text("hello\n", encoding="utf-8")

    read_file = ReadFile(workspace_root=str(tmp_path))
    out = read_file.execute({"path": "demo.txt"})
    assert out["status"] == "success"
    assert "hello" in out["content"]

    glob = Glob(workspace_root=str(tmp_path))
    glob_out = glob.execute({"pattern": "*.txt"})
    assert glob_out["status"] == "success"
    assert glob_out["files"] == ["demo.txt"]

    shell = RunCommand(workspace_root=str(tmp_path))
    assert shell.spec.name == "run_command"

    assert "edit_file" in editor_tools_builder(str(tmp_path)).list_tools()
    assert "glob" in coding_tools_builder(str(tmp_path)).list_tools()
    assert "audit_inventory" in security_audit_tools_builder(str(tmp_path)).list_tools()
    assert "click" in computer_use_tools_builder().list_tools()


def test_include_toolset_accepts_mixed_tools_toolsets_and_registries(tmp_path):
    registry = ToolRegistry()
    registry.include_toolset(
        [
            RunCommand(workspace_root=str(tmp_path)),
            CodebaseToolSet(workspace_root=str(tmp_path)),
            report_tools_builder(str(tmp_path)),
        ]
    )
    names = registry.list_tools()
    assert "run_command" in names
    assert "glob" in names
    assert "read_file" in names
    assert "generate_report" in names


def test_static_toolset_and_toolset_from_tools_register_cleanly(tmp_path):
    static = StaticToolSet(
        [
            ReadFile(workspace_root=str(tmp_path)),
            Glob(workspace_root=str(tmp_path)),
        ],
        name="bundle",
        version="1",
    )
    registry = ToolRegistry().register_toolset(static, namespace="")
    assert "read_file" in registry.list_tools()
    assert "glob" in registry.list_tools()

    helper = toolset_from_tools(
        [ReadFile(workspace_root=str(tmp_path))], name="helper_bundle", version="2"
    )
    helper_registry = ToolRegistry().include_toolset(helper)
    assert "read_file" in helper_registry.list_tools()


def test_agent_module_can_be_initialized_with_toolset_list(tmp_path):
    sample = tmp_path / "demo.txt"
    sample.write_text("hello\n", encoding="utf-8")

    class _ToolsetAgent(AgentModule[_ToolState, dict[str, Any], Action]):
        def __init__(self):
            super().__init__(
                toolset=[
                    ReadFile(workspace_root=str(tmp_path)),
                    toolset_from_tools(
                        [Glob(workspace_root=str(tmp_path))], name="glob"
                    ),
                ]
            )

        def init_state(self, task: str, **kwargs: Any) -> _ToolState:
            return _ToolState(task=task, max_steps=2)

        def decide(
            self, state: _ToolState, observation: dict[str, Any]
        ) -> Decision[Action]:
            if state.current_step == 0:
                return Decision.act(
                    [Action(name="read_file", args={"path": "demo.txt"})]
                )
            return Decision.final("done")

        def reduce(
            self,
            state: _ToolState,
            observation: dict[str, Any],
            decision: Decision[Action],
        ) -> _ToolState:
            _ = observation
            _ = decision
            return state

    result = Engine(agent=_ToolsetAgent(), budget=RuntimeBudget(max_steps=2)).run("x")
    assert result.state.stop_reason == "final"


def test_computer_use_toolset_atomic_tools_are_callable() -> None:
    registry = computer_use_tools_builder()
    assert "click" in registry.list_tools()
    tool_obj = registry.get("click")
    assert isinstance(tool_obj, Click)


def test_tool_registry_requires_exact_names_but_can_suggest_a_canonical_name() -> None:
    registry = ToolRegistry()

    @tool(name="coding.run_command")
    def run_command(command: str) -> str:
        return command

    @tool(name="coding.list_files")
    def list_files(path: str = ".") -> list[str]:
        return [path]

    registry.register(run_command)
    registry.register(list_files)

    assert registry.get("coding.run_command") is not None
    assert registry.get("run_command") is None
    assert registry.get("coding=run_command") is None
    assert registry.get("coding-run_command") is None
    assert "coding.list_files" in registry.suggest("coding=list_filez")


def test_class_tool_must_implement_execute() -> None:
    def legacy_run(self: BaseTool, **kwargs: Any) -> Any:
        return kwargs

    run_only_tool = type("RunOnlyTool", (BaseTool,), {"run": legacy_run})

    with pytest.raises(TypeError, match="abstract method execute"):
        run_only_tool(ToolSpec(name="run_only", description="legacy"))
