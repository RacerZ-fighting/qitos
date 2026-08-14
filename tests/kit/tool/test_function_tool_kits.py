"""Test that epub/report/thinking/skill tools are FunctionTool instances with correct metadata."""

from qitos.core.tool import FunctionTool
from qitos.kit.tool.epub.toolset import EpubToolSet
from qitos.kit.tool.report.toolset import ReportToolSet
from qitos.kit.tool.thinking.toolset import ThinkingToolSet
from qitos.kit.tool.skill.toolset import SkillToolSet


class TestEpubToolMigration:
    """Epub tools should be FunctionTool instances with read_only=True."""

    def setup_method(self):
        self.ts = EpubToolSet(workspace_root=".")

    def test_list_chapters_is_function_tool(self):
        tool = self.ts.list_chapters
        assert isinstance(
            tool, FunctionTool
        ), f"Expected FunctionTool, got {type(tool)}"

    def test_read_chapter_is_function_tool(self):
        tool = self.ts.read_chapter
        assert isinstance(
            tool, FunctionTool
        ), f"Expected FunctionTool, got {type(tool)}"

    def test_search_is_function_tool(self):
        tool = self.ts.search
        assert isinstance(
            tool, FunctionTool
        ), f"Expected FunctionTool, got {type(tool)}"

    def test_list_chapters_read_only(self):
        tool = self.ts.list_chapters
        assert tool.meta.read_only is True

    def test_read_chapter_read_only(self):
        tool = self.ts.read_chapter
        assert tool.meta.read_only is True

    def test_search_read_only(self):
        tool = self.ts.search
        assert tool.meta.read_only is True


class TestReportToolMigration:
    """Report tools should be FunctionTool instances with needs_approval=True."""

    def setup_method(self):
        self.ts = ReportToolSet(workspace_root=".")

    def test_finding_add_is_function_tool(self):
        assert isinstance(self.ts.finding_add, FunctionTool)

    def test_attack_map_is_function_tool(self):
        assert isinstance(self.ts.attack_map, FunctionTool)

    def test_summary_generate_is_function_tool(self):
        assert isinstance(self.ts.summary_generate, FunctionTool)

    def test_generate_report_is_function_tool(self):
        assert isinstance(self.ts.generate_report, FunctionTool)

    def test_finding_export_is_function_tool(self):
        assert isinstance(self.ts.finding_export, FunctionTool)

    def test_finding_add_needs_approval(self):
        assert self.ts.finding_add.meta.needs_approval is True

    def test_attack_map_needs_approval(self):
        assert self.ts.attack_map.meta.needs_approval is True

    def test_summary_generate_needs_approval(self):
        assert self.ts.summary_generate.meta.needs_approval is True

    def test_generate_report_needs_approval(self):
        assert self.ts.generate_report.meta.needs_approval is True

    def test_finding_export_needs_approval(self):
        assert self.ts.finding_export.meta.needs_approval is True


class TestThinkingToolMigration:
    """Thinking tools should be FunctionTool instances with default approval."""

    def setup_method(self):
        self.ts = ThinkingToolSet()

    def test_sequential_thinking_is_function_tool(self):
        assert isinstance(self.ts.sequential_thinking, FunctionTool)

    def test_get_thoughts_is_function_tool(self):
        assert isinstance(self.ts.get_thoughts, FunctionTool)

    def test_clear_thoughts_is_function_tool(self):
        assert isinstance(self.ts.clear_thoughts, FunctionTool)

    def test_sequential_thinking_no_approval_needed(self):
        assert self.ts.sequential_thinking.meta.needs_approval is False

    def test_get_thoughts_no_approval_needed(self):
        assert self.ts.get_thoughts.meta.needs_approval is False

    def test_clear_thoughts_no_approval_needed(self):
        assert self.ts.clear_thoughts.meta.needs_approval is False


class TestSkillToolMigration:
    """Bundled Skill tools are read-only FunctionTool instances."""

    def setup_method(self):
        self.ts = SkillToolSet()

    def test_disclosure_tools_are_function_tools(self):
        assert isinstance(self.ts.list_skills, FunctionTool)
        assert isinstance(self.ts.load_skill, FunctionTool)
        assert isinstance(self.ts.read_skill_resource, FunctionTool)

    def test_read_only_tools(self):
        assert self.ts.list_skills.meta.read_only is True
        assert self.ts.load_skill.meta.read_only is True
        assert self.ts.read_skill_resource.meta.read_only is True
