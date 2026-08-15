"""Tests for Phase 3: Permission/Safety System.

Covers: PermissionMode, PermissionPipeline, BashCommandAnalyzer,
ReadBeforeWriteEnforcer, AutoPermissionClassifier, protected paths,
and ActionExecutor integration.
"""

import os
import tempfile
import time
from unittest.mock import MagicMock

import pytest

from qitos.core.tool import (
    ToolPermission,
    ToolPermissionContext,
    ToolPermissionRule,
    ToolSpec,
    tool,
)
from qitos.core.tool_registry import ToolRegistry
from qitos.kit.permission.pipeline import (
    PermissionMode,
    PermissionPipeline,
)
from qitos.kit.permission.bash_analyzer import (
    BashCommandAnalyzer,
    CommandSafety,
)
from qitos.kit.permission.read_before_write import ReadBeforeWriteEnforcer
from qitos.kit.permission.auto_classifier import AutoPermissionClassifier
from qitos.kit.permission.rules import (
    build_default_deny_rules,
    build_default_ask_rules,
    is_protected_path,
)


# ── PermissionMode ─────────────────────────────────────────────────────────────


class TestPermissionMode:
    def test_enum_values(self):
        assert PermissionMode.DEFAULT.value == "default"
        assert PermissionMode.PLAN.value == "plan"
        assert PermissionMode.ACCEPT_EDITS.value == "accept_edits"
        assert PermissionMode.AUTONOMOUS.value == "autonomous"
        assert PermissionMode.BYPASS.value == "bypass"
        assert PermissionMode.AUTO.value == "auto"

    def test_string_comparison(self):
        assert PermissionMode.DEFAULT == "default"
        assert PermissionMode.PLAN == "plan"


# ── PermissionPipeline ─────────────────────────────────────────────────────────


class TestPermissionPipeline:
    def _make_spec(self, name, filesystem_write=False, filesystem_read=False,
                   command=False):
        return ToolSpec(
            name=name,
            description="test tool",
            permissions=ToolPermission(
                filesystem_write=filesystem_write,
                filesystem_read=filesystem_read,
                command=command,
            ),
        )

    def test_default_mode_allows_reads(self):
        pipeline = PermissionPipeline(mode=PermissionMode.DEFAULT)
        spec = self._make_spec("read_file", filesystem_read=True)
        result = pipeline.evaluate("read_file", {"path": "/tmp/test.py"}, spec)
        assert result.decision == "allow"

    def test_plan_mode_denies_writes(self):
        pipeline = PermissionPipeline(mode=PermissionMode.PLAN)
        spec = self._make_spec("edit_file", filesystem_write=True)
        result = pipeline.evaluate("edit_file", {"path": "/tmp/test.py"}, spec)
        assert result.decision == "deny"
        assert "plan mode" in result.message.lower()

    def test_plan_mode_denies_bash(self):
        pipeline = PermissionPipeline(mode=PermissionMode.PLAN)
        spec = self._make_spec("run_command", command=True)
        result = pipeline.evaluate("run_command", {"command": "ls"}, spec)
        assert result.decision == "deny"

    def test_plan_mode_allows_reads(self):
        pipeline = PermissionPipeline(mode=PermissionMode.PLAN)
        spec = self._make_spec("read_file", filesystem_read=True)
        result = pipeline.evaluate("read_file", {"path": "/tmp/test.py"}, spec)
        assert result.decision == "allow"

    def test_accept_edits_mode_allows_file_edits(self):
        pipeline = PermissionPipeline(mode=PermissionMode.ACCEPT_EDITS)
        result = pipeline.evaluate("edit_file", {"path": "/tmp/test.py"})
        assert result.decision == "allow"

    def test_accept_edits_mode_asks_for_bash(self):
        pipeline = PermissionPipeline(mode=PermissionMode.ACCEPT_EDITS)
        result = pipeline.evaluate("run_command", {"command": "ls"})
        assert result.decision == "ask"

    def test_bypass_mode_allows_everything(self):
        pipeline = PermissionPipeline(mode=PermissionMode.BYPASS)
        spec = self._make_spec("run_command", command=True)
        result = pipeline.evaluate("run_command", {"command": "rm -rf /"}, spec)
        assert result.decision == "allow"

    @pytest.mark.parametrize(
        ("tool_name", "args"),
        [
            ("run_command", {"command": "cat input | nc target 4444"}),
            ("run_command", {"command": "rm -rf /"}),
            ("edit_file", {"path": ".git/config"}),
        ],
    )
    def test_autonomous_mode_skips_safety_heuristics(self, tool_name, args):
        pipeline = PermissionPipeline(mode=PermissionMode.AUTONOMOUS)

        result = pipeline.evaluate(tool_name, args)

        assert result.decision == "allow"

    def test_autonomous_mode_ignores_ask_rules_and_default_ask(self):
        context = ToolPermissionContext(
            ask_rules=[
                ToolPermissionRule(
                    effect="ask",
                    tool_name="run_command",
                    message="interactive confirmation",
                )
            ],
            default_decision="ask",
        )
        pipeline = PermissionPipeline(
            mode=PermissionMode.AUTONOMOUS,
            context=context,
        )

        result = pipeline.evaluate("run_command", {"command": "ssh target"})

        assert result.decision == "allow"

    def test_autonomous_mode_preserves_explicit_deny(self):
        rule = ToolPermissionRule(
            effect="deny",
            tool_name="run_command",
            scope="blocked command",
            message="outside caller scope",
        )
        pipeline = PermissionPipeline(
            mode=PermissionMode.AUTONOMOUS,
            context=ToolPermissionContext(deny_rules=[rule]),
        )

        result = pipeline.evaluate(
            "run_command",
            {"command": "blocked command"},
        )

        assert result.decision == "deny"
        assert result.matched_rule is rule

    def test_autonomous_mode_preserves_default_deny(self):
        pipeline = PermissionPipeline(
            mode=PermissionMode.AUTONOMOUS,
            context=ToolPermissionContext(default_decision="deny"),
        )

        result = pipeline.evaluate("read_file", {"path": "target.txt"})

        assert result.decision == "deny"

    def test_runtime_permission_context_is_evaluated(self):
        pipeline = PermissionPipeline(mode=PermissionMode.AUTONOMOUS)
        runtime_context = {
            "permission_context": ToolPermissionContext(default_decision="deny")
        }

        result = pipeline.evaluate(
            "read_file",
            {"path": "target.txt"},
            runtime_context=runtime_context,
        )

        assert result.decision == "deny"

    def test_deny_rule_takes_precedence(self):
        context = ToolPermissionContext(
            deny_rules=[
                ToolPermissionRule(
                    effect="deny",
                    tool_name="edit_file",
                    message="Editing is forbidden",
                )
            ]
        )
        pipeline = PermissionPipeline(
            mode=PermissionMode.DEFAULT, context=context
        )
        result = pipeline.evaluate("edit_file", {"path": "/tmp/test.py"})
        assert result.decision == "deny"
        assert "Editing is forbidden" in result.message

    def test_protected_path_denied(self):
        pipeline = PermissionPipeline(mode=PermissionMode.DEFAULT)
        result = pipeline.evaluate(
            "edit_file", {"path": ".git/config"}
        )
        assert result.decision == "deny"
        assert "protected" in result.message.lower()

    def test_bash_unsafe_command_denied(self):
        pipeline = PermissionPipeline(mode=PermissionMode.DEFAULT)
        result = pipeline.evaluate("run_command", {"command": "rm -rf /"})
        assert result.decision == "deny"

    def test_bash_needs_review_asks(self):
        pipeline = PermissionPipeline(mode=PermissionMode.DEFAULT)
        result = pipeline.evaluate(
            "run_command", {"command": "cat file.txt | grep pattern"}
        )
        assert result.decision == "ask"

    def test_bash_safe_command_allowed(self):
        pipeline = PermissionPipeline(mode=PermissionMode.DEFAULT)
        result = pipeline.evaluate("run_command", {"command": "ls -la"})
        assert result.decision == "allow"


# ── BashCommandAnalyzer ────────────────────────────────────────────────────────


class TestBashCommandAnalyzer:
    def setup_method(self):
        self.analyzer = BashCommandAnalyzer()

    def test_empty_command_safe(self):
        result = self.analyzer.analyze("")
        assert result.safety == CommandSafety.SAFE

    def test_ls_safe(self):
        result = self.analyzer.analyze("ls -la")
        assert result.safety == CommandSafety.SAFE
        assert result.is_read_only is True

    def test_cat_safe(self):
        result = self.analyzer.analyze("cat file.txt")
        assert result.safety == CommandSafety.SAFE
        assert result.is_read_only is True

    def test_rm_rf_unsafe(self):
        result = self.analyzer.analyze("rm -rf /")
        assert result.safety == CommandSafety.UNSAFE

    def test_sudo_rm_unsafe(self):
        result = self.analyzer.analyze("sudo rm file.txt")
        assert result.safety == CommandSafety.UNSAFE

    def test_mkfs_unsafe(self):
        result = self.analyzer.analyze("mkfs.ext4 /dev/sda1")
        assert result.safety == CommandSafety.UNSAFE

    def test_fork_bomb_unsafe(self):
        result = self.analyzer.analyze(":(){ :|:& }:")
        assert result.safety == CommandSafety.UNSAFE

    def test_git_force_push_unsafe(self):
        result = self.analyzer.analyze("git push origin --force")
        assert result.safety == CommandSafety.UNSAFE

    def test_git_reset_hard_unsafe(self):
        result = self.analyzer.analyze("git reset --hard HEAD~1")
        assert result.safety == CommandSafety.UNSAFE

    def test_pipe_needs_review(self):
        result = self.analyzer.analyze("cat file.txt | grep pattern")
        assert result.safety == CommandSafety.NEEDS_REVIEW
        assert "pipe" in result.explanation.lower()

    def test_command_substitution_needs_review(self):
        result = self.analyzer.analyze("echo $(date)")
        assert result.safety == CommandSafety.NEEDS_REVIEW

    def test_backtick_substitution_needs_review(self):
        result = self.analyzer.analyze("echo `date`")
        assert result.safety == CommandSafety.NEEDS_REVIEW

    def test_interactive_command_needs_review(self):
        result = self.analyzer.analyze("vim file.txt")
        assert result.safety == CommandSafety.NEEDS_REVIEW

    def test_ssh_needs_review(self):
        result = self.analyzer.analyze("ssh user@host")
        assert result.safety == CommandSafety.NEEDS_REVIEW

    def test_git_status_safe(self):
        result = self.analyzer.analyze("git status")
        assert result.safety == CommandSafety.SAFE
        assert result.is_read_only is True

    def test_git_log_safe(self):
        result = self.analyzer.analyze("git log --oneline -10")
        assert result.safety == CommandSafety.SAFE

    def test_extract_paths(self):
        paths = self.analyzer.extract_paths("cat /tmp/test.py | grep foo")
        assert "/tmp/test.py" in paths

    def test_extract_paths_relative(self):
        paths = self.analyzer.extract_paths("ls src/main.py")
        assert "src/main.py" in paths

    def test_is_read_only_grep(self):
        assert self.analyzer.is_read_only("grep -r pattern .") is True

    def test_is_read_only_sed_i(self):
        assert self.analyzer.is_read_only("sed -i 's/old/new/' file") is False

    def test_is_read_only_mkdir(self):
        assert self.analyzer.is_read_only("mkdir new_dir") is False

    def test_dd_unsafe(self):
        result = self.analyzer.analyze("dd if=/dev/zero of=/dev/sda")
        assert result.safety == CommandSafety.UNSAFE

    def test_chmod_777_unsafe(self):
        result = self.analyzer.analyze("chmod -R 777 /var/www")
        assert result.safety == CommandSafety.UNSAFE

    def test_network_curl_needs_review(self):
        result = self.analyzer.analyze("curl https://example.com")
        assert result.safety == CommandSafety.NEEDS_REVIEW

    def test_obfuscation_hex_escape_needs_review(self):
        result = self.analyzer.analyze("echo '\\x41'")
        assert result.safety == CommandSafety.NEEDS_REVIEW

    def test_eval_needs_review(self):
        result = self.analyzer.analyze("eval 'echo hi'")
        assert result.safety == CommandSafety.NEEDS_REVIEW


# ── ReadBeforeWriteEnforcer ────────────────────────────────────────────────────


class TestReadBeforeWriteEnforcer:
    def test_unread_file_rejected(self):
        enforcer = ReadBeforeWriteEnforcer()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            allowed, reason = enforcer.check_write(path)
            assert allowed is False
            assert "not been read" in reason
        finally:
            os.unlink(path)

    def test_read_file_allowed(self):
        enforcer = ReadBeforeWriteEnforcer()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            enforcer.record_read(path, "hello")
            allowed, reason = enforcer.check_write(path)
            assert allowed is True
            assert reason == ""
        finally:
            os.unlink(path)

    def test_stale_file_rejected(self):
        enforcer = ReadBeforeWriteEnforcer()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            enforcer.record_read(path, "hello")
            # Modify the file (change content and mtime)
            time.sleep(0.01)
            with open(path, "w") as f:
                f.write("modified")
            allowed, reason = enforcer.check_write(path)
            assert allowed is False
            assert "modified since read" in reason
        finally:
            os.unlink(path)

    def test_mtime_change_same_content_allowed(self):
        enforcer = ReadBeforeWriteEnforcer()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            enforcer.record_read(path, "hello")
            # Touch the file to change mtime but keep content the same
            time.sleep(0.01)
            with open(path, "w") as f:
                f.write("hello")  # Same content
            allowed, reason = enforcer.check_write(path)
            assert allowed is True
        finally:
            os.unlink(path)

    def test_new_file_allowed(self):
        enforcer = ReadBeforeWriteEnforcer()
        allowed, reason = enforcer.check_write("/nonexistent/path/new_file.py")
        assert allowed is True

    def test_invalidate_after_write(self):
        enforcer = ReadBeforeWriteEnforcer()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            enforcer.record_read(path, "hello")
            enforcer.invalidate(path)
            allowed, reason = enforcer.check_write(path)
            assert allowed is False
            assert "not been read" in reason
        finally:
            os.unlink(path)

    def test_is_read(self):
        enforcer = ReadBeforeWriteEnforcer()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            assert enforcer.is_read(path) is False
            enforcer.record_read(path, "hello")
            assert enforcer.is_read(path) is True
        finally:
            os.unlink(path)

    def test_tracked_files_count(self):
        enforcer = ReadBeforeWriteEnforcer()
        assert enforcer.tracked_files == 0
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("hello")
            path = f.name
        try:
            enforcer.record_read(path, "hello")
            assert enforcer.tracked_files == 1
        finally:
            os.unlink(path)

    def test_clear(self):
        enforcer = ReadBeforeWriteEnforcer()
        enforcer.record_read("/some/file.py", "content")
        enforcer.clear()
        assert enforcer.tracked_files == 0


# ── AutoPermissionClassifier ───────────────────────────────────────────────────


class TestAutoPermissionClassifier:
    def test_safe_read_tools_auto_allowed(self):
        clf = AutoPermissionClassifier()
        for tool_name in ("read_file", "glob", "grep"):
            result = clf.classify(tool_name, {"path": "/tmp/test.py"})
            assert result == "allow", f"{tool_name} should be auto-allowed"

    def test_write_tool_without_read_asks(self):
        clf = AutoPermissionClassifier()
        result = clf.classify("edit_file", {"path": "/tmp/test.py"})
        assert result == "ask"

    def test_write_tool_after_read_allowed(self):
        clf = AutoPermissionClassifier()
        clf.record_read("/tmp/test.py")
        result = clf.classify("edit_file", {"path": "/tmp/test.py"})
        assert result == "allow"

    def test_bash_safe_command_allowed(self):
        clf = AutoPermissionClassifier()
        result = clf.classify("run_command", {"command": "ls -la"})
        assert result == "allow"

    def test_bash_dangerous_command_denied(self):
        clf = AutoPermissionClassifier()
        result = clf.classify("run_command", {"command": "rm -rf /"})
        assert result == "deny"

    def test_unknown_tool_asks(self):
        clf = AutoPermissionClassifier()
        result = clf.classify("unknown_tool", {})
        assert result == "ask"

    def test_denial_tracking_consecutive(self):
        clf = AutoPermissionClassifier()
        for _ in range(3):
            clf.record_denial()
        assert clf.consecutive_denials == 3
        assert clf.is_locked_out is True

    def test_denial_tracking_total(self):
        clf = AutoPermissionClassifier()
        for _ in range(20):
            clf.record_denial()
        assert clf.total_denials == 20
        assert clf.is_locked_out is True

    def test_approval_resets_consecutive(self):
        clf = AutoPermissionClassifier()
        clf.record_denial()
        clf.record_denial()
        assert clf.consecutive_denials == 2
        clf.record_approval()
        assert clf.consecutive_denials == 0
        assert clf.is_locked_out is False

    def test_lockout_affects_auto_mode(self):
        pipeline = PermissionPipeline(
            mode=PermissionMode.AUTO,
            auto_classifier=AutoPermissionClassifier(),
        )
        # Trigger lockout
        for _ in range(3):
            pipeline._auto_classifier.record_denial()
        result = pipeline.evaluate("edit_file", {"path": "/tmp/test.py"})
        assert result.decision == "ask"
        assert "locked out" in result.message.lower()


# ── Protected Paths ────────────────────────────────────────────────────────────


class TestProtectedPaths:
    def test_git_dir_protected(self):
        assert is_protected_path(".git/config") is True

    def test_qitos_dir_protected(self):
        assert is_protected_path(".qitos/settings.json") is True

    def test_bashrc_protected(self):
        assert is_protected_path(".bashrc") is True

    def test_env_protected(self):
        assert is_protected_path(".env") is True

    def test_credentials_protected(self):
        assert is_protected_path("credentials.json") is True

    def test_pem_protected(self):
        assert is_protected_path("server.pem") is True

    def test_ssh_key_protected(self):
        assert is_protected_path("id_rsa") is True

    def test_normal_path_not_protected(self):
        assert is_protected_path("src/main.py") is False

    def test_empty_path_not_protected(self):
        assert is_protected_path("") is False

    def test_aws_credentials_protected(self):
        assert is_protected_path(".aws/credentials") is True

    def test_build_deny_rules(self):
        rules = build_default_deny_rules()
        assert len(rules) > 0
        for rule in rules:
            assert rule.effect == "deny"

    def test_build_ask_rules(self):
        rules = build_default_ask_rules()
        assert len(rules) > 0
        for rule in rules:
            assert rule.effect == "ask"


# ── ActionExecutor Integration ─────────────────────────────────────────────────


class TestActionExecutorIntegration:
    def test_hook_dispatch_uses_correct_attribute(self):
        """Verify the bug fix: _dispatch_tool_hook reads 'hooks' not '_hooks'."""
        from qitos.engine.action_executor import ActionExecutor

        engine_mock = MagicMock()
        engine_mock.hooks = [MagicMock()]

        executor = ActionExecutor(
            tool_registry=ToolRegistry(), engine=engine_mock
        )

        # This should find hooks via engine.hooks, not engine._hooks
        executor._dispatch_tool_hook(
            "on_before_tool_use", "test_tool", {"arg": "value"}
        )

        # The hook method should have been called
        hook = engine_mock.hooks[0]
        assert hook.on_before_tool_use.called

    @pytest.mark.asyncio
    async def test_permission_pipeline_in_executor(self):
        """Test that the executor uses PermissionPipeline when provided."""
        from qitos.engine.action_executor import ActionExecutor
        from qitos.core.action import Action, ActionStatus

        # Create a pipeline in PLAN mode that denies writes
        pipeline = PermissionPipeline(mode=PermissionMode.PLAN)

        @tool(
            name="edit_file",
            permissions=ToolPermission(filesystem_write=True),
        )
        def edit_file(path: str) -> str:
            return path

        registry = ToolRegistry().register(edit_file)

        executor = ActionExecutor(
            tool_registry=registry,
            permission_pipeline=pipeline,
        )

        action = Action(name="edit_file", args={"path": "test.py"})
        results = await executor.execute([action])
        assert results[0].status == ActionStatus.DENIED

    @pytest.mark.asyncio
    async def test_autonomous_pipeline_skips_executor_rbw_and_records_mode(self):
        from qitos.core.action import Action, ActionStatus
        from qitos.engine.action_executor import ActionExecutor

        rbw = ReadBeforeWriteEnforcer()

        @tool(name="edit_file")
        def edit_file(path: str) -> str:
            return path

        registry = ToolRegistry().register(edit_file)
        executor = ActionExecutor(
            tool_registry=registry,
            permission_pipeline=PermissionPipeline(
                mode=PermissionMode.AUTONOMOUS
            ),
            read_before_write_enforcer=rbw,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("existing")
            path = f.name

        try:
            [result] = await executor.execute(
                [Action(name="edit_file", args={"path": path})]
            )
        finally:
            os.unlink(path)

        assert result.status is ActionStatus.SUCCESS
        assert result.metadata["permission"] == {
            "mode": "autonomous",
            "decision": "allow",
            "message": "",
            "scope": path,
            "matched_rule": None,
        }
        assert rbw.tracked_files == 0

    @pytest.mark.asyncio
    async def test_rbw_enforcer_in_executor(self):
        """Test that read-before-write enforcer blocks writes to unread files."""
        from qitos.engine.action_executor import ActionExecutor
        from qitos.core.action import Action, ActionStatus

        rbw = ReadBeforeWriteEnforcer()

        @tool(name="edit_file")
        def edit_file(path: str) -> str:
            return path

        registry = ToolRegistry().register(edit_file)

        executor = ActionExecutor(
            tool_registry=registry,
            read_before_write_enforcer=rbw,
        )

        # Create a real temp file that exists but hasn't been read
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("hello")
            path = f.name

        try:
            action = Action(name="edit_file", args={"path": path})
            results = await executor.execute([action])
            assert results[0].status == ActionStatus.DENIED
            assert results[0].output.get("error_category") == "read_before_write"
        finally:
            os.unlink(path)
