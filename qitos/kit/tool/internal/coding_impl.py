"""Canonical coding-oriented toolset backed by method-style tool definitions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from qitos.core.env import (
    AtomicFileWrite,
    CommandCapability,
    FileRevisionConflictError,
    FileSystemCapability,
    RuntimeCapabilitySnapshot,
    RuntimeCapabilityUnavailableError,
)
from qitos.core.function_tool_decorator import function_tool
from qitos.core.process import (
    ProcessHandle,
    ProcessSnapshot,
    ProcessStatus,
    ProcessTerminalNotifier,
)
from qitos.core.runtime_input import process_terminal_runtime_input
from qitos.core.tool_result import ToolResult, ToolResultStatus
from qitos.kit.env.host_env import HostCommandCapability, HostFSCapability
from qitos.kit._html import extract_html_text
from qitos.kit.tool.internal.coding_utils import (
    build_diff,
    default_rule_scope,
    detect_line_ending,
    resolve_tool_workspace_path,
    truncate_text,
    utc_now,
)
from qitos.kit.tool.internal.results import error_result, tool_result
from qitos.kit.tool.internal.work_plan import UpdateWorkPlanTool
from qitos.kit.tool.internal.runtime_ops import select_runtime_ops
from qitos.kit.tool.notebook import NotebookToolSet

TASK_STATUSES = {"pending", "in_progress", "blocked", "completed", "cancelled"}
_MAX_SEARCH_RESULTS = 2000
_PROCESS_MODEL_SUMMARY_MAX_CHARS = 8_000


def _utc_now() -> str:
    return utc_now()


def _resolve_workspace_path(root_dir: str, path: str) -> Path:
    return resolve_tool_workspace_path(root_dir, path)


def _detect_line_ending(raw: bytes) -> str:
    return detect_line_ending(raw)


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    return truncate_text(text, max_chars)


def _build_diff(old_content: str, new_content: str, path: str) -> str:
    return build_diff(old_content, new_content, path)


def _default_rule_scope(args: Dict[str, Any]) -> Optional[str]:
    return default_rule_scope(args)


def _process_rule_scope(args: Dict[str, Any]) -> Optional[str]:
    process_id = str(args.get("process_id") or "").strip()
    return f"process:{process_id}" if process_id else None


def _non_negative_seconds(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return seconds


def _bounded_head_tail(content: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(content) <= max_chars:
        return content

    marker = "\n\n--- output omitted ---\n\n"
    for _ in range(3):
        content_budget = max(0, max_chars - len(marker))
        head_chars = content_budget // 4
        tail_chars = content_budget - head_chars
        omitted = max(0, len(content) - head_chars - tail_chars)
        next_marker = f"\n\n--- omitted {omitted:,} chars ---\n\n"
        if next_marker == marker:
            break
        marker = next_marker

    content_budget = max(0, max_chars - len(marker))
    head_chars = content_budget // 4
    tail_chars = content_budget - head_chars
    omitted = max(0, len(content) - head_chars - tail_chars)
    marker = f"\n\n--- omitted {omitted:,} chars ---\n\n"
    content_budget = max(0, max_chars - len(marker))
    head_chars = content_budget // 4
    tail_chars = content_budget - head_chars
    return (
        content[:head_chars]
        + marker
        + (content[-tail_chars:] if tail_chars else "")
    )[:max_chars]


def _tool_status_for_process(snapshot: ProcessSnapshot) -> str:
    if snapshot.status is ProcessStatus.RUNNING:
        return "running"
    if snapshot.status is ProcessStatus.EXITED:
        return "success" if snapshot.exit_code == 0 else "partial"
    if snapshot.status is ProcessStatus.TERMINATED:
        return "success"
    return "error"


def _process_snapshot_payload(snapshot: ProcessSnapshot) -> Dict[str, Any]:
    output = snapshot.output
    state = "running" if snapshot.status is ProcessStatus.RUNNING else "terminal"
    lines = [
        f"[process {state}]",
        f"process_status: {snapshot.status.value}",
        f"process_id: {snapshot.handle.process_id}",
        f"cwd: {snapshot.cwd}",
        f"exit_code: {snapshot.exit_code if snapshot.exit_code is not None else 'n/a'}",
        f"output_bytes: {output.total_bytes}",
        f"omitted_bytes: {output.omitted_bytes}",
        f"full_log: {output.log_path}",
    ]
    if snapshot.error:
        lines.append(f"error: {snapshot.error}")
    if snapshot.status is ProcessStatus.RUNNING:
        lines.append(
            "next: continue other work or use process_read/process_wait with this process_id"
        )
    header = "\n".join(lines)
    if output.content:
        output_header = "\n\nOutput:\n"
        content_budget = max(
            0,
            _PROCESS_MODEL_SUMMARY_MAX_CHARS - len(header) - len(output_header),
        )
        summary = header + output_header + _bounded_head_tail(
            output.content,
            content_budget,
        )
    else:
        summary = header
    payload = snapshot.to_dict()
    payload["process_status"] = snapshot.status.value
    payload["status"] = _tool_status_for_process(snapshot)
    payload["model_summary"] = summary[:_PROCESS_MODEL_SUMMARY_MAX_CHARS]
    return payload


def _process_tool_output(snapshot: ProcessSnapshot) -> Dict[str, Any] | ToolResult:
    """Project one process snapshot at the Coding Tool boundary."""

    payload = _process_snapshot_payload(snapshot)
    status = _tool_status_for_process(snapshot)
    if status == "partial":
        return tool_result(payload, status="partial")
    if status == "error":
        return tool_result(payload, status="error")
    # A running background process is domain state; starting it completed the
    # current Tool call successfully.
    return payload


def _foreground_command_output(
    payload: Mapping[str, Any],
) -> Dict[str, Any] | ToolResult:
    """Convert the documented CommandCapability failure states explicitly."""

    output = dict(payload)
    status = str(output.get("status") or "").strip().lower()
    if status == "timed_out":
        return tool_result(output, status="timed_out")
    if status == "cancelled":
        return tool_result(output, status="cancelled")
    if status == "error":
        return tool_result(output, status="error")
    return output


def _join_capability_path(base: str, child: str) -> str:
    normalized_base = str(base or ".").replace("\\", "/").strip("/")
    normalized_child = str(child or "").replace("\\", "/")
    while normalized_child.startswith("./"):
        normalized_child = normalized_child[2:]
    if not normalized_child or normalized_child.startswith("/"):
        raise ValueError(f"process returned invalid workspace path: {child}")
    parts = normalized_child.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"process returned invalid workspace path: {child}")
    if normalized_base in {"", "."}:
        return normalized_child
    return f"{normalized_base}/{normalized_child}"


def _capability_basename(path: str) -> str:
    return str(path or ".").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


class _SearchProcessError(RuntimeError):
    """Structured failure raised by the ripgrep process boundary."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        status: str = "error",
        exit_code: int | None = None,
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status = status
        self.exit_code = exit_code
        self.stderr = stderr


def _search_limit(value: int) -> int:
    limit = int(value)
    if not 1 <= limit <= _MAX_SEARCH_RESULTS:
        raise ValueError(f"limit must be between 1 and {_MAX_SEARCH_RESULTS}")
    return limit


def _search_process_result(
    result: Dict[str, Any], operation: str
) -> tuple[int, str, str]:
    raw_exit_code = result.get("returncode")
    status = str(result.get("status") or "").strip().lower()
    stderr = str(result.get("stderr") or result.get("error") or "")
    if status in {"timed_out", "cancelled", "error"}:
        category = {
            "timed_out": "process_timeout",
            "cancelled": "process_cancelled",
            "error": "process_error",
        }[status]
        raise _SearchProcessError(
            stderr.strip() or f"{operation} process {status}",
            category=category,
            status=status,
            exit_code=int(raw_exit_code) if raw_exit_code is not None else None,
            stderr=stderr,
        )
    if raw_exit_code is None:
        raise _SearchProcessError(
            f"{operation} process returned no exit code",
            category="invalid_process_result",
            stderr=stderr,
        )

    exit_code = int(raw_exit_code)
    if exit_code not in {0, 1}:
        raise _SearchProcessError(
            stderr.strip() or f"ripgrep exited with code {exit_code}",
            category="search_process_error",
            exit_code=exit_code,
            stderr=stderr,
        )
    return exit_code, str(result.get("stdout") or ""), stderr


def _search_error_payload(
    error: Exception,
    *,
    pattern: str,
    path: str,
) -> ToolResult:
    if isinstance(error, _SearchProcessError):
        payload = {
            "status": error.status,
            "message": str(error),
            "error_category": error.category,
            "exit_code": error.exit_code,
            "stderr": error.stderr,
            "pattern": pattern,
            "path": path,
        }
        status: ToolResultStatus
        if error.status == "timed_out":
            status = "timed_out"
        elif error.status == "cancelled":
            status = "cancelled"
        else:
            status = "error"
        return tool_result(payload, status=status)
    if isinstance(error, subprocess.TimeoutExpired):
        stderr = error.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        payload = {
            "status": "timed_out",
            "message": str(error),
            "error_category": "process_timeout",
            "exit_code": None,
            "stderr": stderr,
            "pattern": pattern,
            "path": path,
        }
        return tool_result(payload, status="timed_out")
    if isinstance(error, FileNotFoundError):
        payload = {
            "status": "error",
            "message": f"ripgrep is unavailable: {error}",
            "error_category": "search_tool_unavailable",
            "exit_code": None,
            "stderr": "",
            "pattern": pattern,
            "path": path,
        }
        return tool_result(payload, status="error")
    payload = {
        "status": "error",
        "message": str(error),
        "error_category": (
            "invalid_search_arguments"
            if isinstance(error, (TypeError, ValueError))
            else "search_error"
        ),
        "pattern": pattern,
        "path": path,
    }
    return tool_result(payload, status="error")


class CodingToolSet:
    """Canonical coding toolset with one stable, traditional tool surface."""

    name = "coding"
    version = "2"
    _PROFILE_TOOL_NAMES = {
        "workspace": (
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "hex_view",
            "list_files",
            "list_tree",
            "make_directory",
        ),
        "editor": (
            "read_file",
            "write_file",
            "edit_file",
            "list_files",
            "list_tree",
            "make_directory",
        ),
        "codebase": (
            "read_file",
            "glob",
            "grep",
            "hex_view",
            "list_files",
            "list_tree",
        ),
        "files": (
            "read_file",
            "write_file",
            "edit_file",
            "list_files",
            "list_tree",
            "make_directory",
        ),
        "shell": (
            "run_command",
            "process_list",
            "process_read",
            "process_write",
            "process_wait",
            "process_terminate",
        ),
        "web": ("web_fetch",),
        "full": (
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "hex_view",
            "list_files",
            "list_tree",
            "make_directory",
            "run_command",
            "process_list",
            "process_read",
            "process_write",
            "process_wait",
            "process_terminate",
            "web_fetch",
            "ask_user_choice",
            "update_plan",
            "tool_search",
            "enter_plan_mode",
            "exit_plan_mode",
            "enter_worktree",
            "exit_worktree",
            "mcp_list_resources",
            "mcp_read_resource",
            "cron_create",
            "cron_delete",
            "cron_list",
        ),
    }

    def __init__(
        self,
        workspace_root: str = ".",
        shell_timeout: int = 30,
        include_notebook: bool = True,
        *,
        enable_lsp: bool = True,
        enable_tasks: bool = True,
        enable_web: bool = True,
        profile: str = "full",
        include_http_tools: bool = False,
        auto_approve: bool = False,
        allow_local_fallback: bool = True,
        process_env: Mapping[str, str] | None = None,
    ):
        self.workspace_root = os.path.abspath(workspace_root)
        self.shell_timeout = int(shell_timeout)
        self.include_notebook = bool(include_notebook)
        self.enable_lsp = bool(enable_lsp)
        self.enable_tasks = bool(enable_tasks)
        self.enable_web = bool(enable_web)
        self.profile = str(profile or "full")
        if self.profile not in self._PROFILE_TOOL_NAMES:
            supported = ", ".join(sorted(self._PROFILE_TOOL_NAMES))
            raise ValueError(
                f"Unsupported coding tool profile {self.profile!r}; expected {supported}"
            )
        self.include_http_tools = bool(include_http_tools)
        self.auto_approve = bool(auto_approve)
        self.allow_local_fallback = bool(allow_local_fallback)
        self._local_file_ops = (
            HostFSCapability(self.workspace_root) if self.allow_local_fallback else None
        )
        self._local_process_ops = (
            HostCommandCapability(self.workspace_root, env=process_env)
            if self.allow_local_fallback
            else None
        )
        self._notebook = (
            NotebookToolSet(workspace_root=self.workspace_root)
            if self.include_notebook
            else None
        )
        self._session_tasks: Dict[str, Dict[str, Any]] = {}
        self._task_counter = 0
        self.update_plan = UpdateWorkPlanTool()

    def setup(self, context: Dict[str, Any]) -> None:
        _ = context

    def teardown(self, context: Dict[str, Any]) -> None:
        _ = context

    async def ateardown(self, context: Dict[str, Any]) -> None:
        _ = context
        if self._local_process_ops is not None:
            await self._local_process_ops.aclose()

    def tools(self) -> List[Any]:
        tool_names = self._PROFILE_TOOL_NAMES[self.profile]
        items = [
            getattr(self, name)
            for name in tool_names
            if self.enable_web or name != "web_fetch"
        ]
        if (
            self.profile in {"full", "web"}
            and self.enable_web
            and self.include_http_tools
        ):
            items.extend(
                [
                    self.http_request,
                    self.http_get,
                    self.http_post,
                    self.extract_web_text,
                ]
            )
        if self.profile == "full":
            if self.enable_lsp:
                items.append(self.lsp_query)
            if self.enable_tasks:
                items.extend(
                    [self.task_create, self.task_get, self.task_list, self.task_update]
                )
            if self._notebook is not None:
                items.extend(self._notebook.tools())
        if not self.allow_local_fallback or self.auto_approve:
            bound_items: List[Any] = []
            for item in items:
                isolated = copy(item)
                isolated.spec = copy(item.spec)
                if hasattr(item, "meta"):
                    isolated.meta = copy(item.meta)
                if not self.allow_local_fallback:
                    isolated.spec.required_ops = list(
                        dict.fromkeys(
                            [
                                *list(isolated.spec.required_ops or []),
                                *list(isolated.spec.environment_ops or []),
                            ]
                        )
                    )
                if self.auto_approve:
                    if hasattr(isolated, "meta"):
                        isolated.meta.needs_approval = False
                    isolated.spec.needs_approval = False
                bound_items.append(isolated)
            items = bound_items
        return items

    def _file_ops(
        self, runtime_context: Optional[Dict[str, Any]]
    ) -> FileSystemCapability:
        return select_runtime_ops(runtime_context, "file", self._local_file_ops)

    def _process_ops(
        self, runtime_context: Optional[Dict[str, Any]]
    ) -> CommandCapability:
        return select_runtime_ops(runtime_context, "process", self._local_process_ops)

    async def _managed_process_ops(
        self,
        runtime_context: Optional[Dict[str, Any]],
    ) -> tuple[CommandCapability, str]:
        context = runtime_context or {}
        self._require_runtime_facility(context, "process.background")
        run_id = str(context.get("run_id") or "").strip()
        if not run_id:
            raise RuntimeError("managed process tools require an active Run")
        process_ops = self._process_ops(runtime_context)
        journal = context.get("journal")
        if journal is not None:
            required = ("run_id", "replay", "append")
            if any(not hasattr(journal, name) for name in required):
                raise TypeError("runtime journal does not satisfy SessionJournal")
            if str(journal.run_id or "") != run_id:
                raise RuntimeError("managed process journal belongs to another Run")
            await process_ops.arecover(owner_run_id=run_id, journal=journal)
        return process_ops, run_id

    @staticmethod
    def _require_runtime_facility(
        runtime_context: Mapping[str, Any],
        facility: str,
    ) -> None:
        snapshot = runtime_context.get("runtime_capabilities")
        if snapshot is None:
            return
        if not isinstance(snapshot, RuntimeCapabilitySnapshot):
            raise TypeError("runtime_capabilities must be a RuntimeCapabilitySnapshot")
        if not snapshot.has_facility(facility):
            raise RuntimeCapabilityUnavailableError(
                facility,
                backend=snapshot.backend,
            )

    @staticmethod
    def _process_error_payload(
        exc: Exception,
        **fields: Any,
    ) -> ToolResult:
        payload: Dict[str, Any] = {
            "status": "error",
            "message": str(exc),
            **fields,
        }
        if isinstance(exc, RuntimeCapabilityUnavailableError):
            payload.update(
                {
                    "error_category": "capability_unavailable",
                    "backend": exc.backend,
                    "facility": exc.facility,
                }
            )
        return tool_result(payload, status="error")

    @staticmethod
    def _remember_process_interruption(
        runtime_context: Dict[str, Any],
        snapshot: ProcessSnapshot,
    ) -> None:
        """Keep a bounded process receipt available if the tool task is cancelled."""

        projected = _process_tool_output(snapshot)
        runtime_context["interruption_result"] = ToolResult.from_value(projected)

    async def _refresh_process_interruption(
        self,
        runtime_context: Dict[str, Any],
        process_ops: CommandCapability,
        handle: ProcessHandle,
    ) -> None:
        try:
            snapshot = await asyncio.shield(process_ops.apoll(handle))
        except Exception:
            return
        self._remember_process_interruption(runtime_context, snapshot)

    @staticmethod
    def _process_handle(process_id: str, owner_run_id: str) -> ProcessHandle:
        return ProcessHandle(
            process_id=str(process_id or "").strip(),
            owner_run_id=owner_run_id,
        )

    @staticmethod
    def _process_terminal_notifier(
        runtime_context: Optional[Dict[str, Any]],
    ) -> ProcessTerminalNotifier | None:
        post_runtime_event = (runtime_context or {}).get("post_runtime_event")
        if not callable(post_runtime_event):
            return None

        async def notify(snapshot: ProcessSnapshot) -> bool:
            event = process_terminal_runtime_input(snapshot)
            return bool(await post_runtime_event(event))

        return notify

    @staticmethod
    def _remaining_seconds(
        runtime_context: Optional[Dict[str, Any]],
    ) -> float | None:
        remaining = (runtime_context or {}).get("remaining_seconds")
        if not callable(remaining):
            return None
        value = remaining()
        return None if value is None else max(0.0, float(value))

    def _require_search_directory(
        self,
        path: str,
        runtime_context: Optional[Dict[str, Any]],
    ) -> None:
        try:
            info = self._file_ops(runtime_context).stat(path)
        except FileNotFoundError as exc:
            raise _SearchProcessError(
                f"Search path not found: {path}",
                category="search_path_not_found",
            ) from exc
        if not info.is_directory:
            raise _SearchProcessError(
                f"Path is not a directory: {path}",
                category="invalid_search_path",
            )

    def _read_text_file(
        self,
        path: str,
        runtime_context: Optional[Dict[str, Any]],
    ) -> tuple[str, str, float, str]:
        file_ops = self._file_ops(runtime_context)
        raw = file_ops.read_bytes(path)
        modified_at = file_ops.stat(path).modified_at
        return (
            raw.decode("utf-8", errors="strict"),
            _detect_line_ending(raw),
            float(modified_at or 0.0),
            hashlib.sha256(raw).hexdigest(),
        )

    def _write_text_file(
        self,
        path: str,
        content: str,
        line_ending: str,
        runtime_context: Optional[Dict[str, Any]],
        *,
        expected_sha256: str | None = None,
    ) -> AtomicFileWrite:
        normalized = (
            content.replace("\r\n", "\n").replace("\r", "\n").replace("\n", line_ending)
        )
        return self._file_ops(runtime_context).write_text_atomic(
            path,
            normalized,
            expected_sha256=expected_sha256,
        )

    async def _run_rg_files(
        self,
        target_dir: str,
        pattern: str,
        include_hidden: bool,
        include_ignored: bool,
        runtime_context: Optional[Dict[str, Any]],
    ) -> tuple[List[str], int, str]:
        cmd = ["rg", "--files", "--sort=path", "--null", "--glob", pattern]
        if include_hidden:
            cmd.append("--hidden")
        if include_ignored:
            cmd.append("--no-ignore")
        cmd.append(".")
        result = await self._process_ops(runtime_context).arun_argv(
            cmd,
            timeout=self.shell_timeout,
            cwd=target_dir,
        )
        exit_code, stdout, stderr = _search_process_result(result, "glob")
        matches = sorted(
            _join_capability_path(target_dir, row)
            for row in stdout.split("\0")
            if row
        )
        return matches, exit_code, stderr

    async def _run_rg_grep(
        self,
        pattern: str,
        target_dir: str,
        glob: Optional[str],
        case_sensitive: bool,
        regex: bool,
        files_with_matches: bool,
        context: int,
        file_type: Optional[str],
        include_hidden: bool,
        include_ignored: bool,
        runtime_context: Optional[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, str]:
        cmd = [
            "rg",
            "--color",
            "never",
            "--sort=path",
            "--max-columns",
            "2000",
            "--max-columns-preview",
        ]
        if files_with_matches:
            cmd.extend(["--files-with-matches", "--null"])
        else:
            cmd.append("--json")
        if not regex:
            cmd.append("-F")
        if not case_sensitive:
            cmd.append("-i")
        if glob:
            cmd.extend(["--glob", glob])
        if file_type:
            cmd.extend(["--type", file_type])
        if include_hidden:
            cmd.append("--hidden")
        if include_ignored:
            cmd.append("--no-ignore")
        if context > 0 and not files_with_matches:
            cmd.extend(["--context", str(context)])
        cmd.extend(["--", pattern, "."])
        result = await self._process_ops(runtime_context).arun_argv(
            cmd,
            timeout=self.shell_timeout,
            cwd=target_dir,
        )
        exit_code, stdout, stderr = _search_process_result(result, "grep")
        matches: List[Dict[str, Any]] = []
        if files_with_matches:
            files = sorted(
                _join_capability_path(target_dir, row)
                for row in stdout.split("\0")
                if row
            )
            return ([{"path": path} for path in files], [], exit_code, stderr)

        records: List[Dict[str, Any]] = []
        for row in stdout.splitlines():
            if not row.strip():
                continue
            try:
                event = json.loads(row)
            except json.JSONDecodeError as exc:
                raise _SearchProcessError(
                    f"ripgrep returned invalid JSON: {exc}",
                    category="invalid_search_output",
                    exit_code=exit_code,
                    stderr=stderr,
                ) from exc
            event_type = str(event.get("type") or "")
            if event_type not in {"match", "context"}:
                continue
            data = event.get("data") or {}
            raw_path = str((data.get("path") or {}).get("text") or "")
            line_number = int(data.get("line_number") or 0)
            text = str((data.get("lines") or {}).get("text") or "").rstrip("\r\n")
            record = {
                "kind": event_type,
                "path": _join_capability_path(target_dir, raw_path),
                "line": line_number,
                "text": text,
            }
            records.append(record)
            if event_type == "match":
                matches.append(
                    {key: value for key, value in record.items() if key != "kind"}
                )
        matches.sort(key=lambda item: (item["path"], item["line"], item["text"]))
        records.sort(
            key=lambda item: (
                item["path"],
                item["line"],
                0 if item["kind"] == "match" else 1,
                item["text"],
            )
        )
        return matches, records, exit_code, stderr

    def _tree_lines(
        self,
        path: str,
        depth: int,
        runtime_context: Optional[Dict[str, Any]],
    ) -> List[str]:
        file_ops = self._file_ops(runtime_context)
        display_name = path.rstrip("/").rsplit("/", maxsplit=1)[-1] or "."
        lines: List[str] = [f"{display_name}/"]

        def walk(current: str, indent: str, current_depth: int) -> None:
            if current_depth >= depth:
                return
            items = sorted(
                [
                    entry
                    for entry in file_ops.list_entries(current)
                    if not _capability_basename(entry.path).startswith(".")
                ],
                key=lambda entry: (
                    entry.is_file,
                    _capability_basename(entry.path),
                ),
            )
            for index, item in enumerate(items):
                is_last = index == len(items) - 1
                connector = "`-- " if is_last else "|-- "
                lines.append(
                    f"{indent}{connector}{_capability_basename(item.path)}"
                )
                if item.is_directory:
                    walk(
                        item.path,
                        indent + ("    " if is_last else "|   "),
                        current_depth + 1,
                    )

        walk(path, "", 1)
        return lines

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        verify_tls: bool = True,
        allow_redirects: bool = True,
        max_content_chars: int = 120_000,
    ) -> Dict[str, Any] | ToolResult:
        parsed = urlparse(str(url or ""))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            payload = {
                "status": "error",
                "message": "URL must be an absolute http or https URL",
                "url": url,
            }
            return tool_result(payload, status="error")
        try:
            async with httpx.AsyncClient(
                verify=verify_tls,
                follow_redirects=allow_redirects,
            ) as client:
                response = await client.request(
                    method=str(method or "GET").upper(),
                    url=url,
                    params=params,
                    data=data,
                    json=json_data,
                    headers=headers,
                    timeout=int(timeout or 30),
                )
            text, truncated = _truncate_text(response.text, max_content_chars)
            payload: Dict[str, Any] = {
                "status": "success" if response.status_code < 400 else "error",
                "ok": bool(response.ok),
                "method": str(method or "GET").upper(),
                "url": str(response.url),
                "status_code": response.status_code,
                "reason": response.reason,
                "headers": dict(response.headers),
                "content_type": response.headers.get("Content-Type", ""),
                "content": text,
                "content_length": len(text),
                "truncated": truncated,
                "history": [str(item.url) for item in response.history],
            }
            try:
                payload["json"] = response.json()
            except Exception:
                pass
            if response.status_code >= 400:
                return tool_result(payload, status="error")
            return payload
        except Exception as e:
            payload = {
                "status": "error",
                "message": str(e),
                "url": url,
                "method": method,
            }
            return tool_result(payload, status="error")

    def _extract_html_text(self, html: str) -> Dict[str, Any]:
        extracted = extract_html_text(html or "", layout="lines")
        return {
            "status": "success",
            "title": extracted.title or "",
            "text": extracted.text,
        }

    def _next_task_id(self) -> str:
        self._task_counter += 1
        return f"task-{self._task_counter:03d}"

    @function_tool(
        name="run_command",
        needs_approval=True,
        supports_background=True,
        environment_ops=["process"],
        rule_scope_builder=_default_rule_scope,
    )
    async def run_command(
        self,
        command: str,
        run_in_background: bool = False,
        tty: bool = False,
        yield_time_ms: int = 10000,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """
        Execute one shell command inside the configured working directory.

        :param command: Shell command string to execute.
        :param run_in_background: Detach the command and return its task handle.
        :param tty: Allocate a pseudo-terminal for a managed background command.
        :param yield_time_ms: Initial wait before a live command returns its handle.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        text = str(command or "").strip()
        if not text:
            return tool_result(
                {"status": "error", "message": "Command cannot be empty"},
                status="error",
            )
        try:
            yield_seconds = _non_negative_seconds(
                yield_time_ms,
                name="yield_time_ms",
            ) / 1000
            context = runtime_context or {}
            run_id = str(context.get("run_id") or "").strip()
            if not run_id and not run_in_background and not tty:
                self._require_runtime_facility(context, "process.foreground")
                process_ops = self._process_ops(runtime_context)
                timeout = float(self.shell_timeout)
                remaining = self._remaining_seconds(runtime_context)
                if remaining is not None:
                    timeout = min(timeout, remaining)
                if timeout <= 0:
                    payload = {
                        "status": "error",
                        "message": "command deadline expired before execution",
                        "command": text,
                    }
                    return tool_result(payload, status="error")
                return _foreground_command_output(
                    await process_ops.arun(text, timeout=timeout)
                )
            if tty:
                self._require_runtime_facility(
                    context,
                    "process.pty",
                )
            process_ops, run_id = await self._managed_process_ops(runtime_context)
            snapshot = await process_ops.astart(
                text,
                owner_run_id=run_id,
                tty=bool(tty),
                journal=context.get("journal"),
                terminal_notifier=self._process_terminal_notifier(context),
            )
            self._remember_process_interruption(context, snapshot)
            if run_in_background:
                return _process_tool_output(snapshot)
            deadline = asyncio.get_running_loop().time() + yield_seconds
            raw_deadline = context.get("deadline_monotonic")
            if raw_deadline is not None:
                deadline = min(deadline, float(raw_deadline))
            try:
                snapshot = await process_ops.await_process(
                    snapshot.handle,
                    deadline_monotonic=deadline,
                )
            except asyncio.CancelledError:
                await self._refresh_process_interruption(
                    context,
                    process_ops,
                    snapshot.handle,
                )
                raise
            self._remember_process_interruption(context, snapshot)
            return _process_tool_output(snapshot)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._process_error_payload(exc, command=text)

    @function_tool(
        name="process_list",
        read_only=True,
        concurrency_safe=True,
        environment_ops=["process"],
    )
    async def process_list(
        self,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """List background processes owned by the active Run."""

        try:
            process_ops, run_id = await self._managed_process_ops(runtime_context)
            snapshots = await process_ops.alist(owner_run_id=run_id)
            return {
                "status": "success",
                "processes": [snapshot.to_dict() for snapshot in snapshots],
            }
        except Exception as exc:
            return self._process_error_payload(exc)

    @function_tool(
        name="process_read",
        read_only=True,
        concurrency_safe=True,
        environment_ops=["process"],
    )
    async def process_read(
        self,
        process_id: str,
        cursor: int = 0,
        wait_seconds: float = 0.0,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """Read incremental output from one active-Run process."""

        try:
            process_ops, run_id = await self._managed_process_ops(runtime_context)
            context = runtime_context or {}
            remaining = self._remaining_seconds(runtime_context)
            bounded_wait = _non_negative_seconds(
                wait_seconds,
                name="wait_seconds",
            )
            if remaining is not None:
                bounded_wait = min(bounded_wait, remaining)
            handle = self._process_handle(process_id, run_id)
            initial = await process_ops.apoll(handle)
            self._remember_process_interruption(context, initial)
            try:
                snapshot = await process_ops.aread(
                    handle,
                    cursor=int(cursor),
                    wait_seconds=bounded_wait,
                )
            except asyncio.CancelledError:
                await self._refresh_process_interruption(context, process_ops, handle)
                raise
            self._remember_process_interruption(context, snapshot)
            return _process_tool_output(snapshot)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._process_error_payload(exc, process_id=process_id)

    @function_tool(
        name="process_write",
        needs_approval=True,
        environment_ops=["process"],
        rule_scope_builder=_process_rule_scope,
    )
    async def process_write(
        self,
        process_id: str,
        data: str,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """Write UTF-8 input to one active-Run process."""

        try:
            process_ops, run_id = await self._managed_process_ops(runtime_context)
            snapshot = await process_ops.awrite(
                self._process_handle(process_id, run_id),
                data,
            )
            return _process_tool_output(snapshot)
        except Exception as exc:
            return self._process_error_payload(exc, process_id=process_id)

    @function_tool(
        name="process_wait",
        read_only=True,
        concurrency_safe=True,
        environment_ops=["process"],
    )
    async def process_wait(
        self,
        process_id: str,
        timeout_seconds: Optional[float] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """Wait for one process without exceeding the current absolute deadline."""

        try:
            process_ops, run_id = await self._managed_process_ops(runtime_context)
            context = runtime_context or {}
            raw_deadline = context.get("deadline_monotonic")
            deadline = None if raw_deadline is None else float(raw_deadline)
            if timeout_seconds is not None:
                relative = asyncio.get_running_loop().time() + _non_negative_seconds(
                    timeout_seconds,
                    name="timeout_seconds",
                )
                deadline = relative if deadline is None else min(deadline, relative)
            handle = self._process_handle(process_id, run_id)
            initial = await process_ops.apoll(handle)
            self._remember_process_interruption(context, initial)
            try:
                snapshot = await process_ops.await_process(
                    handle,
                    deadline_monotonic=deadline,
                )
            except asyncio.CancelledError:
                await self._refresh_process_interruption(context, process_ops, handle)
                raise
            self._remember_process_interruption(context, snapshot)
            return _process_tool_output(snapshot)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._process_error_payload(exc, process_id=process_id)

    @function_tool(
        name="process_terminate",
        environment_ops=["process"],
        rule_scope_builder=_process_rule_scope,
    )
    async def process_terminate(
        self,
        process_id: str,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """Terminate one active-Run process group and await cleanup."""

        try:
            process_ops, run_id = await self._managed_process_ops(runtime_context)
            snapshot = await process_ops.aterminate(
                self._process_handle(process_id, run_id)
            )
            return _process_tool_output(snapshot)
        except Exception as exc:
            return self._process_error_payload(exc, process_id=process_id)

    def _read_file_chunk(
        self,
        path: str,
        offset: int = 0,
        limit: int = 200,
        max_chars: int = 20_000,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """
        Read one workspace file as a bounded whole-line text chunk.

        :param path: Path relative to the workspace root.
        :param offset: Zero-based starting line offset.
        :param limit: Maximum number of lines to return.
        :param max_chars: Soft maximum characters; the returned chunk stops at a line
            boundary just after reaching this value.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        try:
            file_ops = self._file_ops(runtime_context)
            info = file_ops.stat(path)
            if info.is_directory:
                return error_result(
                    {"status": "error", "message": f"Path is a directory: {path}"}
                )
            start = max(0, int(offset))
            size = min(1000, max(1, int(limit)))
            byte_limit = 100 * 1024
            if int(max_chars) > 0:
                byte_limit = min(byte_limit, int(max_chars))
            chunk = file_ops.read_text_chunk(
                path,
                offset=start,
                limit=size,
                max_bytes=byte_limit,
                max_line_bytes=2000,
            )
            return {
                "status": "success",
                "path": str(path),
                "content": chunk.content,
                "line_ending": chunk.line_ending,
                "offset": chunk.offset,
                "limit": chunk.line_count,
                "total_lines": chunk.total_lines,
                "size_bytes": chunk.size_bytes,
                "content_sha256": chunk.content_sha256,
                "has_more": chunk.has_more,
                "truncated": chunk.truncated,
            }
        except FileNotFoundError:
            return error_result(
                {"status": "error", "message": f"File not found: {path}"}
            )
        except Exception as e:
            return error_result({"status": "error", "message": str(e), "path": path})

    @function_tool(
        name="read_file",
        read_only=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def read_file(
        self,
        path: str,
        line_offset: int = 0,
        line_count: int = 1000,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """
        Read a bounded text chunk from a workspace file.

        :param path: Path relative to the workspace root.
        :param line_offset: Zero-based first line to return.
        :param line_count: Maximum whole lines to return, capped at 1000.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        result = self._read_file_chunk(
            path=path,
            offset=line_offset,
            limit=line_count,
            max_chars=100 * 1024,
            runtime_context=runtime_context,
        )
        if isinstance(result, ToolResult):
            return result
        offset = int(result.get("offset", line_offset))
        content = str(result.get("content", ""))
        result["numbered_content"] = "\n".join(
            f"{number}\t{line}"
            for number, line in enumerate(content.splitlines(), start=offset + 1)
        )
        result["line_offset"] = offset
        result["line_count"] = int(result.get("limit", 0))
        result.pop("limit", None)
        result.pop("offset", None)
        return result

    @function_tool(
        name="list_files",
        read_only=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def list_files(
        self, path: str = ".", runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any] | ToolResult:
        """
        List files and directories under a workspace-relative path.

        :param path: Directory path relative to the workspace root.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        try:
            file_ops = self._file_ops(runtime_context)
            if not file_ops.stat(path).is_directory:
                return error_result(
                    {
                        "status": "error",
                        "message": f"Path is not a directory: {path}",
                    }
                )
            items = []
            for item in sorted(
                file_ops.list_entries(path),
                key=lambda entry: (
                    entry.is_file,
                    _capability_basename(entry.path),
                ),
            ):
                name = _capability_basename(item.path)
                if name.startswith("."):
                    continue
                items.append(
                    {
                        "name": name,
                        "type": "directory" if item.is_directory else "file",
                        "size": item.size if item.is_file else None,
                    }
                )
            return {
                "status": "success",
                "path": path,
                "count": len(items),
                "files": items,
            }
        except Exception as e:
            return error_result({"status": "error", "message": str(e), "path": path})

    @function_tool(
        name="write_file",
        needs_approval=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def write_file(
        self,
        path: str,
        content: str,
        expected_sha256: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """
        Write text content to a workspace file.

        :param path: Path relative to the workspace root.
        :param content: Full text content to write into the file.
        :param expected_sha256: Optional complete-file revision required before replace.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        try:
            write = self._write_text_file(
                path,
                str(content),
                "\n",
                runtime_context,
                expected_sha256=expected_sha256,
            )
            return {
                "status": "success",
                "path": path,
                "size": write.size_bytes,
                "content_sha256": write.content_sha256,
                "previous_sha256": write.previous_sha256,
                "created": write.created,
            }
        except FileRevisionConflictError as exc:
            return self._revision_conflict_result(exc)
        except Exception as e:
            return error_result({"status": "error", "message": str(e), "path": path})

    @function_tool(
        name="edit_file",
        needs_approval=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
        expected_mtime: Optional[float] = None,
        expected_sha256: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """Replace exact text in one workspace file.

        :param path: Path relative to the workspace root.
        :param old_text: Exact text to replace. It must be unique by default.
        :param new_text: Replacement text.
        :param replace_all: Replace every occurrence instead of requiring uniqueness.
        :param expected_mtime: Optional optimistic concurrency check.
        :param expected_sha256: Optional complete-file revision required before edit.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        try:
            file_ops = self._file_ops(runtime_context)
            if not file_ops.stat(path).is_file:
                return error_result(
                    {"status": "error", "message": f"Not a file: {path}"}
                )
            old_content, line_ending, current_mtime, current_sha256 = self._read_text_file(
                path,
                runtime_context,
            )
            if (
                expected_mtime is not None
                and abs(float(expected_mtime) - float(current_mtime)) > 1e-6
            ):
                return error_result(
                    {
                        "status": "error",
                        "message": "File was modified since the expected mtime.",
                        "path": path,
                    }
                )
            if not old_text:
                return error_result(
                    {
                        "status": "error",
                        "message": "old_text cannot be empty",
                        "path": path,
                    }
                )
            if old_text == new_text:
                return error_result(
                    {
                        "status": "error",
                        "message": "old_text and new_text are identical",
                        "path": path,
                    }
                )
            count = old_content.count(old_text)
            if count == 0:
                return error_result(
                    {
                        "status": "error",
                        "message": f"Text not found in {path}",
                        "path": path,
                    }
                )
            if count > 1 and not replace_all:
                return error_result(
                    {
                        "status": "error",
                        "message": "Text replacement must be unique",
                        "path": path,
                        "occurrences": count,
                    }
                )
            replacement_count = count if replace_all else 1
            new_content = old_content.replace(
                old_text,
                new_text,
                -1 if replace_all else 1,
            )
            write = self._write_text_file(
                path,
                new_content,
                line_ending,
                runtime_context,
                expected_sha256=expected_sha256 or current_sha256,
            )
            return {
                "status": "success",
                "path": path,
                "message": (
                    f"Replaced {replacement_count} occurrences in {path}"
                    if replace_all
                    else f"Replaced one occurrence in {path}"
                ),
                "diff": _build_diff(old_content, new_content, path),
                "line_ending": line_ending,
                "expected_mtime": expected_mtime,
                "current_mtime": file_ops.stat(path).modified_at,
                "previous_sha256": write.previous_sha256,
                "content_sha256": write.content_sha256,
            }
        except FileRevisionConflictError as exc:
            return self._revision_conflict_result(exc)
        except FileNotFoundError:
            return error_result(
                {
                    "status": "error",
                    "message": f"File not found: {path}",
                    "path": path,
                }
            )
        except Exception as e:
            return error_result({"status": "error", "message": str(e), "path": path})

    @staticmethod
    def _revision_conflict_result(exc: FileRevisionConflictError) -> ToolResult:
        return error_result(
            {
                "status": "error",
                "error_category": "file_revision_conflict",
                "message": str(exc),
                "path": exc.path,
                "expected_sha256": exc.expected_sha256,
                "current_sha256": exc.current_sha256,
            }
        )

    @function_tool(
        name="make_directory",
        needs_approval=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def make_directory(
        self, path: str, runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any] | ToolResult:
        """
        Create a directory inside the workspace.

        :param path: Directory path relative to the workspace root.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        try:
            self._file_ops(runtime_context).make_directory(path, parents=True)
            return {"status": "success", "path": path}
        except Exception as e:
            return error_result({"status": "error", "message": str(e), "path": path})

    @function_tool(
        name="glob",
        read_only=True,
        environment_ops=["file", "process"],
        rule_scope_builder=_default_rule_scope,
    )
    async def glob(
        self,
        pattern: str,
        path: str = ".",
        include_hidden: bool = False,
        include_ignored: bool = False,
        limit: int = 200,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """
        Find files under the workspace that match a glob pattern.

        :param pattern: Glob pattern such as `*.py` or `src/**/*.md`.
        :param path: Directory path, relative to the workspace root, to search in.
        :param include_hidden: Whether to include hidden files and directories.
        :param include_ignored: Whether to bypass ignore files and VCS exclusions.
        :param limit: Maximum number of matching files to return.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        if not str(pattern or "").strip():
            return error_result(
                {"status": "error", "message": "Pattern cannot be empty"}
            )
        try:
            result_limit = _search_limit(limit)
            self._require_search_directory(path, runtime_context)
            matches, exit_code, stderr = await self._run_rg_files(
                path,
                pattern,
                include_hidden,
                include_ignored,
                runtime_context,
            )
            capped = matches[:result_limit]
            return {
                "status": "success",
                "pattern": pattern,
                "path": path,
                "files": capped,
                "match_count": len(capped),
                "total_count": len(matches),
                "returned_count": len(capped),
                "truncated": len(matches) > len(capped),
                "exit_code": exit_code,
                "stderr": stderr,
                "context": {
                    "include_hidden": include_hidden,
                    "include_ignored": include_ignored,
                    "binary_files": "skipped",
                },
            }
        except Exception as error:
            return _search_error_payload(error, pattern=pattern, path=path)

    @function_tool(
        name="grep",
        read_only=True,
        environment_ops=["file", "process"],
        rule_scope_builder=_default_rule_scope,
    )
    async def grep(
        self,
        pattern: str,
        path: str = ".",
        glob: Optional[str] = None,
        case_sensitive: bool = False,
        regex: bool = True,
        files_with_matches: bool = False,
        limit: int = 100,
        context: int = 0,
        file_type: Optional[str] = None,
        include_hidden: bool = False,
        include_ignored: bool = False,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """
        Search workspace files for a regex or literal text pattern.

        :param pattern: Regex or literal text to search for.
        :param path: Directory path relative to the workspace root.
        :param glob: Optional glob filter applied before reading candidate files.
        :param case_sensitive: Whether matching should preserve case.
        :param regex: Whether pattern should be interpreted as a regex.
        :param files_with_matches: Whether to return only matching file paths.
        :param limit: Maximum number of returned matches.
        :param context: Number of surrounding context lines to return as records.
        :param file_type: Optional ripgrep file type filter.
        :param include_hidden: Whether to include hidden files and directories.
        :param include_ignored: Whether to bypass ignore files and VCS exclusions.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        if not str(pattern or "").strip():
            return error_result(
                {"status": "error", "message": "Pattern cannot be empty"}
            )
        try:
            result_limit = _search_limit(limit)
            context_lines = int(context)
            if context_lines < 0:
                raise ValueError("context must be non-negative")
            self._require_search_directory(path, runtime_context)
            matches, records, exit_code, stderr = await self._run_rg_grep(
                pattern,
                path,
                glob,
                case_sensitive,
                regex,
                files_with_matches,
                context_lines,
                file_type,
                include_hidden,
                include_ignored,
                runtime_context,
            )
            capped = matches[:result_limit]
            returned_keys = {(item["path"], item.get("line")) for item in capped}
            capped_records = (
                [
                    item
                    for item in records
                    if any(
                        item["path"] == path_value
                        and isinstance(line_value, int)
                        and abs(item["line"] - line_value) <= context_lines
                        for path_value, line_value in returned_keys
                    )
                ]
                if context_lines > 0
                else []
            )
            return {
                "status": "success",
                "pattern": pattern,
                "path": path,
                "matches": capped,
                "records": capped_records,
                "match_count": len(capped),
                "total_count": len(matches),
                "returned_count": len(capped),
                "truncated": len(matches) > len(capped),
                "exit_code": exit_code,
                "stderr": stderr,
                "context": {
                    "glob": glob,
                    "case_sensitive": case_sensitive,
                    "regex": regex,
                    "files_with_matches": files_with_matches,
                    "line_count": context_lines,
                    "include_hidden": include_hidden,
                    "include_ignored": include_ignored,
                    "binary_files": "skipped",
                },
            }
        except Exception as error:
            return _search_error_payload(error, pattern=pattern, path=path)

    @function_tool(
        name="hex_view",
        read_only=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def hex_view(
        self,
        path: str,
        offset: int = 0,
        length: int = 256,
        width: int = 16,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """Inspect an exact bounded byte range from a workspace file.

        :param path: Path relative to the workspace root.
        :param offset: Zero-based byte offset.
        :param length: Number of bytes to display, from 1 through 4096.
        :param width: Bytes per row: 8, 16, or 32.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        try:
            start = int(offset)
            size = int(length)
            row_width = int(width)
            if start < 0:
                raise ValueError("offset must be non-negative")
            if not 1 <= size <= 4096:
                raise ValueError("length must be between 1 and 4096")
            if row_width not in {8, 16, 32}:
                raise ValueError("width must be 8, 16, or 32")
            file_ops = self._file_ops(runtime_context)
            info = file_ops.stat(path)
            if not info.is_file:
                return error_result(
                    {"status": "error", "message": f"Not a file: {path}"}
                )
            if start >= info.size:
                return {
                    "status": "success",
                    "path": path,
                    "offset": start,
                    "length": 0,
                    "file_size": info.size,
                    "content": "",
                    "has_more": False,
                }
            raw = file_ops.read_bytes(path, limit=size, offset=start)
            rows: List[str] = []
            for index in range(0, len(raw), row_width):
                chunk = raw[index : index + row_width]
                hexadecimal = " ".join(f"{byte:02x}" for byte in chunk)
                hexadecimal = hexadecimal.ljust(row_width * 3 - 1)
                printable = "".join(
                    chr(byte) if 32 <= byte < 127 else "." for byte in chunk
                )
                rows.append(
                    f"{start + index:08x}  {hexadecimal}  |{printable}|"
                )
            return {
                "status": "success",
                "path": path,
                "offset": start,
                "length": len(raw),
                "file_size": info.size,
                "content": "\n".join(rows),
                "has_more": start + len(raw) < info.size,
            }
        except Exception as e:
            return error_result({"status": "error", "message": str(e), "path": path})

    @function_tool(
        name="list_tree",
        read_only=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def list_tree(
        self,
        path: str = ".",
        depth: int = 3,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """
        List directory structure in a tree format.

        :param path: Directory path relative to the workspace root.
        :param depth: Maximum depth to traverse.
        """
        try:
            if not self._file_ops(runtime_context).stat(path).is_directory:
                return error_result(
                    {
                        "status": "error",
                        "message": f"Path is not a directory: {path}",
                    }
                )
            normalized_depth = min(max(int(depth), 1), 10)
            lines = self._tree_lines(path, normalized_depth, runtime_context)
            return {
                "status": "success",
                "path": path,
                "depth": normalized_depth,
                "tree": "\n".join(lines),
                "lines": lines,
            }
        except Exception as e:
            return error_result({"status": "error", "message": str(e), "path": path})

    @function_tool(
        name="http_request",
        rule_scope_builder=_default_rule_scope,
    )
    async def http_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        verify_tls: bool = True,
        allow_redirects: bool = True,
        max_content_chars: int = 120_000,
    ) -> Dict[str, Any] | ToolResult:
        """
        Execute an HTTP request and return a structured response payload.

        :param method: HTTP method such as GET or POST.
        :param url: Absolute http or https URL.
        :param params: Optional query parameters.
        :param data: Optional form-like request body.
        :param json_data: Optional JSON request body.
        :param headers: Optional per-request headers.
        :param timeout: Optional timeout override in seconds.
        :param verify_tls: Whether TLS certificates should be verified.
        :param allow_redirects: Whether redirects should be followed automatically.
        :param max_content_chars: Maximum number of response characters to keep.
        """
        return await self._request(
            method=method,
            url=url,
            params=params,
            data=data,
            json_data=json_data,
            headers=headers,
            timeout=timeout,
            verify_tls=verify_tls,
            allow_redirects=allow_redirects,
            max_content_chars=max_content_chars,
        )

    @function_tool(
        name="http_get",
        needs_approval=True,
        rule_scope_builder=_default_rule_scope,
    )
    async def http_get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        verify_tls: bool = True,
        allow_redirects: bool = True,
    ) -> Dict[str, Any] | ToolResult:
        """
        Execute one HTTP GET request.

        :param url: Absolute URL to request.
        :param params: Optional query parameters.
        :param headers: Optional request headers.
        :param timeout: Optional timeout override in seconds.
        :param verify_tls: Whether TLS certificates should be verified.
        :param allow_redirects: Whether redirects should be followed automatically.
        """
        return await self.http_request(
            method="GET",
            url=url,
            params=params,
            headers=headers,
            timeout=timeout,
            verify_tls=verify_tls,
            allow_redirects=allow_redirects,
        )

    @function_tool(
        name="http_post",
        rule_scope_builder=_default_rule_scope,
    )
    async def http_post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        verify_tls: bool = True,
        allow_redirects: bool = True,
    ) -> Dict[str, Any] | ToolResult:
        """
        Execute one HTTP POST request.

        :param url: Absolute URL to request.
        :param data: Optional form-like request body.
        :param json_data: Optional JSON request body.
        :param headers: Optional request headers.
        :param timeout: Optional timeout override in seconds.
        :param verify_tls: Whether TLS certificates should be verified.
        :param allow_redirects: Whether redirects should be followed automatically.
        """
        return await self.http_request(
            method="POST",
            url=url,
            data=data,
            json_data=json_data,
            headers=headers,
            timeout=timeout,
            verify_tls=verify_tls,
            allow_redirects=allow_redirects,
        )

    @function_tool(name="extract_web_text")
    def extract_web_text(self, html: str) -> Dict[str, Any]:
        """
        Extract readable text from raw HTML.

        :param html: Raw HTML string to process.
        """
        payload = self._extract_html_text(html)
        return {
            "status": payload.get("status", "success"),
            "title": payload.get("title", ""),
            "content": payload.get("text", ""),
        }

    @function_tool(
        name="web_fetch",
        needs_approval=True,
        rule_scope_builder=_default_rule_scope,
    )
    async def web_fetch(
        self,
        url: str,
        prompt: str = "",
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """
        Fetch one URL and extract concise text for coding workflows.

        :param url: Absolute URL to fetch.
        :param prompt: Optional task-specific extraction hint.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        _ = runtime_context
        response = await self.http_get(url=url, allow_redirects=False)
        if isinstance(response, ToolResult):
            return response
        if response.get("status_code") in {301, 302, 303, 307, 308}:
            headers = response.get("headers", {})
            redirect_url = headers.get("Location") or response.get("url")
            return {
                "status": "success",
                "redirect_url": redirect_url,
                "url": response.get("url", url),
            }
        extracted = self.extract_web_text(html=str(response.get("content", "")))
        text = str(extracted.get("content", ""))
        result = text
        if prompt.strip():
            keywords = [item.lower() for item in prompt.split() if item.strip()]
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            picked = [
                line
                for line in lines
                if any(token in line.lower() for token in keywords)
            ]
            if picked:
                result = "\n".join(picked[:6])
        auth_hint = ""
        if "github.com" in str(response.get("url", url)):
            auth_hint = "This host may require authentication or a raw-content URL."
        return {
            "status": "success",
            "url": response.get("url", url),
            "content": result,
            "title": extracted.get("title", ""),
            "auth_hint": auth_hint,
        }

    @function_tool(name="ask_user_choice", requires_user_interaction=True)
    def ask_user_choice(
        self,
        questions: List[Dict[str, Any]],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Emit a structured user-input request.

        :param questions: One to three structured user questions.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        _ = runtime_context
        return tool_result(
            {"status": "needs_input", "questions": list(questions or [])},
            status="needs_input",
        )

    @function_tool(name="tool_search", read_only=True)
    def tool_search(
        self, query: str, runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search the current tool registry by name or description.

        :param query: Case-insensitive substring to search for.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        registry = (runtime_context or {}).get("tool_registry")
        needle = str(query or "").lower()
        results: List[Dict[str, Any]] = []
        if registry is not None and hasattr(registry, "list_tools"):
            for name in registry.list_tools():
                desc = ""
                try:
                    desc = str(
                        (registry.describe_tool(name) or {}).get("description", "")
                    )
                except Exception:
                    desc = ""
                if needle in name.lower() or needle in desc.lower():
                    results.append({"name": name, "description": desc})
        return {"status": "success", "count": len(results), "results": results}

    @function_tool(
        name="enter_plan_mode",
        prompt="Use this tool proactively when you need to plan a non-trivial implementation before starting. This transitions into a read-only mode where you can analyze the codebase without making changes.",
    )
    def enter_plan_mode(
        self, reason: str = "", runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Switch runtime state into plan mode.

        :param reason: Optional reason for entering plan mode.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        state = (runtime_context or {}).get("state")
        if state is not None and hasattr(state, "metadata"):
            state.metadata["mode"] = "plan"
            state.metadata["plan_reason"] = reason
        return {"status": "success", "current_mode": "plan", "reason": reason}

    @function_tool(name="exit_plan_mode")
    def exit_plan_mode(
        self, runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Switch runtime state out of plan mode.

        :param runtime_context: Optional runtime context injected by the executor.
        """
        state = (runtime_context or {}).get("state")
        if state is not None and hasattr(state, "metadata"):
            state.metadata["mode"] = "work"
        return {"status": "success", "current_mode": "work"}

    @function_tool(name="enter_worktree")
    def enter_worktree(
        self, runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record that the agent entered worktree mode.

        :param runtime_context: Optional runtime context injected by the executor.
        """
        state = (runtime_context or {}).get("state")
        if state is not None and hasattr(state, "metadata"):
            state.metadata["worktree_mode"] = True
        return {"status": "success", "current_mode": "worktree"}

    @function_tool(name="exit_worktree")
    def exit_worktree(
        self, runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record that the agent exited worktree mode.

        :param runtime_context: Optional runtime context injected by the executor.
        """
        state = (runtime_context or {}).get("state")
        if state is not None and hasattr(state, "metadata"):
            state.metadata["worktree_mode"] = False
        return {"status": "success", "current_mode": "workspace"}

    @function_tool(name="lsp_query", read_only=True)
    def lsp_query(
        self,
        operation: str,
        symbol: str = "",
        runtime_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any] | ToolResult:
        """
        Query an injected LSP backend.

        :param operation: LSP operation such as `definition` or `references`.
        :param symbol: Optional symbol or identifier hint.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        ops = (runtime_context or {}).get("ops") or {}
        lsp = ops.get("lsp")
        if lsp is None or not hasattr(lsp, "query"):
            return error_result(
                {"status": "error", "message": "LSP capability unavailable"}
            )
        payload = lsp.query(operation=operation, symbol=symbol, **kwargs)
        if isinstance(payload, Mapping) and payload.get("status") == "error":
            return error_result(payload)
        return payload

    @function_tool(
        name="task_create",
        prompt="Use this tool proactively when you're about to start a non-trivial implementation task. Creating tasks helps you track progress and organize complex work.",
    )
    def task_create(
        self,
        subject: str,
        description: str,
        active_form: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "pending",
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """
        Create a session-native task record.

        :param subject: Short task title.
        :param description: Longer task description.
        :param active_form: Optional active-form wording.
        :param metadata: Optional structured metadata.
        :param status: Initial task status.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        _ = runtime_context
        normalized_status = str(status or "pending").strip()
        if normalized_status not in TASK_STATUSES:
            return error_result(
                {
                    "status": "error",
                    "message": f"Unsupported status: {normalized_status}",
                }
            )
        task_id = self._next_task_id()
        task = {
            "id": task_id,
            "subject": subject,
            "description": description,
            "status": normalized_status,
            "active_form": active_form,
            "blocks": [],
            "blocked_by": [],
            "notes": [],
            "metadata": dict(metadata or {}),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        self._session_tasks[task_id] = task
        return {"status": "success", "task": dict(task)}

    @function_tool(name="task_get", read_only=True)
    def task_get(
        self, task_id: str, runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any] | ToolResult:
        """
        Fetch one session-native task by id.

        :param task_id: Task identifier.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        _ = runtime_context
        task = self._session_tasks.get(str(task_id))
        if task is None:
            return error_result(
                {"status": "error", "message": f"Task not found: {task_id}"}
            )
        return {"status": "success", "task": dict(task)}

    @function_tool(name="task_list", read_only=True)
    def task_list(
        self,
        status: str = "",
        include_completed: bool = True,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        List session-native tasks.

        :param status: Optional status filter.
        :param include_completed: Whether completed tasks should remain in the result.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        _ = runtime_context
        tasks = list(self._session_tasks.values())
        if status:
            tasks = [task for task in tasks if task.get("status") == status]
        if not include_completed:
            tasks = [task for task in tasks if task.get("status") != "completed"]
        return {
            "status": "success",
            "tasks": [dict(task) for task in tasks],
            "count": len(tasks),
        }

    @function_tool(name="task_update")
    def task_update(
        self,
        task_id: str,
        status: str = "",
        add_blocks: Optional[List[str]] = None,
        remove_blocks: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | ToolResult:
        """
        Update a session-native task.

        :param task_id: Task identifier.
        :param status: Optional new task status.
        :param add_blocks: Optional task ids to add to the blocks list.
        :param remove_blocks: Optional task ids to remove from the blocks list.
        :param metadata: Optional metadata merge payload.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        _ = runtime_context
        task = self._session_tasks.get(str(task_id))
        if task is None:
            return error_result(
                {"status": "error", "message": f"Task not found: {task_id}"}
            )
        if status:
            normalized_status = str(status).strip()
            if normalized_status not in TASK_STATUSES:
                return error_result(
                    {
                        "status": "error",
                        "message": f"Unsupported status: {normalized_status}",
                    }
                )
            task["status"] = normalized_status
        blocks = list(task.get("blocks", []))
        for item in list(add_blocks or []):
            if item not in blocks:
                blocks.append(item)
        for item in list(remove_blocks or []):
            if item in blocks:
                blocks.remove(item)
        task["blocks"] = blocks
        if metadata:
            task["metadata"] = {**dict(task.get("metadata", {})), **dict(metadata)}
        task["updated_at"] = _utc_now()
        self._session_tasks[str(task_id)] = task
        return {"status": "success", "task": dict(task)}

    @function_tool(name="mcp_list_resources", read_only=True)
    def mcp_list_resources(
        self, runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        List injected MCP resources.

        :param runtime_context: Optional runtime context injected by the executor.
        """
        return {
            "status": "success",
            "resources": dict((runtime_context or {}).get("mcp_resources") or {}),
        }

    @function_tool(name="mcp_read_resource", read_only=True)
    def mcp_read_resource(
        self, server: str, uri: str, runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any] | ToolResult:
        """
        Read one MCP resource from injected snapshots.

        :param server: MCP server name.
        :param uri: Resource URI.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        resources = dict((runtime_context or {}).get("mcp_resources") or {})
        for item in list(resources.get(server) or []):
            if isinstance(item, dict) and str(item.get("uri", "")) == str(uri):
                return {"status": "success", "resource": item}
        return error_result(
            {"status": "error", "message": f"Resource not found: {server}:{uri}"}
        )

    @function_tool(name="cron_create")
    def cron_create(
        self, runtime_context: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Stub cron-create tool.

        :param runtime_context: Optional runtime context injected by the executor.
        """
        _ = runtime_context
        return {"status": "success", "created": True, "job": kwargs}

    @function_tool(name="cron_delete")
    def cron_delete(
        self, runtime_context: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Stub cron-delete tool.

        :param runtime_context: Optional runtime context injected by the executor.
        """
        _ = runtime_context
        return {"status": "success", "deleted": True, "request": kwargs}

    @function_tool(name="cron_list", read_only=True)
    def cron_list(
        self, runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Stub cron-list tool.

        :param runtime_context: Optional runtime context injected by the executor.
        """
        _ = runtime_context
        return {"status": "success", "jobs": []}


__all__ = ["CodingToolSet", "TASK_STATUSES", "_resolve_workspace_path"]
