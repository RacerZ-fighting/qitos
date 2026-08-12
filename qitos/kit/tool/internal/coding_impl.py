"""Canonical coding-oriented toolset backed by method-style tool definitions."""

from __future__ import annotations

import json
import os
import re
import subprocess
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from qitos.core.env import CommandCapability, FileSystemCapability
from qitos.core.function_tool_decorator import function_tool
from qitos.kit.env.host_env import HostCommandCapability, HostFSCapability
from qitos.kit.tool.internal.coding_utils import (
    build_diff,
    default_rule_scope,
    detect_line_ending,
    resolve_tool_workspace_path,
    truncate_text,
    utc_now,
)
from qitos.kit.tool.internal.runtime_ops import select_runtime_ops
from qitos.kit.tool.notebook import NotebookToolSet

try:  # optional dependency
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore[assignment]


TASK_STATUSES = {"pending", "in_progress", "blocked", "completed", "cancelled"}
_MAX_SEARCH_RESULTS = 2000


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
) -> Dict[str, Any]:
    if isinstance(error, _SearchProcessError):
        return {
            "status": error.status,
            "message": str(error),
            "error_category": error.category,
            "exit_code": error.exit_code,
            "stderr": error.stderr,
            "pattern": pattern,
            "path": path,
        }
    if isinstance(error, subprocess.TimeoutExpired):
        stderr = error.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "status": "timed_out",
            "message": str(error),
            "error_category": "process_timeout",
            "exit_code": None,
            "stderr": stderr,
            "pattern": pattern,
            "path": path,
        }
    if isinstance(error, FileNotFoundError):
        return {
            "status": "error",
            "message": f"ripgrep is unavailable: {error}",
            "error_category": "search_tool_unavailable",
            "exit_code": None,
            "stderr": "",
            "pattern": pattern,
            "path": path,
        }
    return {
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
        "shell": ("run_command",),
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
            "web_fetch",
            "ask_user_choice",
            "todo_write",
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
            HostCommandCapability(self.workspace_root)
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

    def setup(self, context: Dict[str, Any]) -> None:
        _ = context

    def teardown(self, context: Dict[str, Any]) -> None:
        _ = context

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
    ) -> tuple[str, str, float]:
        file_ops = self._file_ops(runtime_context)
        raw = file_ops.read_bytes(path)
        modified_at = file_ops.stat(path).modified_at
        return (
            raw.decode("utf-8", errors="strict"),
            _detect_line_ending(raw),
            float(modified_at or 0.0),
        )

    def _write_text_file(
        self,
        path: str,
        content: str,
        line_ending: str,
        runtime_context: Optional[Dict[str, Any]],
    ) -> None:
        normalized = (
            content.replace("\r\n", "\n").replace("\r", "\n").replace("\n", line_ending)
        )
        self._file_ops(runtime_context).write_text(path, normalized)

    def _run_rg_files(
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
        result = self._process_ops(runtime_context).run_argv(
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

    def _run_rg_grep(
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
        result = self._process_ops(runtime_context).run_argv(
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

    def _request(
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
    ) -> Dict[str, Any]:
        parsed = urlparse(str(url or ""))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return {
                "status": "error",
                "message": "URL must be an absolute http or https URL",
                "url": url,
            }
        try:
            response = requests.request(
                method=str(method or "GET").upper(),
                url=url,
                params=params,
                data=data,
                json=json_data,
                headers=headers,
                timeout=int(timeout or 30),
                verify=verify_tls,
                allow_redirects=allow_redirects,
            )
            text, truncated = _truncate_text(response.text, max_content_chars)
            payload: Dict[str, Any] = {
                "status": "success" if response.status_code < 400 else "error",
                "ok": bool(response.ok),
                "method": str(method or "GET").upper(),
                "url": response.url,
                "status_code": response.status_code,
                "reason": response.reason,
                "headers": dict(response.headers),
                "content_type": response.headers.get("Content-Type", ""),
                "content": text,
                "content_length": len(text),
                "truncated": truncated,
                "history": [item.url for item in response.history],
            }
            try:
                payload["json"] = response.json()
            except Exception:
                pass
            return payload
        except Exception as e:
            return {"status": "error", "message": str(e), "url": url, "method": method}

    def _extract_html_text(self, html: str) -> Dict[str, Any]:
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html or "", "html.parser")
            title = (
                soup.title.string.strip() if soup.title and soup.title.string else ""
            )
            text = "\n".join(
                line.strip()
                for line in soup.get_text("\n").splitlines()
                if line.strip()
            )
            return {"status": "success", "title": title, "text": text}
        title_match = re.search(
            r"<title>(.*?)</title>", html or "", re.IGNORECASE | re.DOTALL
        )
        title = title_match.group(1).strip() if title_match else ""
        text = re.sub(r"<[^>]+>", " ", html or "")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return {"status": "success", "title": title, "text": text}

    def _next_task_id(self) -> str:
        self._task_counter += 1
        return f"task-{self._task_counter:03d}"

    def _run_bash_command(
        self,
        command: str,
        read_only: bool = False,
        allow_destructive: bool = False,
        run_in_background: bool = False,
        allow_needs_review: bool = False,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = str(command or "").strip()
        if not text:
            return {"status": "error", "message": "Command cannot be empty"}

        # Use BashCommandAnalyzer for safety classification
        from qitos.kit.permission.bash_analyzer import BashCommandAnalyzer, CommandSafety

        analyzer = BashCommandAnalyzer()
        analysis = analyzer.analyze(text)

        if not allow_destructive and analysis.safety == CommandSafety.UNSAFE:
            return {
                "status": "error",
                "message": f"Destructive command blocked: {analysis.explanation}",
                "error_category": "destructive_command",
                "detected_patterns": analysis.detected_patterns,
            }

        if read_only and not analysis.is_read_only:
            return {
                "status": "error",
                "message": "Command appears to write to the workspace in read-only mode",
            }

        python_inline_smoke = text.startswith(("python -c ", "python3 -c "))
        if (
            analysis.safety == CommandSafety.NEEDS_REVIEW
            and not allow_destructive
            and not allow_needs_review
            and not self.auto_approve
            and not python_inline_smoke
        ):
            return {
                "status": "needs_approval",
                "message": f"Command needs review: {analysis.explanation}",
                "detected_patterns": analysis.detected_patterns,
            }
        try:
            process_ops = self._process_ops(runtime_context)
            if run_in_background:
                return process_ops.start(text)
            return process_ops.run(text, timeout=self.shell_timeout)
        except Exception as e:
            return {"status": "error", "message": str(e), "command": text}

    @function_tool(
        name="run_command",
        needs_approval=True,
        supports_background=True,
        environment_ops=["process"],
        rule_scope_builder=_default_rule_scope,
    )
    def run_command(
        self,
        command: str,
        read_only: bool = False,
        allow_destructive: bool = False,
        run_in_background: bool = False,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute one shell command inside the configured working directory.

        :param command: Shell command string to execute.
        :param read_only: Reject commands that appear to mutate the workspace.
        :param allow_destructive: Explicitly allow commands classified as destructive.
        :param run_in_background: Detach the command and return its task handle.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        return self._run_bash_command(
            command=command,
            read_only=read_only,
            allow_destructive=allow_destructive,
            run_in_background=run_in_background,
            runtime_context=runtime_context,
        )

    def _read_file_chunk(
        self,
        path: str,
        offset: int = 0,
        limit: int = 200,
        max_chars: int = 20_000,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
                return {"status": "error", "message": f"Path is a directory: {path}"}
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
                "has_more": chunk.has_more,
                "truncated": chunk.truncated,
            }
        except FileNotFoundError:
            return {"status": "error", "message": f"File not found: {path}"}
        except Exception as e:
            return {"status": "error", "message": str(e), "path": path}

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
    ) -> Dict[str, Any]:
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
        if result.get("status") != "success":
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
    ) -> Dict[str, Any]:
        """
        List files and directories under a workspace-relative path.

        :param path: Directory path relative to the workspace root.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        try:
            file_ops = self._file_ops(runtime_context)
            if not file_ops.stat(path).is_directory:
                return {
                    "status": "error",
                    "message": f"Path is not a directory: {path}",
                }
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
            return {"status": "error", "message": str(e), "path": path}

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
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Write text content to a workspace file.

        :param path: Path relative to the workspace root.
        :param content: Full text content to write into the file.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        try:
            self._write_text_file(path, str(content), "\n", runtime_context)
            return {"status": "success", "path": path, "size": len(content)}
        except Exception as e:
            return {"status": "error", "message": str(e), "path": path}

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
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Replace exact text in one workspace file.

        :param path: Path relative to the workspace root.
        :param old_text: Exact text to replace. It must be unique by default.
        :param new_text: Replacement text.
        :param replace_all: Replace every occurrence instead of requiring uniqueness.
        :param expected_mtime: Optional optimistic concurrency check.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        try:
            file_ops = self._file_ops(runtime_context)
            if not file_ops.stat(path).is_file:
                return {"status": "error", "message": f"Not a file: {path}"}
            old_content, line_ending, current_mtime = self._read_text_file(
                path,
                runtime_context,
            )
            if (
                expected_mtime is not None
                and abs(float(expected_mtime) - float(current_mtime)) > 1e-6
            ):
                return {
                    "status": "error",
                    "message": "File was modified since the expected mtime.",
                    "path": path,
                }
            if not old_text:
                return {
                    "status": "error",
                    "message": "old_text cannot be empty",
                    "path": path,
                }
            if old_text == new_text:
                return {
                    "status": "error",
                    "message": "old_text and new_text are identical",
                    "path": path,
                }
            count = old_content.count(old_text)
            if count == 0:
                return {
                    "status": "error",
                    "message": f"Text not found in {path}",
                    "path": path,
                }
            if count > 1 and not replace_all:
                return {
                    "status": "error",
                    "message": "Text replacement must be unique",
                    "path": path,
                    "occurrences": count,
                }
            replacement_count = count if replace_all else 1
            new_content = old_content.replace(
                old_text,
                new_text,
                -1 if replace_all else 1,
            )
            self._write_text_file(path, new_content, line_ending, runtime_context)
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
            }
        except FileNotFoundError:
            return {
                "status": "error",
                "message": f"File not found: {path}",
                "path": path,
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "path": path}

    @function_tool(
        name="make_directory",
        needs_approval=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def make_directory(
        self, path: str, runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a directory inside the workspace.

        :param path: Directory path relative to the workspace root.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        try:
            self._file_ops(runtime_context).make_directory(path, parents=True)
            return {"status": "success", "path": path}
        except Exception as e:
            return {"status": "error", "message": str(e), "path": path}

    @function_tool(
        name="glob",
        read_only=True,
        environment_ops=["file", "process"],
        rule_scope_builder=_default_rule_scope,
    )
    def glob(
        self,
        pattern: str,
        path: str = ".",
        include_hidden: bool = False,
        include_ignored: bool = False,
        limit: int = 200,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
            return {"status": "error", "message": "Pattern cannot be empty"}
        try:
            result_limit = _search_limit(limit)
            self._require_search_directory(path, runtime_context)
            matches, exit_code, stderr = self._run_rg_files(
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
    def grep(
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
    ) -> Dict[str, Any]:
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
            return {"status": "error", "message": "Pattern cannot be empty"}
        try:
            result_limit = _search_limit(limit)
            context_lines = int(context)
            if context_lines < 0:
                raise ValueError("context must be non-negative")
            self._require_search_directory(path, runtime_context)
            matches, records, exit_code, stderr = self._run_rg_grep(
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
    ) -> Dict[str, Any]:
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
                return {"status": "error", "message": f"Not a file: {path}"}
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
            return {"status": "error", "message": str(e), "path": path}

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
    ) -> Dict[str, Any]:
        """
        List directory structure in a tree format.

        :param path: Directory path relative to the workspace root.
        :param depth: Maximum depth to traverse.
        """
        try:
            if not self._file_ops(runtime_context).stat(path).is_directory:
                return {
                    "status": "error",
                    "message": f"Path is not a directory: {path}",
                }
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
            return {"status": "error", "message": str(e), "path": path}

    @function_tool(
        name="http_request",
        rule_scope_builder=_default_rule_scope,
    )
    def http_request(
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
    ) -> Dict[str, Any]:
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
        return self._request(
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
    def http_get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        verify_tls: bool = True,
        allow_redirects: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute one HTTP GET request.

        :param url: Absolute URL to request.
        :param params: Optional query parameters.
        :param headers: Optional request headers.
        :param timeout: Optional timeout override in seconds.
        :param verify_tls: Whether TLS certificates should be verified.
        :param allow_redirects: Whether redirects should be followed automatically.
        """
        return self.http_request(
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
    def http_post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        verify_tls: bool = True,
        allow_redirects: bool = True,
    ) -> Dict[str, Any]:
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
        return self.http_request(
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
    def web_fetch(
        self,
        url: str,
        prompt: str = "",
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Fetch one URL and extract concise text for coding workflows.

        :param url: Absolute URL to fetch.
        :param prompt: Optional task-specific extraction hint.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        _ = runtime_context
        response = self.http_get(url=url, allow_redirects=False)
        if response.get("status") == "error":
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
    ) -> Dict[str, Any]:
        """
        Emit a structured user-input request.

        :param questions: One to three structured user questions.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        _ = runtime_context
        return {"status": "needs_input", "questions": list(questions or [])}

    @function_tool(name="todo_write")
    def todo_write(
        self,
        todos: List[Dict[str, Any]],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Write lightweight todo items into runtime state metadata.

        :param todos: Todo item list.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        state = (runtime_context or {}).get("state")
        normalized = [dict(item) for item in list(todos or [])]
        if state is not None and hasattr(state, "metadata"):
            state.metadata["todos"] = normalized
        return {"status": "success", "count": len(normalized), "todos": normalized}

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
    ) -> Dict[str, Any]:
        """
        Query an injected LSP backend.

        :param operation: LSP operation such as `definition` or `references`.
        :param symbol: Optional symbol or identifier hint.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        ops = (runtime_context or {}).get("ops") or {}
        lsp = ops.get("lsp")
        if lsp is None or not hasattr(lsp, "query"):
            return {"status": "error", "message": "LSP capability unavailable"}
        return lsp.query(operation=operation, symbol=symbol, **kwargs)

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
    ) -> Dict[str, Any]:
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
            return {
                "status": "error",
                "message": f"Unsupported status: {normalized_status}",
            }
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
    ) -> Dict[str, Any]:
        """
        Fetch one session-native task by id.

        :param task_id: Task identifier.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        _ = runtime_context
        task = self._session_tasks.get(str(task_id))
        if task is None:
            return {"status": "error", "message": f"Task not found: {task_id}"}
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
    ) -> Dict[str, Any]:
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
            return {"status": "error", "message": f"Task not found: {task_id}"}
        if status:
            normalized_status = str(status).strip()
            if normalized_status not in TASK_STATUSES:
                return {
                    "status": "error",
                    "message": f"Unsupported status: {normalized_status}",
                }
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
    ) -> Dict[str, Any]:
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
        return {"status": "error", "message": f"Resource not found: {server}:{uri}"}

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
