"""Canonical coding-oriented toolset backed by method-style tool definitions."""

from __future__ import annotations

import json
import os
import re
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from qitos.core.env import CommandCapability, FileSystemCapability
from qitos.core.function_tool_decorator import function_tool
from qitos.core.tool import ToolPermission
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


def _utc_now() -> str:
    return utc_now()


def _resolve_workspace_path(root_dir: str, path: str) -> Path:
    return resolve_tool_workspace_path(root_dir, path)


def _detect_line_ending(raw: bytes) -> str:
    return detect_line_ending(raw)


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    return truncate_text(text, max_chars)


def _select_line_chunk(
    lines: List[str], start: int, max_lines: int, max_chars: int
) -> tuple[List[str], bool]:
    end = min(len(lines), start + max_lines)
    chunk: List[str] = []
    char_count = 0
    enforce_chars = max_chars > 0
    for line in lines[start:end]:
        char_count += len(line) + (1 if chunk else 0)
        chunk.append(line)
        if enforce_chars and char_count >= max_chars:
            break
    truncated = bool(enforce_chars and start + len(chunk) < end)
    return chunk, truncated


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


class CodingToolSet:
    """Canonical coding toolset with one stable, traditional tool surface."""

    name = "coding"
    version = "2"

    def __init__(
        self,
        workspace_root: str = ".",
        shell_timeout: int = 30,
        include_notebook: bool = True,
        *,
        enable_lsp: bool = True,
        enable_tasks: bool = True,
        enable_web: bool = True,
        expose_legacy_aliases: bool = True,
        expose_modern_names: bool = False,
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
        self.expose_legacy_aliases = bool(expose_legacy_aliases)
        self.expose_modern_names = bool(expose_modern_names)
        self.profile = str(profile or "full")
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
        items: List[Any] = []
        if self.profile == "workspace":
            items.extend(
                [
                    self.read_file,
                    self.write_file,
                    self.edit_file,
                    self.glob,
                    self.grep,
                    self.hex_view,
                    self.list_files,
                    self.list_tree,
                    self.make_directory,
                ]
            )
        # Claude Code modern-name aliases (Read, Edit, Write, Glob, Grep, Bash, etc.)
        elif self.expose_modern_names:
            items.extend(
                [
                    self.Read,
                    self.Edit,
                    self.Write,
                    self.Glob,
                    self.Grep,
                    self.Bash,
                    self.WebFetch,
                    self.AskUserQuestion,
                ]
            )
        if self.profile in {"full", "editor"} and self.expose_legacy_aliases:
            items.extend(
                [
                    self.view,
                    self.create,
                    self.str_replace,
                    self.insert,
                    self.search,
                    self.list_tree,
                    self.replace_lines,
                ]
            )
        if self.profile in {"full", "codebase"} and self.expose_legacy_aliases:
            items.extend(
                [
                    self.glob_files,
                    self.grep_files,
                    self.read_file_range,
                    self.append_file,
                    self.make_directory,
                ]
            )
        if self.profile in {"full", "codebase", "files"} and self.expose_legacy_aliases:
            items.extend([self.read_file, self.write_file, self.list_files])
        if self.profile in {"full", "shell"} and self.expose_legacy_aliases:
            items.append(self.run_command)
        if self.profile in {"full", "web"} and self.enable_web:
            if self.expose_legacy_aliases:
                items.append(self.web_fetch)
            if self.include_http_tools:
                items.extend(
                    [
                        self.http_request,
                        self.http_get,
                        self.http_post,
                        self.extract_web_text,
                    ]
                )
        if self.profile == "full":
            items.extend(
                [
                    self.ask_user_choice,
                    self.todo_write,
                    self.tool_search,
                    self.enter_plan_mode,
                    self.exit_plan_mode,
                    self.enter_worktree,
                    self.exit_worktree,
                    self.mcp_list_resources,
                    self.mcp_read_resource,
                    self.agent_spawn,
                    self.cron_create,
                    self.cron_delete,
                    self.cron_list,
                ]
            )
            if self.enable_lsp:
                items.append(self.lsp_query)
            if self.enable_tasks:
                items.extend(
                    [self.task_create, self.task_get, self.task_list, self.task_update]
                )
            if self._notebook is not None:
                items.extend(self._notebook.tools())
        if not self.allow_local_fallback:
            bound_items: List[Any] = []
            for item in items:
                isolated = copy(item)
                isolated.spec = copy(item.spec)
                if hasattr(item, "meta"):
                    isolated.meta = copy(item.meta)
                isolated.spec.required_ops = list(
                    dict.fromkeys(
                        [
                            *list(isolated.spec.required_ops or []),
                            *list(isolated.spec.environment_ops or []),
                        ]
                    )
                )
                bound_items.append(isolated)
            items = bound_items
        if self.auto_approve:
            for item in items:
                if hasattr(item, "meta") and getattr(item.meta, "needs_approval", False):
                    item.meta.needs_approval = False
                if hasattr(item, "spec") and getattr(item.spec, "needs_approval", False):
                    item.spec.needs_approval = False
        return items

    def _file_ops(
        self, runtime_context: Optional[Dict[str, Any]]
    ) -> FileSystemCapability:
        return select_runtime_ops(runtime_context, "file", self._local_file_ops)

    def _process_ops(
        self, runtime_context: Optional[Dict[str, Any]]
    ) -> CommandCapability:
        return select_runtime_ops(runtime_context, "process", self._local_process_ops)

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
        runtime_context: Optional[Dict[str, Any]],
    ) -> List[str]:
        cmd = ["rg", "--files", "--sort=path", "--glob", pattern]
        if include_hidden:
            cmd.append("--hidden")
        cmd.append(".")
        result = self._process_ops(runtime_context).run_argv(
            cmd,
            timeout=self.shell_timeout,
            cwd=target_dir,
        )
        returncode = int(result.get("returncode", 1))
        if returncode not in {0, 1}:
            message = str(result.get("stderr") or result.get("error") or "rg failed")
            raise RuntimeError(f"Glob failed: {message.strip()}")
        rows = [line.strip() for line in str(result.get("stdout", "")).splitlines()]
        return sorted(
            _join_capability_path(target_dir, row)
            for row in rows
            if row.strip()
        )

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
        runtime_context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        cmd = ["rg", "--color", "never"]
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
        if context > 0 and not files_with_matches:
            cmd.extend(["--context", str(context)])
        cmd.extend(["--", pattern, "."])
        result = self._process_ops(runtime_context).run_argv(
            cmd,
            timeout=self.shell_timeout,
            cwd=target_dir,
        )
        returncode = int(result.get("returncode", 1))
        if returncode not in {0, 1}:
            message = str(result.get("stderr") or result.get("error") or "rg failed")
            raise RuntimeError(f"Grep failed: {message.strip()}")

        stdout = str(result.get("stdout", ""))
        matches: List[Dict[str, Any]] = []
        if files_with_matches:
            return [
                {"path": _join_capability_path(target_dir, row)}
                for row in stdout.split("\0")
                if row
            ]

        for row in stdout.splitlines():
            if not row.strip():
                continue
            event = json.loads(row)
            if event.get("type") != "match":
                continue
            data = event.get("data") or {}
            raw_path = str((data.get("path") or {}).get("text") or "")
            line_number = int(data.get("line_number") or 0)
            text = str((data.get("lines") or {}).get("text") or "").rstrip("\r\n")
            matches.append(
                {
                    "path": _join_capability_path(target_dir, raw_path),
                    "line": line_number,
                    "text": text,
                }
            )
        return matches

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

    @function_tool(
        name="bash_v2",
        needs_approval=True,
        supports_background=True,
        environment_ops=["process"],
        rule_scope_builder=_default_rule_scope,
    )
    def bash_v2(
        self,
        command: str,
        read_only: bool = False,
        allow_destructive: bool = False,
        run_in_background: bool = False,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run one shell command inside the workspace.

        :param command: Shell command to execute.
        :param read_only: Whether the command should avoid mutating the workspace.
        :param allow_destructive: Whether destructive commands are explicitly allowed.
        :param run_in_background: Whether to detach the command and return a log path.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        return self._run_bash_command(
            command=command,
            read_only=read_only,
            allow_destructive=allow_destructive,
            run_in_background=run_in_background,
            runtime_context=runtime_context,
        )

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
        environment_ops=["process"],
        rule_scope_builder=_default_rule_scope,
    )
    def run_command(
        self, command: str, runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute one shell command inside the configured working directory.

        :param command: Shell command string to execute.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        return self.bash_v2(command=command, runtime_context=runtime_context)

    @function_tool(
        name="file_read_v2",
        read_only=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def file_read_v2(
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
        result = self.file_read_v2(
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
        name="view",
        read_only=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def view(
        self,
        path: str,
        view_range: Optional[List[int]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        View a file or directory under the workspace root.

        :param path: Path relative to the workspace root (e.g., `src/main.py` or `src/`).
        :param view_range: Optional inclusive line range `[start, end]` to show for files.

        For files, returns structured line content. For directories, returns a
        readable listing of immediate child entries.
        """
        try:
            file_ops = self._file_ops(runtime_context)
            info = file_ops.stat(path)
            if info.is_directory:
                entries = []
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
                    entries.append(
                        {
                            "name": name,
                            "type": "directory" if item.is_directory else "file",
                        }
                    )
                return {
                    "status": "success",
                    "kind": "directory",
                    "path": path,
                    "entries": entries,
                    "count": len(entries),
                }
            start = 0
            limit = 200
            if isinstance(view_range, list) and len(view_range) == 2:
                view_start = int(view_range[0])
                view_end = int(view_range[1])
                start = max(0, view_start - 1)
                limit = 100_000 if view_end == -1 else max(1, view_end - view_start + 1)
            return self.file_read_v2(
                path=path,
                offset=start,
                limit=limit,
                runtime_context=runtime_context,
            )
        except FileNotFoundError:
            return {"status": "error", "message": f"File not found: {path}"}
        except Exception as e:
            return {"status": "error", "message": str(e), "path": path}

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
        name="create",
        needs_approval=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def create(
        self,
        path: str,
        content: str = "",
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new file with the given content.

        :param path: Path relative to the workspace root (e.g., `new_file.py`).
        :param content: Content to write to the new file.
        """
        result = self.write_file(
            path=path,
            content=content,
            runtime_context=runtime_context,
        )
        if result.get("status") != "success":
            return result
        return {
            "status": "success",
            "path": path,
            "message": f"Created file: {path}",
            "size": len(content),
        }

    @function_tool(
        name="file_edit_v2",
        needs_approval=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def file_edit_v2(
        self,
        path: str,
        action: str,
        old_text: str = "",
        new_text: str = "",
        insert_line: int = 0,
        start_line: int = 0,
        end_line: int = 0,
        replacement: str = "",
        replace_all: bool = False,
        expected_mtime: Optional[float] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Edit one workspace file using a structured action.

        :param path: Path relative to the workspace root.
        :param action: Edit action such as `str_replace`, `insert`, or `replace_lines`.
        :param old_text: Old text for `str_replace`.
        :param new_text: New text for `str_replace`.
        :param insert_line: Line number after which to insert new text.
        :param start_line: Starting line number for `replace_lines`.
        :param end_line: Ending line number for `replace_lines`.
        :param replacement: Replacement content for `replace_lines`.
        :param replace_all: Replace every occurrence for `str_replace`.
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
            normalized_action = str(action or "").strip()
            if normalized_action == "str_replace":
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
                message = (
                    f"Replaced {replacement_count} occurrences in {path}"
                    if replace_all
                    else f"Replaced one occurrence in {path}"
                )
            elif normalized_action == "insert":
                try:
                    insert_line = int(insert_line)
                except Exception:
                    return {
                        "status": "error",
                        "message": f"Invalid insert_line: {insert_line}",
                        "path": path,
                    }
                lines = old_content.splitlines()
                if insert_line < 0 or insert_line > len(lines):
                    return {
                        "status": "error",
                        "message": f"Invalid insert_line: {insert_line}",
                        "path": path,
                    }
                updated_lines = lines[:insert_line] + [new_text] + lines[insert_line:]
                new_content = "\n".join(updated_lines)
                message = f"Inserted content after line {insert_line} in {path}"
            elif normalized_action == "replace_lines":
                try:
                    start_line = int(start_line)
                    end_line = int(end_line)
                except Exception:
                    return {
                        "status": "error",
                        "message": "Invalid line range",
                        "path": path,
                    }
                lines = old_content.splitlines()
                if start_line <= 0 or end_line < start_line or end_line > len(lines):
                    return {
                        "status": "error",
                        "message": "Invalid line range",
                        "path": path,
                    }
                if (
                    isinstance(replacement, str)
                    and replacement
                    and not replacement[:1].isspace()
                    and start_line == end_line
                ):
                    old_line = lines[start_line - 1]
                    indent = old_line[: len(old_line) - len(old_line.lstrip())]
                    if indent:
                        replacement = indent + replacement
                updated_lines = (
                    lines[: start_line - 1] + [replacement] + lines[end_line:]
                )
                new_content = "\n".join(updated_lines)
                message = f"Replaced lines {start_line}-{end_line} in {path}"
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported action: {normalized_action}",
                    "path": path,
                }
            self._write_text_file(path, new_content, line_ending, runtime_context)
            return {
                "status": "success",
                "path": path,
                "message": message,
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
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Replace exact text in one workspace file.

        :param path: Path relative to the workspace root.
        :param old_text: Exact text to replace. It must be unique by default.
        :param new_text: Replacement text.
        :param replace_all: Replace every occurrence instead of requiring uniqueness.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        return self.file_edit_v2(
            path=path,
            action="str_replace",
            old_text=old_text,
            new_text=new_text,
            replace_all=replace_all,
            runtime_context=runtime_context,
        )

    @function_tool(
        name="str_replace",
        needs_approval=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def str_replace(
        self,
        path: str,
        old_str: str,
        new_str: str = "",
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Replace one unique string fragment in a file.

        :param path: Path relative to the workspace root.
        :param old_str: The exact string to replace. Must be unique in the file.
        :param new_str: The new string to replace old_str with.
        """
        return self.file_edit_v2(
            path=path,
            action="str_replace",
            old_text=old_str,
            new_text=new_str,
            runtime_context=runtime_context,
        )

    @function_tool(
        name="insert",
        needs_approval=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def insert(
        self,
        path: str,
        insert_line: int,
        new_str: str,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Insert new text after a given line number.

        :param path: Path relative to the workspace root.
        :param insert_line: Line number after which to insert new_str.
        :param new_str: String to insert.
        """
        return self.file_edit_v2(
            path=path,
            action="insert",
            insert_line=insert_line,
            new_text=new_str,
            runtime_context=runtime_context,
        )

    @function_tool(
        name="replace_lines",
        needs_approval=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def replace_lines(
        self,
        path: str,
        start_line: int,
        end_line: int,
        replacement: str = "",
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Replace an inclusive line range with new content.

        :param path: Path relative to the workspace root.
        :param start_line: Starting line number.
        :param end_line: Ending line number, inclusive.
        :param replacement: Text to replace the specified lines with.
        """
        return self.file_edit_v2(
            path=path,
            action="replace_lines",
            start_line=start_line,
            end_line=end_line,
            replacement=replacement,
            runtime_context=runtime_context,
        )

    @function_tool(
        name="append_file",
        needs_approval=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def append_file(
        self,
        path: str,
        content: str,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Append text to the end of a workspace file.

        :param path: File path relative to the workspace root.
        :param content: Text to append.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        try:
            file_ops = self._file_ops(runtime_context)
            file_ops.append_text(path, content)
            return {
                "status": "success",
                "path": path,
                "appended_size": len(content),
                "size": file_ops.stat(path).size,
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
        name="glob_v2",
        read_only=True,
        environment_ops=["file", "process"],
        rule_scope_builder=_default_rule_scope,
    )
    def glob_v2(
        self,
        pattern: str,
        path: str = ".",
        include_hidden: bool = False,
        limit: int = 100,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Find files under the workspace that match a glob pattern.

        :param pattern: Glob pattern such as `*.py` or `src/**/*.md`.
        :param path: Directory path, relative to the workspace root, to search in.
        :param include_hidden: Whether to include hidden files and directories.
        :param limit: Maximum number of matching files to return.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        if not str(pattern or "").strip():
            return {"status": "error", "message": "Pattern cannot be empty"}
        try:
            if not self._file_ops(runtime_context).stat(path).is_directory:
                return {
                    "status": "error",
                    "message": f"Path is not a directory: {path}",
                }
            matches = self._run_rg_files(
                path,
                pattern,
                include_hidden,
                runtime_context,
            )
            capped = matches[: max(1, int(limit))]
            return {
                "status": "success",
                "pattern": pattern,
                "path": path,
                "files": capped,
                "match_count": len(capped),
                "truncated": len(matches) > len(capped),
                "context": {"include_hidden": include_hidden},
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "pattern": pattern,
                "path": path,
            }

    @function_tool(
        name="glob_files",
        read_only=True,
        environment_ops=["file", "process"],
        rule_scope_builder=_default_rule_scope,
    )
    def glob_files(
        self,
        pattern: str,
        path: str = ".",
        include_hidden: bool = False,
        limit: int = 100,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Find files under the workspace that match a glob pattern.

        :param pattern: Glob pattern such as `*.py`.
        :param path: Directory path relative to the workspace root.
        :param include_hidden: Whether to include hidden files and directories.
        :param limit: Maximum number of matching files to return.
        """
        result = self.glob_v2(
            pattern=pattern,
            path=path,
            include_hidden=include_hidden,
            limit=limit,
            runtime_context=runtime_context,
        )
        if result.get("status") == "success":
            result["num_files"] = result.get("match_count", 0)
        return result

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
        limit: int = 200,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Find workspace files with ripgrep's file inventory.

        :param pattern: Glob pattern relative to path.
        :param path: Workspace-relative directory to search.
        :param include_hidden: Include hidden paths when true.
        :param limit: Maximum sorted paths to return.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        return self.glob_v2(
            pattern=pattern,
            path=path,
            include_hidden=include_hidden,
            limit=limit,
            runtime_context=runtime_context,
        )

    @function_tool(
        name="grep_v2",
        read_only=True,
        environment_ops=["file", "process"],
        rule_scope_builder=_default_rule_scope,
    )
    def grep_v2(
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
        :param context: Reserved context-line count for future expansion.
        :param file_type: Optional ripgrep file type filter.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        if not str(pattern or "").strip():
            return {"status": "error", "message": "Pattern cannot be empty"}
        try:
            if not self._file_ops(runtime_context).stat(path).is_directory:
                return {
                    "status": "error",
                    "message": f"Path is not a directory: {path}",
                }
            matches = self._run_rg_grep(
                pattern,
                path,
                glob,
                case_sensitive,
                regex,
                files_with_matches,
                max(0, int(context)),
                file_type,
                runtime_context,
            )
            capped = matches[: max(1, int(limit))]
            return {
                "status": "success",
                "pattern": pattern,
                "path": path,
                "matches": capped,
                "match_count": len(capped),
                "truncated": len(matches) > len(capped),
                "context": {
                    "glob": glob,
                    "case_sensitive": case_sensitive,
                    "regex": regex,
                    "files_with_matches": files_with_matches,
                },
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "pattern": pattern}

    @function_tool(
        name="grep_files",
        read_only=True,
        environment_ops=["file", "process"],
        rule_scope_builder=_default_rule_scope,
    )
    def grep_files(
        self,
        pattern: str,
        path: str = ".",
        glob: Optional[str] = None,
        case_sensitive: bool = False,
        regex: bool = True,
        files_with_matches: bool = False,
        limit: int = 100,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search workspace files for a regex or literal text pattern.

        :param pattern: Regex or literal text to search for.
        :param path: Directory path relative to the workspace root.
        :param glob: Optional glob filter applied before reading candidate files.
        :param case_sensitive: Whether matching should preserve case.
        :param regex: Whether pattern should be interpreted as a regex.
        :param files_with_matches: Whether to return only one entry per matching file.
        :param limit: Maximum number of returned matches.
        """
        result = self.grep_v2(
            pattern=pattern,
            path=path,
            glob=glob,
            case_sensitive=case_sensitive,
            regex=regex,
            files_with_matches=files_with_matches,
            limit=limit,
            runtime_context=runtime_context,
        )
        if result.get("status") == "success":
            result["num_matches"] = result.get("match_count", 0)
        return result

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
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Search workspace text with fixed-argv ripgrep.

        :param pattern: Regular expression or literal text to find.
        :param path: Workspace-relative directory to search.
        :param glob: Optional file glob filter.
        :param case_sensitive: Preserve case when true.
        :param regex: Interpret pattern as regex when true, literal text otherwise.
        :param files_with_matches: Return only matching paths when true.
        :param limit: Maximum returned matches.
        :param context: Number of context lines requested from ripgrep.
        :param file_type: Optional ripgrep file type such as py or rust.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        return self.grep_v2(
            pattern=pattern,
            path=path,
            glob=glob,
            case_sensitive=case_sensitive,
            regex=regex,
            files_with_matches=files_with_matches,
            limit=limit,
            context=context,
            file_type=file_type,
            runtime_context=runtime_context,
        )

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
        name="read_file_range",
        read_only=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
    )
    def read_file_range(
        self,
        path: str,
        offset: int = 0,
        limit: int = 200,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Read a bounded line range from one workspace file.

        :param path: File path relative to the workspace root.
        :param offset: Zero-based starting line offset.
        :param limit: Maximum number of lines to return.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        result = self.file_read_v2(
            path=path, offset=offset, limit=limit, runtime_context=runtime_context
        )
        if result.get("status") != "success":
            return result
        return {
            "status": "success",
            "path": path,
            "offset": result.get("offset", offset),
            "limit": result.get("limit", limit),
            "total_lines": result.get("total_lines", 0),
            "content": result.get("content", ""),
            "has_more": result.get("has_more", False),
            "truncated": result.get("truncated", False),
        }

    @function_tool(
        name="search",
        read_only=True,
        environment_ops=["file", "process"],
        rule_scope_builder=_default_rule_scope,
    )
    def search(
        self,
        path: str,
        keyword: str,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search for a keyword inside files within a directory tree.

        :param path: Directory path relative to the workspace root.
        :param keyword: Keyword to search for.
        """
        return self.grep_v2(
            pattern=keyword,
            path=path,
            regex=False,
            limit=15,
            runtime_context=runtime_context,
        )

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
        name="web_fetch_v2",
        needs_approval=True,
        rule_scope_builder=_default_rule_scope,
    )
    def web_fetch_v2(
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
            "result": result,
            "title": extracted.get("title", ""),
            "auth_hint": auth_hint,
        }

    @function_tool(
        name="web_fetch",
        needs_approval=True,
        rule_scope_builder=_default_rule_scope,
    )
    def web_fetch(
        self, url: str, runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fetch one web page and extract readable text.

        :param url: Absolute URL to fetch.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        payload = self.web_fetch_v2(url=url, prompt="", runtime_context=runtime_context)
        if payload.get("status") != "success":
            return payload
        return {
            "status": "success",
            "url": payload.get("url", url),
            "redirect_url": payload.get("redirect_url"),
            "title": payload.get("title", ""),
            "content": payload.get("result", ""),
            "auth_hint": payload.get("auth_hint", ""),
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

    @function_tool(
        name="agent_spawn",
        prompt=(
            "Launch a new agent to handle a sub-task autonomously. "
            "The agent runs in an isolated context with its own tool set.\n\n"
            "Available agent types:\n"
            "- explore: Fast codebase search agent (Read, Glob, Grep). Use for finding files, "
            "searching code, or answering questions about the codebase.\n"
            "- plan: Read-only architecture planning agent. Use for designing implementation approaches.\n"
            "- general: General-purpose agent with full tool access. Use for complex multi-step tasks.\n\n"
            "The prompt should be self-contained — the agent won't see this conversation. "
            "Include all context the agent needs (file paths, what to look for, etc.)."
        ),
    )
    def agent_spawn(
        self,
        task: str = "",
        subagent_type: str = "explore",
        max_steps: int = 8,
        run_in_background: bool = False,
        runtime_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Spawn a sub-agent to handle a task autonomously.

        Creates a child Engine with the sub-agent's toolset and runs it.
        Returns the agent's final answer and step summary.

        :param task: The task prompt for the sub-agent.
        :param subagent_type: Agent type (explore, plan, general).
        :param max_steps: Maximum steps for the sub-agent.
        :param run_in_background: If True, run agent in background thread.
        :param runtime_context: Runtime context from the executor.
        """
        if not task:
            return {"status": "error", "message": "No task provided for sub-agent."}

        # Get the parent agent's LLM and protocol
        state_obj = (runtime_context or {}).get("state")
        llm = None
        model_parser = None
        model_protocol = None

        # Try to get LLM from the parent engine's agent
        engine = (runtime_context or {}).get("engine")
        if engine is None and runtime_context:
            # Walk up to find the engine
            tool_registry = runtime_context.get("tool_registry")
            if tool_registry and hasattr(tool_registry, "_engine"):
                engine = tool_registry._engine

        if engine is not None:
            parent_agent = getattr(engine, "agent", None)
            if parent_agent is not None:
                llm = getattr(parent_agent, "llm", None)
                model_parser = getattr(parent_agent, "model_parser", None)
                model_protocol = getattr(parent_agent, "model_protocol", None)

        if llm is None:
            return {"status": "error", "message": "No LLM available for sub-agent."}

        try:
            agent = self._create_sub_agent(
                subagent_type=subagent_type,
                llm=llm,
                max_steps=max_steps,
                model_parser=model_parser,
                model_protocol=model_protocol,
            )
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        if run_in_background:
            return self._run_agent_background(agent, task)
        return self._run_agent_sync(agent, task)

    def _create_sub_agent(
        self,
        subagent_type: str,
        llm: Any,
        max_steps: int,
        model_parser: Any = None,
        model_protocol: Any = None,
    ) -> Any:
        """Create a sub-agent instance based on type."""
        subagent_type = subagent_type.lower().strip()

        if subagent_type == "explore":
            from qitos.kit.tool.internal.subagents import ExploreAgent
            return ExploreAgent(
                llm=llm,
                workspace_root=self.workspace_root,
                max_steps=max_steps,
                model_parser=model_parser,
                model_protocol=model_protocol,
            )
        elif subagent_type == "plan":
            from qitos.kit.tool.internal.subagents import PlanAgent
            return PlanAgent(
                llm=llm,
                workspace_root=self.workspace_root,
                max_steps=max_steps,
                model_parser=model_parser,
                model_protocol=model_protocol,
            )
        elif subagent_type in ("general", "general-purpose"):
            from qitos.kit.tool.internal.subagents import GeneralAgent
            return GeneralAgent(
                llm=llm,
                workspace_root=self.workspace_root,
                max_steps=max_steps,
                model_parser=model_parser,
                model_protocol=model_protocol,
            )
        else:
            raise ValueError(
                f"Unknown sub-agent type: '{subagent_type}'. "
                f"Available types: explore, plan, general"
            )

    def _run_agent_sync(self, agent: Any, task: str) -> Dict[str, Any]:
        """Run a sub-agent synchronously and return results."""
        from qitos.engine.engine import Engine
        from qitos.engine.states import ContextConfig, RuntimeBudget

        # Propagate permission pipeline and RBW enforcer from parent engine
        parent_pipeline = None
        parent_rbw = None
        if self._engine is not None and self._engine.executor is not None:
            parent_pipeline = getattr(self._engine.executor, "_pipeline", None)
            parent_rbw = getattr(self._engine.executor, "_rbw_enforcer", None)

        engine = Engine(
            agent=agent,
            budget=RuntimeBudget(max_steps=agent.max_steps),
            permission_pipeline=parent_pipeline,
            read_before_write_enforcer=parent_rbw,
            context_config=ContextConfig(
                tool_result_max_chars=50000,
                tool_result_per_message_max_chars=200000,
                reactive_compact=True,
            ),
        )
        result = engine.run(task)

        final_answer = ""
        if result.task_result is not None:
            final_answer = str(getattr(result.task_result, "final_output", "")) or ""
        if not final_answer:
            final_answer = str(getattr(result.state, "final_result", "")) or ""

        step_summaries = []
        for s in result.step_summaries:
            step_summaries.append({
                "step": s.step_id,
                "tool": s.tool_name,
                "status": s.status,
            })

        return {
            "status": "success",
            "spawned": True,
            "subagent_type": agent.name,
            "final_answer": final_answer[:8000],
            "step_count": result.step_count,
            "step_summaries": step_summaries,
            "total_tokens": result.total_tokens,
            "runtime_seconds": round(result.runtime_seconds, 2),
        }

    def _run_agent_background(self, agent: Any, task: str) -> Dict[str, Any]:
        """Run a sub-agent in a background thread."""
        import threading

        task_id = f"agent_{id(agent)}_{threading.get_ident()}"

        # Store in session tasks
        self._session_tasks[task_id] = {
            "status": "running",
            "agent_name": agent.name,
            "task": task[:200],
        }

        def _run():
            try:
                result_dict = self._run_agent_sync(agent, task)
                self._session_tasks[task_id] = {
                    **result_dict,
                    "status": "completed",
                }
            except Exception as exc:
                self._session_tasks[task_id] = {
                    "status": "error",
                    "error": str(exc),
                }

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        return {
            "status": "success",
            "spawned": True,
            "background": True,
            "task_id": task_id,
            "message": f"Agent running in background. Use task_get with task_id='{task_id}' to check results.",
        }

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

    # ── Claude Code modern-name aliases ────────────────────────────────────────
    # These match Claude Code's exact tool names and signatures for compatibility.

    @function_tool(
        name="Read",
        read_only=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
        prompt=(
            "Reads a file from the local filesystem. You can access any file directly by using this tool.\n"
            "Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid.\n"
            "Usage:\n"
            "- The file_path parameter must be an absolute path, not a relative path\n"
            "- By default, it reads up to 2000 lines starting from the beginning of the file\n"
            "- You can optionally specify a line offset and limit, but it's recommended to read the whole file by not providing these parameters\n"
            "- When you already know which part of the file you need, only read that part. This can be important for larger files.\n"
            "- This tool can only read files, not directories. To read a directory, use an ls command via the Bash tool.\n"
            "- If you read a file that exists but has empty contents you will receive a system reminder warning."
        ),
    )
    def Read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        *,
        pages: Optional[str] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str | Dict[str, Any]:
        """Read a file, image, PDF, or notebook. Returns content with line numbers.

        :param file_path: Absolute or relative path to the file.
        :param offset: Line number to start reading from (0-based).
        :param limit: Maximum number of lines to read.
        :param pages: Page range for PDF files (e.g., "1-5", "3").
        :param runtime_context: Optional runtime context injected by the executor.
        """
        result = self.file_read_v2(
            path=file_path,
            offset=offset,
            limit=limit,
            max_chars=200_000,
            runtime_context=runtime_context,
        )
        if result.get("status") != "success":
            return result
        content = str(result.get("content", ""))
        # Add line numbers like Claude Code
        lines = content.splitlines() if content else []
        numbered = []
        start = int(result.get("offset", offset))
        for i, line in enumerate(lines, start=start + 1):
            numbered.append(f"{i}\t{line}")
        if result.get("has_more"):
            next_offset = start + len(lines)
            numbered.append(
                f"[truncated: use offset={next_offset} to continue; "
                f"total_lines={result.get('total_lines', '?')}]"
            )
        return "\n".join(numbered)

    @function_tool(
        name="Edit",
        needs_approval=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
        prompt=(
            "Performs exact string replacements in files.\n"
            "Usage:\n"
            "- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.\n"
            "- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. Never include any part of the line number prefix in the old_string or new_string.\n"
            "- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n"
            "- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.\n"
            "- Use `replace_all` for replacing and renaming strings across the file."
        ),
    )
    def Edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str | Dict[str, Any]:
        """Replace old_string with new_string in a file. old_string must be unique unless replace_all=True.

        :param file_path: Absolute or relative path to the file.
        :param old_string: Text to find and replace. Must appear exactly once unless replace_all=True.
        :param new_string: Replacement text.
        :param replace_all: Replace all occurrences of old_string.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        result = self.file_edit_v2(
            path=file_path,
            action="str_replace",
            old_text=old_string,
            new_text=new_string,
            replace_all=replace_all,
            runtime_context=runtime_context,
        )
        if result.get("status") != "success":
            return result
        return str(result.get("message") or "Edit applied successfully")

    @function_tool(
        name="Write",
        needs_approval=True,
        environment_ops=["file"],
        rule_scope_builder=_default_rule_scope,
        prompt=(
            "Writes a file to the local filesystem.\n"
            "Usage:\n"
            "- This tool will overwrite the existing file if there is one at the provided path.\n"
            "- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.\n"
            "- Prefer the Edit tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites.\n"
            "- NEVER create documentation files (*.md) or README files unless explicitly requested by the User."
        ),
    )
    def Write(
        self,
        file_path: str,
        content: str,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str | Dict[str, Any]:
        """Write content to a file, creating it if it doesn't exist.

        :param file_path: Absolute or relative path to the file.
        :param content: Content to write.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        result = self.write_file(
            path=file_path,
            content=content,
            runtime_context=runtime_context,
        )
        if result.get("status") != "success":
            return result
        return f"Successfully wrote to {file_path}"

    @function_tool(
        name="Glob",
        read_only=True,
        environment_ops=["file", "process"],
        rule_scope_builder=_default_rule_scope,
        prompt=(
            "Fast file pattern matching tool that works with any codebase size.\n"
            "Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\". Returns matching file paths sorted by modification time.\n"
            "Use this tool when you need to find files by name patterns. When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead."
        ),
    )
    def Glob(
        self,
        pattern: str,
        path: str = ".",
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str | Dict[str, Any]:
        """Find files matching a glob pattern.

        :param pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts").
        :param path: Directory to search in.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        result = self.glob_v2(
            pattern=pattern,
            path=path,
            runtime_context=runtime_context,
        )
        if result.get("status") != "success":
            return result
        files = result.get("files", [])
        return "\n".join(files)

    @function_tool(
        name="Grep",
        read_only=True,
        environment_ops=["file", "process"],
        rule_scope_builder=_default_rule_scope,
        prompt=(
            "A powerful search tool built on ripgrep.\n"
            "Usage:\n"
            "- ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.\n"
            "- Supports full regex syntax (e.g., \"log.*Error\", \"function\\\\s+\\\\w+\")\n"
            "- Filter files with glob parameter (e.g., \"*.js\", \"**/*.tsx\") or type parameter\n"
            "- Output modes: \"content\" shows matching lines, \"files_with_matches\" shows only file paths (default), \"count\" shows match counts\n"
            "- Use Agent tool for open-ended searches requiring multiple rounds"
        ),
    )
    def Grep(
        self,
        pattern: str,
        path: str = ".",
        glob: Optional[str] = None,
        type: Optional[str] = None,
        output_mode: str = "content",
        context: int = 0,
        head_limit: int = 100,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str | Dict[str, Any]:
        """Search file contents using regex patterns.

        :param pattern: Regular expression pattern to search for.
        :param path: Directory or file to search in.
        :param glob: File pattern filter (e.g., "*.py").
        :param type: File type filter (js, py, rust, etc.).
        :param output_mode: "content", "files_with_matches", or "count".
        :param context: Number of lines of context before/after matches.
        :param head_limit: Maximum number of results.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        # Map Claude Code's output_mode to grep_v2 parameters
        files_with_matches = output_mode == "files_with_matches"
        result = self.grep_v2(
            pattern=pattern,
            path=path,
            glob=glob,
            case_sensitive=False,
            regex=True,
            files_with_matches=files_with_matches,
            limit=head_limit,
            context=context,
            file_type=type,
            runtime_context=runtime_context,
        )
        if result.get("status") != "success":
            return result
        if files_with_matches:
            matches = result.get("matches", [])
            return "\n".join(
                str(match.get("path", ""))
                for match in matches
                if isinstance(match, dict)
            )
        matches = result.get("matches", [])
        if output_mode == "count":
            counts: Dict[str, int] = {}
            for match in matches:
                if isinstance(match, dict):
                    match_path = str(match.get("path", ""))
                    counts[match_path] = counts.get(match_path, 0) + 1
            return "\n".join(
                f"{match_path}:{count}" for match_path, count in sorted(counts.items())
            )
        lines = []
        for m in matches:
            if isinstance(m, dict):
                lines.append(
                    f"{m.get('path', '')}:{m.get('line', '')}:{m.get('text', '')}"
                )
            else:
                lines.append(str(m))
        return "\n".join(lines)

    @function_tool(
        name="Bash",
        needs_approval=True,
        supports_background=True,
        environment_ops=["process"],
        rule_scope_builder=_default_rule_scope,
        prompt=(
            "Executes a given bash command and returns its output.\n"
            "The working directory persists between commands, but shell state does not. The shell environment is initialized from the user's profile (bash or zsh).\n"
            "IMPORTANT: Avoid using this tool to run `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task. Instead, use the appropriate dedicated tool as this will provide a much better experience for the user:\n"
            " - File search: Use Glob (NOT find or ls)\n"
            " - Content search: Use Grep (NOT grep or rg)\n"
            " - Read files: Use Read (NOT cat/head/tail)\n"
            " - Edit files: Use Edit (NOT sed/awk)\n"
            " - Write files: Use Write (NOT echo >/cat <<EOF)\n"
            "If your command will create new directories or files, first use this tool to run `ls` to verify the parent directory exists. Try to maintain your current working directory throughout the session by using absolute paths. You may specify an optional timeout in milliseconds. You can use `run_in_background` to run commands in the background.\n"
            "For git commands: Prefer to create a new commit rather than amending an existing commit. Before running destructive operations, consider whether there is a safer alternative. Never skip hooks (--no-verify) unless the user has explicitly asked for it.\n"
            "For git commit messages, use HEREDOC format: git commit -m \"$(cat <<'EOF'\\n  Commit message here.\\n  EOF\\n  )\""
        ),
    )
    def Bash(
        self,
        command: str,
        description: str = "",
        timeout: Optional[int] = None,
        run_in_background: bool = False,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str | Dict[str, Any]:
        """Execute a shell command.

        :param command: Shell command to execute.
        :param description: Brief description of what the command does.
        :param timeout: Timeout in milliseconds (max 600000).
        :param run_in_background: Run command in background and return task ID.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        result = self.bash_v2(
            command=command,
            read_only=False,
            allow_destructive=False,
            run_in_background=run_in_background,
            runtime_context=runtime_context,
        )
        if result.get("status") in {"error", "needs_input", "needs_approval"}:
            return result
        if result.get("status") != "success":
            error = result.get("error") or result.get("message", "")
            returncode = result.get("returncode", 1)
            stdout = result.get("stdout", "")
            if stdout:
                return f"Exit code {returncode}:\n{stdout}\n{error}"
            return f"Error: {error}"
        stdout = result.get("stdout", "")
        returncode = result.get("returncode", 0)
        if returncode != 0:
            stderr = result.get("stderr", "")
            return f"Exit code {returncode}:\n{stdout}\n{stderr}"
        return stdout

    @function_tool(
        name="WebFetch",
        needs_approval=True,
        rule_scope_builder=_default_rule_scope,
        prompt=(
            "Fetches content from a specified URL and processes it using an AI model. Takes a URL and a prompt as input. Fetches the URL content, converts HTML to markdown. Processes the content with the prompt using a small, fast model.\n"
            "Usage notes:\n"
            "- The URL must be a fully-formed valid URL. HTTP URLs will be automatically upgraded to HTTPS.\n"
            "- The prompt should describe what information you want to extract from the page.\n"
            "- This tool is read-only and does not modify any files.\n"
            "- Results may be summarized if the content is very large.\n"
            "- Includes a self-cleaning 15-minute cache for faster responses.\n"
            "- For GitHub URLs, prefer using the gh CLI via Bash instead."
        ),
    )
    def WebFetch(
        self,
        url: str,
        prompt: str = "",
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Fetch a URL and convert to markdown, optionally summarizing with AI.

        :param url: URL to fetch.
        :param prompt: Optional prompt for AI summarization of the content.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        result = self.web_fetch_v2(
            url=url,
            prompt=prompt,
            runtime_context=runtime_context,
        )
        if result.get("status") != "success":
            return f"Error fetching URL: {result.get('error', 'unknown error')}"
        return result.get("content", "")

    @function_tool(
        name="AskUserQuestion",
        requires_user_interaction=True,
        prompt=(
            "Use this tool when you need to ask the user questions during execution. This allows you to:\n"
            "1. Gather user preferences or requirements\n"
            "2. Clarify ambiguous instructions\n"
            "3. Get decisions on implementation choices as you work\n"
            "4. Offer choices to the user about what direction to take.\n"
            "Usage notes:\n"
            "- Users will always be able to select \"Other\" to provide custom text input\n"
            "- Use multiSelect: true to allow multiple answers to be selected for a question\n"
            "- If you recommend a specific option, make that the first option in the list and add \"(Recommended)\" at the end of the label"
        ),
    )
    def AskUserQuestion(
        self,
        questions: List[Dict[str, Any]],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ask the user one or more questions with optional choices.

        :param questions: List of question dicts with 'question', 'options', and optional 'preview'.
        :param runtime_context: Optional runtime context injected by the executor.
        """
        return self.ask_user_choice(
            questions=questions,
            runtime_context=runtime_context,
        )


__all__ = ["CodingToolSet", "TASK_STATUSES", "_resolve_workspace_path"]
