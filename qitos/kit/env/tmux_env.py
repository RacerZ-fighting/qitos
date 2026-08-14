"""Tmux-backed interactive terminal environment."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from qitos.core.action import Action
from qitos.core.env import (
    Env,
    EnvObservation,
    EnvStepResult,
    RuntimeCapabilitySnapshot,
    TerminalCapability,
)


class TmuxTerminalCapability(TerminalCapability):
    """Interactive terminal ops backed by a local tmux session."""

    def __init__(
        self,
        session_name: str,
        cwd: str = ".",
        *,
        auto_start: bool = True,
        auto_kill: bool = True,
    ):
        self.session_name = session_name
        self.cwd = str(Path(cwd).resolve())
        self.auto_start = bool(auto_start)
        self.auto_kill = bool(auto_kill)
        self._previous_buffer: Optional[str] = None
        self._started_here = False
        if auto_start:
            self.reset_session()

    def reset_session(self, cwd: Optional[str] = None) -> None:
        if cwd:
            self.cwd = str(Path(cwd).resolve())
        self._ensure_tmux()
        if self.is_session_alive():
            self._previous_buffer = None
            return
        cmd = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            self.session_name,
            "-c",
            self.cwd,
            "bash",
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        subprocess.run(
            ["tmux", "set-option", "-t", self.session_name, "history-limit", "50000"],
            capture_output=True,
            text=True,
            check=False,
        )
        self._started_here = True
        self._previous_buffer = None

    def close_session(self) -> None:
        if not self.is_session_alive():
            return
        if self.auto_kill and self._started_here:
            subprocess.run(
                ["tmux", "kill-session", "-t", self.session_name],
                capture_output=True,
                text=True,
                check=False,
            )

    def send_keys(
        self,
        keys: str | list[str],
        min_timeout_sec: float = 0.0,
        block: bool = False,
        max_timeout_sec: float = 180.0,
    ) -> Dict[str, Any]:
        self._ensure_alive()
        sent_keys = self._normalize_keys(keys)
        if sent_keys:
            result = subprocess.run(
                ["tmux", "send-keys", "-t", self.session_name, *sent_keys],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return {
                    "status": "error",
                    "error": (
                        result.stderr or result.stdout or "tmux send-keys failed"
                    ).strip(),
                    "session_name": self.session_name,
                }
        wait_seconds = float(min_timeout_sec or 0.0)
        if block and wait_seconds <= 0.0:
            wait_seconds = min(float(max_timeout_sec), 0.2)
        if wait_seconds > 0.0:
            time.sleep(min(wait_seconds, float(max_timeout_sec)))
        return {
            "status": "success",
            "session_name": self.session_name,
            "cwd": self.cwd,
            "keys": sent_keys,
            "waited_seconds": wait_seconds,
            "block": bool(block),
        }

    def capture_screen(self) -> str:
        self._ensure_alive()
        return self._capture(capture_entire=False)

    def capture_buffer(self) -> str:
        self._ensure_alive()
        return self._capture(capture_entire=True)

    def get_incremental_output(self) -> str:
        current_buffer = self.capture_buffer()
        if self._previous_buffer is None:
            self._previous_buffer = current_buffer
            return f"Current Terminal Screen:\n{self.capture_screen()}"

        new_content = self._find_new_content(current_buffer)
        self._previous_buffer = current_buffer
        if new_content is not None and new_content.strip():
            return f"New Terminal Output:\n{new_content}"
        return f"Current Terminal Screen:\n{self.capture_screen()}"

    def is_session_alive(self) -> bool:
        self._ensure_tmux()
        result = subprocess.run(
            ["tmux", "has-session", "-t", self.session_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0

    def get_timestamp(self) -> float | None:
        return time.time()

    def _capture(self, capture_entire: bool) -> str:
        extra = ["-S", "-"] if capture_entire else []
        result = subprocess.run(
            ["tmux", "capture-pane", "-p", *extra, "-t", self.session_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                (result.stderr or result.stdout or "tmux capture-pane failed").strip()
            )
        return result.stdout

    def _normalize_keys(self, keys: str | list[str]) -> list[str]:
        if isinstance(keys, list):
            return [str(x) for x in keys]
        text = str(keys)
        if not text:
            return []
        if text in {"C-c", "C-d", "Enter"}:
            return [text]
        out: list[str] = []
        current = []
        for ch in text:
            if ch in "\r\n":
                if current:
                    out.append("".join(current))
                    current = []
                out.append("Enter")
            else:
                current.append(ch)
        if current:
            out.append("".join(current))
        return out

    def _find_new_content(self, current_buffer: str) -> Optional[str]:
        previous = (self._previous_buffer or "").strip()
        if not previous:
            return None
        if previous in current_buffer:
            idx = current_buffer.index(previous) + len(previous)
            return current_buffer[idx:].lstrip("\n")
        return None

    def _ensure_tmux(self) -> None:
        if shutil.which("tmux") is None:
            raise RuntimeError("tmux is not installed or not available on PATH")

    def _ensure_alive(self) -> None:
        if not self.is_session_alive():
            raise RuntimeError(f"tmux session is not alive: {self.session_name}")


class TmuxEnv(Env):
    """Environment exposing an interactive tmux-backed terminal."""

    name = "tmux_env"
    version = "1.0"

    def __init__(
        self,
        workspace_root: str = ".",
        session_name: Optional[str] = None,
        terminal: Optional[TerminalCapability] = None,
        *,
        auto_kill: bool = True,
    ):
        self.workspace_root = str(Path(workspace_root).resolve())
        self.session_name = (
            session_name or f"qitos_{Path(self.workspace_root).name or 'terminal'}"
        )
        self.auto_kill = bool(auto_kill)
        self.terminal = terminal or TmuxTerminalCapability(
            session_name=self.session_name,
            cwd=self.workspace_root,
            auto_start=False,
            auto_kill=auto_kill,
        )
        self._last_error: Optional[str] = None

    def setup(
        self, task: Any = None, workspace: Optional[str] = None, **kwargs: Any
    ) -> None:
        if workspace:
            self.workspace_root = str(Path(workspace).resolve())
        Path(self.workspace_root).mkdir(parents=True, exist_ok=True)
        reset = getattr(self.terminal, "reset_session", None)
        if callable(reset):
            reset(cwd=self.workspace_root)

    def reset(
        self, task: Any = None, workspace: Optional[str] = None, **kwargs: Any
    ) -> EnvObservation:
        self.setup(task=task, workspace=workspace, **kwargs)
        self._last_error = None
        return self.observe(state=None)

    def observe(self, state: Any = None) -> EnvObservation:
        try:
            payload = self._terminal_payload()
        except Exception as exc:
            self._last_error = str(exc)
            payload = {
                "output": f"Terminal error: {exc}",
                "screen": "",
                "session_alive": False,
                "timestamp": None,
                "backend": "tmux",
            }
        return EnvObservation(
            data={
                "terminal": payload,
                "workspace_root": self.workspace_root,
                "last_error": self._last_error,
            },
            metadata={"session_name": self.session_name},
        )

    def step(self, action: Any, state: Any = None) -> EnvStepResult:
        obs = self.observe(state=state)
        terminal = obs.data.get("terminal", {}) if isinstance(obs.data, dict) else {}
        alive = bool(terminal.get("session_alive", False))
        error = self._last_error or (
            None if alive else f"terminal session is not alive: {self.session_name}"
        )
        return EnvStepResult(
            observation=obs,
            done=not alive,
            reward=None,
            info={"action_seen": self._to_action_name(action), "backend": "tmux"},
            error=error,
        )

    def health_check(self) -> Dict[str, Any]:
        if shutil.which("tmux") is None:
            return {
                "ok": False,
                "message": "tmux is not installed or not available on PATH",
            }
        return {
            "ok": True,
            "workspace_root": self.workspace_root,
            "session_name": self.session_name,
        }

    def get_ops(self, group: str) -> Any:
        if group == "terminal":
            return self.terminal
        return None

    def capability_snapshot(self) -> RuntimeCapabilitySnapshot:
        return RuntimeCapabilitySnapshot(
            backend="tmux",
            working_directory=self.workspace_root,
            operation_groups=("terminal",),
            facilities=("terminal.interactive",),
        )

    def teardown(self) -> None:
        close = getattr(self.terminal, "close_session", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _terminal_payload(self) -> Dict[str, Any]:
        return {
            "output": self.terminal.get_incremental_output(),
            "screen": self.terminal.capture_screen(),
            "session_alive": self.terminal.is_session_alive(),
            "timestamp": self.terminal.get_timestamp(),
            "backend": "tmux",
        }

    def _to_action_name(self, action: Any) -> str:
        if isinstance(action, Action):
            return action.name
        if isinstance(action, dict):
            return str(action.get("name", ""))
        return ""


__all__ = ["TmuxTerminalCapability", "TmuxEnv"]
