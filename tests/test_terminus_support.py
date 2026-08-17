from __future__ import annotations

from pathlib import Path

import pytest

from qitos.core import TerminalCapability
from qitos.kit import (
    SendTerminalKeys,
    TmuxEnv,
)

class FakeTerminal(TerminalCapability):
    def __init__(self):
        self.alive = True
        self.screen = "$ "
        self.buffer = "$ "
        self.previous = None
        self.sent: list[str] = []
        self.waits: list[float] = []
        self.reset_calls = 0
        self.closed = False
        self.ts = 0.0

    def reset_session(self, cwd: str | None = None) -> None:
        self.reset_calls += 1
        self.screen = "$ "
        self.buffer = "$ "
        self.previous = None
        self.alive = True

    def close_session(self) -> None:
        self.closed = True
        self.alive = False

    def send_keys(
        self,
        keys: str | list[str],
        min_timeout_sec: float = 0.0,
        block: bool = False,
        max_timeout_sec: float = 180.0,
    ) -> dict:
        text = "".join(keys) if isinstance(keys, list) else str(keys)
        self.sent.append(text)
        self.waits.append(float(min_timeout_sec))
        self.ts += 1.0
        if text.strip() == "pwd":
            update = "/workspace\n$ "
        elif "ls" in text:
            update = "README.txt\nnotes.txt\n$ "
        elif not text:
            update = self.screen
        else:
            update = f"executed: {text.strip()}\n$ "
        self.buffer += update
        self.screen = update
        return {
            "status": "success",
            "keys": text,
            "waited_seconds": min_timeout_sec,
            "block": block,
        }

    def capture_screen(self) -> str:
        return self.screen

    def capture_buffer(self) -> str:
        return self.buffer

    def get_incremental_output(self) -> str:
        current = self.buffer
        if self.previous is None:
            self.previous = current
            return f"Current Terminal Screen:\n{self.screen}"
        if self.previous in current:
            idx = current.index(self.previous) + len(self.previous)
            delta = current[idx:].lstrip("\n")
        else:
            delta = self.screen
        self.previous = current
        if delta.strip():
            return f"New Terminal Output:\n{delta}"
        return f"Current Terminal Screen:\n{self.screen}"

    def is_session_alive(self) -> bool:
        return self.alive

    def get_timestamp(self) -> float | None:
        return self.ts


@pytest.mark.asyncio
async def test_send_terminal_keys_tool_uses_terminal_ops() -> None:
    terminal = FakeTerminal()
    tool = SendTerminalKeys()
    result = await tool.execute(
        {"keystrokes": "ls\n", "duration_sec": 0.25},
        runtime_context={"ops": {"terminal": terminal}},
    )
    assert result["status"] == "success"
    assert terminal.sent == ["ls\n"]
    assert terminal.waits == [0.25]


@pytest.mark.asyncio
async def test_send_terminal_keys_submit_appends_newline_once() -> None:
    terminal = FakeTerminal()
    tool = SendTerminalKeys()
    result = await tool.execute(
        {
            "keystrokes": "pwd",
            "duration_sec": 0.1,
            "submit": True,
        },
        runtime_context={"ops": {"terminal": terminal}},
    )
    assert result["status"] == "success"
    assert result["submit"] is True
    assert terminal.sent == ["pwd\n"]


def test_tmux_env_can_wrap_custom_terminal_backend(tmp_path: Path) -> None:
    terminal = FakeTerminal()
    env = TmuxEnv(
        workspace_root=str(tmp_path),
        session_name="test-terminus",
        terminal=terminal,
        auto_kill=False,
    )
    obs = env.reset(workspace=str(tmp_path))
    terminal_payload = obs.data["terminal"]
    assert terminal_payload["backend"] == "tmux"
    assert terminal_payload["session_alive"] is True
    step = env.step({"name": "send_terminal_keys"})
    assert step.done is False
    env.teardown()
    assert terminal.closed is True
