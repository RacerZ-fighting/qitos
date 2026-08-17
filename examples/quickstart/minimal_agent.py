"""Minimal QitOS quickstart: one Agent façade run with one Tool.

The agent loop is ``Message -> Model -> ToolCall -> ToolResult``; the ``Agent``
façade owns the transcript and lifecycle. This example composes them directly —
no Engine, no AgentModule, no parser.

Run it against any OpenAI-compatible endpoint:

    export OPENAI_API_KEY="your_api_key"
    export OPENAI_BASE_URL="https://api.openai.com/v1"
    export QITOS_MODEL="gpt-4o-mini"          # any served model name
    python examples/quickstart/minimal_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from qitos.core.agent import Agent
from qitos.core.agent_loop import AgentRunStatus
from qitos.core.message import AssistantMessage
from qitos.core.tool import tool
from qitos.core.tool_registry import ToolRegistry
from qitos.models import OpenAICompatibleModel


@tool(name="echo")
def echo(text: str) -> str:
    """Repeat the given text back to the caller."""

    return text


async def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY (and OPENAI_BASE_URL) to run this example.")
        return 1
    model = OpenAICompatibleModel(
        model=os.getenv("QITOS_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    agent = Agent(
        model=model,
        tool_registry=ToolRegistry().register(echo),
        system_prompt="You are a minimal helpful agent. Answer concisely.",
        max_turns=4,
    )
    result = await agent.prompt(
        "Call the echo tool once with a short greeting, then reply with what it returned."
    )
    if result.status is not AgentRunStatus.COMPLETED:
        print(f"run ended with status={result.status.value}: {result.error or ''}")
        return 1
    for message in reversed(result.messages):
        if isinstance(message, AssistantMessage) and message.text.strip():
            print(message.text)
            break
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
