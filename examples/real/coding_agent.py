"""Practical coding agent: memory-backed ReAct plus self-reflection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qitos import (
    Action,
    AgentModule,
    Decision,
    HistoryPolicy,
    StateSchema,
    ToolRegistry,
)
from qitos.kit import (
    CodingToolSet,
    MarkdownFileMemory,
    ReActSelfReflectionCritic,
    ReActTextParser,
    format_action,
    render_prompt,
)
from qitos.models import OpenAICompatibleModel

TASK = "Fix the bug in buggy_module.py and make the verification command pass."
WORKSPACE = Path("./playground/coding_agent")
MODEL_NAME = os.getenv("QITOS_MODEL", "Qwen/Qwen3-8B")
MODEL_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1/")
MAX_STEPS = 14
TARGET_FILE = "buggy_module.py"
TEST_COMMAND = 'python -c "import buggy_module; assert buggy_module.add(20, 22) == 42"'
EXPECTED_SNIPPET = "return a + b"

SYSTEM_PROMPT = """You are a production-grade coding agent.

Mission:
- Repair code in the workspace.
- Validate using tests or check commands.
- Keep patches minimal and reversible.

Execution protocol:
1. Start by inspecting the target file.
2. Apply one precise edit per step.
3. Run verification frequently.
4. If a command succeeds and confirms the requirement, end with Final Answer.

Available tools:
{tool_schema}

Return format (strict):
Thought: <short reasoning>
Action: <tool_name>(arg=value, ...)
or
Final Answer: <what changed + verification proof>
"""


@dataclass
class CodingState(StateSchema):
    scratchpad: list[str] = field(default_factory=list)
    target_file: str = TARGET_FILE
    test_command: str = TEST_COMMAND
    expected_snippet: str = EXPECTED_SNIPPET


class CodingMemoryReactAgent(AgentModule[CodingState, dict[str, Any], Action]):
    def __init__(
        self, llm: Any, workspace_root: str, memory: MarkdownFileMemory | None = None
    ):
        registry = ToolRegistry()
        registry.include(
            CodingToolSet(
                workspace_root=workspace_root,
                include_notebook=False,
                enable_lsp=False,
                enable_tasks=False,
                enable_web=False,
            )
        )
        super().__init__(
            tool_registry=registry,
            llm=llm,
            model_parser=ReActTextParser(),
            memory=memory,
        )

    def init_state(self, task: str, **kwargs: Any) -> CodingState:
        return CodingState(
            task=task,
            max_steps=int(kwargs.get("max_steps", MAX_STEPS)),
            target_file=str(kwargs.get("target_file", TARGET_FILE)),
            test_command=str(kwargs.get("test_command", TEST_COMMAND)),
            expected_snippet=str(kwargs.get("expected_snippet", EXPECTED_SNIPPET)),
        )

    def build_system_prompt(self, state: CodingState) -> str | None:
        return render_prompt(
            SYSTEM_PROMPT, {"tool_schema": self.tool_registry.get_tool_descriptions()}
        )

    def prepare(self, state: CodingState) -> str:
        lines = [
            f"Task: {state.task}",
            f"Target file: {state.target_file}",
            f"Expected snippet: {state.expected_snippet}",
            f"Verification command: {state.test_command}",
            f"Step: {state.current_step}/{state.max_steps}",
        ]
        if self.memory is not None:
            memory_rows = (
                self.memory.retrieve(query=None, state=state, observation=None) or []
            )
            if memory_rows:
                lines.append("Retrieved memory:")
                for item in memory_rows[-4:]:
                    lines.append(f"- {getattr(item, 'content', item)}")
        if state.scratchpad:
            lines.append("Recent trajectory:")
            lines.extend(state.scratchpad[-10:])
        return "\n".join(lines)

    def reduce(
        self,
        state: CodingState,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> CodingState:
        action_results = (
            observation.get("action_results", [])
            if isinstance(observation, dict)
            else []
        )
        if decision.rationale:
            state.scratchpad.append(f"Thought: {decision.rationale}")
        if decision.actions:
            state.scratchpad.append(f"Action: {format_action(decision.actions[0])}")
        if action_results:
            first = action_results[0]
            state.scratchpad.append(f"Observation: {first}")
            if isinstance(first, dict) and int(first.get("returncode", 1)) == 0:
                state.final_result = f"Verification succeeded for {state.target_file}."
        state.scratchpad = state.scratchpad[-40:]
        return state


def build_model() -> OpenAICompatibleModel:
    api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("QITOS_API_KEY") or "").strip()
    if not api_key:
        raise ValueError(
            "Set OPENAI_API_KEY or QITOS_API_KEY before running this example."
        )
    return OpenAICompatibleModel(
        model=MODEL_NAME,
        api_key=api_key,
        base_url=MODEL_BASE_URL,
        temperature=0.2,
        max_tokens=2048,
    )


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    target = WORKSPACE / TARGET_FILE
    if not target.exists():
        target.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

    memory_path = WORKSPACE / "memory.md"
    agent = CodingMemoryReactAgent(
        llm=build_model(),
        workspace_root=str(WORKSPACE),
        memory=MarkdownFileMemory(path=str(memory_path)),
    )
    result = agent.run(
        task=TASK,
        workspace=str(WORKSPACE),
        target_file=TARGET_FILE,
        test_command=TEST_COMMAND,
        expected_snippet=EXPECTED_SNIPPET,
        max_steps=MAX_STEPS,
        critics=[ReActSelfReflectionCritic(max_retries=2)],
        history_policy=HistoryPolicy(max_messages=12),
        return_state=True,
    )

    print("workspace:", WORKSPACE)
    print("final_result:", result.state.final_result)
    print("stop_reason:", result.state.stop_reason)
    print("memory_md:", memory_path)


if __name__ == "__main__":
    main()
