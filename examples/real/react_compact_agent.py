"""Focused example: enable CompactHistory on a normal ReAct agent."""

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
    CompactHistory,
    CodingToolSet,
    REACT_SYSTEM_PROMPT,
    ReActTextParser,
    format_action,
    render_prompt,
)
from qitos.models import OpenAICompatibleModel

TASK = "Open buggy_module.py, fix add(a, b) so it returns a + b, then run verification."
WORKSPACE = Path("./playground/react_compact_agent")
MODEL_NAME = os.getenv("QITOS_MODEL", "Qwen/Qwen3-8B")
MODEL_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1/")
MAX_STEPS = 8


@dataclass
class CompactReactState(StateSchema):
    scratchpad: list[str] = field(default_factory=list)
    target_file: str = "buggy_module.py"
    test_command: str = (
        'python -c "import buggy_module; assert buggy_module.add(20, 22) == 42"'
    )


class CompactReactAgent(AgentModule[CompactReactState, dict[str, Any], Action]):
    def __init__(self, llm: Any, workspace_root: str):
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
            history=CompactHistory(
                llm=llm,
                max_tokens=2200,
                keep_last_rounds=2,
                hard_window=48,
            ),
        )

    def init_state(self, task: str, **kwargs: Any) -> CompactReactState:
        return CompactReactState(
            task=task, max_steps=int(kwargs.get("max_steps", MAX_STEPS))
        )

    def build_system_prompt(self, state: CompactReactState) -> str | None:
        return render_prompt(
            REACT_SYSTEM_PROMPT,
            {"tool_schema": self.tool_registry.get_tool_descriptions()},
        )

    def prepare(self, state: CompactReactState) -> str:
        lines = [
            f"Task: {state.task}",
            f"Target file: {state.target_file}",
            f"Verification command: {state.test_command}",
            f"Step: {state.current_step}/{state.max_steps}",
        ]
        if state.scratchpad:
            lines.append("Recent trajectory:")
            lines.extend(state.scratchpad[-10:])
        return "\n".join(lines)

    def reduce(
        self,
        state: CompactReactState,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> CompactReactState:
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
                state.final_result = "Patch applied and verification passed."
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
    target = WORKSPACE / "buggy_module.py"
    if not target.exists():
        target.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

    model = build_model()
    agent = CompactReactAgent(llm=model, workspace_root=str(WORKSPACE))
    result = agent.run(
        task=TASK,
        workspace=str(WORKSPACE),
        max_steps=MAX_STEPS,
        history_policy=HistoryPolicy(max_messages=16, max_tokens=2200),
        return_state=True,
    )
    print("workspace:", WORKSPACE)
    print("final_result:", result.state.final_result)
    print("stop_reason:", result.state.stop_reason)


if __name__ == "__main__":
    main()
