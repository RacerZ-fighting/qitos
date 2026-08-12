"""Pattern: Plan-Act with an explicit plan builder and the default Engine model path."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qitos import Action, AgentModule, Decision, StateSchema, ToolRegistry
from qitos.kit import (
    CodingToolSet,
    NumberedPlanBuilder,
    PLAN_DRAFT_PROMPT,
    PLAN_EXEC_SYSTEM_PROMPT,
    ReActTextParser,
    format_action,
    render_prompt,
)
from qitos.models import OpenAICompatibleModel

TASK = "Fix buggy_module.py and verify the fix with the provided command."
WORKSPACE = Path("./playground/planact_pattern")
MODEL_NAME = os.getenv("QITOS_MODEL", "Qwen/Qwen3-8B")
MODEL_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1/")
MAX_STEPS = 10
TEST_COMMAND = 'python -c "import buggy_module; assert buggy_module.add(20, 22) == 42"'


@dataclass
class PlanActState(StateSchema):
    plan_steps: list[str] = field(default_factory=list)
    cursor: int = 0
    target_file: str = "buggy_module.py"
    test_command: str = TEST_COMMAND
    scratchpad: list[str] = field(default_factory=list)


class PlanActAgent(AgentModule[PlanActState, dict[str, Any], Action]):
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
            tool_registry=registry, llm=llm, model_parser=ReActTextParser()
        )
        self.plan_builder = NumberedPlanBuilder()

    def init_state(self, task: str, **kwargs: Any) -> PlanActState:
        return PlanActState(
            task=task, max_steps=int(kwargs.get("max_steps", MAX_STEPS))
        )

    def build_system_prompt(self, state: PlanActState) -> str | None:
        return render_prompt(
            PLAN_EXEC_SYSTEM_PROMPT,
            {
                "current_step": self._current_step_text(state),
                "tool_schema": self.tool_registry.get_tool_descriptions(),
            },
        )

    def prepare(self, state: PlanActState) -> str:
        lines = [
            f"Task: {state.task}",
            f"Plan cursor: {state.cursor}/{len(state.plan_steps)}",
            f"Current plan step: {self._current_step_text(state)}",
            f"Step: {state.current_step}/{state.max_steps}",
        ]
        if state.plan_steps:
            lines.append("Plan:")
            for idx, item in enumerate(state.plan_steps):
                marker = "->" if idx == state.cursor else "  "
                lines.append(f"{marker} [{idx}] {item}")
        if state.scratchpad:
            lines.append("Recent trajectory:")
            lines.extend(state.scratchpad[-10:])
        return "\n".join(lines)

    def decide(self, state: PlanActState, observation: dict[str, Any]):
        if not state.plan_steps or state.cursor >= len(state.plan_steps):
            if not self._plan(state):
                return Decision.final("Failed to build a valid plan.")
            return Decision.wait("plan_ready")
        return None

    def reduce(
        self,
        state: PlanActState,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> PlanActState:
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
            if isinstance(first, dict) and first.get("status") == "success":
                state.cursor += 1
            if isinstance(first, dict) and int(first.get("returncode", 1)) == 0:
                state.final_result = "Verification passed."
                state.cursor = len(state.plan_steps)
        state.scratchpad = state.scratchpad[-40:]
        return state

    def _plan(self, state: PlanActState) -> bool:
        prompt = render_prompt(
            PLAN_DRAFT_PROMPT,
            {
                "task": (
                    f"{state.task}\n"
                    f"Target file: {state.target_file}\n"
                    f"Last step must run: {state.test_command}"
                ),
            },
        )
        plan = self.plan_builder.build(self.llm, prompt)
        if not plan:
            return False
        state.plan_steps = plan
        state.cursor = 0
        state.scratchpad.append("Plan: " + " | ".join(plan))
        return True

    def _current_step_text(self, state: PlanActState) -> str:
        if state.cursor < 0 or state.cursor >= len(state.plan_steps):
            return "none"
        return state.plan_steps[state.cursor]


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

    agent = PlanActAgent(llm=build_model(), workspace_root=str(WORKSPACE))
    result = agent.run(
        task=TASK,
        workspace=str(WORKSPACE),
        max_steps=MAX_STEPS,
        return_state=True,
    )
    print("workspace:", WORKSPACE)
    print("plan:", result.state.plan_steps)
    print("final_result:", result.state.final_result)
    print("stop_reason:", result.state.stop_reason)


if __name__ == "__main__":
    main()
