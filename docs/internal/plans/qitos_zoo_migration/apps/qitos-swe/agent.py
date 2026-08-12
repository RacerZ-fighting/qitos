"""qitos_swe — Formal SWE agent with DynamicTreeSearch and Phase 2 integration.

Upgraded from experimental/swe_agent.py with:
- DynamicTreeSearch for branch exploration
- Checkpoint support for long-running fixes
- Tracing integration for debugging
- Replan-on-failure with bounded replan count
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from qitos import (
    Action,
    AgentModule,
    Decision,
    EnvSpec,
    StateSchema,
    Task,
    TaskBudget,
    TaskResource,
    ToolRegistry,
)
from qitos.kit import (
    CodingToolSet,
    DynamicTreeSearch,
    RepoEnv,
    XmlDecisionParser,
    format_action,
    render_prompt,
)
from qitos.kit.planning import parse_numbered_plan

PLAN_PROMPT = """You are planning a software bug-fix workflow.

Task: {task}
Target file: {file}
Verification command: {test_command}

Return a numbered plan with 3-6 short steps.
The final step must run the verification command.
Output only numbered lines.
"""

EXEC_PROMPT = """You are a SWE execution agent.

Current plan step: {current_plan_step}

Rules:
- Exactly one tool call per step.
- Prefer the smallest correct edit.
- Run verification after code changes.
- Use final answer only when the issue is resolved.

Available tools:
{tool_schema}

Output must be valid XML with exactly one <decision> root.

Act mode:
<decision mode="act">
  <action name="tool_name">
    <arg name="key">value</arg>
  </action>
</decision>

Final mode:
<decision mode="final">
  <final_answer>summary + proof</final_answer>
</decision>

Wait mode:
<decision mode="wait"></decision>
"""


@dataclass
class SWEState(StateSchema):
    """State for the SWE agent."""

    plan_steps: List[str] = field(default_factory=list)
    cursor: int = 0
    scratchpad: List[str] = field(default_factory=list)
    target_file: str = ""
    test_command: str = ""
    replan_count: int = 0
    checkpoint_id: Optional[str] = None


class QitOSSWEAgent(AgentModule[SWEState, Dict[str, Any], Action]):
    """SWE agent with dynamic planning and branch search.

    Features:
    - LLM-generated numbered plans with cursor tracking
    - DynamicTreeSearch for exploring multiple fix candidates
    - Automatic replan on verification failure
    - Checkpoint support for long-running fixes
    - Tracing for debugging
    """

    def __init__(
        self,
        llm: Any,
        workspace_root: str = "./playground/qitos_swe",
        *,
        max_replans: int = 3,
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
        self._workspace = workspace_root
        self._max_replans = max_replans
        super().__init__(
            tool_registry=registry, llm=llm, model_parser=XmlDecisionParser()
        )

    def init_state(self, task: str, **kwargs: Any) -> SWEState:
        return SWEState(
            task=task,
            max_steps=int(kwargs.get("max_steps", 25)),
            target_file=str(kwargs.get("target_file", "")),
            test_command=str(kwargs.get("test_command", "")),
        )

    def build_system_prompt(self, state: SWEState) -> str | None:
        return render_prompt(
            EXEC_PROMPT,
            {
                "tool_schema": self.tool_registry.get_tool_descriptions(),
                "current_plan_step": self._current_step_text(state),
            },
        )

    def prepare(self, state: SWEState) -> str:
        lines = [
            f"Task: {state.task}",
            f"Target file: {state.target_file}",
            f"Verification: {state.test_command}",
            f"Plan cursor: {state.cursor}/{len(state.plan_steps)}",
            f"Current step: {self._current_step_text(state)}",
            f"Replan count: {state.replan_count}/{self._max_replans}",
        ]
        if state.plan_steps:
            lines.append("Full plan:")
            for idx, item in enumerate(state.plan_steps):
                marker = "->" if idx == state.cursor else "  "
                lines.append(f"{marker} [{idx}] {item}")
        if state.scratchpad:
            lines.append("Recent execution trace:")
            lines.extend(state.scratchpad[-10:])
        return "\n".join(lines)

    def decide(self, state: SWEState, observation: Dict[str, Any]):
        if not state.plan_steps or state.cursor >= len(state.plan_steps):
            if not self._make_or_refresh_plan(state):
                return Decision.final("Failed to generate a valid plan.")
            return Decision.wait("plan_ready")

        step_text = self._current_step_text(state).lower()
        candidates: List[Decision[Action]] = []

        llm_decision = self._llm_step_action(state)
        if llm_decision is not None and llm_decision.mode == "act":
            llm_decision.meta = dict(llm_decision.meta or {})
            llm_decision.meta.setdefault("score", 0.92)
            llm_decision.rationale = llm_decision.rationale or "llm_step_action"
            candidates.append(llm_decision)

        if any(token in step_text for token in ["inspect", "read", "check"]):
            candidates.append(
                Decision.act(
                    [Action(name="view", args={"path": state.target_file})],
                    rationale="inspect_target_file",
                    meta={"score": 0.8},
                )
            )
        if any(token in step_text for token in ["test", "verify", "validation"]):
            candidates.append(
                Decision.act(
                    [Action(name="run_command", args={"command": state.test_command})],
                    rationale="run_verification",
                    meta={"score": 0.95},
                )
            )

        if not candidates:
            return None
        return Decision.branch(
            candidates=candidates, rationale=f"dynamic_plan_step_{state.cursor}"
        )

    def reduce(
        self,
        state: SWEState,
        observation: Dict[str, Any],
        decision: Decision[Action],
    ) -> SWEState:
        action_results = (
            observation.get("action_results", [])
            if isinstance(observation, dict)
            else []
        )
        if decision.rationale:
            state.scratchpad.append(f"Thought: {decision.rationale}")
        if decision.actions:
            state.scratchpad.append(f"Action: {format_action(decision.actions[0])}")

        should_advance = False
        if action_results:
            first = action_results[0]
            state.scratchpad.append(f"Observation: {first}")
            if isinstance(first, dict):
                if first.get("status") == "success":
                    should_advance = True
                if "returncode" in first:
                    if int(first.get("returncode", 1)) == 0:
                        should_advance = True
                        state.final_result = "Verification passed. Patch looks correct."
                    else:
                        state.replan_count += 1
                        if state.replan_count <= self._max_replans:
                            state.cursor = len(state.plan_steps)
                        else:
                            state.final_result = "Max replan count exceeded."

        if should_advance and state.cursor < len(state.plan_steps):
            state.cursor += 1

        state.scratchpad = state.scratchpad[-50:]
        return state

    def _current_step_text(self, state: SWEState) -> str:
        if state.cursor < 0 or state.cursor >= len(state.plan_steps):
            return "none"
        return state.plan_steps[state.cursor]

    def _make_or_refresh_plan(self, state: SWEState) -> bool:
        raw = self.llm(
            [
                {"role": "system", "content": "Return a numbered plan only."},
                {
                    "role": "user",
                    "content": render_prompt(
                        PLAN_PROMPT,
                        {
                            "task": state.task,
                            "file": state.target_file,
                            "test_command": state.test_command,
                        },
                    ),
                },
            ]
        )
        plan = parse_numbered_plan(str(raw))
        if not plan:
            return False
        state.plan_steps = plan
        state.cursor = 0
        state.scratchpad.append(f"Plan: {plan}")
        return True

    def _llm_step_action(self, state: SWEState) -> Decision[Action] | None:
        step_text = self._current_step_text(state)
        if step_text == "none":
            return None
        raw = self.llm(
            [
                {"role": "system", "content": self.build_system_prompt(state) or ""},
                {
                    "role": "user",
                    "content": (
                        f"Task: {state.task}\n"
                        f"Current plan step: {step_text}\n"
                        f"Target file: {state.target_file}\n"
                        f"Verification command: {state.test_command}\n"
                        "Return exactly one XML <decision> object."
                    ),
                },
            ]
        )
        try:
            return self.model_parser.parse(str(raw))
        except Exception:
            return None


__all__ = ["QitOSSWEAgent", "SWEState"]
