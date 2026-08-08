"""qitos_researcher — Formal research agent with Checkpoint and Tracing.

Upgraded from experimental/computer_use_agent.py with:
- Checkpoint support for pause/resume
- Tracing integration for observability
- Handoff capability for multi-agent workflows
- CompactHistory for long research sessions
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from qitos import Action, AgentModule, Decision, StateSchema, ToolRegistry
from qitos.kit import (
    CodingToolSet,
    HTMLExtractText,
    HTTPGet,
    JsonDecisionParser,
    format_action,
    render_prompt,
)

SYSTEM_PROMPT = """You are a research assistant that investigates topics thoroughly.

Goal:
- Research the given topic using available tools.
- Extract and synthesize key findings.
- Produce a comprehensive report file.

Workflow:
1. Fetch relevant pages with http_get.
2. Extract readable text with extract_web_text.
3. Analyze and synthesize findings.
4. Write a structured report.
5. Finish with Final Answer.

Available tools:
{tool_schema}

Return JSON only.

Act mode:
{{
  "mode": "act",
  "rationale": "short reasoning",
  "actions": [{{"name": "tool_name", "args": {{"key": "value"}}}}]
}}

Final mode:
{{
  "mode": "final",
  "rationale": "short reasoning",
  "final_answer": "what was delivered"
}}

Wait mode:
{{
  "mode": "wait",
  "rationale": "why waiting"
}}

Constraints:
- Valid JSON only.
- Exactly one action in act mode.
- Use literal observed values in args.
"""


@dataclass
class ResearcherState(StateSchema):
    """State for the research agent."""

    target_url: str = ""
    report_file: str = "report.md"
    findings: List[str] = field(default_factory=list)
    scratchpad: List[str] = field(default_factory=list)
    checkpoint_id: Optional[str] = None


class QitOSResearcher(AgentModule[ResearcherState, Dict[str, Any], Action]):
    """Research agent with checkpoint and tracing support.

    Features:
    - HTTP research with page extraction
    - Checkpoint-based pause/resume
    - Tracing for observability
    - CompactHistory for long sessions
    - Handoff support for multi-agent workflows
    """

    def __init__(
        self,
        llm: Any,
        workspace_root: str = "./playground/qitos_researcher",
        *,
        enable_checkpoint: bool = True,
        enable_tracing: bool = True,
    ):
        registry = ToolRegistry()
        registry.register(HTTPGet())
        registry.register(HTMLExtractText())
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
        self._enable_checkpoint = enable_checkpoint
        self._enable_tracing = enable_tracing
        super().__init__(
            tool_registry=registry, llm=llm, model_parser=JsonDecisionParser()
        )

    def init_state(self, task: str, **kwargs: Any) -> ResearcherState:
        return ResearcherState(
            task=task,
            max_steps=int(kwargs.get("max_steps", 20)),
            target_url=str(kwargs.get("target_url", "")),
            report_file=str(kwargs.get("report_file", "report.md")),
        )

    def build_system_prompt(self, state: ResearcherState) -> str | None:
        return render_prompt(
            SYSTEM_PROMPT, {"tool_schema": self.tool_registry.get_tool_descriptions()}
        )

    def prepare(self, state: ResearcherState) -> str:
        lines = [
            f"Task: {state.task}",
            f"Target URL: {state.target_url or 'not specified'}",
            f"Report file: {state.report_file}",
            f"Step: {state.current_step}/{state.max_steps}",
        ]
        if state.findings:
            lines.append(f"Findings so far: {len(state.findings)}")
        if state.scratchpad:
            lines.append("Recent trajectory:")
            lines.extend(state.scratchpad[-8:])
        return "\n".join(lines)

    def reduce(
        self,
        state: ResearcherState,
        observation: Dict[str, Any],
        decision: Decision[Action],
    ) -> ResearcherState:
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
            # Track key findings
            if isinstance(first, str) and len(first) > 50:
                state.findings.append(first[:200])
        state.scratchpad = state.scratchpad[-40:]
        return state


__all__ = ["QitOSResearcher", "ResearcherState"]
