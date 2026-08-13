from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import pytest

from qitos import AgentModule, Decision, Engine, StateSchema, ToolRegistry, tool
from qitos.core import CompletionAssessment, ModelPricing
from qitos.core.history import HistoryMessage
from qitos.core.model_response import ModelUsage
from qitos.engine import RuntimeBudget
from qitos.models import Model, ModelStreamChunk


@dataclass
class _State(StateSchema):
    pass


class _Agent(AgentModule[_State, dict[str, Any], Any]):
    def __init__(self, *, llm: Model | None = None) -> None:
        registry = ToolRegistry()

        @tool(name="original")
        def original() -> str:
            return "ok"

        registry.register(original)
        super().__init__(tool_registry=registry, llm=llm)

    def init_state(self, task: str, **kwargs: Any) -> _State:
        _ = kwargs
        return _State(task=task, max_steps=4)

    def reduce(
        self,
        state: _State,
        observation: dict[str, Any],
        decision: Decision[Any],
    ) -> _State:
        _ = observation, decision
        return state


def test_turn_snapshot_freezes_history_tools_model_and_budget() -> None:
    first_model = _UsageModel()
    pricing = ModelPricing(1.0, 2.0)
    agent = _Agent(llm=first_model)
    engine = Engine(
        agent,
        budget=RuntimeBudget(max_steps=4, max_tool_concurrency=2, max_children=3),
        model_pricing=pricing,
    )
    engine._active_run_id = "turn-freeze"
    engine._history_append("user", "before", 0)
    state = agent.init_state("freeze")

    first = engine._capture_turn(state, 0)

    @tool(name="later")
    def later() -> str:
        return "later"

    agent.tool_registry.register(later)
    engine._history_append("user", "after", 1)
    second_model = _UsageModel()
    agent.llm = second_model
    engine.budget.max_tool_concurrency = 1
    engine.model_pricing = ModelPricing(3.0, 4.0)
    second = engine._capture_turn(state, 1)

    assert first.model is first_model
    assert second.model is second_model
    assert first.tools.list_tools() == ["original"]
    assert second.tools.list_tools() == ["later", "original"]
    assert [item.content for item in first.history.messages] == ["before"]
    assert [item.content for item in second.history.messages] == ["before", "after"]
    assert first.budget.max_tool_concurrency == 2
    assert second.budget.max_tool_concurrency == 1
    assert first.budget.model_pricing is pricing
    assert second.budget.model_pricing == ModelPricing(3.0, 4.0)
    with pytest.raises(FrozenInstanceError):
        first.budget.max_children = 9  # type: ignore[misc]


class _CompletionAgent(_Agent):
    def decide(
        self,
        state: _State,
        observation: dict[str, Any],
    ) -> Decision[Any]:
        _ = observation
        return Decision.final(f"proposal-{state.current_step}")

    def assess_completion(
        self,
        state: _State,
        decision: Decision[Any],
    ) -> CompletionAssessment:
        _ = decision
        if state.current_step == 0:
            return CompletionAssessment.continue_run(
                "Collect the missing evidence.",
                reason="investigation_incomplete",
            )
        return CompletionAssessment.completed("evidence_complete")


def test_completion_policy_can_reject_a_final_then_complete() -> None:
    agent = _CompletionAgent()
    engine = Engine(agent, budget=RuntimeBudget(max_steps=3))

    result = engine.run("investigate")

    assert result.state.stop_reason == "completed"
    assert result.state.final_result == "proposal-1"
    assert result.step_count == 2
    feedback = [
        message
        for message in engine._history().messages
        if message.metadata.get("source") == "completion_assessment"
    ]
    assert [message.content for message in feedback] == [
        "Collect the missing evidence."
    ]


@pytest.mark.asyncio
async def test_single_step_uses_completion_policy_without_owning_advancement() -> None:
    agent = _CompletionAgent()
    engine = Engine(agent, budget=RuntimeBudget(max_steps=3))
    state, observation = engine.init_session("investigate")

    result = await engine.astep(state, observation)

    assert result.stop is False
    assert result.decision.mode == "final"
    assert state.current_step == 0
    feedback = [
        message
        for message in engine._history().messages
        if message.metadata.get("source") == "completion_assessment"
    ]
    assert [message.content for message in feedback] == [
        "Collect the missing evidence."
    ]


class _BlockedAgent(_CompletionAgent):
    def assess_completion(
        self,
        state: _State,
        decision: Decision[Any],
    ) -> CompletionAssessment:
        _ = state, decision
        return CompletionAssessment.blocked("authorization_required")


def test_completion_policy_distinguishes_blocked_from_completed() -> None:
    result = Engine(_BlockedAgent()).run("investigate")

    assert result.state.stop_reason == "blocked"
    assert result.state.final_result == "proposal-0"


class _UsageModel(Model):
    def __init__(self) -> None:
        super().__init__(model="usage-model", temperature=None)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        deadline_monotonic: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelStreamChunk]:
        _ = messages, deadline_monotonic, kwargs
        yield ModelStreamChunk(
            text="wait",
            done=True,
            usage=ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12),
        )


class _UsageAgent(_Agent):
    def interpret_model_response(self, state, observation, response):
        _ = state, observation, response
        return Decision.wait("continue")


def test_token_budget_stops_before_another_model_turn() -> None:
    model = _UsageModel()
    result = Engine(
        _UsageAgent(llm=model),
        budget=RuntimeBudget(max_steps=3, max_tokens=5),
    ).run("budget")

    assert result.state.stop_reason == "budget_tokens"
    assert result.total_tokens == 12


def test_explicit_pricing_enforces_cost_budget_without_model_name_guessing() -> None:
    pricing = ModelPricing(
        input_usd_per_million=1_000_000,
        output_usd_per_million=1_000_000,
    )
    result = Engine(
        _UsageAgent(llm=_UsageModel()),
        budget=RuntimeBudget(max_steps=3, max_cost_usd=1.0),
        model_pricing=pricing,
    ).run("budget")

    assert result.state.stop_reason == "budget_cost"
    assert result.total_cost_usd == pytest.approx(12.0)


def test_cost_budget_requires_explicit_pricing() -> None:
    with pytest.raises(ValueError, match="explicit model_pricing"):
        Engine(
            _UsageAgent(llm=_UsageModel()),
            budget=RuntimeBudget(max_steps=3, max_cost_usd=1.0),
        )
