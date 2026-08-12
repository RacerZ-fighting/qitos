from dataclasses import dataclass
from typing import Any

from qitos import Action, AgentModule, Decision, Engine, StateSchema, ToolRegistry, tool
from qitos.core.tool_result import ToolResult
from qitos.engine.hooks import EngineHook, HookContext
from qitos.engine.states import RuntimeBudget


def test_tool_result_projects_summary_without_changing_canonical_output() -> None:
    raw = {"evidence": list(range(100))}
    summary = f"evidence items: {len(raw['evidence'])}"
    result = ToolResult(output={**raw, "model_summary": summary})

    visible = result.to_model_dict()

    assert visible["output"] == summary
    assert result.output == {**raw, "model_summary": summary}
    assert len(str(visible["output"])) < len(result.text)


@dataclass
class _ProjectionState(StateSchema):
    pass


class _ProjectionAgent(AgentModule[_ProjectionState, dict[str, Any], Action]):
    def __init__(self, raw_output: dict[str, Any], summary: str) -> None:
        registry = ToolRegistry()

        @tool(name="inspect")
        def inspect() -> dict[str, Any]:
            return {**raw_output, "model_summary": summary}

        registry.register(inspect)
        super().__init__(tool_registry=registry)

    def init_state(self, task: str, **kwargs: Any) -> _ProjectionState:
        _ = kwargs
        return _ProjectionState(task=task, max_steps=2)

    def decide(
        self,
        state: _ProjectionState,
        observation: dict[str, Any],
    ) -> Decision[Action]:
        _ = observation
        if state.current_step == 0:
            return Decision.act(actions=[Action(name="inspect", args={})])
        return Decision.final("done")

    def reduce(
        self,
        state: _ProjectionState,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> _ProjectionState:
        _ = observation, decision
        return state


class _AfterActCapture(EngineHook):
    def __init__(self) -> None:
        self.results: list[Any] = []

    def on_after_act(self, ctx: HookContext, engine: Any) -> None:
        _ = engine
        self.results = list(ctx.action_results or [])


def test_after_act_hook_receives_the_model_projection() -> None:
    raw_output = {"evidence": "x" * 1000}
    summary = f"captured {len(raw_output['evidence'])} characters"
    hook = _AfterActCapture()
    result = Engine(
        agent=_ProjectionAgent(raw_output, summary),
        budget=RuntimeBudget(max_steps=2),
        hooks=[hook],
    ).run("task")

    assert len(hook.results) == 1
    assert hook.results[0]["output"] == summary
    assert result.records[0].action_results[0].output == {
        **raw_output,
        "model_summary": summary,
    }
