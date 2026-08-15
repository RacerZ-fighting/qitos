from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import pytest

from qitos import Action, AgentModule, Decision, Engine, StateSchema, ToolRegistry, tool
from qitos.core.journal import JournalRecordType
from qitos.core.model_response import ModelUsage
from qitos.engine import RuntimeBudget
from qitos.kit import ReActTextParser
from qitos.kit.journal import JsonlSessionJournal
from qitos.models import Model, ModelRequest, ModelStreamEvent, ModelStreamEventType


@dataclass
class _State(StateSchema):
    pass


class _TerminalModel(Model):
    def __init__(
        self,
        responses: list[ModelStreamEvent | Exception],
    ) -> None:
        super().__init__(model="terminal-test", temperature=None)
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        yield response


class _WaitingAgent(AgentModule[_State, dict[str, Any], Action]):
    def __init__(self, model: Model, *, model_driven: bool = False) -> None:
        registry = ToolRegistry()

        @tool(name="work_tool")
        def work_tool() -> str:
            raise AssertionError("terminal synthesis must not execute tools")

        registry.register(work_tool)
        super().__init__(
            tool_registry=registry,
            llm=model,
            model_parser=ReActTextParser(),
        )
        self._model_driven = model_driven

    def init_state(self, task: str, **kwargs: Any) -> _State:
        _ = kwargs
        return _State(task=task, max_steps=8)

    def decide(
        self,
        state: _State,
        observation: dict[str, Any],
    ) -> Decision[Action] | None:
        _ = state, observation
        if self._model_driven:
            return None
        return Decision.wait("continue working")

    def interpret_model_response(
        self,
        state: _State,
        observation: dict[str, Any],
        response: Any,
    ) -> Decision[Action] | None:
        _ = state, observation
        if response.text == "working":
            return Decision.wait("continue working")
        return None

    def reduce(
        self,
        state: _State,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> _State:
        _ = observation, decision
        return state


def _completed(
    text: str = "Final Answer: terminal report",
    *,
    total_tokens: int = 0,
    tool_calls: list[dict[str, Any]] | None = None,
) -> ModelStreamEvent:
    return ModelStreamEvent(
        type=ModelStreamEventType.COMPLETED,
        text=text,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=ModelUsage(
            input_tokens=total_tokens,
            output_tokens=0,
            total_tokens=total_tokens,
        ),
    )


def test_terminal_synthesis_reserves_the_last_step_and_exposes_no_tools() -> None:
    model = _TerminalModel([_completed()])
    result = Engine(
        _WaitingAgent(model),
        budget=RuntimeBudget(max_steps=3, terminal_synthesis=True),
    ).run("investigate")

    assert result.step_count == 3
    assert result.state.stop_reason == "budget_steps"
    assert result.state.final_result == "terminal report"
    assert len(model.requests) == 1
    request = model.requests[0]
    assert "tools" not in request.options
    assert request.continuation is None
    assert "work_tool" not in str(request.message_dicts())
    assert "<terminal_synthesis>" in str(request.message_dicts()[-1]["content"])
    terminal = result.records[-1]
    assert terminal.decision.mode == "final"
    assert terminal.decision.meta["fallback"] is False
    assert terminal.actions == []
    assert terminal.action_results == []


def test_one_step_budget_is_the_terminal_step() -> None:
    model = _TerminalModel([_completed()])
    engine = Engine(
        _WaitingAgent(model),
        budget=RuntimeBudget(max_steps=1, terminal_synthesis=True),
    )
    result = engine.run("investigate")

    assert result.step_count == 1
    assert [record.step_id for record in result.records] == [0]
    assert result.state.current_step == 0
    assert result.state.final_result == "terminal report"


def test_terminal_tool_call_uses_fallback_without_executing_it() -> None:
    model = _TerminalModel(
        [
            _completed(
                text="",
                tool_calls=[
                    {
                        "id": "terminal-call",
                        "type": "function",
                        "function": {"name": "work_tool", "arguments": "{}"},
                    }
                ],
            )
        ]
    )
    engine = Engine(
        _WaitingAgent(model),
        budget=RuntimeBudget(max_steps=1, terminal_synthesis=True),
    )
    result = engine.run("investigate")

    assert len(model.requests) == 1
    assert result.state.stop_reason == "budget_steps"
    assert "budget_steps" in str(result.state.final_result)
    assert result.records[0].decision.meta["fallback"] is True
    assert result.records[0].actions == []
    assert result.records[0].action_results == []
    assert not any(message.tool_calls for message in engine._history().messages)


def test_exhausted_token_budget_skips_terminal_model_request() -> None:
    model = _TerminalModel([_completed("working", total_tokens=12)])
    result = Engine(
        _WaitingAgent(model, model_driven=True),
        budget=RuntimeBudget(
            max_steps=3,
            max_tokens=5,
            terminal_synthesis=True,
        ),
    ).run("investigate")

    assert len(model.requests) == 1
    assert result.step_count == 2
    assert result.state.stop_reason == "budget_tokens"
    assert "budget_tokens" in str(result.state.final_result)
    assert result.records[-1].model_request is None
    assert result.records[-1].decision.meta["model_attempted"] is False


def test_expired_deadline_uses_fallback_without_calling_the_model() -> None:
    model = _TerminalModel([AssertionError("expired Run called the model")])
    result = Engine(
        _WaitingAgent(model),
        budget=RuntimeBudget(
            max_steps=3,
            deadline_monotonic=time.monotonic() - 1,
            terminal_synthesis=True,
        ),
    ).run("investigate")

    assert model.requests == []
    assert result.step_count == 1
    assert result.state.stop_reason == "budget_time"
    assert "budget_time" in str(result.state.final_result)
    assert result.records[0].decision.meta["model_attempted"] is False


def test_terminal_provider_failure_commits_deterministic_fallback() -> None:
    model = _TerminalModel([RuntimeError("provider unavailable")])
    result = Engine(
        _WaitingAgent(model),
        budget=RuntimeBudget(max_steps=1, terminal_synthesis=True),
    ).run("investigate")

    assert len(model.requests) == 1
    assert result.step_count == 1
    assert result.state.stop_reason == "budget_steps"
    assert "budget_steps" in str(result.state.final_result)
    fallback = result.records[0].model_response["terminal_fallback"]
    assert fallback["model_attempted"] is True
    assert fallback["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_terminal_step_is_durable_and_resume_does_not_replay_it(
    tmp_path: Path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    model = _TerminalModel([_completed()])
    completed = await Engine(
        _WaitingAgent(model),
        budget=RuntimeBudget(max_steps=1, terminal_synthesis=True),
        journal=journal,
    ).arun("investigate")

    records = await journal.replay()
    types = [record.type for record in records]
    assert JournalRecordType.MODEL_COMPLETED in types
    assert JournalRecordType.STEP_COMMITTED in types
    assert JournalRecordType.STATE_SNAPSHOT in types
    assert types[-1] is JournalRecordType.RUN_COMPLETED

    resume_model = _TerminalModel([AssertionError("terminal step was replayed")])
    resumed = await Engine(
        _WaitingAgent(resume_model),
        budget=RuntimeBudget(max_steps=1, terminal_synthesis=True),
        journal=JsonlSessionJournal(tmp_path),
    ).aresume_from_journal(completed.run_id)

    assert resume_model.requests == []
    assert resumed.state.stop_reason == "budget_steps"
    assert resumed.state.final_result == "terminal report"
