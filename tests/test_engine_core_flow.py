from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from examples._support import SequenceModel
from qitos import (
    Action,
    AgentModule,
    Decision,
    Engine,
    HistoryPolicy,
    ModelResponse,
    StateSchema,
    ToolRegistry,
    tool,
)
from qitos.core.history import History, HistoryMessage
from qitos.core.model_response import ModelUsage, ModelUsageSource
from qitos.kit.memory import WindowMemory
from qitos.kit.env import ScreenshotEnv
from qitos.kit.history import CompactHistory, WindowHistory
from qitos.kit.parser import ReActTextParser
from qitos.core.memory import Memory, MemoryRecord
from qitos.engine import RuntimeBudget
from qitos.engine.states import ContextConfig
from qitos.models import Model, ModelRequest, ModelStreamChunk
from qitos.trace import runtime_step_to_trace


@dataclass
class DemoState(StateSchema):
    logs: list[str] = field(default_factory=list)


class _ChunkSequenceModel(Model):
    """Deterministic model stream used by Engine behavior tests."""

    def __init__(
        self,
        transactions: list[list[ModelStreamChunk]],
        *,
        model: str = "test-model",
        provider: str = "test",
        context_window: int = 128_000,
        max_tokens: int = 2_048,
    ) -> None:
        super().__init__(
            model=model,
            context_window=context_window,
            max_tokens=max_tokens,
            temperature=None,
        )
        self.provider_name = provider
        self.transactions = [list(chunks) for chunks in transactions]
        self.calls: list[list[dict[str, Any]]] = []
        self.request_options: list[dict[str, Any]] = []

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamChunk]:
        messages = request.message_dicts()
        self.calls.append([dict(message) for message in messages])
        self.request_options.append(request.option_dict())
        if not self.transactions:
            raise AssertionError("no scripted model transaction remains")
        for chunk in self.transactions.pop(0):
            yield chunk


class DemoAgent(AgentModule[DemoState, dict[str, Any], Action]):
    def __init__(self):
        registry = ToolRegistry()

        @tool(name="add")
        def add(a: int, b: int) -> int:
            return a + b

        registry.register(add)
        super().__init__(tool_registry=registry)

    def init_state(self, task: str, **kwargs: Any) -> DemoState:
        return DemoState(task=task, max_steps=3)

    def decide(self, state: DemoState, observation: dict[str, Any]) -> Decision[Action]:
        if state.current_step == 0:
            return Decision.act(
                actions=[Action(name="add", args={"a": 19, "b": 23})],
                rationale="use tool",
            )
        return Decision.final("42")

    def reduce(
        self,
        state: DemoState,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> DemoState:
        action_results = (
            observation.get("action_results", [])
            if isinstance(observation, dict)
            else []
        )
        if action_results:
            state.logs.append(str(action_results[0]))
        return state


def test_engine_happy_path():
    result = Engine(agent=DemoAgent(), budget=RuntimeBudget(max_steps=3)).run("compute")
    assert result.state.final_result == "42"
    assert result.state.stop_reason == "completed"
    assert len(result.records[0].action_results) == 1
    first_result = result.records[0].action_results[0]
    assert first_result.status == "success"
    assert first_result.output == 42
    assert result.step_summaries
    assert result.step_summaries[0].tool_name == "add"
    assert result.to_dict()["tool_calls_by_name"]["add"] == 1


def test_engine_records_local_model_stream_timing(monkeypatch):
    model = _ChunkSequenceModel(
        [
            [
                ModelStreamChunk(event_type="response.created"),
                ModelStreamChunk(text="Final Answer: done", event_type="text.delta"),
                ModelStreamChunk(done=True, finish_reason="stop"),
            ]
        ]
    )

    class _Agent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model

        def decide(self, state: DemoState, observation: dict[str, Any]):
            _ = state
            _ = observation
            return None

    class _MonotonicClock:
        def __init__(self) -> None:
            self.current = 10.0

        def __call__(self) -> float:
            value = self.current
            self.current += 0.01
            return value

    monkeypatch.setattr("qitos.engine._model_runtime.time.monotonic", _MonotonicClock())

    result = Engine(agent=_Agent(), budget=RuntimeBudget(max_steps=1)).run("finish")

    timing = result.records[0].model_response["timing"]
    assert 0 <= timing["time_to_first_event_ms"]
    assert timing["time_to_first_event_ms"] < timing["time_to_first_content_ms"]
    assert timing["time_to_first_content_ms"] <= timing["total_ms"]
    assert runtime_step_to_trace(result.records[0]).model_response["timing"] == timing


def test_tool_loop_detection_can_be_disabled_for_long_running_agents():
    class RepeatingAgent(AgentModule[DemoState, dict[str, Any], Action]):
        def __init__(self) -> None:
            registry = ToolRegistry()
            self.calls = 0

            @tool(name="GLOB")
            def glob() -> dict[str, str]:
                self.calls += 1
                return {
                    "status": "success",
                    "domain_outcome": "no_match",
                    "model_summary": "[GLOB:no_match]\n\nEnumeration complete: yes",
                }

            registry.register(glob)
            super().__init__(tool_registry=registry)

        def init_state(self, task: str, **kwargs: Any) -> DemoState:
            _ = kwargs
            return DemoState(task=task, max_steps=6)

        def decide(
            self, state: DemoState, observation: dict[str, Any]
        ) -> Decision[Action]:
            _ = observation
            if state.current_step < 5:
                return Decision.act(actions=[Action(name="GLOB", args={})])
            return Decision.final("done")

        def reduce(
            self,
            state: DemoState,
            observation: dict[str, Any],
            decision: Decision[Action],
        ) -> DemoState:
            _ = observation, decision
            return state

    agent = RepeatingAgent()
    result = Engine(
        agent=agent,
        budget=RuntimeBudget(max_steps=6),
        context_config=ContextConfig(tool_call_loop_detection_enabled=False),
    ).run("repeat")

    assert result.state.final_result == "done"
    assert agent.calls == 5
    assert all(
        record.action_results[0].status == "success" for record in result.records[:5]
    )
    assert not any(
        event.payload.get("stage") == "tool_call_loop_detected"
        for event in result.events
    )


def test_tool_loop_detection_remains_enabled_by_default():
    class RepeatingAgent(AgentModule[DemoState, dict[str, Any], Action]):
        def __init__(self) -> None:
            registry = ToolRegistry()
            self.calls = 0

            @tool(name="GLOB")
            def glob() -> dict[str, str]:
                self.calls += 1
                return {"model_summary": "[GLOB:no_match]"}

            registry.register(glob)
            super().__init__(tool_registry=registry)

        def init_state(self, task: str, **kwargs: Any) -> DemoState:
            _ = kwargs
            return DemoState(task=task, max_steps=5)

        def decide(
            self, state: DemoState, observation: dict[str, Any]
        ) -> Decision[Action]:
            _ = observation
            if state.current_step < 4:
                return Decision.act(actions=[Action(name="GLOB", args={})])
            return Decision.final("done")

        def reduce(
            self,
            state: DemoState,
            observation: dict[str, Any],
            decision: Decision[Action],
        ) -> DemoState:
            _ = observation, decision
            return state

    agent = RepeatingAgent()
    result = Engine(agent=agent, budget=RuntimeBudget(max_steps=5)).run("repeat")

    assert agent.calls == 3
    assert result.records[3].action_results[0].error == "tool_call_loop_detected"


def test_agent_run_shortcut():
    agent = DemoAgent()
    assert agent.run("compute", trace=False, render=False) == "42"


def test_agent_condition_stop_is_not_automatic_success():
    class StopAgent(DemoAgent):
        def init_state(self, task: str, **kwargs: Any) -> DemoState:
            _ = kwargs
            return DemoState(task=task, max_steps=3)

        def decide(
            self, state: DemoState, observation: dict[str, Any]
        ) -> Decision[Action]:
            _ = observation
            return Decision.act(
                actions=[Action(name="add", args={"a": 1, "b": 1})],
                rationale="take one action then stop",
            )

        def reduce(
            self,
            state: DemoState,
            observation: dict[str, Any],
            decision: Decision[Action],
        ) -> DemoState:
            _ = observation, decision
            return state

        def should_stop(self, state: DemoState) -> bool:
            _ = state
            return True

    result = Engine(agent=StopAgent(), budget=RuntimeBudget(max_steps=3)).run("compute")
    assert result.state.stop_reason == "agent_condition"
    assert result.state.final_result is None
    assert result.task_result is not None
    assert result.task_result.success is False
    assert all(item.passed is False for item in result.task_result.criteria)


def test_agent_run_enables_trace_and_render_by_default(tmp_path):
    workspace = tmp_path / "workspace"
    logdir = tmp_path / "runs"
    workspace.mkdir(parents=True, exist_ok=True)

    agent = DemoAgent()
    result = agent.run(
        "compute",
        workspace=str(workspace),
        trace_logdir=str(logdir),
        return_state=True,
    )

    assert result.state.final_result == "42"
    assert (workspace / "render_events.jsonl").exists()
    run_dirs = [p for p in logdir.iterdir() if p.is_dir()]
    assert run_dirs


def test_agent_run_can_disable_default_trace_and_render(tmp_path):
    workspace = tmp_path / "workspace"
    logdir = tmp_path / "runs"
    workspace.mkdir(parents=True, exist_ok=True)

    agent = DemoAgent()
    result = agent.run(
        "compute",
        workspace=str(workspace),
        trace_logdir=str(logdir),
        trace=False,
        render=False,
        return_state=True,
    )

    assert result.state.final_result == "42"
    assert not (workspace / "render_events.jsonl").exists()
    assert not logdir.exists() or not any(logdir.iterdir())


def test_engine_injects_memory_context_into_env_view():
    agent = DemoAgent()
    agent.memory = WindowMemory(window_size=20)
    result = Engine(agent=agent, budget=RuntimeBudget(max_steps=3)).run("compute")
    assert result.state.final_result == "42"
    assert hasattr(agent, "memory")
    assert agent.memory is not None


def test_engine_default_model_decide_with_prepare():
    model = SequenceModel(["Action: add(a=20, b=22)"])

    class LLMDrivenDemo(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def build_system_prompt(self, state: DemoState) -> str | None:
            return "System prompt"

        def prepare(self, state: DemoState) -> str:
            return f"Task={state.task} Step={state.current_step}"

        def decide(self, state: DemoState, observation: dict[str, Any]):
            if state.current_step == 0:
                return None
            return Decision.final("42")

    result = Engine(agent=LLMDrivenDemo(), budget=RuntimeBudget(max_steps=3)).run(
        "compute"
    )
    assert result.state.final_result == "42"
    seen_messages = model.calls[0]
    assert len(seen_messages) == 2
    assert seen_messages[0]["role"] == "system"
    assert seen_messages[1]["role"] == "user"


def test_engine_includes_current_step_visual_input_in_user_message(tmp_path):
    png_path = tmp_path / "screen.png"
    png_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x04\x00\x00\x00\xb5\x1c\x0c\x02\x00\x00\x00\x0bIDATx\xdac\xfc\xff\x1f\x00\x02\xeb\x01\xf5i\xf6\x81\xb7\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    model = SequenceModel(
        ["Final Answer: visual complete"],
        model="gpt-4.1-mini",
    )

    class VisualDemo(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def build_system_prompt(self, state: DemoState) -> str | None:
            return "Inspect the screenshot and answer."

        def prepare(self, state: DemoState) -> str:
            return "What is visible in the current screenshot?"

        def decide(self, state: DemoState, observation: dict[str, Any]):
            _ = observation
            if state.current_step == 0:
                return None
            return Decision.final("done")

    env = ScreenshotEnv(
        screenshot_path=str(png_path),
        text="The screenshot shows a login page.",
    )
    result = Engine(agent=VisualDemo(), env=env, budget=RuntimeBudget(max_steps=2)).run(
        "inspect"
    )
    assert result.state.final_result == "visual complete"
    user_message = model.calls[0][-1]
    assert user_message["role"] == "user"
    assert isinstance(user_message["content"], list)
    assert user_message["content"][0]["type"] == "text"
    assert user_message["content"][1]["type"] == "image_file"
    record = result.records[0]
    assert record.model_input_visual_count == 1
    assert record.has_screenshot is True
    assert record.observation_modalities == ["text", "screenshot"]


def test_engine_uses_history_messages_for_next_llm_call():
    model = SequenceModel(["Action: add(a=1, b=1)", "Action: add(a=1, b=1)"])

    class MultiTurnLLMDemo(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def build_system_prompt(self, state: DemoState) -> str | None:
            return "System prompt"

        def prepare(self, state: DemoState) -> str:
            return f"Task={state.task} Step={state.current_step}"

        def decide(self, state: DemoState, observation: dict[str, Any]):
            if state.current_step < 2:
                return None
            return Decision.final("42")

    agent = MultiTurnLLMDemo()
    agent.history = WindowHistory(window_size=50)
    result = Engine(
        agent=agent,
        budget=RuntimeBudget(max_steps=4),
        history_policy=HistoryPolicy(max_messages=4),
    ).run("compute")
    assert result.state.final_result == "42"
    calls = model.calls
    assert len(calls) == 2
    assert calls[0][0]["role"] == "system"
    assert calls[0][-1]["role"] == "user"
    # second call should include history (previous user+assistant)
    assert len(calls[1]) >= 4
    assert calls[1][1]["role"] == "user"
    assert calls[1][2]["role"] == "assistant"


def test_engine_can_start_from_a_history_snapshot() -> None:
    model = SequenceModel(["Final Answer: continued"])

    class _Agent(DemoAgent):
        def __init__(self) -> None:
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()
            self.history = CompactHistory(auto_compact=False)

        def decide(self, state: DemoState, observation: dict[str, Any]):
            _ = state, observation
            return None

    parent = CompactHistory(auto_compact=False)
    inherited_user = HistoryMessage(
        role="user", content={"source": "parent", "sequence": 1}, step_id=0
    )
    inherited_assistant = HistoryMessage(
        role="assistant", content={"source": "parent", "sequence": 2}, step_id=0
    )
    parent.append(inherited_user)
    parent.append(inherited_assistant)

    result = Engine(agent=_Agent(), budget=RuntimeBudget(max_steps=1)).run(
        "continue", history_snapshot=parent.snapshot()
    )

    assert result.state.final_result == "continued"
    assert any(
        message.get("content") == inherited_assistant.content
        for message in model.calls[0]
    )
    assert parent.messages == [inherited_user, inherited_assistant]


def test_engine_emits_parser_events_and_records_step_diagnostics():
    model = SequenceModel(
        [
            "Thought only without action",
            "Action: add(a=20, b=22)",
            "Final Answer: 42",
        ]
    )

    class ParserDiagDemo(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def build_system_prompt(self, state: DemoState) -> str | None:
            return "System prompt"

        def prepare(self, state: DemoState) -> str:
            return f"Task={state.task} Step={state.current_step}"

        def decide(self, state: DemoState, observation: dict[str, Any]):
            if state.current_step < 3:
                return None
            return Decision.final("42")

    result = Engine(agent=ParserDiagDemo(), budget=RuntimeBudget(max_steps=5)).run(
        "compute"
    )
    assert result.state.final_result == "42"
    parser_result_events = [
        e
        for e in result.events
        if getattr(e.phase, "value", e.phase) == "DECIDE"
        and (e.payload or {}).get("stage") == "parser_result"
    ]
    parser_diag_events = [
        e
        for e in result.events
        if getattr(e.phase, "value", e.phase) == "DECIDE"
        and (e.payload or {}).get("stage") == "parser_diagnostics"
    ]
    assert parser_result_events
    assert parser_diag_events
    assert result.records[0].parser_diagnostics["code"] == "missing_action_or_final"
    assert result.records[0].parser_contract == "react_text_v1"
    assert result.records[0].parser_salvage_applied is False


def test_engine_interpret_model_response_bypasses_parser_and_records_summary():
    seen: list[ModelResponse] = []
    model = _ChunkSequenceModel(
        [
            [
                ModelStreamChunk(
                    text="model said to use the add tool",
                    done=True,
                    usage={
                        "prompt_tokens": 12,
                        "completion_tokens": 5,
                        "total_tokens": 17,
                    },
                    finish_reason="stop",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 20, "b": 22}',
                            },
                        }
                    ],
                )
            ]
        ],
        model="demo-model",
        provider="demo-provider",
    )

    class _NeverParser:
        def parse(self, raw_output, context=None):
            _ = raw_output
            _ = context
            raise AssertionError(
                "parser should not be called when interpret_model_response returns Decision"
            )

    class _InterpretAgent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = _NeverParser()

        def decide(self, state: DemoState, observation: dict[str, Any]):
            _ = observation
            if state.current_step > 0:
                return Decision.final("42")
            return None

        def interpret_model_response(
            self,
            state: DemoState,
            observation: dict[str, Any],
            response: ModelResponse,
        ) -> Decision[Action] | None:
            _ = state
            _ = observation
            seen.append(response)
            return Decision.act(
                actions=[Action(name="add", args={"a": 20, "b": 22})],
                rationale=response.text,
            )

    result = Engine(agent=_InterpretAgent(), budget=RuntimeBudget(max_steps=3)).run(
        "compute"
    )
    assert result.state.final_result == "42"
    assert seen
    response = seen[0]
    assert response.text == "model said to use the add tool"
    assert response.usage == {
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    }
    assert response.finish_reason == "stop"
    assert response.model_name == "demo-model"
    assert response.provider == "demo-provider"
    assert result.records[0].model_response["text"] == "model said to use the add tool"
    assert "raw" not in result.records[0].model_response
    model_output_events = [
        e
        for e in result.events
        if getattr(e.phase, "value", e.phase) == "DECIDE"
        and (e.payload or {}).get("stage") == "model_output"
    ]
    assert model_output_events
    assert (
        model_output_events[0].payload["raw_output"] == "model said to use the add tool"
    )
    assert model_output_events[0].payload["model_response"]["finish_reason"] == "stop"
    traced = runtime_step_to_trace(result.records[0]).to_dict()
    assert traced["model_response"]["model_name"] == "demo-model"
    assert "raw" not in traced["model_response"]


def test_engine_interpret_model_response_can_fall_back_to_parser():
    seen: list[ModelResponse] = []
    model = _ChunkSequenceModel(
        [
            [
                ModelStreamChunk(
                    text="Final Answer: 42",
                    done=True,
                    usage={
                        "prompt_tokens": 9,
                        "completion_tokens": 3,
                        "total_tokens": 12,
                    },
                    finish_reason="stop",
                )
            ]
        ],
        model="demo-model",
    )

    class _TrackingParser(ReActTextParser):
        def __init__(self):
            super().__init__()
            self.calls: list[Any] = []

        def parse(self, raw_output: Any, context=None):
            self.calls.append(raw_output)
            return super().parse(raw_output, context=context)

    parser = _TrackingParser()

    class _InterpretAgent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = parser

        def decide(self, state: DemoState, observation: dict[str, Any]):
            _ = state
            _ = observation
            return None

        def interpret_model_response(
            self,
            state: DemoState,
            observation: dict[str, Any],
            response: ModelResponse,
        ) -> Decision[Action] | None:
            _ = state
            _ = observation
            seen.append(response)
            return None

    result = Engine(agent=_InterpretAgent(), budget=RuntimeBudget(max_steps=2)).run(
        "compute"
    )
    assert result.state.final_result == "42"
    assert seen and isinstance(seen[0], ModelResponse)
    assert parser.calls == ["Final Answer: 42"]
    assert result.records[0].model_response["usage"]["total_tokens"] == 12


def test_engine_uses_history_retrieve_contract():
    class ContractHistory(History):
        def __init__(self):
            self._messages: list[HistoryMessage] = []
            self.retrieve_called = 0

        def append(self, message: HistoryMessage) -> None:
            self._messages.append(message)

        def retrieve(self, query=None, state=None, observation=None):
            self.retrieve_called += 1
            return [{"role": "assistant", "content": "history_hint"}]

        def summarize(self, max_items: int = 5) -> str:
            return ""

        def evict(self) -> int:
            return 0

        def reset(self, run_id=None) -> None:
            self._messages = []

        @property
        def messages(self) -> list[HistoryMessage]:
            return list(self._messages)

    class ContractMemory(Memory):
        def __init__(self):
            self._records: list[MemoryRecord] = []
            self.retrieve_called = 0

        def append(self, record: MemoryRecord) -> None:
            self._records.append(record)

        def retrieve(self, query=None, state=None, observation=None):
            self.retrieve_called += 1
            return []

        def summarize(self, max_items: int = 5) -> str:
            return ""

        def evict(self) -> int:
            return 0

        def reset(self, run_id=None) -> None:
            self._records = []

    model = SequenceModel(["Final Answer: 42"])

    class LLMOnceAgent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def build_system_prompt(self, state: DemoState) -> str | None:
            return "System prompt"

        def prepare(self, state: DemoState) -> str:
            return "solve"

        def decide(self, state: DemoState, observation: dict[str, Any]):
            return None

    mem = ContractMemory()
    hist = ContractHistory()
    agent = LLMOnceAgent()
    agent.memory = mem
    agent.history = hist
    result = Engine(agent=agent, budget=RuntimeBudget(max_steps=2)).run("compute")
    assert result.state.final_result == "42"
    assert hist.retrieve_called >= 1
    assert mem.retrieve_called == 0
    assert any(m.get("content") == "history_hint" for m in model.calls[0])


def test_memory_and_history_streams_are_strictly_separated():
    class CaptureMemory(Memory):
        def __init__(self):
            self.records: list[MemoryRecord] = []

        def append(self, record: MemoryRecord) -> None:
            self.records.append(record)

        def retrieve(self, query=None, state=None, observation=None):
            return list(self.records)

        def summarize(self, max_items: int = 5) -> str:
            return ""

        def evict(self) -> int:
            return 0

        def reset(self, run_id=None) -> None:
            self.records = []

    class CaptureHistory(History):
        def __init__(self):
            self.messages: list[HistoryMessage] = []

        def append(self, message: HistoryMessage) -> None:
            self.messages.append(message)

        def retrieve(self, query=None, state=None, observation=None):
            return list(self.messages)

        def summarize(self, max_items: int = 5) -> str:
            return ""

        def evict(self) -> int:
            return 0

        def reset(self, run_id=None) -> None:
            self.messages = []

    model = SequenceModel(["Final Answer: ok"])

    class OneShotLLMAgent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def build_system_prompt(self, state: DemoState) -> str | None:
            return "System prompt"

        def prepare(self, state: DemoState) -> str:
            return "solve"

        def decide(self, state: DemoState, observation: dict[str, Any]):
            return None

    mem = CaptureMemory()
    hist = CaptureHistory()
    agent = OneShotLLMAgent()
    agent.memory = mem
    agent.history = hist
    result = Engine(agent=agent, budget=RuntimeBudget(max_steps=2)).run("compute")
    assert result.state.stop_reason == "completed"

    mem_roles = {r.role for r in mem.records}
    assert {"task", "state", "decision", "next_state", "observation"}.issubset(
        mem_roles
    )
    assert "message" not in mem_roles

    hist_roles = [m.role for m in hist.messages]
    assert "user" in hist_roles
    assert "assistant" in hist_roles


def test_engine_records_context_telemetry_and_defaults_to_compact_runtime_history():
    model = _ChunkSequenceModel(
        [[ModelStreamChunk(text="Final Answer: ok", done=True)]],
        model="dummy-context",
        max_tokens=128,
        context_window=4096,
    )

    class _Agent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def build_system_prompt(self, state: DemoState) -> str | None:
            return "System prompt"

        def prepare(self, state: DemoState) -> str:
            return f"Task={state.task}\n" + ("verbose context " * 20)

        def decide(self, state: DemoState, observation: dict[str, Any]):
            return None

    engine = Engine(agent=_Agent(), budget=RuntimeBudget(max_steps=2))
    result = engine.run("demo")
    assert result.state.final_result == "ok"
    assert engine._runtime_history.__class__.__name__ == "CompactHistory"
    assert result.records
    assert result.records[0].context.get("input_tokens_total", 0) > 0
    assert result.records[0].context.get("context_window") == 4096
    assert result.records[0].context["usage_source"] == "absent"


def test_engine_prefers_provider_usage_for_context_totals():
    model = _ChunkSequenceModel(
        [
            [
                ModelStreamChunk(
                    text="Final Answer: exact",
                    done=True,
                    usage={
                        "prompt_tokens": 123,
                        "completion_tokens": 17,
                        "total_tokens": 140,
                        "cache_creation_input_tokens": 11,
                        "cache_read_input_tokens": 19,
                        "output_tokens_details": {"reasoning_tokens": 7},
                    },
                )
            ]
        ],
        model="dummy-usage",
        max_tokens=128,
        context_window=8192,
    )

    class _Agent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def build_system_prompt(self, state: DemoState) -> str | None:
            return "System prompt"

        def prepare(self, state: DemoState) -> str:
            return "Hello"

        def decide(self, state: DemoState, observation: dict[str, Any]):
            return None

    result = Engine(agent=_Agent(), budget=RuntimeBudget(max_steps=2)).run("demo")
    ctx = result.records[0].context
    assert result.state.final_result == "exact"
    # Provider usage is authoritative after completion; the pre-request
    # estimate remains separately auditable when a meter provided one.
    assert ctx["counting_mode"] == "provider_usage"
    assert ctx["input_tokens_total"] == 123
    assert ctx["provider_prompt_tokens"] == 123
    assert ctx["provider_completion_tokens"] == 17
    assert ctx["provider_total_tokens"] == 140
    assert ctx["output_tokens"] == 17
    assert ctx["tokens_total"] == 140
    assert ctx["cached_tokens"] == 19
    assert ctx["cache_write_tokens"] == 11
    assert ctx["reasoning_tokens"] == 7
    assert ctx["usage_source"] == ModelUsageSource.PROVIDER.value


def test_engine_preserves_zero_provider_input_usage() -> None:
    model = _ChunkSequenceModel(
        [
            [
                ModelStreamChunk(
                    text="Final Answer: exact",
                    done=True,
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 3,
                        "total_tokens": 3,
                    },
                )
            ]
        ],
        model="dummy-zero-usage",
        max_tokens=128,
        context_window=8192,
    )

    class _Agent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def decide(self, state: DemoState, observation: dict[str, Any]):
            return None

    result = Engine(agent=_Agent(), budget=RuntimeBudget(max_steps=2)).run("demo")
    context = result.records[0].context

    assert context["provider_prompt_tokens"] == 0
    assert context["input_tokens_total"] == 0
    assert context["prompt_tokens_total"] == 0
    assert context["tokens_total"] == context["provider_total_tokens"]


def test_engine_keeps_estimated_usage_distinct_from_provider_usage() -> None:
    input_tokens = 20 + uuid4().int % 10
    output_tokens = 5 + uuid4().int % 5
    cache_read_tokens = uuid4().int % input_tokens
    cache_write_tokens = uuid4().int % input_tokens
    reasoning_tokens = uuid4().int % output_tokens
    total_tokens = input_tokens + output_tokens
    usage = ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=reasoning_tokens,
        source=ModelUsageSource.ESTIMATE,
    )
    model = _ChunkSequenceModel(
        [[ModelStreamChunk(text="Final Answer: estimate", done=True, usage=usage)]],
        model="dummy-estimated-usage",
        max_tokens=128,
        context_window=8192,
    )

    class _Agent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def decide(self, state: DemoState, observation: dict[str, Any]):
            return None

    result = Engine(agent=_Agent(), budget=RuntimeBudget(max_steps=2)).run("demo")
    context = result.records[0].context

    assert context["usage_source"] == ModelUsageSource.ESTIMATE.value
    assert context["provider_prompt_tokens"] is None
    assert context["provider_completion_tokens"] is None
    assert context["provider_total_tokens"] is None
    assert context["input_tokens_total"] == input_tokens
    assert context["output_tokens"] == output_tokens
    assert context["tokens_total"] == total_tokens
    assert context["cached_tokens"] == cache_read_tokens
    assert context["cache_write_tokens"] == cache_write_tokens
    assert context["reasoning_tokens"] == reasoning_tokens
    assert context["counting_mode"] == "usage_estimate"


def test_engine_accumulates_each_provider_transaction_once() -> None:
    first_input = 10 + uuid4().int % 10
    first_output = 2 + uuid4().int % 4
    second_input = first_input + 10 + uuid4().int % 10
    second_output = 2 + uuid4().int % 4
    first_total = first_input + first_output
    second_total = second_input + second_output
    model = _ChunkSequenceModel(
        [
            [
                ModelStreamChunk(
                    done=True,
                    tool_calls=[
                        {
                            "id": "call_usage",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 20, "b": 22}',
                            },
                        }
                    ],
                    usage=ModelUsage(
                        input_tokens=first_input,
                        output_tokens=first_output,
                        total_tokens=first_total,
                    ),
                    finish_reason="tool_calls",
                )
            ],
            [
                ModelStreamChunk(
                    text="Final Answer: 42",
                    done=True,
                    usage=ModelUsage(
                        input_tokens=second_input,
                        output_tokens=second_output,
                        total_tokens=second_total,
                    ),
                    finish_reason="stop",
                )
            ],
        ],
        model="dummy-multi-turn-usage",
    )

    class _Agent(DemoAgent):
        def __init__(self) -> None:
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def decide(self, state: DemoState, observation: dict[str, Any]):
            _ = state, observation
            return None

    result = Engine(agent=_Agent(), budget=RuntimeBudget(max_steps=3)).run("compute")

    assert result.state.final_result == "42"
    assert len(result.records) == 2
    first_context = result.records[0].context
    second_context = result.records[1].context
    assert first_context["prompt_tokens_total"] == first_input
    assert first_context["completion_tokens_total"] == first_output
    assert first_context["tokens_total"] == first_total
    assert second_context["prompt_tokens_total"] == first_input + second_input
    assert second_context["completion_tokens_total"] == first_output + second_output
    assert second_context["tokens_total"] == first_total + second_total


def test_engine_uses_model_stream_native_tool_calls_before_parser():
    model = _ChunkSequenceModel(
        [
            [
                ModelStreamChunk(
                    text="",
                    event_type="tool_call.delta",
                    event_metadata={
                        "index": 0,
                        "call_id": "call_1",
                        "arguments_delta": '{"a": 20',
                    },
                ),
                ModelStreamChunk(
                    done=True,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 20, "b": 22}',
                            },
                        }
                    ],
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "total_tokens": 14,
                    },
                    finish_reason="tool_calls",
                ),
            ]
        ],
        model="qwen-plus",
        provider="openai-compatible",
    )
    model.qitos_harness_metadata = {
        "family_preset": "qwen",
        "tool_policy": {
            "primary_delivery": "api_parameter",
            "fallback_delivery": "prompt_injection",
            "native_tool_call_preferred": True,
        },
    }

    class _NeverParser:
        def parse(self, raw_output, context=None):
            _ = raw_output
            _ = context
            raise AssertionError(
                "parser should be bypassed when native tool calls are used"
            )

    class _Agent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = _NeverParser()

        def decide(self, state: DemoState, observation: dict[str, Any]):
            _ = observation
            if state.current_step > 0:
                return Decision.final("42")
            return None

    class _ChunkHandler:
        def __init__(self) -> None:
            self.events: list[object] = []

        def on_start(self) -> None:
            self.events.append("start")

        def on_delta(self, text: str) -> None:
            self.events.append(text)

        def on_chunk(self, chunk: ModelStreamChunk) -> None:
            self.events.append((chunk.event_type, dict(chunk.event_metadata)))

        def on_end(self) -> None:
            self.events.append("end")

    handler = _ChunkHandler()
    engine = Engine(agent=_Agent(), budget=RuntimeBudget(max_steps=3))
    engine.stream_callback = handler
    result = engine.run("compute")
    assert result.state.final_result == "42"
    record = result.records[0]
    assert record.decision_source == "native_tool_calls"
    assert record.native_tool_call_used is True
    assert record.native_tool_call_fallback_reason is None
    assert record.actions[0].name == "add"
    assert record.actions[0].args == {"a": 20, "b": 22}
    assert record.model_response["tool_calls"][0]["function"]["name"] == "add"
    assert record.model_response["finish_reason"] == "tool_calls"
    assert handler.events[0] == "start"
    assert handler.events[1] == (
        "tool_call.delta",
        {
            "index": 0,
            "call_id": "call_1",
            "arguments_delta": '{"a": 20',
        },
    )
    assert handler.events[-1] == "end"
    native_events = [
        e
        for e in result.events
        if getattr(e.phase, "value", e.phase) == "DECIDE"
        and (e.payload or {}).get("stage") == "native_tool_calls_decision"
    ]
    assert native_events
    traced = runtime_step_to_trace(record).to_dict()
    assert traced["decision_source"] == "native_tool_calls"
    assert traced["native_tool_call_used"] is True


def test_engine_preserves_streamed_reasoning_for_trace_and_tool_follow_up():
    reasoning = "Inspect the arguments before invoking the tool."
    model = _ChunkSequenceModel(
        [
            [
                ModelStreamChunk(reasoning_content=reasoning),
                ModelStreamChunk(
                    done=True,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 20, "b": 22}',
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                ),
            ],
            [ModelStreamChunk(text="Final Answer: done", done=True)],
        ],
        model="kimi-k3",
        provider="openai-compatible",
    )
    model.qitos_harness_metadata = {
        "family_preset": "kimi",
        "tool_policy": {
            "primary_delivery": "api_parameter",
            "fallback_delivery": "prompt_injection",
            "native_tool_call_preferred": True,
        },
    }

    class _Agent(DemoAgent):
        def __init__(self, llm: Model) -> None:
            super().__init__()
            self.llm = llm
            self.model_parser = ReActTextParser()

        def decide(self, state: DemoState, observation: dict[str, Any]):
            _ = state, observation
            return None

    result = Engine(agent=_Agent(model), budget=RuntimeBudget(max_steps=3)).run(
        "compute"
    )

    assert result.state.final_result == "done"
    assert result.records[0].model_response["reasoning_content"] == reasoning
    output_event = next(
        event
        for event in result.events
        if getattr(event.phase, "value", event.phase) == "DECIDE"
        and (event.payload or {}).get("stage") == "model_output"
    )
    assert output_event.payload["reasoning_content"] == reasoning
    assistant = next(
        message for message in model.calls[1] if message.get("role") == "assistant"
    )
    assert assistant["reasoning_content"] == reasoning
    assert assistant["tool_calls"][0]["id"] == "call_1"


def test_engine_sanitizes_submit_poc_native_tool_history_without_mutating_result():
    model = _ChunkSequenceModel(
        [
            [
                ModelStreamChunk(
                    done=True,
                    tool_calls=[
                        {
                            "id": "call_submit",
                            "type": "function",
                            "function": {
                                "name": "submit_poc",
                                "arguments": "{}",
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                )
            ],
            [
                ModelStreamChunk(
                    text="Final Answer: done",
                    done=True,
                    finish_reason="stop",
                )
            ],
        ],
        model="GLM-5.1",
        provider="openai-compatible",
    )
    model.qitos_harness_metadata = {
        "family_preset": "glm",
        "tool_policy": {
            "primary_delivery": "api_parameter",
            "fallback_delivery": "prompt_injection",
            "native_tool_call_preferred": True,
        },
    }

    class _SubmitAgent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model

            @tool(name="submit_poc")
            def submit_poc() -> dict[str, Any]:
                return {
                    "status": "success",
                    "vul_exit_code": 0,
                    "fix_exit_code": 0,
                    "poc_id": "p1",
                    "flag": None,
                    "raw_output": "wrong number of function inputs",
                    "verification_scope": "full",
                    "vul_stderr": "target stderr",
                    "fix_stderr": "hidden stderr",
                    "vul_stdout": "target stdout",
                    "fix_stdout": "hidden stdout",
                    "model_summary": "[submit_poc] attempt.bin\n\nNo vulnerable-target trigger was observed.\nVulnerable target exit: 0\nServer: wrong number of function inputs",
                }

            self.tool_registry.register(submit_poc)

        def decide(self, state: DemoState, observation: dict[str, Any]):
            _ = observation
            return None

        def reduce(
            self,
            state: DemoState,
            observation: dict[str, Any],
            decision: Decision[Action],
        ) -> DemoState:
            _ = observation
            _ = decision
            return state

    result = Engine(agent=_SubmitAgent(), budget=RuntimeBudget(max_steps=3)).run(
        "compute"
    )

    assert result.records[0].action_results[0].output["fix_exit_code"] == 0
    assert len(model.calls) >= 2
    second_call_text = "\n".join(str(message) for message in model.calls[1])
    assert "wrong number of function inputs" in second_call_text
    assert "vul_exit_code" not in second_call_text
    assert "fix_exit_code" not in second_call_text
    assert "fix_stderr" not in second_call_text
    assert "fix_stdout" not in second_call_text
    assert "verification_scope" not in second_call_text
    act_events = [
        e for e in result.events if getattr(e.phase, "value", e.phase) == "ACT"
    ]
    act_event_text = "\n".join(str(e.payload) for e in act_events)
    assert "wrong number of function inputs" in act_event_text
    assert "vul_exit_code" not in act_event_text
    assert "fix_exit_code" not in act_event_text
    assert "fix_stderr" not in act_event_text
    assert "fix_stdout" not in act_event_text
    assert "verification_scope" not in act_event_text


def test_engine_agent_can_block_disallowed_actions_before_execution():
    executed = {"value": False}
    model = _ChunkSequenceModel(
        [
            [
                ModelStreamChunk(
                    done=True,
                    tool_calls=[
                        {
                            "id": "call_blocked",
                            "type": "function",
                            "function": {
                                "name": "blocked_tool",
                                "arguments": "{}",
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                )
            ]
        ],
        model="qwen-plus",
        provider="openai-compatible",
    )
    model.qitos_harness_metadata = {
        "family_preset": "qwen",
        "tool_policy": {
            "primary_delivery": "api_parameter",
            "fallback_delivery": "prompt_injection",
            "native_tool_call_preferred": True,
        },
    }

    class _BlockAgent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model

            @tool(name="blocked_tool")
            def blocked_tool() -> str:
                executed["value"] = True
                return "should not run"

            self.tool_registry.register(blocked_tool)

        def decide(self, state: DemoState, observation: dict[str, Any]):
            _ = observation
            if state.current_step > 0:
                return Decision.final("done")
            return None

        def block_action(self, state: DemoState, action: Action) -> str | None:
            _ = state
            if action.name == "blocked_tool":
                return "blocked for this state"
            return None

    result = Engine(agent=_BlockAgent(), budget=RuntimeBudget(max_steps=3)).run(
        "compute"
    )

    assert executed["value"] is False
    first_result = result.records[0].action_results[0]
    assert first_result.status == "denied"
    assert first_result.error == "action_blocked"
    assert first_result.metadata["error_category"] == "action_blocked"
    assert "blocked for this state" in str(first_result.output)


def test_engine_salvages_glm_text_tool_call_markup_before_parser():
    model = _ChunkSequenceModel(
        [
            [
                ModelStreamChunk(
                    text=(
                        "<tool_call>add"
                        "<arg_key>a</arg_key><arg_value>20</arg_value>"
                        "<arg_key>b</arg_key><arg_value>22</arg_value>"
                        "</tool_call>"
                    ),
                    done=True,
                    finish_reason="tool_calls",
                )
            ]
        ],
        model="GLM-5.1",
        provider="openai-compatible",
    )
    model.qitos_harness_metadata = {
        "family_preset": "glm",
        "tool_policy": {
            "primary_delivery": "api_parameter",
            "fallback_delivery": "prompt_injection",
            "native_tool_call_preferred": True,
        },
    }

    class _NeverParser:
        def parse(self, raw_output, context=None):
            _ = raw_output
            _ = context
            raise AssertionError("GLM text tool-call markup should bypass the parser")

    class _Agent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = _NeverParser()

        def decide(self, state: DemoState, observation: dict[str, Any]):
            _ = observation
            if state.current_step > 0:
                return Decision.final("42")
            return None

    result = Engine(agent=_Agent(), budget=RuntimeBudget(max_steps=3)).run("compute")
    assert result.state.final_result == "42"
    record = result.records[0]
    assert record.decision_source == "native_tool_calls"
    assert record.native_tool_call_used is True
    assert record.actions[0].name == "add"
    assert record.actions[0].args == {"a": 20, "b": 22}
    assert record.model_response["tool_calls"][0]["function"]["name"] == "add"


def test_engine_native_tool_call_lane_returns_paired_error_on_bad_arguments():
    model = _ChunkSequenceModel(
        [
            [
                ModelStreamChunk(
                    text="Final Answer: must not bypass the invalid call",
                    done=True,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": "{not-json",
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                )
            ],
            [ModelStreamChunk(text="Final Answer: recovered", done=True)],
        ],
        model="qwen-plus",
    )
    model.qitos_harness_metadata = {
        "family_preset": "qwen",
        "tool_policy": {
            "primary_delivery": "api_parameter",
            "fallback_delivery": "prompt_injection",
            "native_tool_call_preferred": True,
        },
    }

    class _Agent(DemoAgent):
        def __init__(self):
            super().__init__()
            self.llm = model
            self.model_parser = ReActTextParser()

        def decide(self, state: DemoState, observation: dict[str, Any]):
            _ = state, observation
            return None

    agent = _Agent()
    result = Engine(agent=agent, budget=RuntimeBudget(max_steps=3)).run("compute")

    assert len(model.calls) == 2
    assert result.state.final_result == "recovered"
    record = result.records[0]
    assert record.decision_source == "native_tool_calls"
    assert record.native_tool_call_used is True
    assert record.native_tool_call_fallback_reason is None
    assert record.tool_invocations[0]["action_id"] == "call_1"
    assert record.tool_invocations[0]["attempts"] == 0
    assert record.tool_invocations[0]["error_category"] == (
        "tool_call_arguments_invalid"
    )
    assert record.action_results[0].status == "error"
    assert "TOOL_CALL_ARGUMENTS_INVALID" in record.action_results[0].output
    tool_messages = [
        message for message in model.calls[1] if message.get("role") == "tool"
    ]
    assert [message.get("tool_call_id") for message in tool_messages] == ["call_1"]


def test_engine_native_tool_call_lane_repairs_control_chars_in_arguments():
    model = _ChunkSequenceModel(
        [
            [
                ModelStreamChunk(
                    done=True,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "record_fact",
                                "arguments": '{"evidence":"line1\nline2"}',
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                )
            ]
        ],
        model="qwen-plus",
    )
    model.qitos_harness_metadata = {
        "family_preset": "qwen",
        "tool_policy": {
            "primary_delivery": "api_parameter",
            "fallback_delivery": "prompt_injection",
            "native_tool_call_preferred": True,
        },
    }

    class _NeverParser:
        def parse(self, raw_output, context=None):
            _ = raw_output
            _ = context
            raise AssertionError("native tool-call repair should bypass parser")

    class _Agent(DemoAgent):
        def __init__(self):
            registry = ToolRegistry()

            @tool(name="record_fact")
            def record_fact(evidence: str) -> str:
                return evidence

            registry.register(record_fact)
            AgentModule.__init__(
                self,
                tool_registry=registry,
                llm=model,
                model_parser=_NeverParser(),
            )

        def decide(self, state: DemoState, observation: dict[str, Any]):
            _ = observation
            if state.current_step > 0:
                return Decision.final("done")
            return None

    result = Engine(agent=_Agent(), budget=RuntimeBudget(max_steps=3)).run("record")
    record = result.records[0]
    assert record.decision_source == "native_tool_calls"
    assert record.native_tool_call_used is True
    assert record.native_tool_call_fallback_reason is None
    assert record.actions[0].name == "record_fact"
    assert record.actions[0].args == {"evidence": "line1\nline2"}
    assert record.actions[0].metadata["arguments_repair"] == "escaped_control_chars"
