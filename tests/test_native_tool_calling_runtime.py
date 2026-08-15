from collections.abc import AsyncIterator
from dataclasses import dataclass
import time
from typing import Any

from qitos import (
    Action,
    AgentModule,
    Decision,
    Engine,
    Observation,
    StateSchema,
    ToolRegistry,
    tool,
)
from qitos.engine import RuntimeBudget
from qitos.kit import ReActTextParser
from qitos.models import Model, ModelRequest, ModelStreamEvent, ModelStreamEventType
from qitos.models._openai_responses import _to_responses_input


@dataclass
class _State(StateSchema):
    pass


class _NativeToolModel(Model):
    def __init__(self):
        super().__init__(
            model="test-native",
            max_tokens=256,
            context_window=8192,
            temperature=None,
        )
        self.calls = 0
        self.seen_messages: list[list[dict[str, Any]]] = []
        self.qitos_harness_metadata = {
            "tool_policy": {"native_tool_call_preferred": True},
            "parser": "ReActTextParser",
            "protocol": "react_text_v1",
        }

    def _chunk_for_call(self, call_index: int) -> ModelStreamEvent:
        if call_index == 0:
            return ModelStreamEvent(
                type=ModelStreamEventType.COMPLETED,
                tool_calls=[
                    {
                        "id": "call_native_1",
                        "type": "function",
                        "function": {"name": "weird_tool", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
                event_type="test.completed",
            )
        return ModelStreamEvent(
            text="Final Answer: done",
            type=ModelStreamEventType.COMPLETED,
            finish_reason="stop",
            event_type="test.completed",
        )

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        messages = request.message_dicts()
        self.seen_messages.append(list(messages))
        call_index = self.calls
        self.calls += 1
        yield self._chunk_for_call(call_index)


class _NativeToolAgent(AgentModule[_State, Observation, Action]):
    def __init__(self, llm: Any):
        registry = ToolRegistry()

        @tool(name="weird_tool")
        def weird_tool() -> dict[str, Any]:
            return {"payload": {1, 2}}

        registry.register(weird_tool)
        super().__init__(
            tool_registry=registry, llm=llm, model_parser=ReActTextParser()
        )

    def init_state(self, task: str, **kwargs: Any) -> _State:
        return _State(task=task, max_steps=3)

    def decide(
        self, state: _State, observation: Observation
    ) -> Decision[Action] | None:
        _ = state
        _ = observation
        return None

    def reduce(
        self, state: _State, observation: Observation, decision: Decision[Action]
    ) -> _State:
        _ = observation
        _ = decision
        return state


class _HarnessAwareModel(Model):
    def __init__(self):
        super().__init__(model="harness-aware", temperature=None)
        self.qitos_harness_metadata = {
            "parser": "ReActTextParser",
            "protocol": "react_text_v1",
        }

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = request
        yield ModelStreamEvent(
            text="Final Answer: auto harness parser worked",
            type=ModelStreamEventType.COMPLETED,
            finish_reason="stop",
            event_type="test.completed",
        )


class _HarnessAgent(AgentModule[_State, Observation, Action]):
    def __init__(self):
        super().__init__(llm=_HarnessAwareModel())

    def init_state(self, task: str, **kwargs: Any) -> _State:
        _ = kwargs
        return _State(task=task, max_steps=2)

    def decide(
        self, state: _State, observation: Observation
    ) -> Decision[Action] | None:
        _ = state
        _ = observation
        return None

    def reduce(
        self, state: _State, observation: Observation, decision: Decision[Action]
    ) -> _State:
        _ = observation
        _ = decision
        return state


def test_native_tool_chain_preserves_tool_call_history_and_non_json_result() -> None:
    llm = _NativeToolModel()
    agent = _NativeToolAgent(llm=llm)
    result = Engine(agent=agent, budget=RuntimeBudget(max_steps=3)).run("native")
    assert result.state.final_result == "done"
    assert len(llm.seen_messages) >= 2
    second_call = llm.seen_messages[1]
    assistant_msgs = [msg for msg in second_call if msg.get("role") == "assistant"]
    tool_msgs = [msg for msg in second_call if msg.get("role") == "tool"]
    assert assistant_msgs
    assert tool_msgs
    assert assistant_msgs[-1].get("tool_calls")
    assert tool_msgs[-1].get("tool_call_id") == "call_native_1"
    tool_content = str(tool_msgs[-1].get("content", ""))
    assert "1" in tool_content and "2" in tool_content


def test_native_tool_timeout_remains_timed_out_in_result_trace_and_history() -> None:
    class _SlowToolModel(_NativeToolModel):
        def _chunk_for_call(self, call_index: int) -> ModelStreamEvent:
            if call_index == 0:
                return ModelStreamEvent(
                    type=ModelStreamEventType.COMPLETED,
                    tool_calls=[
                        {
                            "id": "call_slow_1",
                            "type": "function",
                            "function": {
                                "name": "slow_tool",
                                "arguments": "{}",
                            },
                        }
                    ],
                    finish_reason="tool_calls",
                    event_type="test.completed",
                )
            return super()._chunk_for_call(call_index)

    class _SlowToolAgent(AgentModule[_State, Observation, Action]):
        def __init__(self, llm: Any) -> None:
            registry = ToolRegistry()

            @tool(name="slow_tool", timeout_s=0.01)
            def slow_tool() -> str:
                time.sleep(0.05)
                return "late"

            registry.register(slow_tool)
            super().__init__(
                tool_registry=registry,
                llm=llm,
                model_parser=ReActTextParser(),
            )

        def init_state(self, task: str, **kwargs: Any) -> _State:
            _ = kwargs
            return _State(task=task, max_steps=3)

        def decide(
            self, state: _State, observation: Observation
        ) -> Decision[Action] | None:
            _ = state, observation
            return None

        def reduce(
            self,
            state: _State,
            observation: Observation,
            decision: Decision[Action],
        ) -> _State:
            _ = observation, decision
            return state

    llm = _SlowToolModel()
    result = Engine(
        agent=_SlowToolAgent(llm),
        budget=RuntimeBudget(max_steps=3),
    ).run("time out")

    tool_result = result.records[0].action_results[0]
    assert tool_result.status == "timed_out"
    assert tool_result.is_success is False
    assert result.success_rate == 0.0
    assert result.step_summaries[0].status == "timed_out"
    action_result_events = [
        event
        for event in result.events
        if event.payload.get("stage") == "action_results"
    ]
    assert action_result_events[0].payload["action_results"][0]["status"] == (
        "timed_out"
    )

    tool_messages = [
        message for message in llm.seen_messages[1] if message.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert "timed_out" in str(tool_messages[0].get("content", ""))


def test_normal_turns_append_complete_native_tool_transactions() -> None:
    class _VariableNativeToolModel(Model):
        qitos_harness_metadata = {
            "tool_policy": {"native_tool_call_preferred": True},
            "parser": "ReActTextParser",
            "protocol": "react_text_v1",
        }

        def __init__(self) -> None:
            super().__init__(
                model="test-variable-native",
                max_tokens=256,
                context_window=8192,
                temperature=None,
            )
            self.calls = 0
            self.messages_by_call: list[list[dict[str, Any]]] = []
            self.orphan_ids_by_call: list[list[str]] = []

        async def stream(
            self,
            request: ModelRequest,
        ) -> AsyncIterator[ModelStreamEvent]:
            messages = request.message_dicts()
            self.messages_by_call.append(messages)
            assistant_ids = {
                str(tool_call["id"])
                for message in messages
                if message.get("role") == "assistant"
                for tool_call in message.get("tool_calls", [])
                if isinstance(tool_call, dict) and tool_call.get("id")
            }
            tool_result_ids = {
                str(message["tool_call_id"])
                for message in messages
                if message.get("role") == "tool" and message.get("tool_call_id")
            }
            self.orphan_ids_by_call.append(sorted(tool_result_ids - assistant_ids))

            call_index = self.calls
            self.calls += 1
            if call_index >= 12:
                yield ModelStreamEvent(
                    text="Final Answer: done",
                    type=ModelStreamEventType.COMPLETED,
                    finish_reason="stop",
                    event_type="test.completed",
                )
                return

            tool_call_count = 3 if call_index == 0 else 1
            yield ModelStreamEvent(
                type=ModelStreamEventType.COMPLETED,
                tool_calls=[
                    {
                        "id": f"call_{call_index}_{offset}",
                        "type": "function",
                        "function": {
                            "name": "probe",
                            "arguments": ('{"value": %d}' % (call_index * 10 + offset)),
                        },
                    }
                    for offset in range(tool_call_count)
                ],
                finish_reason="tool_calls",
                event_type="test.completed",
            )

    class _VariableNativeToolAgent(AgentModule[_State, Observation, Action]):
        def __init__(self, llm: Any):
            registry = ToolRegistry()

            @tool(name="probe")
            def probe(value: int) -> dict[str, int]:
                return {"value": value}

            registry.register(probe)
            super().__init__(
                tool_registry=registry,
                llm=llm,
                model_parser=ReActTextParser(),
            )

        def init_state(self, task: str, **kwargs: Any) -> _State:
            _ = kwargs
            return _State(task=task, max_steps=16)

        def prepare(self, state: _State) -> str:
            return f"continue step {state.current_step}"

        def decide(
            self, state: _State, observation: Observation
        ) -> Decision[Action] | None:
            _ = state, observation
            return None

        def reduce(
            self,
            state: _State,
            observation: Observation,
            decision: Decision[Action],
        ) -> _State:
            _ = observation, decision
            return state

    llm = _VariableNativeToolModel()
    result = Engine(
        agent=_VariableNativeToolAgent(llm),
        budget=RuntimeBudget(max_steps=16),
    ).run("exercise variable native tool rounds")

    assert result.state.final_result == "done"
    assert llm.calls == 13
    assert llm.orphan_ids_by_call == [[] for _ in range(13)]
    for previous, current in zip(
        llm.messages_by_call,
        llm.messages_by_call[1:],
        strict=False,
    ):
        assert current[: len(previous)] == previous
    model_input_events = [
        event.payload
        for event in result.events
        if event.payload.get("stage") == "model_input"
    ]
    assert len(model_input_events) == 13
    system_digests = set()
    for index, payload in enumerate(model_input_events):
        parity = payload["tool_transaction_parity"]
        assert parity["valid"] is True
        assert parity["orphan_result_ids"] == []
        cache = payload["request_cache"]
        assert cache["affinity_enabled"] is True
        assert "cache_affinity" not in cache
        system_digests.add(cache["system_prompt_digest"])
        if index == 0:
            assert cache["previous_request_available"] is False
            assert cache["previous_messages_are_prefix"] is None
            continue
        assert cache["previous_request_available"] is True
        assert cache["previous_messages_are_prefix"] is True
        assert cache["common_prefix_messages"] == len(
            llm.messages_by_call[index - 1]
        )
        assert cache["common_prefix_tokens"] > 0
        assert cache["system_prompt_stable"] is True
        assert cache["tool_schema_stable"] is True
        assert cache["prefix_cache_reusable"] is True
    assert len(system_digests) == 1


def test_responses_native_items_survive_engine_tool_round() -> None:
    class _ResponsesNativeModel(_NativeToolModel):
        def _chunk_for_call(self, call_index: int) -> ModelStreamEvent:
            if call_index == 0:
                return ModelStreamEvent(
                    type=ModelStreamEventType.COMPLETED,
                    tool_calls=[
                        {
                            "id": "call_native_1",
                            "type": "function",
                            "function": {
                                "name": "weird_tool",
                                "arguments": "{}",
                            },
                        }
                    ],
                    native_items=[
                        {
                            "type": "reasoning",
                            "id": "reasoning_1",
                            "summary": [],
                        },
                        {
                            "type": "function_call",
                            "id": "function_1",
                            "call_id": "call_native_1",
                            "name": "weird_tool",
                            "arguments": "{}",
                        },
                    ],
                    finish_reason="tool_calls",
                    event_type="test.completed",
                )
            return super()._chunk_for_call(call_index)

    llm = _ResponsesNativeModel()
    result = Engine(
        agent=_NativeToolAgent(llm=llm),
        budget=RuntimeBudget(max_steps=3),
    ).run("native responses")

    assert result.state.final_result == "done"
    replay_items = _to_responses_input(llm.seen_messages[1])
    replay_types = [item.get("type") for item in replay_items]
    reasoning_index = replay_types.index("reasoning")
    assert replay_types[reasoning_index : reasoning_index + 3] == [
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert replay_items[reasoning_index + 2]["call_id"] == "call_native_1"


def test_agent_run_auto_applies_harness_parser_defaults() -> None:
    agent = _HarnessAgent()
    output = agent.run("auto-parser", trace=False, render=False)
    assert output == "auto harness parser worked"
