"""Tests for functional API — @task, @agent, compose, infer (Task 3.1)."""

from __future__ import annotations

import asyncio
import time
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from qitos.func.task import task, TaskFunction
from qitos.func.agent import agent, AgentFunction, _FunctionAgent, _AgentState
from qitos.func.compose import compose
from qitos.func.infer import (
    infer_parameters,
    infer_state_type,
    infer_tool_list,
    infer_model,
)
from qitos.core.agent_module import AgentModule
from qitos.core.state import StateSchema


# ---------------------------------------------------------------------------
# 3.1.1 @task decorator
# ---------------------------------------------------------------------------


class TestTaskDecorator:
    def test_bare_decorator(self):
        @task
        def add(a: int, b: int) -> int:
            return a + b

        assert isinstance(add, TaskFunction)
        assert add.name == "add"

    def test_decorator_with_kwargs(self):
        @task(name="search")
        def my_search(query: str) -> list[str]:
            return [query]

        assert isinstance(my_search, TaskFunction)
        assert my_search.name == "search"

    def test_name_defaults_to_function_name(self):
        @task
        def my_special_task(x: int) -> int:
            return x

        assert my_special_task.name == "my_special_task"

    def test_direct_call(self):
        @task
        def double(x: int) -> int:
            return x * 2

        assert double(5) == 10

    def test_submit_for_parallel(self):
        @task
        def slow_add(a: int, b: int) -> int:
            time.sleep(0.01)
            return a + b

        future = slow_add.submit(None, 1, 2)
        assert future.result(timeout=5) == 3

    def test_async_call(self):
        @task
        async def async_double(x: int) -> int:
            return x * 2

        assert async_double.is_async
        result = asyncio.run(async_double.acall(5))
        assert result == 10

    def test_repr(self):
        @task
        def my_task(x: int) -> int:
            return x

        assert "my_task" in repr(my_task)


# ---------------------------------------------------------------------------
# 3.1.2 @agent decorator
# ---------------------------------------------------------------------------


class TestAgentDecorator:
    def test_bare_decorator(self):
        @agent
        def my_agent(task: str) -> str:
            return f"processed: {task}"

        assert isinstance(my_agent, AgentFunction)
        assert my_agent.name == "my_agent"

    def test_decorator_with_kwargs(self):
        @agent(name="researcher", max_steps=5)
        def my_agent(task: str) -> str:
            return task

        assert isinstance(my_agent, AgentFunction)
        assert my_agent.name == "researcher"

    def test_direct_call(self):
        @agent
        def echo(task: str) -> str:
            return f"echo: {task}"

        assert echo("hello") == "echo: hello"

    def test_as_agent_module(self):
        @agent
        def my_agent(task: str) -> str:
            """A test agent."""
            return f"done: {task}"

        module = my_agent._as_agent_module()
        assert isinstance(module, AgentModule)
        assert module.name == "my_agent"

    def test_agent_module_init_state(self):
        @agent
        def my_agent(task: str) -> str:
            return task

        module = my_agent._as_agent_module()
        state = module.init_state("test task")
        assert state.task == "test task"

    def test_agent_module_build_prompt(self):
        @agent
        def documented_agent(task: str) -> str:
            """This is a documented agent."""
            return task

        module = documented_agent._as_agent_module()
        state = module.init_state("test")
        prompt = module.build_system_prompt(state)
        assert "documented agent" in prompt

    def test_repr(self):
        @agent
        def my_agent(task: str) -> str:
            return task

        assert "my_agent" in repr(my_agent)


# ---------------------------------------------------------------------------
# 3.1.3 Task composition
# ---------------------------------------------------------------------------


class TestComposition:
    def test_compose_run_parallel(self):
        @task
        def task_a(x: int) -> int:
            return x + 1

        @task
        def task_b(x: int) -> int:
            return x * 2

        composer = compose(task_a, task_b)
        results = composer.run(5)
        assert results == [6, 10]

    def test_compose_map(self):
        @task
        def add_one(x: int) -> int:
            return x + 1

        @task
        def double(x: int) -> int:
            return x * 2

        composer = compose(add_one, double)
        results = composer.map([10, 3])
        assert results == [11, 6]

    def test_compose_map_wrong_length(self):
        @task
        def task_a(x: int) -> int:
            return x

        composer = compose(task_a)
        with pytest.raises(ValueError, match="Expected 1 items"):
            composer.map([1, 2])

    def test_compose_arun(self):
        @task
        def sync_task(x: int) -> int:
            return x + 1

        composer = compose(sync_task)
        results = asyncio.run(composer.arun(5))
        assert results == [6]


# ---------------------------------------------------------------------------
# 3.1.4 Parameter inference
# ---------------------------------------------------------------------------


class _MyTestState(StateSchema):
    value: int = 0


class TestInference:
    def test_infer_state_type(self):
        def my_func(task: str, state: _MyTestState) -> str:
            return task

        assert infer_state_type(my_func) is _MyTestState

    def test_infer_state_type_none(self):
        def my_func(task: str) -> str:
            return task

        assert infer_state_type(my_func) is None

    def test_infer_tool_list(self):
        def my_func(task: str, tools: list = None) -> str:
            return task

        # None default is not a list, so returns []
        assert infer_tool_list(my_func) == []

    def test_infer_tool_list_with_default(self):
        def my_func(task: str, tools: list = None) -> str:
            return task

        # None is not a list, so returns []
        result = infer_tool_list(my_func)
        assert result == []  # None is not isinstance(list)

    def test_infer_model(self):
        def my_func(task: str, model: str = "gpt-4") -> str:
            return task

        assert infer_model(my_func) == "gpt-4"

    def test_infer_model_none(self):
        def my_func(task: str) -> str:
            return task

        assert infer_model(my_func) is None

    def test_infer_parameters(self):
        def my_func(task: str, state: _MyTestState, model: str = "gpt-4", temperature: float = 0.7) -> str:
            return task

        params = infer_parameters(my_func)
        assert params["state_type"] is _MyTestState
        assert params["model"] == "gpt-4"
        assert params["temperature"] == 0.7


# ---------------------------------------------------------------------------
# 3.1.5 AgentModule interop
# ---------------------------------------------------------------------------


class TestAgentModuleInterop:
    def test_function_agent_is_agent_module(self):
        @agent
        def my_agent(task: str) -> str:
            return task

        module = my_agent._as_agent_module()
        assert isinstance(module, AgentModule)

    def test_function_agent_can_delegate(self):
        """@agent can coexist with AgentModule subclasses in an agent_registry."""
        from qitos.core.agent_module import AgentModule
        from qitos.core.decision import Decision

        class SubAgent(AgentModule):
            name = "sub"

            def init_state(self, task, **kw):
                from dataclasses import dataclass
                @dataclass
                class S(StateSchema):
                    pass
                return S(task=task, max_steps=5)

            def reduce(self, state, obs, decision):
                return state

        @agent(name="func")
        def func_agent(task: str) -> str:
            return task

        func_module = func_agent._as_agent_module()
        sub_module = SubAgent(llm=MagicMock())

        # Both can be listed in a registry
        agents = [func_module, sub_module]
        assert len(agents) == 2
        assert agents[0].name == "func"
        assert agents[1].name == "sub"

    def test_function_agent_state(self):
        @agent
        def my_agent(task: str) -> str:
            return task

        module = my_agent._as_agent_module()
        state = module.init_state("test")
        assert isinstance(state, _AgentState)
        assert state.result is None
        assert state.steps_taken == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_task_preserves_metadata(self):
        @task
        def documented_task(x: int) -> int:
            """This is a documented task."""
            return x

        assert documented_task.__doc__ == "This is a documented task."

    def test_agent_preserves_metadata(self):
        @agent
        def documented_agent(task: str) -> str:
            """This is a documented agent."""
            return task

        assert documented_agent.__doc__ == "This is a documented agent."

    def test_task_with_no_return_type(self):
        @task
        def no_return(x):
            pass

        assert no_return(1) is None

    def test_async_agent_acall(self):
        @agent
        async def async_agent(task: str) -> str:
            return f"async: {task}"

        result = asyncio.run(async_agent.acall("test"))
        assert result == "async: test"
