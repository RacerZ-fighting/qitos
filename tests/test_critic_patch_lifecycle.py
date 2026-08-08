"""Tests for critic patch lifecycle — no leaks between runs."""
from __future__ import annotations

import pytest
from qitos import AgentModule, StateSchema
from qitos.engine.engine import Engine


class _MinimalAgent(AgentModule):
    name = "test_critic_lifecycle"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_state(self, task, **kwargs):
        return StateSchema(task=task, max_steps=3)

    def build_system_prompt(self, state):
        return "You are a test agent."

    def prepare(self, state):
        return f"Task: {state.task}"

    def reduce(self, state, observation, decision):
        return state


def test_critic_variables_initialized_in_init():
    """_critic_modified_prompt and _critic_instruction_patch are None after __init__."""
    from unittest.mock import MagicMock
    agent = _MinimalAgent(llm=MagicMock())
    engine = Engine(agent=agent)
    assert engine._critic_modified_prompt is None
    assert engine._critic_instruction_patch is None


def test_reset_run_state_clears_critic_patches():
    """_reset_run_state() clears both critic patch variables."""
    from unittest.mock import MagicMock
    agent = _MinimalAgent(llm=MagicMock())
    engine = Engine(agent=agent)
    engine._critic_modified_prompt = "modified"
    engine._critic_instruction_patch = "instruction"
    engine._reset_run_state()
    assert engine._critic_modified_prompt is None
    assert engine._critic_instruction_patch is None


def test_instruction_patch_does_not_leak_between_runs():
    """Setting _critic_instruction_patch in one run doesn't leak to the next."""
    from unittest.mock import MagicMock
    agent = _MinimalAgent(llm=MagicMock())
    engine = Engine(agent=agent)
    engine._critic_instruction_patch = "leaked_patch"
    engine._reset_run_state()
    assert engine._critic_instruction_patch is None


def test_handoff_method_does_not_clear_instruction_patch_inline():
    """_intercept_handoff_action returns Decision without side effects on instruction patch."""
    from qitos.core.action import Action
    from unittest.mock import MagicMock
    agent = _MinimalAgent(llm=MagicMock())
    engine = Engine(agent=agent)
    engine._critic_instruction_patch = "some_patch"
    # Call the handoff check
    action = Action(name="transfer_to_agent_b", args={"rationale": "test"})
    result = engine._intercept_handoff_action(action)
    # The patch should NOT be cleared by this method (it's handled in _reset_run_state)
    assert engine._critic_instruction_patch == "some_patch"
    assert result is not None


def test_no_dead_code_after_return():
    """Verify the dead code line (self._critic_instruction_patch = None after return) is removed."""
    import inspect
    source = inspect.getsource(Engine._intercept_handoff_action)
    lines = source.strip().split('\n')
    # After the return line, there should be no assignment to _critic_instruction_patch
    found_return = False
    for line in lines:
        if 'return Decision.handoff' in line:
            found_return = True
        elif found_return and '_critic_instruction_patch' in line and '=' in line:
            pytest.fail("Dead code after return: _critic_instruction_patch assignment found after return")
