"""Tests for interrupt/resume mechanism."""

import pytest
from qitos.engine.interrupt import (
    EngineInterrupt,
    InterruptInfo,
    interrupt,
    _reset_interrupt_context,
    _set_resume_values,
    _clear_resume_values,
)
from qitos.checkpoint import (
    Checkpoint,
    CheckpointConfig,
    CheckpointId,
    CheckpointMetadata,
    InMemoryCheckpointStore,
)
from qitos.core.errors import StopReason
from qitos.engine.states import StepResult


class TestInterruptFunction:
    """Test the interrupt() function directly."""

    def setup_method(self):
        _reset_interrupt_context()
        _clear_resume_values()

    def test_interrupt_raises_on_first_call(self):
        with pytest.raises(EngineInterrupt) as exc_info:
            interrupt("What is your name?")
        assert exc_info.value.value == "What is your name?"
        assert exc_info.value.interrupt_id == "int_1"

    def test_interrupt_returns_resume_value(self):
        _set_resume_values({"int_1": "Alice"})
        result = interrupt("What is your name?")
        assert result == "Alice"

    def test_multiple_interrupts(self):
        # First interrupt
        with pytest.raises(EngineInterrupt) as exc1:
            interrupt("Question 1")
        assert exc1.value.interrupt_id == "int_1"

        # Reset for re-execution
        _reset_interrupt_context()
        _set_resume_values({"int_1": "Answer 1", "int_2": "Answer 2"})

        # Re-execute: first interrupt returns value
        result1 = interrupt("Question 1")
        assert result1 == "Answer 1"

        # Second interrupt returns value
        result2 = interrupt("Question 2")
        assert result2 == "Answer 2"

    def test_partial_resume_values(self):
        _reset_interrupt_context()
        _set_resume_values({"int_1": "Answer 1"})

        # First interrupt satisfied
        result = interrupt("Question 1")
        assert result == "Answer 1"

        # Second interrupt not satisfied — raises again
        with pytest.raises(EngineInterrupt) as exc:
            interrupt("Question 2")
        assert exc.value.interrupt_id == "int_2"

    def test_interrupt_counter_resets(self):
        with pytest.raises(EngineInterrupt):
            interrupt("Q1")
        _reset_interrupt_context()
        # Counter is reset, so first interrupt again
        with pytest.raises(EngineInterrupt) as exc:
            interrupt("Q2")
        assert exc.value.interrupt_id == "int_1"


class TestEngineInterrupt:
    """Test EngineInterrupt exception."""

    def test_engine_interrupt_fields(self):
        ei = EngineInterrupt(value="question", interrupt_id="int_1")
        assert ei.value == "question"
        assert ei.interrupt_id == "int_1"
        assert ei.checkpoint_id is None

    def test_engine_interrupt_is_exception(self):
        ei = EngineInterrupt(value="test")
        assert isinstance(ei, Exception)


class TestInterruptInfo:
    """Test InterruptInfo dataclass."""

    def test_interrupt_info(self):
        info = InterruptInfo(
            interrupt_id="int_1",
            checkpoint_id=CheckpointId("cp1"),
            value="question",
        )
        assert info.interrupt_id == "int_1"
        assert info.checkpoint_id == "cp1"
        assert info.value == "question"


class TestInterruptWithCheckpointStore:
    """Test interrupt with a real checkpoint store for resume flow."""

    def test_step_result_has_interrupt_info(self):
        """Verify StepResult can carry interrupt_info."""
        info = InterruptInfo(
            interrupt_id="int_1",
            checkpoint_id=CheckpointId("cp1"),
            value="approval needed",
        )
        result = StepResult(
            step_id=1,
            decision=None,
            record=None,  # type: ignore
            observation=None,
            stop=True,
            stop_reason=StopReason.INTERRUPT,
            interrupt_info=info,
        )
        assert result.interrupt_info is not None
        assert result.interrupt_info.interrupt_id == "int_1"
        assert result.stop_reason == StopReason.INTERRUPT

    @pytest.mark.asyncio
    async def test_interrupt_with_store(self):
        """Test that interrupt checkpoint can be saved and resumed."""
        store = InMemoryCheckpointStore()
        cp = Checkpoint(
            id=CheckpointId("cp1"),
            thread_id="t1",
            step=3,
            state_data={"task": "hello", "current_step": 3},
        )
        config = CheckpointConfig(thread_id="t1")
        meta: CheckpointMetadata = {"source": "interrupt", "step": 3}
        await store.put(config, cp, meta)

        # Verify we can retrieve it
        got = await store.get(CheckpointConfig(thread_id="t1"))
        assert got is not None
        assert got.step == 3
        assert got.state_data["task"] == "hello"
