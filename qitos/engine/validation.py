"""Runtime state validation gates for Engine execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

from ..core.errors import StopReason
from ..core.state import StateSchema, StateValidationError


Validator = Callable[[StateSchema], None]


def validate_step_bounds(state: StateSchema) -> None:
    if state.current_step > state.max_steps:
        raise StateValidationError(
            f"current_step={state.current_step} exceeds max_steps={state.max_steps}"
        )


def validate_optional_plan_fields(state: StateSchema) -> None:
    plan_steps = getattr(state, "plan_steps", None)
    cursor = getattr(state, "cursor", None)
    if isinstance(plan_steps, list) and cursor is not None:
        if int(cursor) < 0:
            raise StateValidationError("cursor must be >= 0 when plan_steps is present")
        if int(cursor) > len(plan_steps):
            raise StateValidationError("cursor exceeds available plan steps")


def validate_final_consistency(state: StateSchema) -> None:
    if state.stop_reason:
        try:
            StopReason(str(state.stop_reason))
        except ValueError as exc:
            raise StateValidationError(
                "stop_reason must be one of StopReason values"
            ) from exc
    result_reasons = {
        StopReason.FINAL.value,
        StopReason.COMPLETED.value,
        StopReason.BLOCKED.value,
    }
    if state.stop_reason in result_reasons and not state.final_result:
        raise StateValidationError(
            f"stop_reason={state.stop_reason} requires final_result"
        )


DEFAULT_STATE_VALIDATORS: List[Validator] = [
    validate_step_bounds,
    validate_optional_plan_fields,
    validate_final_consistency,
]


@dataclass
class StateValidatorChain:
    validators: List[Validator]

    def validate(self, state: StateSchema) -> None:
        state.validate()
        for validator in self.validators:
            validator(state)


class StateValidationGate:
    """Run validation checks before and after each engine phase."""

    def __init__(self, validators: Iterable[Validator] = DEFAULT_STATE_VALIDATORS):
        self.chain = StateValidatorChain(list(validators))

    def before_phase(self, state: StateSchema, phase: str) -> None:
        self.chain.validate(state)

    def after_phase(self, state: StateSchema, phase: str) -> None:
        self.chain.validate(state)


__all__ = [
    "Validator",
    "DEFAULT_STATE_VALIDATORS",
    "StateValidatorChain",
    "StateValidationGate",
    "validate_step_bounds",
    "validate_optional_plan_fields",
    "validate_final_consistency",
]
