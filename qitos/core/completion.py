"""Typed completion policy values for the canonical agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CompletionDisposition(str, Enum):
    """The product decision for one model-proposed final answer."""

    COMPLETED = "completed"
    BLOCKED = "blocked"
    CONTINUE = "continue"


@dataclass(frozen=True, slots=True)
class CompletionAssessment:
    """Accept, classify, or reject one proposed final answer.

    ``feedback`` is appended to canonical history only for ``CONTINUE`` so the
    following turn can address the missing completion condition.
    """

    disposition: CompletionDisposition
    reason: str = ""
    feedback: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, CompletionDisposition):
            raise TypeError("disposition must be a CompletionDisposition")
        if self.disposition is CompletionDisposition.CONTINUE and not self.feedback:
            raise ValueError("continue assessments require feedback")

    @classmethod
    def completed(cls, reason: str = "") -> "CompletionAssessment":
        return cls(CompletionDisposition.COMPLETED, reason=reason)

    @classmethod
    def blocked(cls, reason: str = "") -> "CompletionAssessment":
        return cls(CompletionDisposition.BLOCKED, reason=reason)

    @classmethod
    def continue_run(
        cls, feedback: str, *, reason: str = ""
    ) -> "CompletionAssessment":
        return cls(
            CompletionDisposition.CONTINUE,
            reason=reason,
            feedback=feedback,
        )


__all__ = ["CompletionAssessment", "CompletionDisposition"]
