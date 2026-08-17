"""Model-facing error taxonomy for QitOS runtime."""

from __future__ import annotations


class ModelTransportError(Exception):
    """A model request that exhausted its adapter-owned transport retries."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.retryable = retryable
        self.status_code = status_code


class ModelRequestDeadlineExceeded(TimeoutError):
    """The run deadline expired while a model request was in flight."""


class ModelRequestCancelled(Exception):
    """Cooperative cancellation stopped waiting for a model request."""


class ModelContinuationRejected(Exception):
    """A Provider rejected an optional server-side continuation handle."""
