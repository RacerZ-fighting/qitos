"""Environment composed from named capability providers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qitos.core.env import (
    Env,
    EnvObservation,
    EnvStepResult,
    RuntimeCapabilitySnapshot,
)


class CapabilityEnv(Env):
    """Expose a fixed set of named operations through the canonical Env API.

    The environment does not own provider lifecycles. Composition roots remain
    responsible for closing attempt-scoped, shared, or externally managed
    providers after the Engine has finished using the environment.

    Args:
        ops: Mapping from capability group names to concrete providers.
        name: Human-readable environment name used by traces and diagnostics.
        snapshot: Immutable facts for the initialized backend. Its operation
            groups must match the supplied providers exactly.
        attestation: Legacy environment metadata retained for callers that
            still record attempt identity separately from runtime facts.
    """

    version = "1.0"

    def __init__(
        self,
        ops: Mapping[str, Any],
        *,
        name: str = "capability_env",
        snapshot: RuntimeCapabilitySnapshot | None = None,
        attestation: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise ValueError("CapabilityEnv name must be non-empty")

        normalized_ops: dict[str, Any] = {}
        for raw_group, provider in ops.items():
            group = str(raw_group or "").strip()
            if not group:
                raise ValueError("Capability group names must be non-empty")
            if provider is None:
                raise ValueError(f"Capability provider must be non-null: {group}")
            normalized_ops[group] = provider

        self.name = normalized_name
        self._ops = normalized_ops
        self.attestation = dict(attestation or {})
        self._snapshot = snapshot or RuntimeCapabilitySnapshot(
            backend=normalized_name,
            working_directory=".",
            operation_groups=tuple(sorted(normalized_ops)),
        )
        if set(self._snapshot.operation_groups) != set(normalized_ops):
            raise ValueError(
                "Runtime snapshot operation groups must match capability providers"
            )
        self._last_action: str | None = None

    @property
    def capability_groups(self) -> tuple[str, ...]:
        """Return the available capability group names in stable order."""

        return tuple(sorted(self._ops))

    def reset(
        self,
        task: Any = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> EnvObservation:
        """Reset only the environment observation state.

        Concrete provider lifecycles are deliberately outside this generic
        composition object.
        """

        _ = task, workspace, kwargs
        self._last_action = None
        return self.observe()

    def observe(self, state: Any = None) -> EnvObservation:
        """Return a bounded description of available capabilities."""

        return EnvObservation(
            data={
                "capability_groups": list(self.capability_groups),
                "last_action": self._last_action,
            },
            metadata={"state_step": getattr(state, "current_step", None)},
        )

    def step(self, action: Any, state: Any = None) -> EnvStepResult:
        """Record the last action without interpreting tool behavior."""

        self._last_action = _action_name(action)
        return EnvStepResult(observation=self.observe(state=state), done=False)

    def health_check(self) -> dict[str, Any]:
        """Run optional provider health checks and report exact failures."""

        checks: dict[str, dict[str, Any]] = {}
        failed: list[str] = []
        for group in self.capability_groups:
            provider = self._ops[group]
            probe = getattr(provider, "health_check", None)
            if not callable(probe):
                checks[group] = {"ok": True}
                continue
            try:
                raw_result = probe()
            except Exception as exc:
                checks[group] = {"ok": False, "message": str(exc)}
                failed.append(group)
                continue
            result = dict(raw_result) if isinstance(raw_result, Mapping) else {
                "ok": bool(raw_result)
            }
            checks[group] = result
            if not bool(result.get("ok", False)):
                failed.append(group)

        return {
            "ok": not failed,
            "capability_groups": list(self.capability_groups),
            "checks": checks,
            "failed_groups": failed,
            **(
                {"message": "Capability provider health check failed"}
                if failed
                else {}
            ),
        }

    def get_ops(self, group: str) -> Any:
        """Return the exact provider registered for a capability group."""

        return self._ops.get(str(group))

    def capability_snapshot(self) -> RuntimeCapabilitySnapshot:
        """Return the exact immutable snapshot bound to these providers."""

        return self._snapshot


def _action_name(action: Any) -> str | None:
    if isinstance(action, Mapping):
        actions = action.get("actions")
        if isinstance(actions, list) and actions:
            first = actions[0]
            if isinstance(first, Mapping):
                value = first.get("name")
                return str(value) if value else None
            value = getattr(first, "name", None)
            return str(value) if value else None
        value = action.get("name")
        return str(value) if value else None
    value = getattr(action, "name", None)
    return str(value) if value else None


__all__ = ["CapabilityEnv"]
