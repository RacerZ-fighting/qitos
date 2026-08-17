"""Dependency-aware execution Plan shared by Root and Child Agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from .child import ChildHandle
from .tool_result import ToolResult

MAX_PLAN_NODES = 64
MAX_PLAN_NODE_ID_CHARS = 128
MAX_PLAN_DESCRIPTION_CHARS = 512
MAX_PLAN_EXPLANATION_CHARS = 2_000
UPDATE_PLAN_TOOL_NAME = "update_plan"


class PlanContractError(ValueError):
    """Raised when a Plan value or replacement violates the public contract."""


class PlanStatus(str, Enum):
    """Durable node states; readiness is deliberately derived, not stored."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


def _normalized_text(
    value: str,
    field: str,
    limit: int,
    *,
    single_line: bool = True,
) -> str:
    if not isinstance(value, str):
        raise PlanContractError(f"{field} must be text")
    normalized = value.strip()
    if not normalized:
        raise PlanContractError(f"{field} must be non-empty text")
    if len(normalized) > limit:
        raise PlanContractError(f"{field} must be at most {limit} characters")
    if single_line and any(
        ord(character) < 32 or ord(character) == 127 for character in normalized
    ):
        raise PlanContractError(f"{field} must be a single printable line")
    return normalized


@dataclass(frozen=True, slots=True)
class PlanNode:
    """One stable unit of work in a Plan dependency graph."""

    node_id: str
    description: str
    status: PlanStatus = PlanStatus.PENDING
    dependencies: tuple[str, ...] = ()
    owner: ChildHandle | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "node_id",
            _normalized_text(
                self.node_id,
                "Plan node_id",
                MAX_PLAN_NODE_ID_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "description",
            _normalized_text(
                self.description,
                "Plan description",
                MAX_PLAN_DESCRIPTION_CHARS,
            ),
        )
        if not isinstance(self.status, PlanStatus):
            raise PlanContractError("Plan status is invalid")
        if not isinstance(self.dependencies, tuple):
            raise PlanContractError("Plan dependencies must be an immutable tuple")
        normalized_dependencies = tuple(
            _normalized_text(
                dependency,
                "Plan dependency",
                MAX_PLAN_NODE_ID_CHARS,
            )
            for dependency in self.dependencies
        )
        if len(normalized_dependencies) != len(set(normalized_dependencies)):
            raise PlanContractError("Plan dependencies must be unique")
        object.__setattr__(self, "dependencies", normalized_dependencies)
        if self.owner is not None and not isinstance(self.owner, ChildHandle):
            raise PlanContractError("Plan owner must be a ChildHandle or None")
        if self.owner is not None and self.status is not PlanStatus.IN_PROGRESS:
            raise PlanContractError("Only an in-progress Plan node may have an owner")


@dataclass(frozen=True, slots=True)
class Plan:
    """Immutable dependency graph; an empty value represents no active steps."""

    nodes: tuple[PlanNode, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple):
            raise PlanContractError("Plan nodes must be an immutable tuple")
        if len(self.nodes) > MAX_PLAN_NODES:
            raise PlanContractError(
                f"Plan must contain at most {MAX_PLAN_NODES} nodes"
            )
        if any(not isinstance(node, PlanNode) for node in self.nodes):
            raise PlanContractError("Plan contains an invalid node")
        by_id = {node.node_id: node for node in self.nodes}
        if len(by_id) != len(self.nodes):
            raise PlanContractError("Plan node ids must be unique")
        for node in self.nodes:
            if node.node_id in node.dependencies:
                raise PlanContractError("A Plan node cannot depend on itself")
            unknown = set(node.dependencies) - set(by_id)
            if unknown:
                names = ", ".join(sorted(unknown))
                raise PlanContractError(
                    f"Plan node {node.node_id!r} has unknown dependencies: {names}"
                )
        ordered = _topological_nodes(self.nodes, by_id)
        if len(ordered) != len(self.nodes):
            raise PlanContractError("Plan dependencies contain a cycle")
        active_owners: set[ChildHandle | None] = set()
        for node in self.nodes:
            if node.status in {
                PlanStatus.IN_PROGRESS,
                PlanStatus.COMPLETED,
                PlanStatus.FAILED,
                PlanStatus.BLOCKED,
            } and any(
                by_id[dependency].status is not PlanStatus.COMPLETED
                for dependency in node.dependencies
            ):
                raise PlanContractError(
                    f"Plan node {node.node_id!r} advanced before its dependencies"
                )
            if node.status is PlanStatus.IN_PROGRESS:
                if node.owner in active_owners:
                    raise PlanContractError(
                        "Plan allows at most one in-progress node per owner"
                    )
                active_owners.add(node.owner)

    def node(self, node_id: str) -> PlanNode:
        """Return one node or fail with a stable contract error."""

        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise PlanContractError(f"Unknown Plan node: {node_id}")

    def is_ready(self, node_id: str) -> bool:
        """Return derived readiness for one pending node."""

        node = self.node(node_id)
        if node.status is not PlanStatus.PENDING:
            return False
        by_id = {candidate.node_id: candidate for candidate in self.nodes}
        return all(
            by_id[dependency].status is PlanStatus.COMPLETED
            for dependency in node.dependencies
        )

    @property
    def ready_node_ids(self) -> tuple[str, ...]:
        """Ready node ids in deterministic topological order."""

        return tuple(
            node.node_id
            for node in self.topological_nodes()
            if self.is_ready(node.node_id)
        )

    def topological_nodes(self) -> tuple[PlanNode, ...]:
        """Return a stable dependency-first ordering, breaking ties by node id."""

        return _topological_nodes(
            self.nodes,
            {node.node_id: node for node in self.nodes},
        )

    def transition_node(
        self,
        node_id: str,
        *,
        status: PlanStatus,
        owner: ChildHandle | None,
    ) -> "Plan":
        """Return a complete Plan value with one node's lifecycle replaced."""

        current = self.node(node_id)
        replacement = PlanNode(
            node_id=current.node_id,
            description=current.description,
            status=status,
            dependencies=current.dependencies,
            owner=owner,
        )
        return Plan(
            tuple(
                replacement if node.node_id == node_id else node
                for node in self.nodes
            )
        )


def _topological_nodes(
    nodes: Sequence[PlanNode],
    by_id: Mapping[str, PlanNode],
) -> tuple[PlanNode, ...]:
    dependents: dict[str, list[str]] = {node_id: [] for node_id in by_id}
    remaining = {node.node_id: len(node.dependencies) for node in nodes}
    for node in nodes:
        for dependency in node.dependencies:
            dependents[dependency].append(node.node_id)
    ready = sorted(node_id for node_id, count in remaining.items() if count == 0)
    ordered: list[PlanNode] = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(by_id[node_id])
        for dependent in sorted(dependents[node_id]):
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                ready.append(dependent)
                ready.sort()
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class PlanUpdate:
    plan: Plan
    explanation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, Plan):
            raise PlanContractError("Plan update value is invalid")
        if self.explanation is not None:
            object.__setattr__(
                self,
                "explanation",
                _normalized_text(
                    self.explanation,
                    "Plan explanation",
                    MAX_PLAN_EXPLANATION_CHARS,
                    single_line=False,
                ),
            )


def _parse_owner(value: object) -> ChildHandle | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PlanContractError("Plan owner must be an object or null")
    try:
        return ChildHandle.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise PlanContractError("Plan owner is not a valid ChildHandle") from exc


def parse_plan_update(arguments: Mapping[str, object]) -> PlanUpdate:
    """Parse the strict model-facing whole-graph replacement shape."""

    unknown = set(arguments) - {"plan", "explanation"}
    if unknown:
        fields = ", ".join(sorted(unknown))
        raise PlanContractError(f"Unknown Plan fields: {fields}")
    raw_plan = arguments.get("plan")
    if not isinstance(raw_plan, list):
        raise PlanContractError("Plan must be a list")
    nodes: list[PlanNode] = []
    required = {"node_id", "description", "status", "dependencies"}
    for raw_node in raw_plan:
        if not isinstance(raw_node, Mapping):
            raise PlanContractError("Each Plan node must be an object")
        if not required.issubset(raw_node) or not set(raw_node).issubset(
            required | {"owner"}
        ):
            raise PlanContractError(
                "Each Plan node requires node_id, description, status, and dependencies"
            )
        raw_dependencies = raw_node["dependencies"]
        if not isinstance(raw_dependencies, list) or any(
            not isinstance(dependency, str) for dependency in raw_dependencies
        ):
            raise PlanContractError("Plan dependencies must be a list of text ids")
        try:
            status = PlanStatus(str(raw_node["status"]))
        except ValueError as exc:
            raise PlanContractError(
                f"Invalid Plan status: {raw_node['status']}"
            ) from exc
        nodes.append(
            PlanNode(
                node_id=raw_node["node_id"],
                description=raw_node["description"],
                status=status,
                dependencies=tuple(raw_dependencies),
                owner=_parse_owner(raw_node.get("owner")),
            )
        )
    explanation = arguments.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        raise PlanContractError("Plan explanation must be text")
    return PlanUpdate(Plan(tuple(nodes)), explanation)


def plan_to_dict(plan: Plan) -> dict[str, object]:
    """Encode one Plan with an exact durable shape."""

    if not isinstance(plan, Plan):
        raise TypeError("plan must be a Plan")
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "description": node.description,
                "status": node.status.value,
                "dependencies": list(node.dependencies),
                "owner": None if node.owner is None else node.owner.to_dict(),
            }
            for node in plan.nodes
        ]
    }


def plan_from_dict(payload: Mapping[str, object]) -> Plan:
    """Decode one exact durable Plan shape."""

    if set(payload) != {"nodes"}:
        raise PlanContractError("Plan state requires only nodes")
    raw_nodes = payload["nodes"]
    durable_fields = {
        "node_id",
        "description",
        "status",
        "dependencies",
        "owner",
    }
    if not isinstance(raw_nodes, list) or any(
        not isinstance(node, Mapping) or set(node) != durable_fields
        for node in raw_nodes
    ):
        raise PlanContractError("Durable Plan nodes have invalid fields")
    return parse_plan_update({"plan": raw_nodes}).plan


_LEGAL_TRANSITIONS: Mapping[PlanStatus, frozenset[PlanStatus]] = {
    PlanStatus.PENDING: frozenset(
        {
            PlanStatus.PENDING,
            PlanStatus.IN_PROGRESS,
            PlanStatus.COMPLETED,
            PlanStatus.FAILED,
            PlanStatus.BLOCKED,
            PlanStatus.CANCELLED,
        }
    ),
    PlanStatus.IN_PROGRESS: frozenset(
        {
            PlanStatus.PENDING,
            PlanStatus.IN_PROGRESS,
            PlanStatus.COMPLETED,
            PlanStatus.FAILED,
            PlanStatus.BLOCKED,
            PlanStatus.CANCELLED,
        }
    ),
    PlanStatus.FAILED: frozenset(
        {
            PlanStatus.FAILED,
            PlanStatus.PENDING,
            PlanStatus.IN_PROGRESS,
            PlanStatus.CANCELLED,
        }
    ),
    PlanStatus.BLOCKED: frozenset(
        {
            PlanStatus.BLOCKED,
            PlanStatus.PENDING,
            PlanStatus.IN_PROGRESS,
            PlanStatus.CANCELLED,
        }
    ),
    PlanStatus.COMPLETED: frozenset({PlanStatus.COMPLETED}),
    PlanStatus.CANCELLED: frozenset({PlanStatus.CANCELLED}),
}


def validate_plan_transition(current: Plan | None, updated: Plan) -> None:
    """Validate one accepted whole-graph replacement against durable history."""

    if current is not None and not isinstance(current, Plan):
        raise TypeError("current must be a Plan or None")
    if not isinstance(updated, Plan):
        raise TypeError("updated must be a Plan")
    if current is None:
        return
    old = {node.node_id: node for node in current.nodes}
    new = {node.node_id: node for node in updated.nodes}
    removed = set(old) - set(new)
    if removed:
        names = ", ".join(sorted(removed))
        raise PlanContractError(f"Plan replacement removed existing nodes: {names}")
    for node_id, previous in old.items():
        replacement = new[node_id]
        if replacement.description != previous.description:
            raise PlanContractError(
                f"Plan node {node_id!r} cannot rewrite its description"
            )
        if replacement.dependencies != previous.dependencies:
            raise PlanContractError(
                f"Plan node {node_id!r} cannot rewrite its dependencies"
            )
        if replacement.status not in _LEGAL_TRANSITIONS[previous.status]:
            raise PlanContractError(
                f"Illegal Plan transition for {node_id!r}: "
                f"{previous.status.value} -> {replacement.status.value}"
            )
        if (
            previous.status is PlanStatus.IN_PROGRESS
            and replacement.status is PlanStatus.IN_PROGRESS
            and replacement.owner != previous.owner
        ):
            raise PlanContractError(
                f"In-progress Plan node {node_id!r} cannot change owner"
            )


def _action_name(action: object) -> str:
    name = getattr(action, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(action, Mapping):
        return str(action.get("name", ""))
    return ""


def _action_args(action: object) -> Mapping[str, object]:
    for field in ("args", "arguments"):
        value = getattr(action, field, None)
        if isinstance(value, Mapping):
            return value
    if isinstance(action, Mapping):
        raw = action.get("args") or action.get("arguments") or {}
        if isinstance(raw, Mapping):
            return raw
    return {}


def reduce_plan(
    current: Plan | None,
    actions: Sequence[object],
    results: Sequence[ToolResult],
) -> Plan | None:
    """Fold successful update_plan calls in call order with transition checks."""

    if len(actions) != len(results):
        raise ValueError("actions and results must have the same length")
    reduced = current
    for action, raw_result in zip(actions, results):
        result = ToolResult.from_value(raw_result)
        if _action_name(action) == UPDATE_PLAN_TOOL_NAME and result.is_success:
            proposed = parse_plan_update(dict(_action_args(action))).plan
            validate_plan_transition(reduced, proposed)
            reduced = proposed
    return reduced


_MARKERS = {
    PlanStatus.PENDING: "[ ]",
    PlanStatus.IN_PROGRESS: "[~]",
    PlanStatus.COMPLETED: "[x]",
    PlanStatus.FAILED: "[!]",
    PlanStatus.BLOCKED: "[-]",
    PlanStatus.CANCELLED: "[/]",
}


def render_plan_markdown(plan: Plan) -> str | None:
    """Render a deterministic TODO projection, or None for an empty Plan."""

    if not isinstance(plan, Plan):
        raise TypeError("plan must be a Plan")
    if not plan.nodes:
        return None
    graph_details = any(node.dependencies or node.owner for node in plan.nodes)
    lines = ["# TODO", ""]
    for node in plan.topological_nodes():
        label = (
            f"{node.node_id}: {node.description}"
            if graph_details
            else node.description
        )
        details: list[str] = []
        if node.dependencies:
            details.append(f"depends on: {', '.join(node.dependencies)}")
        if node.owner is not None:
            details.append(f"owner: {node.owner.child_id}")
        suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- {_MARKERS[node.status]} {label}{suffix}")
    return "\n".join(lines)


__all__ = [
    "MAX_PLAN_DESCRIPTION_CHARS",
    "MAX_PLAN_EXPLANATION_CHARS",
    "MAX_PLAN_NODE_ID_CHARS",
    "MAX_PLAN_NODES",
    "UPDATE_PLAN_TOOL_NAME",
    "Plan",
    "PlanContractError",
    "PlanNode",
    "PlanStatus",
    "PlanUpdate",
    "parse_plan_update",
    "plan_from_dict",
    "plan_to_dict",
    "reduce_plan",
    "render_plan_markdown",
    "validate_plan_transition",
]
