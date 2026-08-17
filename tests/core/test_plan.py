from __future__ import annotations

from types import SimpleNamespace

import pytest

from qitos.core import (
    ChildHandle,
    Plan,
    PlanContractError,
    PlanNode,
    PlanStatus,
    ToolResult,
    parse_plan_update,
    plan_from_dict,
    plan_to_dict,
    reduce_plan,
    render_plan_markdown,
    validate_plan_transition,
)


def _handle(child_id: str) -> ChildHandle:
    return ChildHandle(child_id=child_id, parent_run_id="parent-run")


def _graph() -> Plan:
    return Plan(
        (
            PlanNode("discover", "Discover services", PlanStatus.COMPLETED),
            PlanNode(
                "validate",
                "Validate the result",
                PlanStatus.IN_PROGRESS,
                dependencies=("discover",),
            ),
            PlanNode(
                "report",
                "Report evidence",
                dependencies=("validate",),
            ),
        )
    )


def test_plan_round_trips_with_derived_readiness() -> None:
    plan = Plan(
        (
            PlanNode("a", "First", PlanStatus.COMPLETED),
            PlanNode("b", "Second", dependencies=("a",)),
        )
    )

    restored = plan_from_dict(plan_to_dict(plan))

    assert restored == plan
    assert restored.ready_node_ids == ("b",)
    assert restored.node("b").status is PlanStatus.PENDING


def test_durable_plan_codec_rejects_missing_node_fields() -> None:
    payload = plan_to_dict(Plan((PlanNode("a", "First"),)))
    raw_nodes = payload["nodes"]
    assert isinstance(raw_nodes, list)
    del raw_nodes[0]["owner"]

    with pytest.raises(PlanContractError):
        plan_from_dict(payload)


@pytest.mark.parametrize(
    "nodes",
    [
        (
            PlanNode("same", "First"),
            PlanNode("same", "Second"),
        ),
        (
            PlanNode("a", "First", dependencies=("missing",)),
        ),
        (
            PlanNode("a", "First", dependencies=("b",)),
            PlanNode("b", "Second", dependencies=("a",)),
        ),
    ],
)
def test_plan_rejects_invalid_graphs(nodes: tuple[PlanNode, ...]) -> None:
    with pytest.raises(PlanContractError):
        Plan(nodes)


def test_plan_allows_independent_owners_but_only_one_node_per_owner() -> None:
    first = _handle("child-a")
    second = _handle("child-b")

    plan = Plan(
        (
            PlanNode("a", "First", PlanStatus.IN_PROGRESS, owner=first),
            PlanNode("b", "Second", PlanStatus.IN_PROGRESS, owner=second),
        )
    )

    assert {node.owner for node in plan.nodes} == {first, second}
    with pytest.raises(PlanContractError):
        Plan(
            (
                PlanNode("a", "First", PlanStatus.IN_PROGRESS, owner=first),
                PlanNode("b", "Second", PlanStatus.IN_PROGRESS, owner=first),
            )
        )


def test_plan_rejects_advancing_a_node_before_dependencies_complete() -> None:
    with pytest.raises(PlanContractError):
        Plan(
            (
                PlanNode("a", "First"),
                PlanNode(
                    "b",
                    "Second",
                    PlanStatus.IN_PROGRESS,
                    dependencies=("a",),
                ),
            )
        )


def test_plan_replacement_preserves_node_history_and_owner() -> None:
    owner = _handle("child-a")
    current = Plan(
        (PlanNode("a", "First", PlanStatus.IN_PROGRESS, owner=owner),)
    )

    validate_plan_transition(
        current,
        Plan((PlanNode("a", "First", PlanStatus.COMPLETED),)),
    )
    with pytest.raises(PlanContractError):
        validate_plan_transition(current, Plan())
    with pytest.raises(PlanContractError):
        validate_plan_transition(
            current,
            Plan(
                (
                    PlanNode(
                        "a",
                        "First",
                        PlanStatus.IN_PROGRESS,
                        owner=_handle("child-b"),
                    ),
                )
            ),
        )


def test_plan_reducer_applies_only_successful_updates_in_call_order() -> None:
    first = {
        "plan": [
            {
                "node_id": "first",
                "description": "First",
                "status": "pending",
                "dependencies": [],
            }
        ]
    }
    last = {
        "plan": [
            {
                "node_id": "first",
                "description": "First",
                "status": "completed",
                "dependencies": [],
            }
        ]
    }

    reduced = reduce_plan(
        None,
        [
            SimpleNamespace(name="update_plan", args=first),
            SimpleNamespace(name="update_plan", args=last),
        ],
        [ToolResult(output={}), ToolResult(output={})],
    )

    assert reduced == parse_plan_update(last).plan


def test_plan_markdown_is_topological_and_flat_plan_is_plain_todo() -> None:
    graph = _graph()
    rendered = render_plan_markdown(graph)

    assert rendered is not None
    assert rendered.index("discover") < rendered.index("validate")
    assert rendered.index("validate") < rendered.index("report")

    flat = Plan((PlanNode("only", "Do the work"),))
    flat_rendered = render_plan_markdown(flat)
    assert flat_rendered is not None
    assert "Do the work" in flat_rendered
    assert "only:" not in flat_rendered
    assert render_plan_markdown(Plan()) is None
