"""Behavior tests for canonical child Agent contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qitos.core.child import (
    AgentConclusion,
    ChildHandle,
    ChildLaunchRequest,
    ChildResult,
    ChildStatus,
)
from qitos.core.journal import JournalRecordRef
from qitos.core.task import TaskBudget


def _request() -> ChildLaunchRequest:
    return ChildLaunchRequest(
        task="Inspect the service",
        description="service inspection",
        name="inspector",
        context="The parent already identified port 443.",
        profile="restricted",
        allowed_tool_groups=("network", "files", "network"),
        working_directory="workspace",
        budget=TaskBudget(
            max_steps=12,
            max_runtime_seconds=30,
            max_tokens=4_000,
            max_cost_usd=0.5,
            max_tool_concurrency=2,
            max_children=1,
        ),
    )


def test_child_launch_request_is_immutable_and_round_trips() -> None:
    request = _request()

    assert request.allowed_tool_groups == ("network", "files")
    assert ChildLaunchRequest.from_dict(request.to_dict()) == request
    with pytest.raises(FrozenInstanceError):
        request.task = "changed"  # type: ignore[misc]


def test_child_launch_request_rejects_payload_without_task_binding_fields() -> None:
    payload = _request().to_dict()
    del payload["parent_task_id"]
    del payload["plan_assignment"]
    with pytest.raises(ValueError):
        ChildLaunchRequest.from_dict(payload)


def test_child_launch_request_round_trips_task_binding() -> None:
    request = ChildLaunchRequest(
        task="Enumerate the service",
        description="enumeration child",
        parent_task_id="root-task",
        plan_assignment="plan-node-1",
    )
    assert ChildLaunchRequest.from_dict(request.to_dict()) == request
    with pytest.raises(ValueError):
        ChildLaunchRequest(
            task="x",
            description="y",
            parent_task_id=" ",
        )


def test_child_result_preserves_scoped_handle_and_evidence() -> None:
    result = ChildResult(
        handle=ChildHandle(child_id="child-1", parent_run_id="parent-1"),
        request=_request(),
        status=ChildStatus.BLOCKED,
        conclusion=AgentConclusion(
            summary="Authentication is required.",
            evidence=(
                JournalRecordRef(run_id="child-run", record_id="record-7"),
            ),
            failure_paths=("Anonymous access was rejected.",),
            unknowns=("No test credential was supplied.",),
            next_steps=("Provide a scoped test account.",),
        ),
        child_run_id="child-run",
        steps=3,
        total_tokens=900,
        total_cost_usd=0.75,
        usage_complete=False,
        cost_complete=True,
        elapsed_seconds=1.25,
    )

    restored = ChildResult.from_dict(result.to_dict())

    assert restored == result
    assert restored.ready is True
    assert restored.succeeded is False
    assert restored.handle.parent_run_id == "parent-1"
    assert restored.total_cost_usd == pytest.approx(0.75)
    assert restored.usage_complete is False
    assert restored.cost_complete is True


def test_legacy_child_result_marks_usage_incomplete() -> None:
    result = ChildResult(
        handle=ChildHandle(child_id="child-1", parent_run_id="parent-1"),
        request=_request(),
        status=ChildStatus.COMPLETED,
        total_tokens=900,
    )
    payload = result.to_dict()
    payload.pop("total_cost_usd")
    payload.pop("usage_complete")
    payload.pop("cost_complete")

    restored = ChildResult.from_dict(payload)

    assert restored.total_tokens == result.total_tokens
    assert restored.total_cost_usd == 0.0
    assert restored.usage_complete is False
    assert restored.cost_complete is False


@pytest.mark.parametrize(
    ("status", "terminal"),
    [
        (ChildStatus.PENDING, False),
        (ChildStatus.RUNNING, False),
        (ChildStatus.CANCEL_REQUESTED, False),
        (ChildStatus.COMPLETED, True),
        (ChildStatus.BLOCKED, True),
        (ChildStatus.BUDGET_EXHAUSTED, True),
        (ChildStatus.FAILED, True),
        (ChildStatus.CANCELLED, True),
        (ChildStatus.INTERRUPTED, True),
        (ChildStatus.UNKNOWN, True),
    ],
)
def test_child_status_defines_stable_terminal_states(
    status: ChildStatus,
    terminal: bool,
) -> None:
    assert status.terminal is terminal


def test_child_result_rejects_inconsistent_ready_projection() -> None:
    value = ChildResult(
        handle=ChildHandle(child_id="child-1", parent_run_id="parent-1"),
        request=_request(),
        status=ChildStatus.RUNNING,
    ).to_dict()
    value["ready"] = True

    with pytest.raises(ValueError, match="ready does not match status"):
        ChildResult.from_dict(value)


def test_child_handle_rejects_empty_parent_scope() -> None:
    with pytest.raises(ValueError, match="parent_run_id"):
        ChildHandle(child_id="child-1", parent_run_id="")
