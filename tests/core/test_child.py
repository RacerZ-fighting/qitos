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
        elapsed_seconds=1.25,
    )

    restored = ChildResult.from_dict(result.to_dict())

    assert restored == result
    assert restored.ready is True
    assert restored.succeeded is False
    assert restored.handle.parent_run_id == "parent-1"


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
