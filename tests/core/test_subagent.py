"""Behavior tests for canonical Subagent contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qitos.core.subagent import (
    AgentConclusion,
    SubagentHandle,
    SubagentLaunchRequest,
    SubagentResult,
    SubagentStatus,
)
from qitos.core.journal import JournalRecord, JournalRecordRef, JournalRecordType
from qitos.core.runtime_input import RuntimeInput
from qitos.core.task import TaskBudget, TaskReference
from qitos.core.tool import ToolPermissionContext, ToolPermissionRule


def _request() -> SubagentLaunchRequest:
    return SubagentLaunchRequest(
        task="Inspect the service",
        description="service inspection",
        name="inspector",
        context="The parent already identified port 443.",
        success_criteria=("Return target-side evidence",),
        constraints={"scope": "engagement-primary"},
        references=(
            TaskReference(kind="artifact", uri="scope://engagement/primary"),
        ),
        permission_context=ToolPermissionContext(
            default_decision="deny",
            allow_rules=(ToolPermissionRule(effect="allow", tool_family="network"),),
        ),
        profile="restricted",
        allowed_tool_groups=("network", "files", "network"),
        working_directory="workspace",
        budget=TaskBudget(
            max_steps=12,
            max_runtime_seconds=30,
            max_tokens=4_000,
            max_cost_usd=0.5,
            max_tool_concurrency=2,
            max_subagents=1,
        ),
    )


def test_subagent_launch_request_is_immutable_and_round_trips() -> None:
    request = _request()

    assert request.allowed_tool_groups == ("network", "files")
    assert request.success_criteria == ("Return target-side evidence",)
    assert request.constraints == {"scope": "engagement-primary"}
    assert request.references[0].uri == "scope://engagement/primary"
    assert request.permission_context is not None
    assert SubagentLaunchRequest.from_dict(request.to_dict()) == request
    with pytest.raises(FrozenInstanceError):
        request.task = "changed"  # type: ignore[misc]


def test_subagent_launch_request_rejects_payload_without_task_binding_fields() -> None:
    payload = _request().to_dict()
    del payload["parent_task_id"]
    del payload["plan_assignment"]
    with pytest.raises(ValueError):
        SubagentLaunchRequest.from_dict(payload)


def test_subagent_launch_request_round_trips_task_binding() -> None:
    request = SubagentLaunchRequest(
        task="Enumerate the service",
        description="enumeration subagent",
        parent_task_id="root-task",
        plan_assignment="plan-node-1",
    )
    assert SubagentLaunchRequest.from_dict(request.to_dict()) == request
    with pytest.raises(ValueError):
        SubagentLaunchRequest(
            task="x",
            description="y",
            parent_task_id=" ",
        )


def test_subagent_result_preserves_scoped_handle_and_evidence() -> None:
    result = SubagentResult(
        handle=SubagentHandle(subagent_id="subagent-1", parent_run_id="parent-1"),
        request=_request(),
        status=SubagentStatus.BLOCKED,
        conclusion=AgentConclusion(
            summary="Authentication is required.",
            evidence=(
                JournalRecordRef(run_id="subagent-run", record_id="record-7"),
            ),
            resource_refs=("subagent-run:resource:session-1",),
            failure_paths=("Anonymous access was rejected.",),
            unknowns=("No test credential was supplied.",),
            next_steps=("Provide a scoped test account.",),
        ),
        subagent_run_id="subagent-run",
        steps=3,
        total_tokens=900,
        total_cost_usd=0.75,
        usage_complete=False,
        cost_complete=True,
        elapsed_seconds=1.25,
    )

    restored = SubagentResult.from_dict(result.to_dict())

    assert restored == result
    assert restored.ready is True
    assert restored.succeeded is False
    assert restored.handle.parent_run_id == "parent-1"
    assert restored.conclusion.resource_refs == (
        "subagent-run:resource:session-1",
    )
    assert restored.total_cost_usd == pytest.approx(0.75)
    assert restored.usage_complete is False
    assert restored.cost_complete is True


def test_legacy_subagent_result_marks_usage_incomplete() -> None:
    result = SubagentResult(
        handle=SubagentHandle(subagent_id="subagent-1", parent_run_id="parent-1"),
        request=_request(),
        status=SubagentStatus.COMPLETED,
        total_tokens=900,
    )
    payload = result.to_dict()
    payload.pop("total_cost_usd")
    payload.pop("usage_complete")
    payload.pop("cost_complete")

    restored = SubagentResult.from_dict(payload)

    assert restored.total_tokens == result.total_tokens
    assert restored.total_cost_usd == 0.0
    assert restored.usage_complete is False
    assert restored.cost_complete is False


def test_legacy_child_result_decodes_to_subagent_contract() -> None:
    result = SubagentResult(
        handle=SubagentHandle(subagent_id="subagent-1", parent_run_id="parent-1"),
        request=_request(),
        status=SubagentStatus.COMPLETED,
        conclusion=AgentConclusion(summary="done"),
        subagent_run_id="subagent-run",
    )
    payload = result.to_dict()
    payload["handle"]["child_id"] = payload["handle"].pop("subagent_id")
    payload["child_run_id"] = payload.pop("subagent_run_id")
    payload["request"]["budget"]["max_children"] = payload["request"][
        "budget"
    ].pop("max_subagents")

    restored = SubagentResult.from_dict(payload)

    assert restored == result
    assert set(restored.to_dict()["handle"]) == {"subagent_id", "parent_run_id"}
    assert "subagent_run_id" in restored.to_dict()
    assert "max_subagents" in restored.request.to_dict()["budget"]


def test_legacy_child_runtime_input_decodes_to_subagent_event() -> None:
    event = RuntimeInput.from_dict(
        {
            "event_id": "child-1:terminal",
            "kind": "agent.child.completed",
            "correlation_id": "child-1",
            "source": "qitos.agent",
            "payload": {
                "handle": {
                    "child_id": "child-1",
                    "parent_run_id": "parent-1",
                },
                "child_id": "child-1",
                "child_status": "completed",
                "run_id": "child-run",
            },
        }
    )

    assert event.kind == "agent.subagent.completed"
    assert event.payload["subagent_id"] == "child-1"
    assert event.payload["subagent_status"] == "completed"
    assert event.payload["subagent_run_id"] == "child-run"
    assert event.payload["handle"] == {
        "subagent_id": "child-1",
        "parent_run_id": "parent-1",
    }


def test_legacy_child_runtime_input_rejects_conflicting_new_fields() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        RuntimeInput.from_dict(
            {
                "event_id": "child-1:terminal",
                "kind": "agent.child.completed",
                "correlation_id": "child-1",
                "source": "qitos.agent",
                "payload": {
                    "child_id": "child-1",
                    "subagent_id": "subagent-other",
                },
            }
        )


@pytest.mark.parametrize(
    ("legacy_type", "canonical_type"),
    [
        ("child.started", JournalRecordType.SUBAGENT_STARTED),
        ("child.terminal", JournalRecordType.SUBAGENT_TERMINAL),
    ],
)
def test_legacy_child_journal_type_decodes_one_way(
    legacy_type: str,
    canonical_type: JournalRecordType,
) -> None:
    record = JournalRecord.from_dict(
        {
            "schema_version": 1,
            "seq": 1,
            "record_id": "parent:legacy:1",
            "type": legacy_type,
            "run_id": "parent",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "payload": {},
        }
    )

    assert record.type is canonical_type
    assert record.to_dict()["type"] == canonical_type.value


@pytest.mark.parametrize(
    ("status", "terminal"),
    [
        (SubagentStatus.PENDING, False),
        (SubagentStatus.RUNNING, False),
        (SubagentStatus.CANCEL_REQUESTED, False),
        (SubagentStatus.COMPLETED, True),
        (SubagentStatus.BLOCKED, True),
        (SubagentStatus.BUDGET_EXHAUSTED, True),
        (SubagentStatus.FAILED, True),
        (SubagentStatus.CANCELLED, True),
        (SubagentStatus.INTERRUPTED, True),
        (SubagentStatus.UNKNOWN, True),
    ],
)
def test_subagent_status_defines_stable_terminal_states(
    status: SubagentStatus,
    terminal: bool,
) -> None:
    assert status.terminal is terminal


def test_subagent_result_rejects_inconsistent_ready_projection() -> None:
    value = SubagentResult(
        handle=SubagentHandle(subagent_id="subagent-1", parent_run_id="parent-1"),
        request=_request(),
        status=SubagentStatus.RUNNING,
    ).to_dict()
    value["ready"] = True

    with pytest.raises(ValueError, match="ready does not match status"):
        SubagentResult.from_dict(value)


def test_subagent_handle_rejects_empty_parent_scope() -> None:
    with pytest.raises(ValueError, match="parent_run_id"):
        SubagentHandle(subagent_id="subagent-1", parent_run_id="")
