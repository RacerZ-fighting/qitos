"""Goal-bearing Task: definition codecs, lifecycle invariants, transitions."""

from __future__ import annotations

from datetime import datetime

import pytest

from qitos.core.model_response import ModelUsage
from qitos.core.task import (
    Task,
    TaskBlocker,
    TaskBudget,
    TaskLifecycle,
    TaskReference,
    TaskStatus,
    validate_task_transition,
)


def _task(**kwargs: object) -> Task:
    return Task(task_id="t1", objective="solve the objective", **kwargs)  # type: ignore[arg-type]


# ── Task definition ────────────────────────────────────────────────────────


def test_task_round_trip_full_definition() -> None:
    task = _task(
        parent_task_id="root-0",
        success_criteria=("service answers", "credentials rotated"),
        constraints={"scope": "10.0.0.0/24", "window": "business-hours"},
        references=(
            TaskReference(kind="file", uri="scope.txt", description="engagement scope"),
            TaskReference(kind="url", uri="https://target.example"),
        ),
        budget=TaskBudget(max_steps=20, max_tokens=5000),
        created_at="2026-08-17T10:00:00+00:00",
        created_by_run_id="run-9",
        plan_assignment="plan-node-3",
    )
    payload = task.to_dict()
    restored = Task.from_dict(payload)
    assert restored == task
    assert restored.to_dict() == payload
    assert restored.constraints is not task.constraints


def test_task_round_trip_defaults() -> None:
    task = _task()
    assert Task.from_dict(task.to_dict()) == task
    payload = task.to_dict()
    assert payload["parent_task_id"] is None
    assert payload["plan_assignment"] is None
    assert payload["success_criteria"] == []
    assert payload["constraints"] == {}
    assert payload["references"] == []


def test_task_created_at_default_is_utc_iso() -> None:
    task = _task()
    parsed = datetime.fromisoformat(task.created_at)
    assert parsed.tzinfo is not None


@pytest.mark.parametrize(
    "field, value",
    [
        ("task_id", ""),
        ("objective", "  "),
        ("parent_task_id", ""),
        ("created_by_run_id", ""),
        ("plan_assignment", ""),
    ],
)
def test_task_rejects_empty_identity_text(field: str, value: str) -> None:
    overrides = {
        "task_id": "t1",
        "objective": "solve the objective",
        field: value,
    }
    with pytest.raises(ValueError):
        Task(**overrides)  # type: ignore[arg-type]


def test_task_rejects_naive_created_at() -> None:
    with pytest.raises(ValueError):
        _task(created_at="2026-08-17T10:00:00")
    with pytest.raises(ValueError):
        _task(created_at="not-a-timestamp")


def test_task_rejects_non_string_constraints() -> None:
    with pytest.raises(TypeError):
        _task(constraints={"scope": 42})
    with pytest.raises(TypeError):
        _task(constraints={1: "x"})


def test_task_constraints_are_frozen() -> None:
    task = _task(constraints={"scope": "lab"})
    with pytest.raises(TypeError):
        task.constraints["scope"] = "other"  # type: ignore[index]


def test_task_from_dict_fail_closed_on_key_sets() -> None:
    payload = _task().to_dict()
    missing = {key: value for key, value in payload.items() if key != "objective"}
    with pytest.raises(ValueError):
        Task.from_dict(missing)
    with pytest.raises(ValueError):
        Task.from_dict({**payload, "metadata": {}})
    with pytest.raises(ValueError):
        Task.from_dict({**payload, "resources": []})


def test_task_from_dict_fail_closed_on_value_types() -> None:
    payload = _task().to_dict()
    with pytest.raises(TypeError):
        Task.from_dict({**payload, "success_criteria": "done"})
    with pytest.raises(TypeError):
        Task.from_dict({**payload, "success_criteria": [""]})
    with pytest.raises(ValueError):
        Task.from_dict({**payload, "references": [{"kind": "file"}]})
    with pytest.raises(ValueError):
        Task.from_dict(
            {
                **payload,
                "references": [
                    {"kind": "host", "uri": "x", "description": ""}
                ],
            }
        )
    with pytest.raises(ValueError):
        Task.from_dict({**payload, "budget": {"max_steps": 3}})


def test_task_reference_codec_fail_closed() -> None:
    reference = TaskReference(kind="artifact", uri="artifact://report")
    assert TaskReference.from_dict(reference.to_dict()) == reference
    with pytest.raises(ValueError):
        TaskReference(kind="blob", uri="x")
    with pytest.raises(ValueError):
        TaskReference(kind="file", uri=" ")
    with pytest.raises(ValueError):
        TaskReference.from_dict({"kind": "file", "uri": "x"})


def test_task_budget_codec_round_trip() -> None:
    budget = TaskBudget(max_steps=5, max_cost_usd=1.5)
    assert TaskBudget.from_dict(budget.to_dict()) == budget
    with pytest.raises(ValueError):
        TaskBudget.from_dict({"max_steps": 5})


# ── Task blocker and lifecycle ─────────────────────────────────────────────


def test_task_blocker_codec_and_validation() -> None:
    blocker = TaskBlocker(awaiting="input", detail="needs authorization")
    assert TaskBlocker.from_dict(blocker.to_dict()) == blocker
    with pytest.raises(ValueError):
        TaskBlocker(awaiting="maybe", detail="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TaskBlocker(awaiting="external", detail="")
    with pytest.raises(ValueError):
        TaskBlocker.from_dict({"awaiting": "input"})


def test_lifecycle_blocker_present_exactly_while_blocked() -> None:
    blocker = TaskBlocker(awaiting="external", detail="awaiting target window")
    blocked = TaskLifecycle(status=TaskStatus.BLOCKED, blocker=blocker)
    assert TaskLifecycle.from_dict(blocked.to_dict()) == blocked
    with pytest.raises(ValueError):
        TaskLifecycle(status=TaskStatus.ACTIVE, blocker=blocker)
    with pytest.raises(ValueError):
        TaskLifecycle(status=TaskStatus.BLOCKED)
    with pytest.raises(ValueError):
        TaskLifecycle(status=TaskStatus.COMPLETED, blocker=blocker, terminal_reason="ok")


def test_lifecycle_terminal_reason_present_exactly_at_terminal() -> None:
    completed = TaskLifecycle(
        status=TaskStatus.COMPLETED, terminal_reason="criteria met"
    )
    assert TaskLifecycle.from_dict(completed.to_dict()) == completed
    with pytest.raises(ValueError):
        TaskLifecycle(status=TaskStatus.ACTIVE, terminal_reason="done")
    with pytest.raises(ValueError):
        TaskLifecycle(status=TaskStatus.FAILED)
    with pytest.raises(ValueError):
        TaskLifecycle(status=TaskStatus.CANCELLED, terminal_reason=" ")


def test_lifecycle_usage_allowed_at_any_status() -> None:
    usage = ModelUsage(total_tokens=120)
    active = TaskLifecycle(status=TaskStatus.ACTIVE, usage=usage)
    restored = TaskLifecycle.from_dict(active.to_dict())
    assert restored.usage is not None
    assert restored.usage.total_tokens == 120
    terminal = TaskLifecycle(
        status=TaskStatus.FAILED, terminal_reason="budget", usage=usage
    )
    assert TaskLifecycle.from_dict(terminal.to_dict()).usage is not None


def test_lifecycle_from_dict_fail_closed_on_key_sets() -> None:
    payload = TaskLifecycle(status=TaskStatus.ACTIVE).to_dict()
    with pytest.raises(ValueError):
        TaskLifecycle.from_dict({"status": "active"})
    with pytest.raises(ValueError):
        TaskLifecycle.from_dict({**payload, "extra": True})
    with pytest.raises(ValueError):
        TaskLifecycle.from_dict({**payload, "status": "paused"})


# ── transition legality ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "from_status, to_status",
    [
        (TaskStatus.ACTIVE, TaskStatus.BLOCKED),
        (TaskStatus.ACTIVE, TaskStatus.COMPLETED),
        (TaskStatus.ACTIVE, TaskStatus.FAILED),
        (TaskStatus.ACTIVE, TaskStatus.CANCELLED),
        (TaskStatus.BLOCKED, TaskStatus.ACTIVE),
        (TaskStatus.BLOCKED, TaskStatus.CANCELLED),
        (TaskStatus.BLOCKED, TaskStatus.COMPLETED),
    ],
)
def test_legal_transitions(from_status: TaskStatus, to_status: TaskStatus) -> None:
    validate_task_transition(from_status, to_status)


@pytest.mark.parametrize(
    "from_status, to_status",
    [
        (TaskStatus.ACTIVE, TaskStatus.ACTIVE),
        (TaskStatus.BLOCKED, TaskStatus.BLOCKED),
        (TaskStatus.COMPLETED, TaskStatus.ACTIVE),
        (TaskStatus.COMPLETED, TaskStatus.FAILED),
        (TaskStatus.FAILED, TaskStatus.BLOCKED),
        (TaskStatus.CANCELLED, TaskStatus.COMPLETED),
    ],
)
def test_illegal_transitions_raise(
    from_status: TaskStatus, to_status: TaskStatus
) -> None:
    with pytest.raises(ValueError):
        validate_task_transition(from_status, to_status)


def test_terminal_statuses_report_terminal() -> None:
    assert not TaskStatus.ACTIVE.terminal
    assert not TaskStatus.BLOCKED.terminal
    for status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
        assert status.terminal
