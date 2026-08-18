"""Journal-backed operations for the canonical dependency-aware Plan."""

from __future__ import annotations

from ..core.subagent import SubagentHandle
from ..core.journal import JournalPosition, JournalRecordType, SessionJournal
from ..core.plan import (
    Plan,
    PlanContractError,
    PlanStatus,
    validate_plan_transition,
)
from .journal.recovery import recover_session
from .journal.turn_recorder import encode_plan_updated


async def load_plan(journal: SessionJournal, task_id: str) -> Plan | None:
    """Replay and return the current Plan without mutating the journal."""

    if not isinstance(journal, SessionJournal):
        raise TypeError("journal must implement SessionJournal")
    if not isinstance(task_id, str) or not task_id.strip():
        raise PlanContractError("Plan requires a non-empty Task id")
    recovered = recover_session(await journal.replay())
    task = recovered.tasks.get(task_id)
    if task is None:
        raise PlanContractError("Plan references an unknown Task")
    if task.lifecycle.status.terminal:
        raise PlanContractError("A terminal Task cannot update its Plan")
    return recovered.plans.get(task_id)


async def _commit_plan(
    journal: SessionJournal,
    task_id: str,
    plan: Plan,
    *,
    record_id: str,
) -> JournalPosition:
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("record_id must be non-empty text")
    return await journal.append(
        JournalRecordType.PLAN_UPDATED,
        encode_plan_updated(task_id, plan),
        record_id=record_id,
    )


def _validate_model_owner_changes(current: Plan | None, proposed: Plan) -> None:
    """Keep creation of Subagent owners inside the durable launch boundary."""

    previous = (
        {} if current is None else {node.node_id: node for node in current.nodes}
    )
    for node in proposed.nodes:
        old = previous.get(node.node_id)
        if node.owner is not None and (old is None or node.owner != old.owner):
            raise PlanContractError(
                "update_plan cannot create or change a Subagent owner; "
                "launch the Subagent with this node as its assignment"
            )


async def commit_model_plan_update(
    journal: SessionJournal,
    task_id: str,
    proposed: Plan,
    *,
    record_id: str,
) -> Plan:
    """Validate and commit one model-authored whole-graph replacement."""

    current = await load_plan(journal, task_id)
    validate_plan_transition(current, proposed)
    _validate_model_owner_changes(current, proposed)
    await _commit_plan(journal, task_id, proposed, record_id=record_id)
    return proposed


async def assign_plan_node(
    journal: SessionJournal,
    node_id: str,
    owner: SubagentHandle,
    *,
    parent_task_id: str | None,
    record_id: str,
) -> Plan:
    """Reserve one ready node before ``subagent.started`` is written."""

    if not isinstance(owner, SubagentHandle):
        raise TypeError("owner must be a SubagentHandle")
    if parent_task_id is None:
        raise PlanContractError("Subagent Plan assignment requires a parent Task")
    recovered = recover_session(await journal.replay())
    parent_task = recovered.unfinished_root
    if (
        parent_task is None
        or parent_task.definition.task_id != parent_task_id
    ):
        raise PlanContractError(
            "Subagent Plan assignment does not match the unfinished parent Task"
        )
    current = recovered.plans.get(parent_task_id)
    if current is None:
        raise PlanContractError("Subagent Plan assignment requires a current Plan")
    node = current.node(node_id)
    if node.status is not PlanStatus.PENDING or not current.is_ready(node_id):
        raise PlanContractError(
            f"Plan node {node_id!r} must be pending and ready before assignment"
        )
    proposed = current.transition_node(
        node_id,
        status=PlanStatus.IN_PROGRESS,
        owner=owner,
    )
    validate_plan_transition(current, proposed)
    await _commit_plan(
        journal,
        parent_task_id,
        proposed,
        record_id=record_id,
    )
    return proposed


async def release_plan_node(
    journal: SessionJournal,
    task_id: str,
    node_id: str,
    owner: SubagentHandle,
    *,
    record_id: str,
) -> Plan:
    """Durably release an assignment whose ``subagent.started`` did not commit."""

    current = await load_plan(journal, task_id)
    if current is None:
        raise PlanContractError("Cannot release an assignment without a Plan")
    node = current.node(node_id)
    if node.status is not PlanStatus.IN_PROGRESS or node.owner != owner:
        raise PlanContractError("Plan assignment no longer belongs to this Subagent")
    proposed = current.transition_node(
        node_id,
        status=PlanStatus.PENDING,
        owner=None,
    )
    validate_plan_transition(current, proposed)
    await _commit_plan(journal, task_id, proposed, record_id=record_id)
    return proposed


__all__ = [
    "assign_plan_node",
    "commit_model_plan_update",
    "load_plan",
    "release_plan_node",
]
