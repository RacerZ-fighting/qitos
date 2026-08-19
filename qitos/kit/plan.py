"""Journal-backed operations for the canonical progress checklist."""

from __future__ import annotations

from ..core.journal import JournalRecordType, SessionJournal
from ..core.plan import Plan, PlanContractError
from .journal.recovery import recover_session
from .journal.turn_recorder import encode_plan_updated


async def load_plan(journal: SessionJournal, task_id: str) -> Plan | None:
    """Replay and return the current checklist without mutating the journal."""

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


async def commit_model_plan_update(
    journal: SessionJournal,
    task_id: str,
    proposed: Plan,
    *,
    record_id: str,
) -> Plan:
    """Commit one model-authored whole-checklist replacement."""

    if not isinstance(proposed, Plan):
        raise TypeError("proposed must be a Plan")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("record_id must be non-empty text")
    await load_plan(journal, task_id)
    await journal.append(
        JournalRecordType.PLAN_UPDATED,
        encode_plan_updated(task_id, proposed),
        record_id=record_id,
    )
    return proposed


__all__ = ["commit_model_plan_update", "load_plan"]
