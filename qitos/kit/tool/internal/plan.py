"""Concrete update_plan tool for the canonical Plan contract."""

from __future__ import annotations

from typing import Any, Dict, Optional

from qitos.core.journal import SessionJournal
from qitos.core.plan import (
    MAX_PLAN_EXPLANATION_CHARS,
    MAX_PLAN_ITEMS,
    MAX_PLAN_STEP_CHARS,
    UPDATE_PLAN_TOOL_NAME,
    PlanContractError,
    parse_plan_update,
    plan_to_dict,
)
from qitos.core.tool import BaseTool, ToolSpec
from qitos.kit.plan import commit_model_plan_update


class UpdatePlanTool(BaseTool):
    """Validate and durably commit a complete progress checklist replacement."""

    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name=UPDATE_PLAN_TOOL_NAME,
                description=(
                    "Replace the current task progress checklist. Steps may be added, "
                    "removed, rewritten, or reordered as the approach changes. Keep at "
                    "most one step in progress."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "array",
                            "maxItems": MAX_PLAN_ITEMS,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": {
                                        "type": "string",
                                        "maxLength": MAX_PLAN_STEP_CHARS,
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": [
                                            "pending",
                                            "in_progress",
                                            "completed",
                                        ],
                                    },
                                },
                                "required": ["step", "status"],
                                "additionalProperties": False,
                            },
                        },
                        "explanation": {
                            "type": "string",
                            "maxLength": MAX_PLAN_EXPLANATION_CHARS,
                        },
                    },
                    "required": ["plan"],
                    "additionalProperties": False,
                },
                read_only=False,
                concurrency_safe=False,
                needs_approval=False,
            )
        )

    async def execute(
        self,
        args: Dict[str, Any],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, object]:
        update = parse_plan_update(args)
        context = runtime_context or {}
        journal = context.get("journal")
        if not isinstance(journal, SessionJournal):
            raise PlanContractError("update_plan requires a Session Journal")
        call_id = context.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            raise PlanContractError("update_plan requires a stable Tool call id")
        task_id = context.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise PlanContractError("update_plan requires the current Task id")
        committed = await commit_model_plan_update(
            journal,
            task_id,
            update.plan,
            record_id=f"{journal.run_id}:plan:tool:{call_id}",
        )
        return {
            "plan": plan_to_dict(committed)["items"],
            "explanation": update.explanation,
        }


__all__ = ["UpdatePlanTool"]
