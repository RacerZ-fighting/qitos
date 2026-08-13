"""Concrete update_plan tool for the canonical WorkPlan contract."""

from __future__ import annotations

from typing import Any, Dict, Optional

from qitos.core.tool import BaseTool, ToolSpec
from qitos.core.work_plan import (
    MAX_WORK_PLAN_EXPLANATION_CHARS,
    MAX_WORK_PLAN_ITEMS,
    MAX_WORK_PLAN_STEP_CHARS,
    UPDATE_PLAN_TOOL_NAME,
    parse_work_plan_update,
)


class UpdateWorkPlanTool(BaseTool):
    """Validate and normalize a complete WorkPlan replacement."""

    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name=UPDATE_PLAN_TOOL_NAME,
                description=(
                    "Replace the current ordered work plan. Keep completed work "
                    "when revising the remaining steps."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "array",
                            "maxItems": MAX_WORK_PLAN_ITEMS,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": {
                                        "type": "string",
                                        "maxLength": MAX_WORK_PLAN_STEP_CHARS,
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
                            "maxLength": MAX_WORK_PLAN_EXPLANATION_CHARS,
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
        _ = runtime_context
        update = parse_work_plan_update(args)
        return {
            "plan": [
                {"step": item.step, "status": item.status.value}
                for item in update.plan.items
            ],
            "explanation": update.explanation,
        }


__all__ = ["UpdateWorkPlanTool"]
