"""Concrete update_plan tool for the canonical Plan contract."""

from __future__ import annotations

from typing import Any, Dict, Optional

from qitos.core.journal import SessionJournal
from qitos.core.plan import (
    MAX_PLAN_DESCRIPTION_CHARS,
    MAX_PLAN_EXPLANATION_CHARS,
    MAX_PLAN_NODE_ID_CHARS,
    MAX_PLAN_NODES,
    UPDATE_PLAN_TOOL_NAME,
    PlanContractError,
    parse_plan_update,
    plan_to_dict,
)
from qitos.core.tool import BaseTool, ToolSpec
from qitos.kit.plan import commit_model_plan_update


class UpdatePlanTool(BaseTool):
    """Validate and durably commit a complete dependency-graph replacement."""

    def __init__(self) -> None:
        owner_schema = {
            "type": "object",
            "properties": {
                "subagent_id": {"type": "string"},
                "parent_run_id": {"type": "string"},
            },
            "required": ["subagent_id", "parent_run_id"],
            "additionalProperties": False,
        }
        super().__init__(
            ToolSpec(
                name=UPDATE_PLAN_TOOL_NAME,
                description=(
                    "Replace the current dependency-aware execution Plan. Keep "
                    "existing nodes when revising work; readiness is derived from "
                    "completed dependencies. Subagent owners are assigned by the Agent "
                    "launch boundary, not invented here."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "array",
                            "maxItems": MAX_PLAN_NODES,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "node_id": {
                                        "type": "string",
                                        "maxLength": MAX_PLAN_NODE_ID_CHARS,
                                    },
                                    "description": {
                                        "type": "string",
                                        "maxLength": MAX_PLAN_DESCRIPTION_CHARS,
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": [
                                            "pending",
                                            "in_progress",
                                            "completed",
                                            "failed",
                                            "blocked",
                                            "cancelled",
                                        ],
                                    },
                                    "dependencies": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "maxLength": MAX_PLAN_NODE_ID_CHARS,
                                        },
                                        "uniqueItems": True,
                                    },
                                    "owner": {
                                        "anyOf": [owner_schema, {"type": "null"}]
                                    },
                                },
                                "required": [
                                    "node_id",
                                    "description",
                                    "status",
                                    "dependencies",
                                ],
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
            "plan": plan_to_dict(committed)["nodes"],
            "explanation": update.explanation,
        }


__all__ = ["UpdatePlanTool"]
