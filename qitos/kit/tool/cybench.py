"""CyBench-specific utility tools."""

from __future__ import annotations

from typing import Any, Dict, Optional

from qitos.core.tool import BaseTool, ToolPermission, ToolSpec


class SubmitAnswer(BaseTool):
    """Record one answer candidate for the current CyBench objective.

    Use this tool when the agent has reached a final answer proposal for the
    active CyBench task or subtask and wants to surface it for evaluation.
    """

    def __init__(self):
        super().__init__(
            ToolSpec(
                name="submit_answer",
                description="Submit one answer candidate for current CyBench objective",
                parameters={
                    "answer": {"type": "string", "description": "candidate answer"},
                    "subtask_index": {
                        "type": "integer",
                        "description": "optional subtask index",
                    },
                },
                required=["answer"],
                permissions=ToolPermission(),
                required_ops=[],
            )
        )

    async def execute(
        self, args: Dict[str, Any], runtime_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Submit one final answer candidate for the active CyBench task.

        :param answer: Proposed answer text.
        :param subtask_index: Optional subtask index for multi-part tasks.
        :param runtime_context: Optional runtime ops injected by the engine.

        This tool records an answer proposal for evaluation; it does not grade it.
        """
        _ = runtime_context
        answer = str(args.get("answer", ""))
        raw_subtask_index = args.get("subtask_index")
        subtask_index = (
            int(raw_subtask_index) if raw_subtask_index is not None else None
        )
        return {
            "status": "success",
            "type": "answer_submission",
            "answer": answer,
            "subtask_index": subtask_index,
        }


__all__ = ["SubmitAnswer"]
