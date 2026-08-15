from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from qitos import Action, AgentModule, ContextSnapshot, Decision, Engine, StateSchema
from qitos import ToolRegistry, tool
from qitos.core.journal import JournalRecordType
from qitos.core.message_builder import MessageBuildRequest, MessageBuildResult
from qitos.core.model_capabilities import ModelAPI, ModelCapabilities
from qitos.core.tool_result import ToolResult
from qitos.kit.journal import JsonlSessionJournal
from qitos.models import Model, ModelRequest, ModelStreamEvent, ModelStreamEventType


@dataclass
class _State(StateSchema):
    revision: int = 0


class _SnapshotBuilder:
    def build_messages(self, request: MessageBuildRequest) -> MessageBuildResult:
        messages = [{"role": "system", "content": "stable system"}]
        messages.extend(
            dict(message)
            for message in request.history
            if message.get("role") != "system"
        )
        history_entries: list[dict[str, Any]] = []
        if not any(
            isinstance(message.get("_metadata"), Mapping)
            and message["_metadata"].get("source") == "task"
            for message in request.history
        ):
            messages.append({"role": "user", "content": request.state.task})
            history_entries.append(
                {
                    "role": "user",
                    "content": request.state.task,
                    "step_id": request.step_id,
                    "metadata": {"source": "task"},
                }
            )
        revision = str(request.state.revision)
        return MessageBuildResult(
            messages=messages,
            history_entries=history_entries,
            context_snapshot=ContextSnapshot(
                revision=revision,
                content=f"state revision {revision}",
            ),
        )


class _SequenceModel(Model):
    def __init__(self, *, call_tool: bool) -> None:
        super().__init__(model="context-snapshot", temperature=None)
        self.call_tool = call_tool
        self.requests: list[ModelRequest] = []
        self.qitos_harness_metadata = {
            "tool_policy": {"native_tool_call_preferred": True},
            "protocol": "react_text_v1",
        }

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            api=ModelAPI.RESPONSES,
            native_tool_calls=True,
        )

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if self.call_tool:
            self.call_tool = False
            yield ModelStreamEvent(
                type=ModelStreamEventType.COMPLETED,
                tool_calls=[
                    {
                        "id": "advance-call",
                        "type": "function",
                        "function": {"name": "advance", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
            )
            return
        yield ModelStreamEvent(
            type=ModelStreamEventType.COMPLETED,
            text="Final Answer: complete",
            finish_reason="stop",
        )


class _Agent(AgentModule[_State, dict[str, Any], Action]):
    def __init__(self, model: Model) -> None:
        registry = ToolRegistry()

        @tool(name="advance")
        def advance() -> str:
            return "advanced"

        registry.register(advance)
        super().__init__(tool_registry=registry, llm=model)
        self.message_builder = _SnapshotBuilder()

    def init_state(self, task: str, **kwargs: Any) -> _State:
        _ = kwargs
        return _State(task=task, max_steps=4)

    def decide(
        self,
        state: _State,
        observation: dict[str, Any],
    ) -> Decision[Action] | None:
        _ = state, observation
        return None

    def reduce_action_result(
        self,
        state: _State,
        action: Action,
        result: ToolResult,
        *,
        step_id: int,
    ) -> _State:
        _ = action, result, step_id
        state.revision += 1
        return state

    def reduce(
        self,
        state: _State,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> _State:
        _ = observation, decision
        return state


@pytest.mark.asyncio
async def test_context_snapshots_are_append_only_and_fork_from_committed_state(
    tmp_path: Path,
) -> None:
    model = _SequenceModel(call_tool=True)
    journal = JsonlSessionJournal(tmp_path)
    completed = await Engine(_Agent(model), journal=journal).arun("inspect")

    assert completed.state.final_result == "complete"
    assert model.requests[0].message_dicts() == model.requests[1].message_dicts()[:3]
    assert model.requests[0].message_dicts()[-1]["content"] == "state revision 0"
    assert model.requests[1].message_dicts()[-1]["content"] == "state revision 1"
    tool_message = next(
        message
        for message in model.requests[1].message_dicts()
        if message.get("role") == "tool"
    )
    assert tool_message["content"] == "advanced"

    records = await journal.replay()
    first_commit = next(
        record
        for record in records
        if record.type is JournalRecordType.STEP_COMMITTED
    )
    first_snapshots = [
        item
        for record in records[: first_commit.position.seq]
        for item in record.payload.get("history_append", [])
        if item.get("metadata", {}).get("source") == "context_snapshot"
    ]
    assert [item["metadata"]["revision"] for item in first_snapshots] == ["0"]

    fork_owner = Engine(
        _Agent(_SequenceModel(call_tool=False)),
        journal=JsonlSessionJournal(tmp_path),
    )
    fork = await fork_owner.afork_journal(
        completed.run_id,
        first_commit.position,
        new_run_id="snapshot-fork",
    )
    fork_model = _SequenceModel(call_tool=False)
    resumed = await Engine(
        _Agent(fork_model),
        journal=fork,
    ).aresume_from_journal(fork.run_id)

    assert resumed.state.revision == 1
    assert len(fork_model.requests) == 1
    contents = [
        message.get("content") for message in fork_model.requests[0].message_dicts()
    ]
    assert contents.count("state revision 0") == 1
    assert contents.count("state revision 1") == 1
