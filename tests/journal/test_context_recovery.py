from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from qitos import Action, AgentModule, Engine, StateSchema, ToolRegistry, tool
from qitos.core import (
    JournalRecordType,
    ModelAPI,
    ModelCapabilities,
    ModelContinuation,
)
from qitos.core.history import HistoryPolicy
from qitos.engine import RuntimeBudget
from qitos.kit.history import CompactConfig, CompactHistory
from qitos.kit.journal import JsonlSessionJournal
from qitos.models import Model, ModelRequest, ModelStreamEvent, ModelStreamEventType


@dataclass
class _ContextState(StateSchema):
    pass


class _ContextAgent(AgentModule[_ContextState, dict[str, Any], Action]):
    def __init__(self, model: Model) -> None:
        registry = ToolRegistry()

        @tool(name="inspect")
        def inspect(label: str) -> str:
            return f"{label}:" + ("x" * 120)

        registry.register(inspect)
        super().__init__(tool_registry=registry)
        self.llm = model
        self.history = CompactHistory(
            config=CompactConfig(
                max_tokens=90,
                keep_last_rounds=1,
                compact_long_messages_over_chars=60,
                microcompact_preview_chars=20,
                summary_max_chars=240,
            )
        )

    def init_state(self, task: str, **kwargs: Any) -> _ContextState:
        _ = kwargs
        return _ContextState(task=task, max_steps=6)

    def decide(
        self,
        state: _ContextState,
        observation: dict[str, Any],
    ) -> None:
        _ = state, observation
        return None

    def reduce(
        self,
        state: _ContextState,
        observation: dict[str, Any],
        decision: Any,
    ) -> _ContextState:
        _ = observation, decision
        return state


class _InterruptibleContinuationModel(Model):
    def __init__(self) -> None:
        super().__init__(model="context-model", temperature=None)
        self.requests: list[ModelRequest] = []
        self.blocked = asyncio.Event()
        self.qitos_harness_metadata = {
            "tool_policy": {"native_tool_call_preferred": True},
            "protocol": "react_text_v1",
        }

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            api=ModelAPI.RESPONSES,
            native_tool_calls=True,
            continuation=True,
        )

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index >= 4:
            self.blocked.set()
            await asyncio.Event().wait()
            return
        call_id = f"call-{index}"
        label = f"evidence-{index}"
        yield ModelStreamEvent(
            type=ModelStreamEventType.COMPLETED,
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "inspect",
                        "arguments": f'{{"label":"{label}"}}',
                    },
                }
            ],
            finish_reason="tool_calls",
            continuation=ModelContinuation(
                run_id=request.run_id,
                provider=request.provider,
                model=request.model,
                protocol=request.protocol,
                response_id=f"response-{index}",
                prefix_items=len(request.messages),
                prefix_digest=request.request_digest,
                settings_digest="settings",
            ),
        )


class _FinalModel(Model):
    def __init__(self) -> None:
        super().__init__(model="context-model", temperature=None)
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
            continuation=True,
        )

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent(
            type=ModelStreamEventType.COMPLETED,
            text="recovered",
            finish_reason="stop",
        )


def _has_compaction_audit(events: Iterable[Any]) -> bool:
    for event in events:
        payload = getattr(event, "payload", None)
        if not isinstance(payload, Mapping) or payload.get("stage") != "model_input":
            continue
        metadata = payload.get("history_messages_meta")
        if isinstance(metadata, Sequence) and any(
            isinstance(item, Mapping)
            and bool(item.get("summary") or item.get("compacted"))
            for item in metadata
        ):
            return True
    return False


def _projected_tool_call_ids(request: ModelRequest) -> set[str]:
    return {
        str(call.get("id"))
        for message in request.messages
        for call in (
            message.get("tool_calls", ())
            if isinstance(message.get("tool_calls"), Sequence)
            else ()
        )
        if isinstance(call, Mapping) and call.get("id")
    }


@pytest.mark.asyncio
async def test_compaction_resume_and_fork_preserve_canonical_transactions(
    tmp_path: Path,
) -> None:
    journal_root = tmp_path / "journal"
    original_model = _InterruptibleContinuationModel()
    original_journal = JsonlSessionJournal(journal_root)
    original_engine = Engine(
        _ContextAgent(original_model),
        journal=original_journal,
        history_policy=HistoryPolicy(max_messages=100, max_tokens=90),
        budget=RuntimeBudget(max_steps=6),
    )

    running = asyncio.create_task(original_engine.arun("collect durable evidence"))
    await asyncio.wait_for(original_model.blocked.wait(), timeout=2)
    original_engine.cancel()
    interrupted = await asyncio.wait_for(running, timeout=3)

    assert interrupted.state.stop_reason == "cancelled_immediate"
    assert _has_compaction_audit(interrupted.events)
    assert _projected_tool_call_ids(original_model.requests[-1]) == {"call-3"}
    assert original_model.requests[-1].continuation is not None
    assert original_model.requests[-1].continuation.response_id == "response-3"

    reader = JsonlSessionJournal(journal_root)
    await reader.open(interrupted.run_id)
    records = await reader.replay()
    await reader.close()
    commits = [
        record
        for record in records
        if record.type is JournalRecordType.STEP_COMMITTED
    ]
    assert len(commits) == 4
    canonical_history = [
        item
        for record in records
        for item in record.payload.get("history_append", [])
    ]
    calls = {
        str(call.get("id"))
        for message in canonical_history
        for call in message.get("tool_calls", [])
    }
    results = {
        str(message.get("tool_call_id"))
        for message in canonical_history
        if message.get("tool_call_id")
    }
    assert calls == results == {"call-0", "call-1", "call-2", "call-3"}
    assert not any(
        bool(message.get("metadata", {}).get("summary"))
        for message in canonical_history
    )

    fork_engine = Engine(
        _ContextAgent(_FinalModel()),
        journal=JsonlSessionJournal(journal_root),
    )
    fork_journal = await fork_engine.afork_journal(
        interrupted.run_id,
        commits[-1].position,
        new_run_id="context-fork",
    )
    fork_model = _FinalModel()
    fork_agent = _ContextAgent(fork_model)
    forked = await Engine(
        fork_agent,
        journal=fork_journal,
        history_policy=HistoryPolicy(max_messages=100, max_tokens=90),
    ).aresume_from_journal(fork_journal.run_id)

    assert forked.state.final_result == "recovered"
    assert fork_model.requests[0].continuation is None
    assert _has_compaction_audit(forked.events)
    assert _projected_tool_call_ids(fork_model.requests[0]) == {"call-3"}
    assert len(fork_agent.history.messages) > len(fork_model.requests[0].messages)

    resumed_model = _FinalModel()
    resumed_agent = _ContextAgent(resumed_model)
    resumed = await Engine(
        resumed_agent,
        journal=JsonlSessionJournal(journal_root),
        history_policy=HistoryPolicy(max_messages=100, max_tokens=90),
    ).aresume_from_journal(interrupted.run_id)

    assert resumed.state.final_result == "recovered"
    assert resumed_model.requests[0].continuation is not None
    assert resumed_model.requests[0].continuation.response_id == "response-3"
    assert _has_compaction_audit(resumed.events)
    assert _projected_tool_call_ids(resumed_model.requests[0]) == {"call-3"}
    assert len(resumed_agent.history.messages) > len(resumed_model.requests[0].messages)
