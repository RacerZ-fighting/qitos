from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from qitos import Action, AgentModule, Decision, Engine, Observation, StateSchema
from qitos.checkpoint import CheckpointConfig, InMemoryCheckpointStore
from qitos.core.artifact import ArtifactStoreError
from qitos.core.tool_registry import ToolRegistry
from qitos.engine import ContextConfig, RuntimeBudget
from qitos.kit import FileArtifactStore, ReActTextParser
from qitos.kit.tool.file import ReadFile
from qitos.models import Model, ModelRequest, ModelStreamEvent, ModelStreamEventType
from qitos import tool


@dataclass
class _State(StateSchema):
    pass


class _ToolModel(Model):
    qitos_harness_metadata = {
        "tool_policy": {"native_tool_call_preferred": True},
        "parser": "ReActTextParser",
        "protocol": "react_text_v1",
    }

    def __init__(self, call_id: str) -> None:
        super().__init__(model="artifact-test", temperature=None)
        self.call_id = call_id
        self.calls = 0
        self.inputs: list[list[dict[str, Any]]] = []

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        messages = request.message_dicts()
        self.inputs.append(list(messages))
        if self.calls == 0:
            self.calls += 1
            yield ModelStreamEvent(
                type=ModelStreamEventType.COMPLETED,
                tool_calls=[
                    {
                        "id": self.call_id,
                        "type": "function",
                        "function": {"name": "produce", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
            )
            return
        self.calls += 1
        yield ModelStreamEvent(text="Final Answer: complete", type=ModelStreamEventType.COMPLETED)


class _FinalModel(Model):
    def __init__(self) -> None:
        super().__init__(model="artifact-resume", temperature=None)
        self.inputs: list[list[dict[str, Any]]] = []

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        messages = request.message_dicts()
        self.inputs.append(list(messages))
        yield ModelStreamEvent(text="Final Answer: resumed", type=ModelStreamEventType.COMPLETED)


class _Agent(AgentModule[_State, Observation, Action]):
    def __init__(self, model: Model, output: str) -> None:
        registry = ToolRegistry()

        @tool(name="produce")
        def produce() -> str:
            return output

        registry.register(produce)
        super().__init__(
            llm=model,
            tool_registry=registry,
            model_parser=ReActTextParser(),
        )

    def init_state(self, task: str, **kwargs: Any) -> _State:
        _ = kwargs
        return _State(task=task, max_steps=3)

    def decide(
        self,
        state: _State,
        observation: Observation,
    ) -> Decision[Action] | None:
        _ = state, observation
        return None

    def reduce(
        self,
        state: _State,
        observation: Observation,
        decision: Decision[Action],
    ) -> _State:
        _ = observation, decision
        return state


def _tool_message(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return next(message for message in messages if message.get("role") == "tool")


@pytest.mark.asyncio
async def test_large_output_is_durable_readable_and_stable_on_resume(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = "\n".join(f"finding {index}" for index in range(1200))
    call_id = "call-artifact"
    model = _ToolModel(call_id)
    checkpoint_store = InMemoryCheckpointStore()
    artifact_store = FileArtifactStore(
        workspace / ".pentestagent" / "artifacts",
        reference_root=workspace,
    )
    context = ContextConfig(
        tool_result_max_chars=len(output) // 8,
        tool_result_batch_max_chars=len(output) // 4,
    )

    result = await Engine(
        _Agent(model, output),
        budget=RuntimeBudget(max_steps=3),
        checkpoint_store=checkpoint_store,
        artifact_store=artifact_store,
        context_config=context,
    ).arun("collect evidence")

    recorded = result.records[0].action_results[0]
    assert recorded.output == output
    assert len(recorded.artifacts) == 1
    artifact = recorded.artifacts[0]
    artifact_path = workspace / artifact.path
    assert artifact_path.read_text(encoding="utf-8") == output
    assert artifact.size_bytes == len(output.encode("utf-8"))
    original_model_message = _tool_message(model.inputs[1])
    model_content = str(original_model_message["content"])
    assert artifact.path in model_content
    assert len(model_content) < len(output)
    assert len(model_content) <= context.tool_result_max_chars
    assert output[:40] in model_content
    assert output[-40:] in model_content
    assert "omitted" in model_content
    assert artifact.sha256
    assert artifact.sha256 not in model_content
    assert original_model_message["tool_call_id"] == call_id

    read_result = await ReadFile(str(workspace)).execute(
        {"path": artifact.path, "line_offset": 0, "line_count": 20}
    )
    assert read_result["status"] == "success"
    assert output.splitlines()[0] in read_result["content"]
    assert read_result["has_more"] is True

    checkpoints = await checkpoint_store.list(
        CheckpointConfig(thread_id=result.run_id)
    )
    tool_checkpoint = next(
        item
        for item in checkpoints
        if item.checkpoint.history is not None
        and not item.checkpoint.state_data.get("stop_reason")
        and any(
            message.role == "tool"
            for message in item.checkpoint.history.messages
        )
    )
    resumed_model = _FinalModel()
    resumed_artifact_root = workspace / "unused-after-resume"
    await Engine(
        _Agent(resumed_model, output),
        checkpoint_store=checkpoint_store,
        artifact_store=FileArtifactStore(resumed_artifact_root),
        context_config=ContextConfig(tool_result_max_chars=len(output) * 2),
    ).aresume_from_checkpoint(tool_checkpoint.config)

    resumed_message = _tool_message(resumed_model.inputs[0])
    assert resumed_message["content"] == original_model_message["content"]
    assert not resumed_artifact_root.exists()
    await checkpoint_store.close()


@pytest.mark.asyncio
async def test_short_output_stays_inline_without_artifact(tmp_path: Path) -> None:
    output = "short result"
    model = _ToolModel("short-call")
    result = await Engine(
        _Agent(model, output),
        artifact_store=FileArtifactStore(tmp_path / "artifacts"),
        context_config=ContextConfig(tool_result_max_chars=len(output) * 4),
    ).arun("small output")

    recorded = result.records[0].action_results[0]
    assert recorded.output == output
    assert recorded.artifacts == ()
    assert _tool_message(model.inputs[1])["content"] == output
    assert not (tmp_path / "artifacts").exists()


class _FailingArtifactStore:
    def write_text(
        self,
        *,
        artifact_id: str,
        content: str,
        media_type: str,
    ) -> Any:
        _ = artifact_id, content, media_type
        raise ArtifactStoreError("storage unavailable")


@pytest.mark.asyncio
async def test_persistence_failure_keeps_canonical_terminal_result() -> None:
    output = "evidence\n" * 1000
    model = _ToolModel("failed-artifact-call")
    result = await Engine(
        _Agent(model, output),
        artifact_store=_FailingArtifactStore(),
        context_config=ContextConfig(tool_result_max_chars=len(output) // 8),
    ).arun("preserve evidence")

    recorded = result.records[0].action_results[0]
    assert recorded.output == output
    assert recorded.artifacts == ()
    assert recorded.status == "success"
    assert len(str(_tool_message(model.inputs[1])["content"])) < len(output)
    assert any(
        event.payload.get("stage") == "artifact_persist_failed"
        for event in result.events
    )


@pytest.mark.asyncio
async def test_parallel_results_share_one_strict_model_budget(
    tmp_path: Path,
) -> None:
    class BatchModel(Model):
        qitos_harness_metadata = _ToolModel.qitos_harness_metadata

        def __init__(self) -> None:
            super().__init__(model="artifact-batch", temperature=None)
            self.inputs: list[list[dict[str, Any]]] = []

        async def stream(
            self,
            request: ModelRequest,
        ) -> AsyncIterator[ModelStreamEvent]:
            self.inputs.append(request.message_dicts())
            if len(self.inputs) == 1:
                yield ModelStreamEvent(
                    type=ModelStreamEventType.COMPLETED,
                    tool_calls=[
                        {
                            "id": f"batch-{label}",
                            "type": "function",
                            "function": {
                                "name": "produce_named",
                                "arguments": f'{{"label":"{label}"}}',
                            },
                        }
                        for label in ("alpha", "bravo", "charlie")
                    ],
                    finish_reason="tool_calls",
                )
                return
            yield ModelStreamEvent(
                text="Final Answer: complete",
                type=ModelStreamEventType.COMPLETED,
            )

    class BatchAgent(AgentModule[_State, Observation, Action]):
        def __init__(self, model: Model) -> None:
            registry = ToolRegistry()

            @tool(name="produce_named")
            def produce_named(label: str) -> str:
                return f"{label}-start\n" + (label * 600) + f"\n{label}-end"

            registry.register(produce_named)
            super().__init__(
                llm=model,
                tool_registry=registry,
                model_parser=ReActTextParser(),
            )

        def init_state(self, task: str, **kwargs: Any) -> _State:
            _ = kwargs
            return _State(task=task, max_steps=3)

        def decide(
            self,
            state: _State,
            observation: Observation,
        ) -> Decision[Action] | None:
            _ = state, observation
            return None

        def reduce(
            self,
            state: _State,
            observation: Observation,
            decision: Decision[Action],
        ) -> _State:
            _ = observation, decision
            return state

    model = BatchModel()
    single_limit = 420
    batch_limit = 900
    result = await Engine(
        BatchAgent(model),
        artifact_store=FileArtifactStore(
            tmp_path / "artifacts",
            reference_root=tmp_path,
        ),
        context_config=ContextConfig(
            tool_result_max_chars=single_limit,
            tool_result_batch_max_chars=batch_limit,
        ),
    ).arun("collect parallel evidence")

    tool_messages = [
        message for message in model.inputs[1] if message.get("role") == "tool"
    ]
    contents = [str(message.get("content") or "") for message in tool_messages]
    assert len(tool_messages) == 3
    assert all(0 < len(content) <= single_limit for content in contents)
    assert sum(map(len, contents)) <= batch_limit
    assert all("Preview (head/tail)" in content for content in contents)
    assert all("omitted" in content for content in contents)
    assert all(len(item.artifacts) == 1 for item in result.records[0].action_results)
