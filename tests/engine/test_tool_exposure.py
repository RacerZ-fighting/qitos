from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from qitos import Action, AgentModule, Decision, Engine, StateSchema, ToolRegistry, tool
from qitos.core import JournalRecordType, ToolExposure
from qitos.kit import ReActTextParser
from qitos.kit.journal import JsonlSessionJournal
from qitos.models import Model, ModelRequest, ModelStreamChunk
from qitos.protocols import ModelProtocol


def test_registry_freeze_is_revisioned_and_definition_isolated() -> None:
    registry = ToolRegistry()

    @tool(name="read", group="filesystem.read")
    def read(path: str) -> str:
        return path

    registry.register(read)
    exposure = registry.freeze(
        ["read"],
        metadata={"profile": "default", "groups": ["filesystem.read"]},
    )
    source_revision = registry.revision

    source_tool = registry.get("read")
    assert source_tool is not None
    source_tool.spec.description = "changed after freeze"

    @tool(name="late")
    def late() -> str:
        return "late"

    registry.register(late)
    projected = exposure.get_all_specs()
    projected[0]["function"]["parameters"]["properties"]["path"]["type"] = "integer"
    described = exposure.describe_tool("read")
    described["input_schema"]["properties"]["path"]["type"] = "boolean"
    isolated = exposure.get("read")
    assert isolated is not None
    isolated.spec.description = "changed through returned tool"

    assert exposure.list_tools() == ["read"]
    assert exposure.source_registry_revision == source_revision
    assert exposure.describe_tool("read")["description"] != "changed after freeze"
    assert (
        exposure.get_all_specs()[0]["function"]["parameters"]["properties"]["path"][
            "type"
        ]
        == "string"
    )
    assert exposure.audit_metadata()["selection"] == {
        "profile": "default",
        "groups": ["filesystem.read"],
    }
    with pytest.raises(TypeError, match="immutable"):
        exposure.unregister("read")
    with pytest.raises(TypeError):
        registry.freeze(metadata={"unsupported": {"set"}})


@dataclass
class _ExposureState(StateSchema):
    pass


def _api_tool_protocol() -> ModelProtocol:
    return ModelProtocol(
        id="exposure_api_tools_v1",
        display_name="Exposure API Tools",
        parser_factory=ReActTextParser,
        prompt_renderer=lambda base_prompt, _tools: str(base_prompt or ""),
        contract_renderer=lambda _protocol: "Final Answer: <answer>",
        tool_schema_renderer=lambda _registry: "",
        tool_schema_delivery="api_parameter",
    )


class _ExposureModel(Model):
    def __init__(self, registry: ToolRegistry) -> None:
        super().__init__(model="exposure-test", temperature=None)
        self.registry = registry
        self.calls = 0
        self.tool_names_by_call: list[list[str]] = []
        self.late_executions = 0
        self.qitos_harness_metadata = {
            "tool_policy": {"native_tool_call_preferred": True},
            "parser": "ReActTextParser",
            "protocol": "exposure_api_tools_v1",
        }

    def supports_tool_schema_delivery(
        self, delivery: str, protocol: Any = None
    ) -> bool:
        _ = protocol
        return delivery == "api_parameter"

    def build_tool_schema_request_options(
        self,
        tool_schema_payload: list[dict[str, Any]] | None,
        *,
        protocol: Any = None,
        delivery: str = "prompt_injection",
    ) -> dict[str, Any]:
        _ = protocol, delivery
        return {"tools": list(tool_schema_payload or [])}

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamChunk]:
        schemas = list(request.option_dict().get("tools") or [])
        self.tool_names_by_call.append(
            [str(item["function"]["name"]) for item in schemas]
        )
        call_index = self.calls
        self.calls += 1
        if call_index == 0:

            @tool(name="late")
            def late() -> str:
                self.late_executions += 1
                return "late"

            self.registry.register(late)
            yield ModelStreamChunk(
                done=True,
                tool_calls=[
                    {
                        "id": "call-late",
                        "type": "function",
                        "function": {"name": "late", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
                event_type="test.completed",
            )
            return
        yield ModelStreamChunk(
            text="Final Answer: done",
            done=True,
            finish_reason="stop",
            event_type="test.completed",
        )


class _ExposureAgent(AgentModule[_ExposureState, dict[str, Any], Action]):
    def __init__(self, registry: ToolRegistry, llm: Model) -> None:
        super().__init__(
            tool_registry=registry,
            llm=llm,
            model_parser=ReActTextParser(),
            model_protocol=_api_tool_protocol(),
        )

    def init_state(self, task: str, **kwargs: Any) -> _ExposureState:
        _ = kwargs
        return _ExposureState(task=task, max_steps=3)

    def reduce(
        self,
        state: _ExposureState,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> _ExposureState:
        _ = observation
        if decision.mode == "final":
            state.final_result = decision.final_answer
        return state


@pytest.mark.asyncio
async def test_engine_uses_one_exposure_for_model_dispatch_and_journal(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()

    @tool(name="initial")
    def initial() -> str:
        return "initial"

    registry.register(initial)
    model = _ExposureModel(registry)
    journal = JsonlSessionJournal(tmp_path)

    result = await Engine(
        agent=_ExposureAgent(registry, model),
        journal=journal,
    ).arun("exercise frozen exposure")

    assert model.tool_names_by_call == [["initial"], ["initial", "late"]]
    assert model.late_executions == 0
    late_result = result.records[0].action_results[0]
    assert late_result.metadata["error_category"] == "tool_not_found"
    assert late_result.metadata["executed"] is False

    completed = [
        item
        for item in await journal.replay()
        if item.type is JournalRecordType.MODEL_COMPLETED
    ]
    first = completed[0].payload["prompt_metadata"]["tool_exposure"]
    second = completed[1].payload["prompt_metadata"]["tool_exposure"]
    assert first["tool_names"] == ["initial"]
    assert second["tool_names"] == ["initial", "late"]
    assert first["registry_revision"] < second["registry_revision"]
    assert first["schema_sha256"] != second["schema_sha256"]
    assert result.records[0].prompt_metadata["tool_exposure"] == first

    resumed = await Engine(
        agent=_ExposureAgent(registry, model),
        journal=JsonlSessionJournal(tmp_path),
    ).aresume_from_journal(result.run_id)
    assert resumed.records[0].prompt_metadata["tool_exposure"] == first
    assert model.calls == 2
    assert isinstance(registry.freeze(), ToolExposure)
