"""Trace reattachment: producer artifacts, schema validation, qita discovery."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from qitos.core.agent_loop import AgentRunStatus
from qitos.core.tool import tool
from qitos.core.tool_registry import ToolRegistry
from qitos.kit.journal import InMemoryJournalStore
from qitos.kit.session import SessionHarness
from qitos.qita._cli_app import _discover_runs
from qitos.trace.schema import TraceSchemaValidator

from tests.core.agent_fakes import (
    ScriptedModel,
    failed_events,
    make_hanging_model,
    text_events,
    tool_call_wire,
    tool_events,
)


@tool(name="echo")
def _echo(text: str) -> str:
    return f"echo:{text}"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _artifacts(trace_dir: Path, run_id: str) -> tuple[dict, list[dict], list[dict]]:
    run_dir = trace_dir / run_id
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    events = _read_jsonl(run_dir / "events.jsonl")
    steps = _read_jsonl(run_dir / "steps.jsonl")
    return manifest, events, steps


@pytest.mark.asyncio
async def test_run_produces_discoverable_valid_trace(tmp_path) -> None:
    store = InMemoryJournalStore()
    trace_dir = tmp_path / "trace"
    harness = SessionHarness(store, trace_directory=trace_dir)
    model = ScriptedModel(
        [
            tool_events([tool_call_wire("c1", "echo", {"text": "hi"})]),
            text_events("final answer", usage={"total_tokens": 120}),
        ]
    )
    session_run = await harness.start(
        model=model, tool_registry=ToolRegistry().register(_echo)
    )
    result = await session_run.prompt("hello")
    assert result.status is AgentRunStatus.COMPLETED
    run_id = session_run.run_id
    await session_run.close()

    manifest, events, steps = _artifacts(trace_dir, run_id)
    assert manifest["status"] == "completed"
    assert manifest["summary"]["stop_reason"] == "completed"
    assert manifest["summary"]["final_result"] == "final answer"
    assert manifest["model_id"] == "scripted-model"

    # Both turns committed one step each; the phases come from the loop
    # vocabulary, and tool events carry call id, name and status.
    assert [step["step_id"] for step in steps] == [0, 1]
    phases = {event["phase"] for event in events}
    assert {"agent", "input", "model", "tool", "turn"} <= phases
    tool_events_ = [event for event in events if event["phase"] == "tool"]
    assert any(
        event["payload"].get("call_id") == "c1"
        and event["payload"].get("name") == "echo"
        and event["payload"].get("status") == "success"
        for event in tool_events_
    )
    assert all(event["run_id"] == run_id for event in events)

    validator = TraceSchemaValidator()
    validator.validate_manifest(manifest)
    validator.validate_events(events)
    validator.validate_steps(steps)

    discovered = _discover_runs(trace_dir)
    assert [run["id"] for run in discovered] == [run_id]
    assert discovered[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_failed_run_finalizes_as_failed(tmp_path) -> None:
    store = InMemoryJournalStore()
    trace_dir = tmp_path / "trace"
    harness = SessionHarness(store, trace_directory=trace_dir)
    model = ScriptedModel([failed_events("provider exploded")])
    session_run = await harness.start(model=model)
    result = await session_run.prompt("boom")
    assert result.status is AgentRunStatus.FAILED
    run_id = session_run.run_id
    await session_run.close()

    manifest, events, steps = _artifacts(trace_dir, run_id)
    assert manifest["status"] == "failed"
    assert manifest["summary"]["failure_report"]
    validator = TraceSchemaValidator()
    validator.validate_manifest(manifest)
    validator.validate_events(events)
    validator.validate_steps(steps)


@pytest.mark.asyncio
async def test_aborted_run_finalizes_as_stopped(tmp_path) -> None:
    store = InMemoryJournalStore()
    trace_dir = tmp_path / "trace"
    harness = SessionHarness(store, trace_directory=trace_dir)
    gate = asyncio.Event()
    model = ScriptedModel([make_hanging_model(gate, first_text="working")])
    session_run = await harness.start(model=model)
    running = asyncio.create_task(session_run.prompt("hang"))
    while not session_run.agent.is_streaming:
        await asyncio.sleep(0)
    session_run.abort()
    result = await running
    assert result.status is AgentRunStatus.ABORTED
    run_id = session_run.run_id
    await session_run.close()
    gate.set()

    manifest, events, steps = _artifacts(trace_dir, run_id)
    assert manifest["status"] == "stopped"
    validator = TraceSchemaValidator()
    validator.validate_manifest(manifest)
    validator.validate_events(events)
    validator.validate_steps(steps)
    # The abort terminalized the open turn with an error message, and that
    # committed turn published its step with a failed marker.
    assert [step["step_id"] for step in steps] == [0]
    turn_events = [event for event in events if event["phase"] == "turn"]
    assert any(event["ok"] is False for event in turn_events)


@pytest.mark.asyncio
async def test_each_leg_traces_its_own_run_directory(tmp_path) -> None:
    store = InMemoryJournalStore()
    trace_dir = tmp_path / "trace"
    harness = SessionHarness(store, trace_directory=trace_dir)
    model = ScriptedModel([text_events("one"), text_events("two")])
    session_run = await harness.start(model=model)
    await session_run.prompt("first")
    first_run_id = session_run.run_id
    await session_run.prompt("second")
    second_run_id = session_run.run_id
    await session_run.close()

    assert first_run_id != second_run_id
    first_manifest, _, _ = _artifacts(trace_dir, first_run_id)
    second_manifest, _, _ = _artifacts(trace_dir, second_run_id)
    assert first_manifest["status"] == "completed"
    assert second_manifest["status"] == "completed"
    assert {run["id"] for run in _discover_runs(trace_dir)} == {
        first_run_id,
        second_run_id,
    }
