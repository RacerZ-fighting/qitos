"""Façade transcript restore: initial messages, set_transcript, continuation seal."""

from __future__ import annotations

import asyncio

import pytest

from qitos.core.agent import Agent, AgentBusyError
from qitos.core.message import (
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from qitos.core.model_request import ModelContinuation
from qitos.core.tool_result import ToolResult

from tests.core.agent_fakes import (
    ScriptedModel,
    failed_events,
    make_hanging_model,
    text_events,
)


def _continuation(run_id: str) -> ModelContinuation:
    return ModelContinuation(
        run_id=run_id,
        provider="scripted",
        model="scripted-model",
        protocol="legacy",
        response_id="resp-1",
        prefix_items=1,
        prefix_digest="digest",
        settings_digest="settings",
    )


def _restored_messages(run_id: str) -> list:
    return [
        UserMessage(content="earlier question"),
        AssistantMessage(text="earlier answer", continuation=_continuation(run_id)),
    ]


def test_initial_messages_seed_the_transcript() -> None:
    seeded = _restored_messages("run-seed")
    agent = Agent(
        model=ScriptedModel([text_events("next")]),
        initial_messages=seeded,
    )
    assert agent.messages == tuple(seeded)


def test_initial_messages_validate_typed_messages() -> None:
    with pytest.raises(TypeError):
        Agent(
            model=ScriptedModel([text_events("next")]),
            initial_messages=[{"role": "user", "content": "not typed"}],
        )


@pytest.mark.asyncio
async def test_set_transcript_replaces_between_runs() -> None:
    model = ScriptedModel([text_events("first"), text_events("second")])
    agent = Agent(model=model)
    result = await agent.prompt("hello")
    assert result.status.value == "completed"
    assert len(agent.messages) == 2

    replacement = _restored_messages("run-other")
    agent.set_transcript(replacement)
    assert agent.messages == tuple(replacement)

    result = await agent.prompt("again")
    assert result.status.value == "completed"
    assert agent.messages[:2] == tuple(replacement)


@pytest.mark.asyncio
async def test_set_transcript_rejects_a_busy_run() -> None:
    gate = asyncio.Event()
    model = ScriptedModel([make_hanging_model(gate, first_text="working")])
    agent = Agent(model=model)
    running = asyncio.create_task(agent.prompt("hello"))
    await asyncio.wait_for(_wait_until_streaming(agent), timeout=1)
    with pytest.raises(AgentBusyError):
        agent.set_transcript(_restored_messages("run-busy"))
    gate.set()
    await running
    # The rejection left the in-flight transcript untouched.
    assert any(
        isinstance(message, UserMessage) and message.content == "hello"
        for message in agent.messages
    )


async def _wait_until_streaming(agent: Agent) -> None:
    while not agent.is_streaming:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_set_transcript_clears_error_projection() -> None:
    model = ScriptedModel([failed_events("provider exploded"), text_events("ok")])
    agent = Agent(model=model)
    result = await agent.prompt("boom")
    assert result.status.value == "failed"
    assert agent.error_message is not None
    agent.set_transcript(_restored_messages("run-clear"))
    assert agent.error_message is None
    assert agent.streaming_message is None
    assert agent.pending_tool_call_ids == frozenset()


def test_set_transcript_validates_typed_messages() -> None:
    agent = Agent(model=ScriptedModel([text_events("ok")]))
    with pytest.raises(TypeError):
        agent.set_transcript(["not a message"])


@pytest.mark.asyncio
async def test_seeded_continuation_chains_without_a_transcript_swap() -> None:
    # A restore that keeps the transcript intact (resume) may still offer
    # the seeded assistant's Provider continuation; the identity and digest
    # guards own correctness there.
    run_id = "run-continuation-kept"
    model = ScriptedModel([text_events("answer")])
    agent = Agent(
        model=model,
        initial_messages=_restored_messages(run_id),
        run_id_factory=lambda: run_id,
    )
    result = await agent.prompt("continue")
    assert result.status.value == "completed"
    assert model.requests[0].continuation == _continuation(run_id)


@pytest.mark.asyncio
async def test_set_transcript_seals_seeded_continuations() -> None:
    # After a wholesale swap (compaction), the next request is always a full
    # request; only assistant messages produced afterwards may chain again.
    run_id = "run-continuation-sealed"
    model = ScriptedModel([text_events("one"), text_events("two")])
    agent = Agent(model=model, run_id_factory=lambda: run_id)
    agent.set_transcript(_restored_messages(run_id))
    result = await agent.prompt("full request please")
    assert result.status.value == "completed"
    assert model.requests[0].continuation is None


@pytest.mark.asyncio
async def test_reset_clears_transcript_and_seal() -> None:
    run_id = "run-reset"
    model = ScriptedModel([text_events("answer")])
    agent = Agent(model=model, run_id_factory=lambda: run_id)
    agent.set_transcript(_restored_messages(run_id))
    agent.reset()
    assert agent.messages == ()
    result = await agent.prompt("fresh")
    assert result.status.value == "completed"
    assert model.requests[0].continuation is None
