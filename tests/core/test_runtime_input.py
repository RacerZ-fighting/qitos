"""Behavior tests for the canonical runtime-input derivations.

Every delivery endpoint steers ``payload["content"]`` and drops an input whose
text is empty, so a derived terminal notification is only real when its text
survives that projection.
"""

from __future__ import annotations

import pytest

from qitos.core.process import (
    ProcessHandle,
    ProcessOutput,
    ProcessSnapshot,
    ProcessStatus,
)
from qitos.core.runtime_input import (
    process_terminal_runtime_input,
    subagent_terminal_runtime_input,
)
from qitos.core.subagent import (
    AgentConclusion,
    SubagentHandle,
    SubagentLaunchRequest,
    SubagentResult,
    SubagentStatus,
)
from qitos.kit.session.runtime_inputs import runtime_input_text


def _subagent_result(
    *,
    status: SubagentStatus = SubagentStatus.COMPLETED,
    summary: str = "",
    error: str | None = None,
    steps: int = 7,
) -> SubagentResult:
    return SubagentResult(
        handle=SubagentHandle(subagent_id="subagent-1", parent_run_id="run-parent"),
        request=SubagentLaunchRequest(
            task="Enumerate the service",
            description="service enumeration",
            name="enumerator",
        ),
        status=status,
        conclusion=AgentConclusion(summary=summary),
        subagent_run_id="run-child",
        error=error,
        steps=steps,
    )


def _process_snapshot(
    *,
    status: ProcessStatus = ProcessStatus.EXITED,
    exit_code: int | None = 0,
    content: str = "scan finished",
    error: str | None = None,
) -> ProcessSnapshot:
    return ProcessSnapshot(
        handle=ProcessHandle(process_id="process-1", owner_run_id="run-parent"),
        status=status,
        command="scan --target service",
        cwd="/workspace",
        pid=101,
        tty=False,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:01:00Z",
        exit_code=exit_code,
        error=error,
        output=ProcessOutput(
            content=content,
            cursor=0,
            next_cursor=len(content),
            total_bytes=len(content),
            omitted_bytes=0,
            truncated=False,
            log_path=None,
        ),
    )


def test_subagent_terminal_input_survives_the_delivery_projection() -> None:
    result = _subagent_result(summary="The service exposes no writable path.")

    event = subagent_terminal_runtime_input(result)
    text = runtime_input_text(event)

    assert text
    assert result.handle.subagent_id in text
    assert result.status.value in text
    assert str(result.steps) in text
    assert result.conclusion.summary in text
    # The structured terminal projection still rides the same payload.
    assert event.payload["subagent_run_id"] == result.subagent_run_id
    assert event.payload["conclusion"] == result.conclusion.to_dict()


def test_subagent_terminal_input_reports_a_failure_without_a_summary() -> None:
    result = _subagent_result(
        status=SubagentStatus.FAILED,
        error="deadline expired before admission",
    )

    text = runtime_input_text(subagent_terminal_runtime_input(result))

    assert text
    assert result.status.value in text
    assert result.error is not None and result.error in text


def test_subagent_terminal_input_announces_a_silent_terminal() -> None:
    """A terminal with neither summary nor error still has to reach the parent."""

    text = runtime_input_text(
        subagent_terminal_runtime_input(_subagent_result(steps=0))
    )

    assert text
    assert "subagent-1" in text


def test_subagent_terminal_input_bounds_an_oversized_conclusion() -> None:
    result = _subagent_result(summary="detail " * 4_000)

    text = runtime_input_text(subagent_terminal_runtime_input(result))

    assert 0 < len(text) < len(result.conclusion.summary)
    assert text.startswith(f"Subagent {result.handle.subagent_id}")


def test_subagent_terminal_input_rejects_a_running_result() -> None:
    with pytest.raises(ValueError):
        subagent_terminal_runtime_input(
            _subagent_result(status=SubagentStatus.RUNNING)
        )


def test_process_terminal_input_survives_the_delivery_projection() -> None:
    snapshot = _process_snapshot(content="443/tcp open")

    event = process_terminal_runtime_input(snapshot)
    text = runtime_input_text(event)

    assert text
    assert snapshot.handle.process_id in text
    assert snapshot.status.value in text
    assert snapshot.output.content in text
    assert event.payload["exit_code"] == snapshot.exit_code


def test_process_terminal_input_reports_a_failure_without_output() -> None:
    snapshot = _process_snapshot(
        status=ProcessStatus.FAILED,
        exit_code=None,
        content="",
        error="the runtime refused to start the command",
    )

    text = runtime_input_text(process_terminal_runtime_input(snapshot))

    assert text
    assert snapshot.error is not None and snapshot.error in text


def test_terminal_inputs_keep_one_stable_identity_per_source() -> None:
    """Delivery is idempotent by event id, so the same terminal derives one id."""

    result = _subagent_result(summary="done")
    snapshot = _process_snapshot()

    assert subagent_terminal_runtime_input(
        result
    ).event_id == subagent_terminal_runtime_input(result).event_id
    assert process_terminal_runtime_input(
        snapshot
    ).event_id == process_terminal_runtime_input(snapshot).event_id
    assert (
        subagent_terminal_runtime_input(result).correlation_id
        == result.handle.subagent_id
    )
