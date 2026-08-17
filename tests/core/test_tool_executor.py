"""Tool batch executor conformance: terminal results, deadlines, cancellation."""

from __future__ import annotations

import asyncio
import time
from typing import List

import pytest

from qitos.core._freeze import thaw_deep
from qitos.core.agent_events import (
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
)
from qitos.core.cancellation import CancelToken
from qitos.core.message import ToolCall
from qitos.core.tool import RetryPolicy, tool
from qitos.core.tool_executor import (
    ToolBatchExecutor,
    ToolExecutionConfig,
)
from qitos.core.tool_registry import ToolRegistry
from qitos.core.tool_result import ToolResult

from .agent_fakes import RecordingTransaction


def _exposure(*items):
    return ToolRegistry().include_toolset(list(items)).freeze()


def _call(name: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args)


@tool(name="echo")
def _echo(text: str) -> str:
    return f"echo:{text}"


@pytest.mark.asyncio
async def test_success_result_carries_call_id_and_events() -> None:
    events = []
    executor = ToolBatchExecutor(
        _exposure(_echo), ToolExecutionConfig(run_id="r"), emit=events.append
    )
    results = await executor.execute_batch([_call("echo", {"text": "hi"})])

    assert len(results) == 1
    result = results[0]
    assert result.status == "success"
    assert result.output == "echo:hi"
    assert result.call_id == "c1"
    assert [event.type for event in events] == [
        "tool_execution_start",
        "tool_execution_end",
    ]
    assert isinstance(events[0], ToolExecutionStart)
    assert isinstance(events[1], ToolExecutionEnd)
    assert events[1].is_error is False


@pytest.mark.asyncio
async def test_transaction_records_started_then_terminal() -> None:
    transaction = RecordingTransaction()
    executor = ToolBatchExecutor(
        _exposure(_echo),
        ToolExecutionConfig(run_id="r", turn=3),
        transaction=transaction,
    )
    await executor.execute_batch([_call("echo", {"text": "x"}, call_id="c9")])
    kinds = [record[0] for record in transaction.records]
    assert kinds == ["tool_started", "tool_terminal"]
    assert transaction.records[0][2] == "c9"
    assert transaction.records[1][2:] == ("c9", "success")


@pytest.mark.asyncio
async def test_plain_values_and_tool_results_both_normalize() -> None:
    @tool(name="structured")
    def _structured(text: str) -> ToolResult:
        return ToolResult(status="partial", output={"rows": [1]})

    executor = ToolBatchExecutor(_exposure(_echo, _structured), ToolExecutionConfig())
    results = await executor.execute_batch(
        [_call("echo", {"text": "a"}, "c1"), _call("structured", {"text": "b"}, "c2")]
    )
    assert results[0].status == "success"
    assert results[1].status == "partial"
    # Terminal results are deeply immutable snapshots crossing the boundary.
    assert thaw_deep(results[1].output) == {"rows": [1]}
    with pytest.raises(TypeError):
        results[1].output["rows"] = []  # type: ignore[index]


@pytest.mark.asyncio
async def test_none_output_becomes_missing_result_error() -> None:
    @tool(name="nothing")
    def _nothing(text: str) -> None:
        return None

    executor = ToolBatchExecutor(_exposure(_nothing), ToolExecutionConfig())
    results = await executor.execute_batch([_call("nothing", {"text": "a"})])
    assert results[0].status == "error"
    assert results[0].metadata["error_code"] == "TOOL_RESULT_MISSING"


@pytest.mark.asyncio
async def test_unknown_tool_and_schema_violation_are_terminal_errors() -> None:
    executor = ToolBatchExecutor(_exposure(_echo), ToolExecutionConfig())
    unknown, invalid = await executor.execute_batch(
        [
            _call("ghost", {}, "c1"),
            _call("echo", {"wrong": 1}, "c2"),
        ]
    )
    assert unknown.metadata["error_category"] == "tool_not_found"
    assert invalid.metadata["error_category"] == "invalid_tool_arguments"
    assert invalid.metadata["started"] is False


@pytest.mark.asyncio
async def test_retry_policy_is_the_single_retry_owner() -> None:
    attempts = 0

    @tool(name="flaky", retry_policy=RetryPolicy(max_attempts=3, backoff_factor=0))
    def _flaky(text: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("transient")
        return "ok"

    executor = ToolBatchExecutor(_exposure(_flaky), ToolExecutionConfig())
    results = await executor.execute_batch([_call("flaky", {"text": "x"})])
    assert attempts == 3
    assert results[0].status == "success"
    assert results[0].metadata["attempts"] == 3


@pytest.mark.asyncio
async def test_non_retryable_exception_runs_exactly_once() -> None:
    attempts = 0

    @tool(
        name="strict",
        retry_policy=RetryPolicy(
            max_attempts=3, backoff_factor=0, retryable_exceptions=(KeyError,)
        ),
    )
    def _strict(text: str) -> str:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("fatal")

    executor = ToolBatchExecutor(_exposure(_strict), ToolExecutionConfig())
    results = await executor.execute_batch([_call("strict", {"text": "x"})])
    assert attempts == 1
    assert results[0].status == "error"
    assert results[0].metadata["error_category"] == "runtime_error"


@pytest.mark.asyncio
async def test_tool_timeout_produces_timed_out_terminal() -> None:
    @tool(name="slow", timeout_s=0.05)
    async def _slow(text: str) -> str:
        await asyncio.sleep(5)
        return text

    executor = ToolBatchExecutor(_exposure(_slow), ToolExecutionConfig())
    results = await executor.execute_batch([_call("slow", {"text": "x"})])
    assert results[0].status == "timed_out"
    assert results[0].metadata["error_category"] == "timeout"
    assert results[0].metadata["started"] is True


@pytest.mark.asyncio
async def test_expired_run_deadline_blocks_admission() -> None:
    executor = ToolBatchExecutor(
        _exposure(_echo),
        ToolExecutionConfig(deadline_monotonic=time.monotonic() - 1),
    )
    results = await executor.execute_batch([_call("echo", {"text": "x"})])
    assert results[0].status == "timed_out"
    assert results[0].metadata["started"] is False


@pytest.mark.asyncio
async def test_pre_cancelled_batch_marks_every_call_not_started() -> None:
    token = CancelToken()
    token.request_cancel("immediate")
    executor = ToolBatchExecutor(
        _exposure(_echo), ToolExecutionConfig(cancel_token=token)
    )
    results = await executor.execute_batch(
        [_call("echo", {"text": "a"}, "c1"), _call("echo", {"text": "b"}, "c2")]
    )
    assert [r.status for r in results] == ["cancelled", "cancelled"]
    assert all(r.metadata["started"] is False for r in results)


@pytest.mark.asyncio
async def test_serial_cancel_aborts_the_rest_of_the_batch() -> None:
    token = CancelToken()

    @tool(name="canceller")
    def _canceller(text: str) -> str:
        token.request_cancel("immediate")
        return text

    executor = ToolBatchExecutor(
        _exposure(_canceller, _echo), ToolExecutionConfig(cancel_token=token)
    )
    results = await executor.execute_batch(
        [
            _call("canceller", {"text": "x"}, "c1"),
            _call("echo", {"text": "y"}, "c2"),
        ]
    )
    # The call that requested cancellation still settles as cancelled, and
    # every later call is cancelled without starting.
    assert results[0].status == "cancelled"
    assert results[0].metadata["started"] is True
    assert results[1].status == "cancelled"
    assert results[1].metadata["started"] is False


@pytest.mark.asyncio
async def test_permission_denied_and_ask_terminal_results() -> None:
    @tool(name="guarded")
    def _guarded(text: str) -> str:
        return text

    from qitos.core.tool import ToolPermissionContext, ToolPermissionRule

    context = ToolPermissionContext(
        deny_rules=[ToolPermissionRule(effect="deny", tool_name="guarded")],
    )
    executor = ToolBatchExecutor(
        _exposure(_guarded),
        ToolExecutionConfig(extra_runtime_context={"permission_context": context}),
    )
    results = await executor.execute_batch([_call("guarded", {"text": "x"})])
    assert results[0].status == "denied"
    assert results[0].metadata["error_category"] == "permission_denied"

    asking = ToolPermissionContext(
        ask_rules=[ToolPermissionRule(effect="ask", tool_name="guarded")],
    )
    executor = ToolBatchExecutor(
        _exposure(_guarded),
        ToolExecutionConfig(extra_runtime_context={"permission_context": asking}),
    )
    results = await executor.execute_batch([_call("guarded", {"text": "x"})])
    assert results[0].status == "needs_approval"


@pytest.mark.asyncio
async def test_parallel_results_keep_input_order_with_mixed_latencies() -> None:
    @tool(name="fast", concurrency_safe=True)
    async def _fast(text: str) -> str:
        await asyncio.sleep(0.01)
        return "fast"

    @tool(name="slow", concurrency_safe=True)
    async def _slow(text: str) -> str:
        await asyncio.sleep(0.1)
        return "slow"

    executor = ToolBatchExecutor(
        _exposure(_fast, _slow),
        ToolExecutionConfig(mode="parallel", max_concurrency=4),
    )
    results = await executor.execute_batch(
        [_call("slow", {"text": "s"}, "c1"), _call("fast", {"text": "f"}, "c2")]
    )
    assert [r.call_id for r in results] == ["c1", "c2"]
    assert [r.output for r in results] == ["slow", "fast"]


@pytest.mark.asyncio
async def test_exclusive_call_is_a_barrier_between_safe_segments() -> None:
    order: list[str] = []

    @tool(name="safe", concurrency_safe=True)
    async def _safe(text: str) -> str:
        await asyncio.sleep(0.02)
        order.append(f"safe:{text}")
        return text

    @tool(name="exclusive")
    def _exclusive(text: str) -> str:
        order.append("exclusive")
        return text

    executor = ToolBatchExecutor(
        _exposure(_safe, _exclusive),
        ToolExecutionConfig(mode="parallel", max_concurrency=4),
    )
    await executor.execute_batch(
        [
            _call("safe", {"text": "a"}, "c1"),
            _call("exclusive", {"text": "x"}, "c2"),
            _call("safe", {"text": "b"}, "c3"),
        ]
    )
    assert order == ["safe:a", "exclusive", "safe:b"]


@pytest.mark.asyncio
async def test_parallel_cancel_drains_started_calls_to_terminal() -> None:
    token = CancelToken()
    release = asyncio.Event()
    started = 0

    @tool(name="waiter", concurrency_safe=True)
    async def _waiter(text: str) -> str:
        nonlocal started
        started += 1
        if started == 2:
            token.request_cancel("immediate")
        await release.wait()
        return text

    executor = ToolBatchExecutor(
        _exposure(_waiter),
        ToolExecutionConfig(
            mode="parallel", max_concurrency=2, cancel_token=token
        ),
    )

    async def _release_soon() -> None:
        while started < 2:
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)
        release.set()

    releaser = asyncio.create_task(_release_soon())
    results = await executor.execute_batch(
        [_call("waiter", {"text": "a"}, "c1"), _call("waiter", {"text": "b"}, "c2")]
    )
    await releaser
    assert len(results) == 2
    assert all(r.status in {"success", "cancelled"} for r in results)
    # Both calls started and settled to a real terminal state despite cancel.
    assert all(r.metadata.get("started") is True for r in results)


@pytest.mark.asyncio
async def test_fail_fast_aborts_remaining_calls() -> None:
    @tool(name="broken")
    def _broken(text: str) -> str:
        raise RuntimeError("boom")

    executor = ToolBatchExecutor(
        _exposure(_broken, _echo),
        ToolExecutionConfig(fail_fast=True),
    )
    results = await executor.execute_batch(
        [_call("broken", {"text": "x"}, "c1"), _call("echo", {"text": "y"}, "c2")]
    )
    assert results[0].status == "error"
    assert results[1].status == "cancelled"
    assert results[1].metadata["cancel_source"] == "fail_fast"


@pytest.mark.asyncio
async def test_needs_approval_tool_never_runs_in_a_parallel_segment() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    log: List[str] = []

    @tool(name="safe", concurrency_safe=True)
    async def _safe(text: str) -> str:
        log.append("safe:start")
        entered.set()
        await release.wait()
        log.append("safe:end")
        return text

    @tool(name="approval", needs_approval=True, concurrency_safe=True)
    async def _approval(text: str) -> str:
        log.append("approval:start")
        log.append("approval:end")
        return text

    executor = ToolBatchExecutor(
        _exposure(_safe, _approval), ToolExecutionConfig(mode="parallel")
    )
    batch = asyncio.create_task(
        executor.execute_batch(
            [_call("safe", {"text": "a"}, "c1"), _call("approval", {"text": "b"}, "c2")]
        )
    )
    await entered.wait()
    # An incorrectly concurrent approval would complete while safe is parked.
    await asyncio.sleep(0)
    release.set()
    results = await batch
    assert [r.status for r in results] == ["success", "success"]
    # The approval tool is an exclusive barrier: it runs only after the safe
    # call has fully finished.
    assert log == ["safe:start", "safe:end", "approval:start", "approval:end"]


@pytest.mark.asyncio
async def test_duplicate_tool_call_ids_are_rejected_before_any_side_effect() -> None:
    calls_run: List[str] = []

    @tool(name="tracked")
    def _tracked(text: str) -> str:
        calls_run.append(text)
        return text

    transaction = RecordingTransaction()
    executor = ToolBatchExecutor(
        _exposure(_tracked, _echo),
        ToolExecutionConfig(run_id="r"),
        transaction=transaction,
    )
    results = await executor.execute_batch(
        [
            _call("echo", {"text": "ok"}, "c1"),
            _call("tracked", {"text": "a"}, "dup"),
            _call("tracked", {"text": "b"}, "dup"),
        ]
    )
    # No handler with a duplicated id ever executed.
    assert calls_run == []
    assert results[0].status == "success"
    assert results[1].status == "error" and results[2].status == "error"
    assert results[1].metadata["error_category"] == "duplicate_tool_call_id"
    assert results[2].metadata["error_category"] == "duplicate_tool_call_id"
    assert results[1].metadata["started"] is False
    # Each duplicated id is journaled exactly once, under its first
    # occurrence, so deterministic record ids stay unique. Batch admission
    # records rejections before any executed call.
    started = [r[2] for r in transaction.records if r[0] == "tool_started"]
    terminals = [r[2] for r in transaction.records if r[0] == "tool_terminal"]
    assert started == ["dup", "c1"]
    assert terminals == ["dup", "c1"]


@pytest.mark.asyncio
async def test_parallel_preflight_is_sequential_and_precedes_execution() -> None:
    log: List[tuple] = []

    def _make(name: str):
        @tool(name=name, concurrency_safe=True)
        async def _safe(text: str) -> str:
            log.append(("run", name))
            return text

        return _safe

    tools = [_make("a"), _make("b"), _make("c")]

    def _before(hook_context) -> None:
        log.append(("before", hook_context.tool_call.name))
        return None

    executor = ToolBatchExecutor(
        _exposure(*tools),
        ToolExecutionConfig(mode="parallel", before_tool_call=_before),
    )
    results = await executor.execute_batch(
        [_call(name, {"text": name}, f"c{i}") for i, name in enumerate(("a", "b", "c"))]
    )
    assert all(r.status == "success" for r in results)
    befores = [entry for entry in log if entry[0] == "before"]
    runs = [entry for entry in log if entry[0] == "run"]
    # Permission/validation/before-hook preflight runs in input order, and no
    # handler starts before every preflight has finished (Pi's parallel order).
    assert befores == [("before", "a"), ("before", "b"), ("before", "c")]
    first_run = log.index(runs[0])
    assert all(log.index(entry) < first_run for entry in befores)
    assert len(runs) == 3


@pytest.mark.asyncio
async def test_before_hook_updated_args_are_revalidated_then_executed() -> None:
    from qitos.core.tool_executor import BeforeToolCallDecision

    executor = ToolBatchExecutor(
        _exposure(_echo),
        ToolExecutionConfig(
            before_tool_call=lambda ctx: BeforeToolCallDecision(
                updated_args={"text": "changed"}
            )
        ),
    )
    results = await executor.execute_batch([_call("echo", {"text": "orig"}, "c1")])
    assert results[0].status == "success"
    assert results[0].output == "echo:changed"

    invalid = ToolBatchExecutor(
        _exposure(_echo),
        ToolExecutionConfig(
            before_tool_call=lambda ctx: BeforeToolCallDecision(
                updated_args={"wrong": 1}
            )
        ),
    )
    results = await invalid.execute_batch([_call("echo", {"text": "orig"}, "c1")])
    # QitOS strengthening: updated args are re-validated against the schema.
    assert results[0].status == "error"
    assert results[0].metadata["error_category"] == "invalid_tool_arguments"
    assert results[0].metadata["started"] is False


@pytest.mark.asyncio
async def test_hung_before_hook_is_cancelled_by_token() -> None:
    hook_started = asyncio.Event()

    async def _hung(ctx) -> None:
        hook_started.set()
        await asyncio.Event().wait()
        return None

    token = CancelToken()
    executor = ToolBatchExecutor(
        _exposure(_echo),
        ToolExecutionConfig(cancel_token=token, before_tool_call=_hung),
    )
    batch = asyncio.create_task(
        executor.execute_batch([_call("echo", {"text": "x"}, "c1")])
    )
    await hook_started.wait()
    token.request_cancel("immediate")
    results = await asyncio.wait_for(batch, timeout=5)
    # A hung hook cannot block abort: the call ends with a cancelled terminal.
    assert results[0].status == "cancelled"
    assert results[0].metadata["cancel_source"] == "cancel_token"


@pytest.mark.asyncio
async def test_external_cancellation_terminalizes_then_reraises() -> None:
    handler_started = asyncio.Event()
    release = asyncio.Event()  # never set

    @tool(name="waiter")
    async def _waiter(text: str) -> str:
        handler_started.set()
        await release.wait()
        return text

    transaction = RecordingTransaction()
    executor = ToolBatchExecutor(
        _exposure(_waiter, _echo),
        ToolExecutionConfig(run_id="r"),
        transaction=transaction,
    )
    batch = asyncio.create_task(
        executor.execute_batch(
            [_call("waiter", {"text": "a"}, "c1"), _call("echo", {"text": "b"}, "c2")]
        )
    )
    await handler_started.wait()
    batch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await batch
    # Started work was terminalized before the cancellation propagated.
    results = executor.last_batch_results
    assert results is not None and len(results) == 2
    assert results[0].status == "cancelled"
    assert results[0].metadata["cancel_source"] == "caller_cancelled"
    assert results[1].status == "cancelled"
    kinds = [record[0] for record in transaction.records]
    assert kinds == [
        "tool_started",
        "tool_terminal",
        "tool_started",
        "tool_terminal",
    ]


@pytest.mark.asyncio
async def test_tool_progress_emits_tool_execution_update_events() -> None:
    @tool(name="progress")
    def _progress(text: str, runtime_context=None) -> str:
        runtime_context["emit_progress"]({"stage": "half"})
        return text

    events = []
    executor = ToolBatchExecutor(
        _exposure(_progress), ToolExecutionConfig(), emit=events.append
    )
    results = await executor.execute_batch([_call("progress", {"text": "x"}, "c1")])
    assert results[0].status == "success"
    updates = [e for e in events if isinstance(e, ToolExecutionUpdate)]
    assert len(updates) == 1
    assert updates[0].tool_call_id == "c1"
    assert updates[0].partial_result == {"stage": "half"}
    # Updates arrive before the execution end event.
    kinds = [type(e).__name__ for e in events]
    assert kinds.index("ToolExecutionUpdate") < kinds.index("ToolExecutionEnd")


@pytest.mark.asyncio
async def test_after_step_cancel_does_not_interrupt_tool_execution() -> None:
    token = CancelToken()

    @tool(name="setter")
    def _setter(text: str) -> str:
        token.request_cancel("after_step")
        return text

    executor = ToolBatchExecutor(
        _exposure(_setter, _echo), ToolExecutionConfig(cancel_token=token)
    )
    results = await executor.execute_batch(
        [_call("setter", {"text": "a"}, "c1"), _call("echo", {"text": "b"}, "c2")]
    )
    # after_step never interrupts an in-flight batch; the agent loop stops
    # the run at the turn boundary instead.
    assert [r.status for r in results] == ["success", "success"]
