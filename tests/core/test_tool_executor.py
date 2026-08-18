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
from qitos.core.journal import (
    JournalAppendCancelled,
    JournalCommitState,
    JournalPosition,
)
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


@pytest.mark.parametrize("phase", ["tool_started", "tool_terminal"])
@pytest.mark.parametrize(
    "commit_state",
    [
        JournalCommitState.COMMITTED,
        JournalCommitState.NOT_COMMITTED,
        JournalCommitState.UNKNOWN,
    ],
)
@pytest.mark.asyncio
async def test_cancelled_journal_append_respects_commit_state_without_reexecution(
    phase: str,
    commit_state: JournalCommitState,
) -> None:
    class CancelOnceTransaction(RecordingTransaction):
        def __init__(self) -> None:
            super().__init__()
            self.started_attempts = 0
            self.terminal_attempts = 0
            self.cancelled = False

        def _cancel(self, record_id: str) -> JournalAppendCancelled:
            position = JournalPosition("r", 1, record_id)
            return JournalAppendCancelled(
                position if commit_state is JournalCommitState.COMMITTED else None,
                commit_state=commit_state,
                pending_position=position,
            )

        async def tool_started(self, turn: int, call: ToolCall) -> None:
            self.started_attempts += 1
            if phase == "tool_started" and not self.cancelled:
                self.cancelled = True
                raise self._cancel(f"{call.id}:started")
            await super().tool_started(turn, call)

        async def tool_terminal(
            self, turn: int, call: ToolCall, result: ToolResult
        ) -> None:
            self.terminal_attempts += 1
            if phase == "tool_terminal" and not self.cancelled:
                self.cancelled = True
                raise self._cancel(f"{call.id}:terminal")
            await super().tool_terminal(turn, call, result)

    executions = 0

    @tool(name="side_effect")
    def _side_effect() -> str:
        nonlocal executions
        executions += 1
        return "done"

    transaction = CancelOnceTransaction()
    executor = ToolBatchExecutor(
        _exposure(_side_effect),
        ToolExecutionConfig(run_id="r"),
        transaction=transaction,
    )

    with pytest.raises(JournalAppendCancelled) as cancelled:
        await executor.execute_batch([_call("side_effect", {}, "c1")])

    assert cancelled.value.commit_state is commit_state
    assert executions == (0 if phase == "tool_started" else 1)
    if phase == "tool_started":
        assert transaction.started_attempts == (
            2 if commit_state is JournalCommitState.NOT_COMMITTED else 1
        )
        assert transaction.terminal_attempts == (
            0 if commit_state is JournalCommitState.UNKNOWN else 1
        )
    else:
        assert transaction.started_attempts == 1
        assert transaction.terminal_attempts == (
            2 if commit_state is JournalCommitState.NOT_COMMITTED else 1
        )
    if commit_state is JournalCommitState.UNKNOWN:
        assert executor.last_batch_results is None
    else:
        assert executor.last_batch_results is not None
        assert executor.last_batch_results[0].status == (
            "cancelled" if phase == "tool_started" else "success"
        )


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
async def test_frozen_nested_arguments_are_thawed_before_schema_and_handler() -> None:
    observed: list[list[str]] = []

    @tool(name="collect")
    def _collect(items: list[str]) -> int:
        observed.append(items)
        return len(items)

    executor = ToolBatchExecutor(_exposure(_collect), ToolExecutionConfig())
    result = (
        await executor.execute_batch(
            [_call("collect", {"items": ["a", "b"]}, "nested")]
        )
    )[0]

    assert result.status == "success"
    assert result.output == 2
    assert observed == [["a", "b"]]


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
    with pytest.raises(ValueError, match="ToolCall ids must be unique"):
        await executor.execute_batch(
            [
                _call("echo", {"text": "ok"}, "c1"),
                _call("tracked", {"text": "a"}, "dup"),
                _call("tracked", {"text": "b"}, "dup"),
            ]
        )
    # The whole malformed assistant batch fails closed before events,
    # journal admission, or handler side effects.
    assert calls_run == []
    assert transaction.records == []


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
async def test_parallel_handlers_commit_terminal_results_in_input_order() -> None:
    first_started = asyncio.Event()
    second_finished = asyncio.Event()
    release_first = asyncio.Event()
    completion_order: List[str] = []

    @tool(name="first", concurrency_safe=True)
    async def _first() -> str:
        first_started.set()
        await release_first.wait()
        completion_order.append("first")
        return "first"

    @tool(name="second", concurrency_safe=True)
    async def _second() -> str:
        await first_started.wait()
        completion_order.append("second")
        second_finished.set()
        return "second"

    transaction = RecordingTransaction()
    executor = ToolBatchExecutor(
        _exposure(_first, _second),
        ToolExecutionConfig(mode="parallel", max_concurrency=2),
        transaction=transaction,
    )
    batch = asyncio.create_task(
        executor.execute_batch(
            [_call("first", {}, "c1"), _call("second", {}, "c2")]
        )
    )

    await second_finished.wait()
    assert completion_order == ["second"]
    assert not any(record[0] == "tool_terminal" for record in transaction.records)

    release_first.set()
    results = await batch

    assert [result.call_id for result in results] == ["c1", "c2"]
    terminal_ids = [
        record[2]
        for record in transaction.records
        if record[0] == "tool_terminal"
    ]
    assert terminal_ids == ["c1", "c2"]


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
async def test_immediate_token_interrupts_hung_async_handler_and_terminalizes() -> None:
    handler_started = asyncio.Event()
    handler_settled = asyncio.Event()

    @tool(name="waiter")
    async def _waiter() -> str:
        try:
            handler_started.set()
            await asyncio.Event().wait()
            return "unreachable"
        finally:
            handler_settled.set()

    token = CancelToken()
    transaction = RecordingTransaction()
    executor = ToolBatchExecutor(
        _exposure(_waiter),
        ToolExecutionConfig(cancel_token=token),
        transaction=transaction,
    )
    batch = asyncio.create_task(
        executor.execute_batch([_call("waiter", {}, "c1")])
    )
    await handler_started.wait()

    token.request_cancel("immediate")
    results = await asyncio.wait_for(batch, timeout=1)

    assert handler_settled.is_set()
    assert results[0].status == "cancelled"
    assert results[0].metadata["cancel_source"] == "cancel_token"
    assert [record[0] for record in transaction.records] == [
        "tool_started",
        "tool_terminal",
    ]


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
async def test_tool_progress_is_observable_while_handler_is_still_running() -> None:
    update_seen = asyncio.Event()
    release = asyncio.Event()

    @tool(name="progress")
    async def _progress(text: str, runtime_context=None) -> str:
        runtime_context["emit_progress"]({"stage": "running"})
        await release.wait()
        return text

    async def _emit(event) -> None:
        if isinstance(event, ToolExecutionUpdate):
            update_seen.set()

    executor = ToolBatchExecutor(
        _exposure(_progress), ToolExecutionConfig(), emit=_emit
    )
    batch = asyncio.create_task(
        executor.execute_batch([_call("progress", {"text": "x"}, "c1")])
    )
    await asyncio.wait_for(update_seen.wait(), timeout=1)
    assert not batch.done()
    release.set()
    results = await batch
    assert results[0].status == "success"


@pytest.mark.asyncio
async def test_handler_raised_cancelled_error_terminalizes_then_reraises() -> None:
    @tool(name="self_cancel")
    async def _self_cancel() -> None:
        raise asyncio.CancelledError()

    transaction = RecordingTransaction()
    executor = ToolBatchExecutor(
        _exposure(_self_cancel), ToolExecutionConfig(), transaction=transaction
    )

    with pytest.raises(asyncio.CancelledError):
        await executor.execute_batch([_call("self_cancel", {}, "c1")])

    results = executor.last_batch_results
    assert results is not None
    assert results[0].status == "cancelled"
    assert [record[0] for record in transaction.records] == [
        "tool_started",
        "tool_terminal",
    ]


@pytest.mark.asyncio
async def test_plain_mapping_status_is_domain_output_not_lifecycle_control() -> None:
    @tool(name="launch")
    def _launch() -> dict[str, object]:
        return {"status": "running", "handle": "process-1"}

    executor = ToolBatchExecutor(_exposure(_launch), ToolExecutionConfig())
    result = (await executor.execute_batch([_call("launch", {}, "c1")]))[0]

    assert result.status == "success"
    assert result.output == {"status": "running", "handle": "process-1"}


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


@pytest.mark.asyncio
async def test_tool_hook_receives_read_only_cancel_signal() -> None:
    token = CancelToken()
    observed = []

    def _before(context) -> None:
        observed.append(context.runtime.cancel_signal)

    executor = ToolBatchExecutor(
        _exposure(_echo),
        ToolExecutionConfig(cancel_token=token, before_tool_call=_before),
    )
    await executor.execute_batch([_call("echo", {"text": "x"}, "c1")])

    assert observed == [token.signal]
    assert not hasattr(observed[0], "request_cancel")
    assert not hasattr(observed[0], "clear")


@pytest.mark.asyncio
async def test_extra_runtime_context_cannot_replace_executor_authority() -> None:
    attacker_exposure = ToolRegistry().freeze()
    observed = {}

    @tool(name="inspect_runtime")
    def _inspect_runtime(runtime_context=None) -> str:
        observed.update(runtime_context)
        return "ok"

    exposure = _exposure(_inspect_runtime)
    sentinel = object()
    executor = ToolBatchExecutor(
        exposure,
        ToolExecutionConfig(
            run_id="authoritative-run",
            extra_runtime_context={
                "tool_registry": attacker_exposure,
                "run_id": "forged-run",
                "deadline_monotonic": -1.0,
                "remaining_seconds": sentinel,
                "agent_cancelled": sentinel,
                "emit_progress": sentinel,
                "record_artifact": sentinel,
                "permission_context": "kept",
            },
        ),
    )

    results = await executor.execute_batch([_call("inspect_runtime", {}, "c1")])

    assert results[0].status == "success"
    assert observed["tool_registry"] is exposure
    assert observed["tool_registry"] is not attacker_exposure
    assert observed["run_id"] == "authoritative-run"
    assert observed["deadline_monotonic"] is None
    assert callable(observed["remaining_seconds"])
    assert callable(observed["agent_cancelled"])
    assert callable(observed["emit_progress"])
    assert callable(observed["record_artifact"])
    assert observed["permission_context"] == "kept"


@pytest.mark.asyncio
async def test_event_fault_terminalizes_batch_before_propagating() -> None:
    executed = []

    @tool(name="side_effect")
    def _side_effect() -> str:
        executed.append(True)
        return "unexpected"

    def _broken(event) -> None:
        if isinstance(event, ToolExecutionStart):
            raise RuntimeError("event sink failed")

    transaction = RecordingTransaction()
    executor = ToolBatchExecutor(
        _exposure(_side_effect, _echo),
        ToolExecutionConfig(run_id="r"),
        emit=_broken,
        transaction=transaction,
    )

    with pytest.raises(RuntimeError, match="event sink failed"):
        await executor.execute_batch(
            [_call("side_effect", {}, "c1"), _call("echo", {"text": "x"}, "c2")]
        )

    assert executed == []
    results = executor.last_batch_results
    assert results is not None
    assert [result.status for result in results] == ["cancelled", "cancelled"]
    assert [record[0] for record in transaction.records] == [
        "tool_started",
        "tool_terminal",
        "tool_started",
        "tool_terminal",
    ]


@pytest.mark.asyncio
async def test_parallel_end_event_fault_is_not_downgraded_to_tool_error() -> None:
    @tool(name="first", concurrency_safe=True)
    async def _first() -> str:
        await asyncio.sleep(0)
        return "first"

    @tool(name="second", concurrency_safe=True)
    async def _second() -> str:
        await asyncio.sleep(0)
        return "second"

    def _broken(event) -> None:
        if isinstance(event, ToolExecutionEnd):
            raise RuntimeError("parallel event sink failed")

    transaction = RecordingTransaction()
    executor = ToolBatchExecutor(
        _exposure(_first, _second),
        ToolExecutionConfig(mode="parallel", max_concurrency=2),
        emit=_broken,
        transaction=transaction,
    )

    with pytest.raises(RuntimeError, match="parallel event sink failed"):
        await executor.execute_batch(
            [_call("first", {}, "c1"), _call("second", {}, "c2")]
        )

    results = executor.last_batch_results
    assert results is not None
    assert [result.status for result in results] == ["success", "success"]
    assert len(
        [record for record in transaction.records if record[0] == "tool_terminal"]
    ) == 2
