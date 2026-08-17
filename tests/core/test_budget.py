from __future__ import annotations

import asyncio

import pytest

from qitos.core.budget import BudgetLedger
from qitos.core.journal import (
    JournalAppendCancelled,
    JournalCommitState,
    JournalError,
    JournalPosition,
    JournalRecordType,
)
from qitos.kit.journal import JsonlSessionJournal


@pytest.mark.asyncio
async def test_budget_ledger_commits_concurrent_usage_once() -> None:
    ledger = BudgetLedger(max_tokens=100, max_cost_usd=10.0)

    await asyncio.gather(
        *(
            ledger.commit(
                origin_run_id=f"child-{index % 2}",
                transaction_id=f"transaction-{index}",
                tokens=3,
                cost_usd=0.25,
                usage_complete=True,
                cost_complete=True,
            )
            for index in range(20)
        )
    )
    await ledger.commit(
        origin_run_id="child-0",
        transaction_id="transaction-0",
        tokens=3,
        cost_usd=0.25,
        usage_complete=True,
        cost_complete=True,
    )

    snapshot = ledger.snapshot()
    assert snapshot.total_tokens == 60
    assert snapshot.total_cost_usd == pytest.approx(5.0)
    assert snapshot.remaining_tokens == 40
    assert snapshot.remaining_cost_usd == pytest.approx(5.0)
    assert snapshot.usage_complete is True
    assert snapshot.cost_complete is True


@pytest.mark.asyncio
async def test_budget_ledger_rejects_conflicting_transaction_reuse() -> None:
    ledger = BudgetLedger()
    await ledger.commit(
        origin_run_id="root",
        transaction_id="transaction",
        tokens=3,
        cost_usd=0.25,
        usage_complete=True,
        cost_complete=True,
    )

    with pytest.raises(ValueError, match="different usage"):
        await ledger.commit(
            origin_run_id="root",
            transaction_id="transaction",
            tokens=4,
            cost_usd=0.25,
            usage_complete=True,
            cost_complete=True,
        )


@pytest.mark.asyncio
async def test_budget_ledger_preserves_each_origins_last_commit_snapshot() -> None:
    ledger = BudgetLedger(max_tokens=10)

    await ledger.commit(
        origin_run_id="child-a",
        transaction_id="a-1",
        tokens=4,
        cost_usd=0.0,
        usage_complete=True,
        cost_complete=False,
    )
    await ledger.commit(
        origin_run_id="child-b",
        transaction_id="b-1",
        tokens=7,
        cost_usd=0.0,
        usage_complete=True,
        cost_complete=False,
    )

    child_a = ledger.snapshot_after_origin("child-a")
    child_b = ledger.snapshot_after_origin("child-b")
    assert child_a is not None
    assert child_a.total_tokens == 4
    assert child_a.tokens_exhausted is False
    assert child_b is not None
    assert child_b.total_tokens == 11
    assert child_b.tokens_exhausted is True
    assert ledger.snapshot_after_origin("child-without-commit") is None


@pytest.mark.asyncio
async def test_failed_journal_append_does_not_advance_budget() -> None:
    class FailingJournal:
        run_id = "root-run"

        async def append(self, *args, **kwargs):
            _ = args, kwargs
            raise JournalError("write failed")

    ledger = BudgetLedger(max_tokens=100)
    ledger.attach(FailingJournal(), root_run_id="root-run")  # type: ignore[arg-type]

    with pytest.raises(JournalError, match="write failed"):
        await ledger.commit(
            origin_run_id="root-run",
            transaction_id="transaction",
            tokens=10,
            cost_usd=0.0,
            usage_complete=True,
            cost_complete=False,
        )

    assert ledger.snapshot().total_tokens == 0


@pytest.mark.parametrize(
    ("commit_state", "expected_tokens"),
    [
        (JournalCommitState.COMMITTED, 10),
        (JournalCommitState.UNKNOWN, 0),
        (JournalCommitState.NOT_COMMITTED, 0),
    ],
)
@pytest.mark.asyncio
async def test_cancelled_budget_append_respects_durable_commit_state(
    commit_state: JournalCommitState,
    expected_tokens: int,
) -> None:
    position = JournalPosition(
        run_id="root-run",
        seq=1,
        record_id="root-run:budget:record",
    )

    class CancelledJournal:
        run_id = "root-run"

        async def append(self, *args, **kwargs):
            _ = args, kwargs
            raise JournalAppendCancelled(
                position if commit_state is JournalCommitState.COMMITTED else None,
                commit_state=commit_state,
                pending_position=position,
            )

    ledger = BudgetLedger(max_tokens=100)
    ledger.attach(
        CancelledJournal(),  # type: ignore[arg-type]
        root_run_id="root-run",
    )

    with pytest.raises(JournalAppendCancelled) as cancelled:
        await ledger.commit(
            origin_run_id="child-run",
            transaction_id="transaction",
            tokens=10,
            cost_usd=0.0,
            usage_complete=True,
            cost_complete=False,
        )

    assert cancelled.value.commit_state is commit_state
    assert ledger.snapshot().total_tokens == expected_tokens
    origin = ledger.snapshot_after_origin("child-run")
    if commit_state is JournalCommitState.COMMITTED:
        assert origin is not None and origin.total_tokens == 10
    else:
        assert origin is None


@pytest.mark.asyncio
async def test_budget_ledger_restores_descendant_usage_from_root_jsonl(
    tmp_path,
) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("root-run", {"agent": "root"})
    ledger = BudgetLedger(max_tokens=100, max_cost_usd=10.0)
    ledger.attach(journal, root_run_id="root-run", records=await journal.replay())

    await ledger.commit(
        origin_run_id="root-run",
        transaction_id="root-transaction",
        tokens=10,
        cost_usd=1.0,
        usage_complete=True,
        cost_complete=True,
    )
    await ledger.commit(
        origin_run_id="child-run",
        transaction_id="child-transaction",
        tokens=30,
        cost_usd=2.0,
        usage_complete=False,
        cost_complete=True,
    )
    records = await journal.replay()
    assert [
        record.type
        for record in records
        if record.type is JournalRecordType.BUDGET_COMMITTED
    ] == [JournalRecordType.BUDGET_COMMITTED, JournalRecordType.BUDGET_COMMITTED]
    await journal.close()

    reopened = JsonlSessionJournal(tmp_path)
    await reopened.open("root-run")
    restored = BudgetLedger(max_tokens=100, max_cost_usd=10.0)
    restored.attach(
        reopened,
        root_run_id="root-run",
        records=await reopened.replay(),
    )

    snapshot = restored.snapshot()
    assert snapshot.total_tokens == 40
    assert snapshot.total_cost_usd == pytest.approx(3.0)
    assert snapshot.usage_complete is False
    assert snapshot.cost_complete is True
    await reopened.close()


@pytest.mark.asyncio
async def test_budget_ledger_attach_rejects_invalid_commit_shapes(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("root-run", {"agent": "root"})
    await journal.append(
        JournalRecordType.BUDGET_COMMITTED,
        {
            "origin_run_id": "root-run",
            "transaction_id": "root-transaction",
            "tokens": 10,
            "cost_usd": 1.0,
        },
        record_id="root-run:budget:invalid",
    )

    ledger = BudgetLedger(max_tokens=100, max_cost_usd=10.0)
    with pytest.raises(ValueError, match="budget.committed fields are invalid"):
        ledger.attach(journal, root_run_id="root-run", records=await journal.replay())
    await journal.close()
