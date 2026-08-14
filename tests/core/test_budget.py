from __future__ import annotations

import asyncio

import pytest

from qitos.core.budget import BudgetLedger
from qitos.core.journal import JournalError, JournalRecordType
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
async def test_budget_ledger_recovers_legacy_root_and_child_usage(tmp_path) -> None:
    journal = JsonlSessionJournal(tmp_path)
    await journal.create("root-run", {"agent": "root"})
    await journal.append(
        JournalRecordType.MODEL_COMPLETED,
        {
            "transaction_id": "root-transaction",
            "model_response": {
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                "cost_usd": 1.5,
            },
        },
        record_id="root-run:model:legacy",
    )
    await journal.append(
        JournalRecordType.CHILD_TERMINAL,
        {
            "child_run_id": "child-run",
            "total_tokens": 30,
        },
        record_id="root-run:child:legacy:terminal",
    )

    ledger = BudgetLedger(max_tokens=100, max_cost_usd=10.0)
    ledger.attach(journal, root_run_id="root-run", records=await journal.replay())

    snapshot = ledger.snapshot()
    assert snapshot.total_tokens == 40
    assert snapshot.total_cost_usd == pytest.approx(1.5)
    assert snapshot.usage_complete is False
    assert snapshot.cost_complete is False
    await journal.close()
