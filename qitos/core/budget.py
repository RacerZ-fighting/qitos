"""Shared Root/Child model-usage accounting."""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass

from .journal import (
    JournalAppendCancelled,
    JournalCommitState,
    JournalRecord,
    JournalRecordType,
    SessionJournal,
    resolve_inherited_record,
)


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Immutable totals and effective remaining Run budget."""

    max_tokens: int | None
    max_cost_usd: float | None
    total_tokens: int
    total_cost_usd: float
    usage_complete: bool
    cost_complete: bool

    @property
    def remaining_tokens(self) -> int | None:
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.total_tokens)

    @property
    def remaining_cost_usd(self) -> float | None:
        if self.max_cost_usd is None:
            return None
        return max(0.0, self.max_cost_usd - self.total_cost_usd)

    @property
    def tokens_exhausted(self) -> bool:
        return self.max_tokens is not None and self.total_tokens >= self.max_tokens

    @property
    def cost_exhausted(self) -> bool:
        return (
            self.max_cost_usd is not None
            and self.total_cost_usd >= self.max_cost_usd
        )


@dataclass(frozen=True, slots=True)
class _BudgetCommit:
    origin_run_id: str
    transaction_id: str
    tokens: int
    cost_usd: float
    usage_complete: bool
    cost_complete: bool

    @property
    def key(self) -> tuple[str, str]:
        return (self.origin_run_id, self.transaction_id)

    def to_payload(self) -> dict[str, object]:
        return {
            "origin_run_id": self.origin_run_id,
            "transaction_id": self.transaction_id,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "usage_complete": self.usage_complete,
            "cost_complete": self.cost_complete,
        }

    @classmethod
    def from_record(cls, record: JournalRecord) -> _BudgetCommit:
        payload = record.payload
        expected = {
            "origin_run_id",
            "transaction_id",
            "tokens",
            "cost_usd",
            "usage_complete",
            "cost_complete",
        }
        if set(payload) != expected:
            raise ValueError("budget.committed fields are invalid")
        return cls.create(
            origin_run_id=payload["origin_run_id"],
            transaction_id=payload["transaction_id"],
            tokens=payload["tokens"],
            cost_usd=payload["cost_usd"],
            usage_complete=payload["usage_complete"],
            cost_complete=payload["cost_complete"],
        )

    @classmethod
    def create(
        cls,
        *,
        origin_run_id: object,
        transaction_id: object,
        tokens: object,
        cost_usd: object,
        usage_complete: object,
        cost_complete: object,
    ) -> _BudgetCommit:
        if not isinstance(origin_run_id, str) or not origin_run_id.strip():
            raise ValueError("origin_run_id must be a non-empty string")
        if not isinstance(transaction_id, str) or not transaction_id.strip():
            raise ValueError("transaction_id must be a non-empty string")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("tokens must be a non-negative integer")
        if (
            isinstance(cost_usd, bool)
            or not isinstance(cost_usd, (int, float))
            or not math.isfinite(float(cost_usd))
            or float(cost_usd) < 0
        ):
            raise ValueError("cost_usd must be a non-negative finite number")
        if not isinstance(usage_complete, bool):
            raise TypeError("usage_complete must be a boolean")
        if not isinstance(cost_complete, bool):
            raise TypeError("cost_complete must be a boolean")
        return cls(
            origin_run_id=origin_run_id.strip(),
            transaction_id=transaction_id.strip(),
            tokens=tokens,
            cost_usd=float(cost_usd),
            usage_complete=usage_complete,
            cost_complete=cost_complete,
        )


class BudgetLedger:
    """One async-safe usage owner shared by a Root Run and its descendants.

    Journal-backed commits are appended to the Root Run JSONL before the
    in-memory projection advances. Reusing a transaction id is idempotent only
    when the complete usage payload is identical.
    """

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer or None")
        if max_cost_usd is not None and (
            isinstance(max_cost_usd, bool)
            or not isinstance(max_cost_usd, (int, float))
            or not math.isfinite(float(max_cost_usd))
            or float(max_cost_usd) <= 0
        ):
            raise ValueError("max_cost_usd must be a positive finite number or None")
        self._max_tokens = max_tokens
        self._max_cost_usd = (
            float(max_cost_usd) if max_cost_usd is not None else None
        )
        self._commits: dict[tuple[str, str], _BudgetCommit] = {}
        self._total_tokens = 0
        self._total_cost_usd = 0.0
        self._usage_complete = True
        self._cost_complete = True
        self._origin_snapshots: dict[str, BudgetSnapshot] = {}
        self._ordered_attribution_complete = True
        self._journal: SessionJournal | None = None
        self._root_run_id = ""
        self._lock = asyncio.Lock()

    def attach(
        self,
        journal: SessionJournal,
        *,
        root_run_id: str,
        records: Iterable[JournalRecord] = (),
    ) -> None:
        """Attach the canonical Root journal and rebuild committed usage."""

        normalized_run_id = str(root_run_id or "").strip()
        if not normalized_run_id:
            raise ValueError("root_run_id must be a non-empty string")
        if journal.run_id != normalized_run_id:
            raise ValueError("BudgetLedger journal must be open for the Root Run")
        if self._journal is not None and (
            self._journal is not journal or self._root_run_id != normalized_run_id
        ):
            raise RuntimeError("BudgetLedger is already attached to another Root Run")
        self._reset_projection()
        effective_records = tuple(resolve_inherited_record(item) for item in records)
        for record in effective_records:
            if record.type is not JournalRecordType.BUDGET_COMMITTED:
                continue
            self._apply(_BudgetCommit.from_record(record))
        self._journal = journal
        self._root_run_id = normalized_run_id

    def reset(self) -> None:
        """Reset an unbound in-memory ledger for a fresh Root execution."""

        if self._journal is not None:
            raise RuntimeError("an attached BudgetLedger cannot be reset")
        self._reset_projection()

    def configure_limits(
        self,
        *,
        max_tokens: int | None,
        max_cost_usd: float | None,
    ) -> None:
        """Set Root limits before the first transaction is committed."""

        if self._commits:
            raise RuntimeError("BudgetLedger limits cannot change after usage")
        replacement = BudgetLedger(
            max_tokens=max_tokens,
            max_cost_usd=max_cost_usd,
        )
        self._max_tokens = replacement._max_tokens
        self._max_cost_usd = replacement._max_cost_usd

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            max_tokens=self._max_tokens,
            max_cost_usd=self._max_cost_usd,
            total_tokens=self._total_tokens,
            total_cost_usd=self._total_cost_usd,
            usage_complete=self._usage_complete,
            cost_complete=self._cost_complete,
        )

    def snapshot_after_origin(self, origin_run_id: str) -> BudgetSnapshot | None:
        """Return the aggregate as of one origin's latest exact commit.

        Recovery uses this to attribute a Root budget crossing to the Child
        that observed it, without applying later sibling usage retroactively.
        """

        normalized = str(origin_run_id or "").strip()
        if not normalized:
            raise ValueError("origin_run_id must be a non-empty string")
        if not self._ordered_attribution_complete:
            return None
        return self._origin_snapshots.get(normalized)

    async def commit(
        self,
        *,
        origin_run_id: str,
        transaction_id: str,
        tokens: int,
        cost_usd: float,
        usage_complete: bool,
        cost_complete: bool,
    ) -> BudgetSnapshot:
        commit = _BudgetCommit.create(
            origin_run_id=origin_run_id,
            transaction_id=transaction_id,
            tokens=tokens,
            cost_usd=cost_usd,
            usage_complete=usage_complete,
            cost_complete=cost_complete,
        )
        async with self._lock:
            existing = self._commits.get(commit.key)
            if existing is not None:
                if existing != commit:
                    raise ValueError(
                        "budget transaction was reused with different usage"
                    )
                return self.snapshot()
            if self._journal is not None:
                digest = hashlib.sha256(
                    f"{commit.origin_run_id}\0{commit.transaction_id}".encode("utf-8")
                ).hexdigest()[:32]
                try:
                    await self._journal.append(
                        JournalRecordType.BUDGET_COMMITTED,
                        commit.to_payload(),
                        record_id=f"{self._root_run_id}:budget:{digest}",
                    )
                except JournalAppendCancelled as cancellation:
                    if cancellation.commit_state is JournalCommitState.COMMITTED:
                        self._apply(commit)
                    raise
            self._apply(commit)
            return self.snapshot()

    def _reset_projection(self) -> None:
        self._commits.clear()
        self._total_tokens = 0
        self._total_cost_usd = 0.0
        self._usage_complete = True
        self._cost_complete = True
        self._origin_snapshots.clear()
        self._ordered_attribution_complete = True

    def _apply(self, commit: _BudgetCommit, *, ordered: bool = True) -> None:
        existing = self._commits.get(commit.key)
        if existing is not None:
            if existing != commit:
                raise ValueError("budget transaction was reused with different usage")
            return
        self._commits[commit.key] = commit
        self._total_tokens += commit.tokens
        self._total_cost_usd += commit.cost_usd
        self._usage_complete = self._usage_complete and commit.usage_complete
        self._cost_complete = self._cost_complete and commit.cost_complete
        if ordered:
            self._origin_snapshots[commit.origin_run_id] = self.snapshot()
        else:
            self._ordered_attribution_complete = False


__all__ = ["BudgetLedger", "BudgetSnapshot"]
