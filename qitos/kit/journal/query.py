"""Read-only projections over committed canonical journal records."""

from __future__ import annotations

from ...core.journal import (
    JournalCorruptionError,
    JournalRecordRef,
    JournalRecordType,
    SessionJournal,
    ToolTransaction,
    resolve_inherited_record,
)
from .turn_recorder import decode_step_committed


async def committed_tool_transactions(
    journal: SessionJournal,
) -> tuple[ToolTransaction, ...]:
    """Return committed Tool transactions in canonical turn/call order.

    The projection follows inherited records back to their origin Run and only
    returns terminals referenced by ``step.committed``. Open, crash-torn, or
    otherwise uncommitted Tool calls are deliberately absent. Payload decoding
    and the terminal/transcript join remain owned by QitOS and fail closed on
    corruption.
    """

    if not isinstance(journal, SessionJournal):
        raise TypeError("journal must implement SessionJournal")
    transactions: list[ToolTransaction] = []
    seen: set[JournalRecordRef] = set()
    for record in await journal.replay():
        effective = resolve_inherited_record(record)
        if effective.type is not JournalRecordType.STEP_COMMITTED:
            continue
        try:
            _turn, _transcript_ids, terminal_ids = decode_step_committed(
                effective.payload
            )
        except ValueError as exc:
            raise JournalCorruptionError(str(exc)) from exc
        for terminal_id in terminal_ids:
            reference = JournalRecordRef(effective.run_id, terminal_id)
            if reference in seen:
                raise JournalCorruptionError(
                    "committed Tool terminal is referenced more than once"
                )
            transaction = journal.find_tool_transaction(reference)
            if transaction is None:
                raise JournalCorruptionError(
                    "step.committed references an unresolved Tool terminal"
                )
            seen.add(reference)
            transactions.append(transaction)
    return tuple(transactions)


__all__ = ["committed_tool_transactions"]
