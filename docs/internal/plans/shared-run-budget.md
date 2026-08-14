# Shared Run Budget

## Goal

Make one Root Run and every descendant Engine consume the same token and cost
budget while preserving each Engine's local usage for audit and Child results.

## Contracts

- A Root Engine owns one `BudgetLedger`; Child Engines receive the same instance.
- An Engine's `RuntimeBudget` remains its local ceiling. The effective remaining
  budget is the smaller of the local remainder and the shared Run remainder.
- Every completed model transaction commits token and cost usage exactly once.
  Commits are keyed by origin Run and model transaction id.
- Journal-backed Runs append `budget.committed` to the Root JSONL before the
  in-memory ledger advances. SQLite remains a rebuildable query projection.
- Resume and fork rebuild the ledger only from canonical JSONL records.
- Provider usage is marked complete only when provider token counts are usable.
  Estimated tokens still consume the budget but make the aggregate incomplete.
- Cost is marked complete only when explicit pricing and input/output counts are
  available. A configured cost limit therefore continues to require pricing.
- `EngineResult` exposes both local and shared totals. `ChildResult` records the
  Child's local totals and completeness, so Parent accounting never adds the
  Child result back into the shared ledger.

## First delivery boundary

The ledger serializes completed transaction settlement and prevents another turn
after the effective budget is exhausted. It does not reserve an estimated amount
before concurrent provider calls. Already-dispatched concurrent calls can
therefore finish above the ceiling. Reservation/lease semantics require a
separate design because providers do not guarantee actual response size.

## Verification

- Root and Child usage accumulates in one ledger without lost concurrent writes.
- Local Child ceilings and shared Run ceilings both stop subsequent turns.
- Token and cost settle in the same idempotent transaction.
- Child result round-trips local token/cost and completeness.
- Root JSONL replay restores descendant usage and does not depend on SQLite.
- Existing single-Engine behavior remains unchanged.

## Outcome

Implemented as planned. `BudgetLedger` is a core contract, `budget.committed` is
canonical Root JSONL, Engine results separate shared and local usage, and Child results
persist local token/cost plus completeness. Legacy journals rebuild conservatively from
model and Child terminal records. Reservation remains outside this delivery boundary.
