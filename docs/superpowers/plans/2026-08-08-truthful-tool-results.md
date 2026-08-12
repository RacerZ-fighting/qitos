# Truthful tool-result lifecycle

## Goal

Replace the duplicated success/error guesses in `ToolResult`, `ActionExecutor`, and
Engine projection with one fail-closed lifecycle contract.

## Scope

- [x] Define the canonical success, partial, running, error, skipped/denied, approval,
  timeout, and cancellation states in `qitos.core`.
- [x] Reject legacy spellings instead of keeping a compatibility normalization path.
- [x] Reject unknown explicit statuses instead of inferring success.
- [x] Keep domain outcomes separate from execution status in maintained built-ins.
- [x] Project terminal tool states to `ActionStatus` without treating background or
  partial snapshots as completed operations.
- [x] Preserve the canonical state through Engine records, observations, model history,
  events/traces, summaries, and success-rate calculation.
- [x] Make renderers, Qita, benchmark classifiers, and inspector diagnostics treat only
  exact `success` as success while retaining the exact lifecycle value.
- [x] Separate verification and other domain outcomes from Qita interaction lifecycle.
- [x] Remove the executor-only error classifier and the second Engine status list.
- [x] Remove flattened legacy result fields so serialization has one exact shape.
- [x] Add focused contract and native timeout tests.

## Non-goals

- Tool registration/exposure policy, schema validation, concurrency policy, and nested
  output truncation remain separate tool-contract slices.
- Provider transport retries and process-level deadline cleanup are unchanged.
- PentestAgent's MCP deployment and authorization policy remain project-owned.

## Acceptance

Only exact `success` contributes to success metrics. `running`, `partial`, `skipped`,
`denied`, `needs_input`, `needs_approval`, `timed_out`, and `cancelled` remain observable
without becoming implicit success. Unknown explicit status values fail closed, while
domain outcomes use their own field. Native tool-call history retains one result for the
original call id and makes the lifecycle status model-visible.
