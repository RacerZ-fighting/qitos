# Durable Process Terminal Mailbox

## Goal

Wake an active Agent when a Run-owned background process becomes terminal without
letting process watchers mutate Agent state or start a model turn directly.

The design follows Pi's turn-boundary queue discipline and Codex's separation between
managed terminal ownership and process-exit notification. QitOS keeps its existing
durable `RuntimeInput` mailbox as the single delivery path.

## Contract

- `CommandCapability.astart()` may receive one async terminal notifier.
- The Host watcher first finishes output collection and persists `process.terminal`.
- Only after that boundary may it submit one bounded `process.completed` RuntimeInput.
- Mailbox acceptance remains owned by `Engine.apost_runtime_event()`, so a Journal-backed
  Run persists the input before it wakes and injects it only at a turn safe point.
- Notification failure does not erase the terminal process fact or make the process live
  again; the terminal remains queryable through process control tools.
- Foreground commands and unsupported container background execution do not gain a
  second execution path.

## Success criteria

- [x] Active background completion produces one terminal RuntimeInput with a stable id.
- [x] `process.terminal` is durable before `runtime_input.posted`.
- [x] A failed terminal append never posts a completion input.
- [x] A closed/rejecting mailbox leaves the terminal snapshot queryable.
- [x] Cancellation and teardown drain watcher/notifier work without orphan Tasks.
- [x] Focused tests, the full QitOS suite, flake8, and mypy pass.
