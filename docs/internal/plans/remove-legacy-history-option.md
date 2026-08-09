# Remove the legacy message-slicing option

## Goal

Remove `keep_last_messages` from the public compact-history API instead of retaining a
compatibility-only no-op. Compact history must expose only the transaction-safe
`keep_last_rounds` retention policy and the explicit `hard_window` count limit.

## Assumptions and tradeoff

- The removal is intentionally breaking: old keyword calls should fail immediately
  instead of appearing to configure behavior that no longer exists.
- No value migration is required because `keep_last_messages` has no runtime effect;
  translating its numeric value into rounds would silently invent new behavior.
- PentestAgent has no call site for the legacy option, so its runtime behavior is
  unchanged.

## Work

- [x] Delete the field, constructor keyword, warnings, and compatibility test.
- [x] Remove the unused keyword from maintained examples and migration snapshots.
- [x] Update the guide, API reference, changelog, and README-facing news.
- [x] Prove the symbol is absent from maintained code and run focused compact-history
      tests plus QitOS's
      repository checks.

## Completion

Completed on 2026-08-09. Maintained QitOS and PentestAgent code have no call sites, the
full QitOS suite passes, and the focused runtime files pass flake8. The changelog and
this plan retain the removed API name for migration history. The separately versioned
`qitos_zoo` submodule still has two downstream call sites and requires its own upstream
migration; it was intentionally left clean and uncommitted because it is not part of
the authorized QitOS/PentestAgent repositories.
