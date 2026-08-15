# Unify runtime manifests and hosted Web fallback

## Decision

Extend the canonical `RuntimeCapabilitySnapshot`; do not add a parallel manifest.
Route an admitted `web_search` exposure through a tested Provider-hosted tool when the
configured model transport declares support. Keep the same managed `web_search` Tool as
the final fallback when the Provider explicitly rejects the hosted tool before emitting
any observable event.

Keep the contract operationally small. Existing Responses `native_items` preserve the
hosted call, message annotations, and citations; QitOS does not add a second Web record,
usage ledger, provenance store, or recovery model for the same transaction.

## Scope

1. Bind Docker command probes and snapshot commands to the target container. Keep the
   controller-side Docker backend file/foreground-only and never emulate background or
   PTY through detached host commands. Application composition roots own profile
   selection and pass the verified `RuntimeCommand` values into the Env; the generic
   `EnvSpec` factory has no profile to verify and therefore truthfully reports no
   command claims instead of trusting caller-supplied attestation.
2. Advertise hosted search only on tested first-party transports: the official OpenAI
   Responses endpoint uses `web_search`, while the official Anthropic Messages endpoint uses
   `web_search_20250305`. When the frozen Tool exposure contains an admitted managed
   `web_search`, send the hosted tool first and retain the managed function schema as a
   request-local fallback.
3. Fall back only for an explicit unsupported/unavailable hosted-tool response before
   any text, reasoning, tool call, or native output item is published. Never replay an
   observable Provider transaction.
4. Preserve hosted output and citations through the existing `native_items` and model
   transaction. Anthropic-compatible endpoints do not inherit Anthropic server tools:
   Kimi keeps its managed `/search` and Chat `$web_search` capabilities, including when
   its model uses Messages transport. Keep managed `web_search` and `web_fetch` as
   ordinary terminal ToolResult paths; do not force both routes into a new common
   metadata shape.

## Verification

- Runtime snapshot validation and round-trip, backend-local Docker probe, and limited
  Docker facility tests without invoking Docker.
- Official OpenAI Responses and official Anthropic Messages request routing, hosted
  success/citations, explicit unavailable fallback, no fallback after a published
  event, Kimi managed routing, and terminal persistence tests.
- Python 3.10/3.11/3.12 suite plus the QitOS stable static and packaging checks.

## Progress

- [x] Confirm current Runtime, model stream, managed Web, Journal, and exposure owners.
- [x] Bind Docker snapshot commands and probes to the selected container.
- [x] Implement official OpenAI and Anthropic hosted request routing, native citations,
      and safe managed fallback without changing Kimi's existing managed paths.
- [x] Add Qwen Responses/Chat hosted routing and its managed `enable_search` fallback.
- [x] Verify Qwen Responses hosted search against the configured DashScope endpoint.
- [x] Update public docs, changelog, and README news.
- [x] Pass Python 3.10/3.11/3.12 tests and the stable flake8/mypy gate.
- [x] Merge the reviewed feature commit into fork `main`.
