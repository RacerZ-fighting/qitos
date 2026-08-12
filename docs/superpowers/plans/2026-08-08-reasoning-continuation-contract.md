# Reasoning effort and continuation contract

## Goal

Keep provider reasoning controls consistent across request paths while preserving
opaque continuation state without treating it as a visible answer.

## Scope

- [x] Resolve GPT-5.6 `max` without changing older OpenAI model policies.
- [x] Remove conflicting reasoning controls when compatible Chat providers require
  thinking to be disabled for a forced tool call.
- [x] Request encrypted reasoning items on OpenAI Responses requests and preserve
  fields delivered before `response.completed`.
- [x] Replay opaque native items while keeping encrypted data out of summaries.
- [x] Keep `reasoning_content` as continuation metadata, not final response text.
- [x] Run focused request, stream, history, and provider tests plus changed-surface
  static checks.

## Non-goals

- No dynamic provider catalog, new PentestAgent provider, or second model loop.
- No `previous_response_id` session state or interpretation of private reasoning.
- No new provider-specific environment variables or reasoning prompt format.

## Acceptance

Provider requests contain one internally consistent reasoning control, GPT-5.6 keeps
`max`, OpenAI Responses tool rounds replay encrypted reasoning items exactly, compatible
providers receive no unsupported OpenAI-only `include`, and private continuation bytes
never enter trace summaries or visible final text.
