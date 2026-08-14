# Runtime terminal recovery contracts

## Goal

Keep Tool, Child, process, and mailbox identity recoverable from the canonical Run
Journal without adding another state store.

## Delivery

- [x] Bind every canonical `ToolResult` to its model `call_id`, while accepting older
  terminal records that do not contain the field.
- [x] Allow Child invocation factories to finish async resource construction before a
  Child Engine starts.
- [x] Derive idempotent background Child and process completion inputs from local
  terminal records; foreground Child results are not redelivered, consumed inputs stay
  consumed, and inherited terminals do not become fork input.
- [x] Re-scan terminal facts after resume-time tool setup so recovered Child terminals
  reach the inbox before the next model transaction.
- [x] Pass focused tests, stable-surface flake8/mypy, and the complete QitOS test suite.
- [x] Synchronize public docs, changelog, and bilingual README news.

## Compatibility

Existing synchronous Child factories remain valid. Older `ToolResult` records recover
their identity from the adjacent canonical Action. New `child.started` records persist
the foreground/background delivery policy. Older Child journals keep replaying any
durable `runtime_input.posted` record but are not guessed to be background when that
marker is absent. Process terminals already contain everything needed to reconstruct the
same bounded completion event. No Journal migration or second replay source is introduced.
