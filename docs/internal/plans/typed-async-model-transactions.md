# Typed async model transactions

## Status

Approved for implementation on `feat/typed-model-transactions`.

This is an intentional public-API replacement. QitOS will have one model
transaction contract when the work is complete. It will not retain legacy
`ModelStreamChunk`, raw SDK responses, synchronous provider fallbacks, `V2`
types, aliases, gateways, or adapter DTOs.

## Goal

Make QitOS the single owner of the reusable model boundary required by agent
runtimes:

- immutable provider-neutral requests, ordered output items, tool calls,
  continuations, usage, failures, and stream events;
- strict JSON values and deterministic, versioned persistence codecs;
- one asynchronous adapter lifecycle for OpenAI Responses, Anthropic Messages,
  and Chat Completions compatibility;
- a native asynchronous Engine model path with request-scoped deadlines,
  cancellation, retries, resource cleanup, and exactly one terminal event;
- no provider SDK objects or long-lived `dict[str, Any]` payloads outside the
  provider wire boundary.

The change does not add a second agent loop. It replaces the model boundary
inside the existing `AgentModule + Engine` mainline.

## Modification layers

### Stable core

`qitos.core` owns:

- strict immutable `JsonValue`, including duplicate-key and non-finite-number
  rejection;
- canonical content, input, and ordered output items;
- `ModelRequest`, generation options, tool snapshots, `ToolCall`,
  `ModelResponse`, `Usage`, continuation variants, and typed failures;
- the `ModelEvent` union and its sequence/terminal/protocol validator;
- deterministic request, response, and event codecs with aggregate schema IDs
  and integer versions.

The wire protocol enum is named `ModelWireProtocol`. The existing
`qitos.protocols.ModelProtocol` describes prompt/parser interaction formats and
is not reused for provider transport.

The codec uses the standard library and small typed helpers. Pydantic is not a
base QitOS dependency, would require coercion controls, and does not solve
duplicate-key preservation. `msgspec` would add a compiled core dependency
without a measured bottleneck. Provider SDKs remain optional model extras.

### Providers

`qitos.models` owns SDK and wire conversion. Existing provider class names are
rewritten to implement the asynchronous core protocol; separate `Async*`
classes are removed.

- OpenAI uses `AsyncOpenAI` with SDK retries disabled. Responses and Chat share
  request/tool/usage/failure helpers while retaining their distinct wire
  semantics.
- Anthropic uses the official asynchronous Messages client. It preserves block
  order, thinking signatures, redacted thinking, `tool_use`, `tool_result`,
  cache usage, and `message_stop` terminal semantics.
- Other shipped providers implement the same contract. A synchronous third-
  party transport may be isolated with `asyncio.to_thread`; it cannot introduce
  a synchronous model API or block the Engine event loop directly.

Each retry attempt owns and closes its stream. Cancellation is re-raised after
cleanup and is never retried. An attempt is published only after a valid
terminal event, so failed-attempt text, reasoning, usage, and partial tool calls
cannot enter Engine state.

### Engine and history projection

The existing `Engine` gains an async-native run path and consumes only
`ModelAdapter.stream(ModelRequest)`. The daemon-thread `AsyncEngine`,
`call_raw`, arbitrary callable probing, synthetic response dictionaries, and
`ModelStreamChunk` assembly are removed.

The synchronous entry, where QitOS still exposes one, may create one event loop
for the complete Engine run and must reject calls from an already-running loop.
It is not used by asynchronous consumers.

This slice projects the existing QitOS history policy into canonical request
items and writes completed typed outputs back at transaction boundaries. A
subsequent session-store change may change physical history storage, but it may
not reintroduce a second model representation or provider SDK state.

### Public consumers

QitOS configuration, cache, harness, trace, hooks, examples, and tests migrate
to the same types. Public exports come from `qitos.core` and `qitos`; model
implementations come from `qitos.models`.

PentestAgent is migrated only after QitOS passes its own checks and is committed.
It updates the QitOS gitlink, imports the QitOS contracts and codecs directly,
and deletes the replaced local model implementation. It does not add re-exports
or compatibility modules.

## Contract decisions

### Request snapshot

`ModelRequest` contains stable request and turn IDs, the selected model, system
blocks, canonical transcript items, strict tool snapshots, common generation
options, optional provider continuation, and an absolute monotonic deadline.
The deadline is process-local and is deliberately omitted from its durable
codec.

Transcript validation is single-pass and order-sensitive. A tool result must
follow a unique assistant call, and every call included in a request must have a
terminal result. Native data must belong to the selected wire protocol.

### Ordered output and continuation

Provider output is an ordered tuple of text, reasoning, tool-call, and opaque
items. `ToolCall.call_id` is the execution correlation ID; a Responses output
item ID is stored separately. Raw arguments are retained even when strict JSON
decoding fails.

Responses continuation retains response IDs and replayable ordered items.
Anthropic continuation retains replayable assistant blocks and signatures.
Continuation is an optimization and protocol requirement, not the only history
source.

### Events and terminal behavior

Events carry a request ID and strictly increasing sequence. They distinguish
model start, text, reasoning, tool-call start/argument/completion, cumulative
usage, continuation updates, completed response, and failed response.

Every normally exhausted stream has exactly one `ModelCompleted` or
`ModelFailed`; terminal is last. A completed response must contain at least one
meaningful output item. Tool execution is allowed only from complete calls in a
committed terminal response.

Expected provider, transport, decode, deadline, and unsupported errors become a
typed terminal failure after adapter-owned retry policy is exhausted.
`asyncio.CancelledError` is exceptional control flow: the adapter closes its
resources and re-raises it so the Engine/session owner can durably record
cancellation at its transaction boundary.

### Retry and timeout ownership

The adapter is the sole owner of model retries and disables SDK retries.
Authentication, billing, invalid request, unsupported, deadline, and
cancellation never retry. Context overflow can only request recovery after
compaction; it cannot replay the same request.

The absolute Engine deadline always wins. Connection timeout, event-idle
timeout, and retry backoff use only the remaining deadline. A new attempt never
receives a fresh full budget.

## Success conditions

1. QitOS exposes one immutable typed model contract and one versioned codec
   implementation from `qitos.core`.
2. `ModelStreamChunk`, raw `ModelResponse.raw`, public provider dictionaries,
   synchronous provider fallback, and the daemon-thread `AsyncEngine` are absent.
3. Responses, Anthropic Messages, and Chat compatibility pass the same adapter
   contract tests, including cancellation, deadline, partial failure, missing
   terminal, reasoning continuation, parallel tool calls, and usage.
4. The Engine never executes a partial tool call and commits only a validated
   terminal transaction.
5. Provider streams and clients close on success, failure, timeout, and
   cancellation; cancellation does not retry and no tasks are left pending.
6. QitOS README, changelog, English and Chinese reference docs, examples, and
   public API tests describe the implemented behavior rather than the removed
   interface.
7. QitOS passes its independent pytest, flake8, mypy, packaging, and diff
   hygiene checks before any PentestAgent gitlink update.
8. PentestAgent then deletes its replaced model JSON/contracts/codec/stream and
   directly consumes the QitOS types while passing its own `make check`.

## Key assumptions

- Python 3.10 remains QitOS's minimum, so enums use the Python 3.10-compatible
  string-enum form and public typing does not rely on 3.11-only syntax.
- Provider SDKs are optional model extras; importing `qitos.core` never imports
  or initializes an SDK client.
- Chat Completions remains a deliberately selected compatibility protocol and
  never acts as a fallback for Responses or Anthropic.
- Existing parser protocols remain QitOS interaction-format policy. They are
  projected into prompts but do not own provider transport state.
- Reference projects are behavioral evidence only. Their source is not copied
  and they are not added as dependencies.

## Implementation sequence

1. Add strict JSON/codec primitives and the final core model contracts with
   malformed-data, round-trip, event-sequence, and cancellation contract tests.
2. Rewrite OpenAI Responses and Chat to the async contract, including strict
   request projection, retry/deadline ownership, terminal buffering, and close
   tests.
3. Rewrite Anthropic Messages to the same contract and add block-order,
   thinking/signature, tool-result, cache-usage, failure, and close tests.
4. Migrate the remaining shipped providers, cache, factory, and configuration;
   remove old model entry points and public exports.
5. Migrate the existing Engine model path and history projection; remove
   `AsyncEngine`, arbitrary response normalization, and old stream types.
6. Update QitOS docs, README/README.zh, CHANGELOG, examples, packaging, and
   quality configuration; run and record the independent QitOS gate.
7. Commit and push QitOS. Then update PentestAgent's gitlink and direct imports,
   delete its duplicate implementation/tests, update its architecture docs, run
   `make check`, and commit/push PentestAgent separately.

## Verification commands

Run from the QitOS repository:

```bash
uv run --no-project --python 3.11 --with-editable '.[models]' \
  --with 'pytest>=7' --with 'pytest-asyncio>=0.23' pytest -q
uv run --no-project --python 3.11 --with-editable '.[models]' \
  --with 'flake8>=6' flake8 qitos/core qitos/engine qitos/models qitos/trace
uv run --no-project --python 3.11 --with-editable '.[models]' \
  --with 'mypy>=1' mypy qitos/core qitos/engine qitos/models qitos/trace
uv run --no-project --python 3.11 --with-editable '.[models]' \
  --with build --with twine python -m build
uv run --no-project --python 3.11 --with twine twine check dist/*
git diff --check
```

Run from PentestAgent only after the QitOS commit and gitlink update:

```bash
uv sync --locked
make check
git diff --check
```
