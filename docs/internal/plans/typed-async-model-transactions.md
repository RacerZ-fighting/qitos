# Model runtime ownership refactor

## Status

Active on `refactor/model-continuation`.

This plan replaces the earlier proposal to build a parallel typed model stack.
QitOS already owns model content, responses, providers, retries, Engine model
execution, history, checkpoints, and traces. Those implementations are the
starting point and must be corrected in place. They are not inputs to a new
gateway, mirror contract, or codec package.

Only the model capability is in the current implementation batch. Tool,
Session/Child, Runtime, Plan/Skill/MCP, and legacy pipeline removal are deferred
until the model batch has passed its own acceptance gate.

The native async transport and transactional retry slices are complete. The active
slice closes the remaining request-snapshot, provider-continuation, canonical replay,
and typed stream-event gaps without adding a second model path.

## Non-negotiable ownership rules

1. Inspect the current owner, production callers, exports, and behavior tests
   before changing a capability.
2. Extend or replace the current owner in place. Do not add a wrapper, gateway,
   compatibility alias, second runtime, mirror DTO, or duplicate encode/decode
   path.
3. Do not create a generic model codec merely because PentestAgent currently has
   one. First migrate QitOS's real History, CheckpointStore, or TraceWriter
   behavior. Keep serialization at that existing persistence boundary unless at
   least two real QitOS callers require the same durable representation.
4. If common serialization is eventually justified, extract one implementation
   from working callers and delete their local copies in the same change. Do not
   expose standalone request, response, or stream-fragment codecs without a
   production persistence requirement.
5. A replacement is complete only when production callers, tests, exports, and
   documentation use the canonical owner and the superseded path is removed.
6. Complete, verify, commit, and push one behavior slice before starting the
   next. Do not leave half-migrated providers or a compatibility dual stack.

## Confirmed requirements

The existing QitOS model path will be improved to provide:

- a genuinely asynchronous Provider-to-Engine path;
- first-class OpenAI Responses and Anthropic Messages behavior, with Chat
  Completions available only when deliberately selected;
- preservation of provider semantics needed for replay, including tool call
  correlation, reasoning/thinking state, continuation, usage, and finish state;
- one retry/deadline owner, SDK retries disabled where QitOS retries, absolute
  deadline propagation, cancellation re-raising, and deterministic cleanup;
- transaction boundaries that prevent partial failed model output from being
  committed or causing tool execution;
- durable replay through the existing History, CheckpointStore, and TraceWriter
  owners, based only on data those production paths actually need;
- terminal checkpoints return their persisted state without invoking a model or tool;
- removal of superseded synchronous/async duplicates once all their callers are
  migrated in the same behavior slice.

Provider SDK objects remain inside provider implementations. QitOS does not
adopt PentestAgent's local model modules as its design.

## Existing owners to inspect and modify

The exact call graph is verified before code changes. The initial owner map is:

- model content: `qitos/core/multimodal.py`;
- completed model result: `qitos/core/model_response.py`;
- model/provider base and construction: `qitos/models/base.py` and existing
  provider modules;
- timeout, retry, and transport resource lifecycle: `qitos/models/transport.py`;
- Engine model execution: `qitos/engine/_model_runtime.py`, `engine.py`, and
  `async_engine.py`;
- durable state: `qitos/core/history.py`, `qitos/checkpoint/store.py`, and
  `qitos/trace/writer.py`.

This list identifies responsibility, not a promise to retain every current file
or class. Files are split or removed only after behavior is working and a
focused size/style review shows that doing so improves ownership rather than
duplicates it.

## Serial execution plan

### 1. Restore and record the clean baseline

- Remove all interrupted parallel implementation artifacts.
- Keep only this plan and the repository-wide single-owner rule.
- Run the complete current QitOS test suite.
- Commit and push this documentation-only change.

Success means the branch contains no untracked model contracts, codec modules,
or half-migrated provider code.

### 2. Prove the current production model path

- Trace every production caller from configuration/factory through Provider,
  Engine, History, checkpoint, and trace.
- Inventory the current sync/async entry points, retry owners, response shapes,
  and exported names.
- Identify the smallest end-to-end behavior slice that can be replaced without
  an alias or temporary parallel contract.
- Add behavior tests at the existing owner/caller boundary before changing its
  implementation. Do not add source-layout or fixed-string tests.

This step may revise the later steps when the observed call graph contradicts
the initial owner map.

### 3. Make the existing model execution path async and transactional

- Refactor the selected existing owners and all their production callers in one
  closed slice.
- Keep one model invocation path and one retry/deadline implementation.
- Prove cancellation, timeout, retry, partial-stream failure, terminal commit,
  ordering, and resource cleanup with controlled async tests.
- Remove the superseded path, exports, and tests in the same change.
- Run focused tests, full tests, static checks, packaging checks, and diff
  review; then commit and push before continuing.

No persistence codec is part of this step unless an existing persistence caller
is necessarily on the execution path and its current representation cannot
express the required committed result.

### 4. Persist the committed model transaction through existing owners

- Change History first, then CheckpointStore and TraceWriter only as required by
  their real replay behavior.
- Preserve complete tool-call correlation and provider continuation needed to
  resume, while keeping local history as the source of truth.
- Introduce shared serialization only if repeated production code now proves it
  necessary; otherwise keep it with its persistence owner.
- Migrate readers and writers together and delete the replaced representation.
- Verify replay/resume and malformed-data failure behavior, then commit and push.

The concrete implementation keeps four boundaries small:

- one immutable `ModelRequest` is built after context projection and is the only value
  passed from Engine to a Provider;
- one typed `ModelContinuation` may accompany a request, but is accepted only when its
  Run, provider, model, protocol, request settings, and canonical input prefix match;
- `model.completed` stores the exact request snapshot and the next continuation handle;
- continuation rejection retries once with the same full canonical request and without
  the handle. Resume may reuse a matching handle, while fork and Provider/model changes
  necessarily fall back to the transcript.

Continuation never replaces History. Responses can send an incremental input only after
validating that the current full input starts with the exact request-and-response prefix
recorded by the preceding committed transaction. Anthropic keeps its thinking signature
and message id in canonical native history, but does not claim server-side continuation
because Messages has no equivalent replay handle.

### 5. Migrate PentestAgent directly to QitOS

Only after the QitOS model batch is committed and passes its own gate:

- update the QitOS gitlink;
- replace PentestAgent model imports with direct QitOS imports;
- delete PentestAgent's duplicate `json_value`, contracts, codec, stream, and
  gateway implementations rather than wrapping them;
- update PentestAgent architecture documentation to match the implemented
  boundary;
- run `make check`, then commit and push the PentestAgent change.

### 6. Review architecture and verify remotely

- Review changed files for style, module length, ownership, dependency choices,
  blocking I/O, task leaks, and accidental GIL-bound work on the event loop.
- Compare only directly relevant behavior with the approved local reference
  projects; do not copy or add them as dependencies.
- Check out the pushed branches in the authorized remote environment and run the
  applicable project gates there.
- Record any measured test-runtime bottleneck before changing test parallelism.

## Acceptance gate for every behavior commit

Before a behavior commit is pushed:

1. The slice has one production implementation and no compatibility dual stack.
2. Every changed behavior has tests at a public or real caller boundary.
3. Focused tests and the complete QitOS suite pass.
4. Flake8 and mypy pass for the changed stable surfaces, or pre-existing baseline
   failures are recorded separately and the change adds none.
5. Packaging and import smoke tests pass when exports or dependencies change.
6. `git diff --check` passes and the diff contains no credentials, local paths,
   `tmp/` artifacts, caches, or unrelated rewrites.
7. The implementation and documentation describe the same single owner.

## Verification commands

Run from the QitOS repository using a cache under the repository's `tmp/`
directory:

```bash
UV_CACHE_DIR=tmp/uv-cache uv run --no-project --python 3.11 \
  --with-editable '.[models]' --with 'pytest>=7' \
  --with 'pytest-asyncio>=0.23' pytest -q
UV_CACHE_DIR=tmp/uv-cache uv run --no-project --python 3.11 \
  --with-editable '.[models]' --with 'flake8>=6' \
  flake8 qitos/core qitos/engine qitos/models qitos/trace
UV_CACHE_DIR=tmp/uv-cache uv run --no-project --python 3.11 \
  --with-editable '.[models]' --with 'mypy>=1' \
  mypy qitos/core qitos/engine qitos/models qitos/trace
UV_CACHE_DIR=tmp/uv-cache uv run --no-project --python 3.11 \
  --with build --with twine python -m build
UV_CACHE_DIR=tmp/uv-cache uv run --no-project --python 3.11 \
  --with twine twine check dist/*
git diff --check
```

## Active progress

- [x] Prove the current Provider, Engine, History, Journal, and trace call graph.
- [x] Replace `messages + **kwargs` dispatch with one immutable request value.
- [x] Persist and validate Responses continuation with full-request fallback.
- [x] Replace ambiguous stream chunks with discriminated typed events.
- [ ] Prove long-history projection, resume, fork, Provider switch, and invalid-handle behavior.
- [ ] Run the complete QitOS gate and merge the feature branch into `main`.

Run from PentestAgent only after the QitOS commit and gitlink update:

```bash
uv sync --locked
make check
git diff --check
```
