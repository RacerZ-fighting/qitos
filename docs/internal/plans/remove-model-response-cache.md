# Remove the implicit model response cache

## Decision

Remove `qitos.cache` and `Engine(cache_backend=...)`. QitOS keeps provider-native
prompt-cache controls and usage facts, but does not replay complete model transactions
from a local Agent-kernel cache.

## Evidence

- The package already declares itself deprecated. Its only runtime attachment is an
  optional `Any` parameter on `Engine`; no QitOS recipe, example, PentestAgent caller,
  or qitos-zoo application supplies it.
- Engine construction mutates `agent.llm` by wrapping it implicitly. Reusing an Agent
  across Engines can therefore inherit process-local behavior that is absent from the
  Run request and trace provenance.
- The file backend persists `time.monotonic()` expiry values, which are not durable
  across process or host restarts, and silently treats corrupt entries and write errors
  as misses.
- Cached transactions replay prior usage and native continuation items as if they came
  from a fresh provider request. This conflicts with live deadline, billing, provider
  continuation, and trace semantics.
- Pi and Hermes use provider-native prompt caching and preserve cache-read/cache-write
  usage. Codex caches explicit resources such as model catalogs, not complete Agent
  model responses.

## Scope

1. Delete `qitos.cache` and its self-only behavior tests.
2. Remove the Engine constructor parameter and implicit model mutation.
3. Retain model capability and provider prompt-cache behavior unchanged.
4. Record the breaking removal in the changelog and bilingual README news.

## Verification

- Repository search has no production import or public API reference to the removed
  package or Engine parameter.
- Full tests and stable-surface static checks pass on Python 3.11.
- Focused Engine/model tests pass on Python 3.11.
- Built wheel and sdist contain no `qitos/cache` package.

## Progress

- [x] Remove the package, Engine hook, and self-only tests.
- [x] Update public and internal documentation.
- [x] Pass full tests, static checks, and distribution validation.
