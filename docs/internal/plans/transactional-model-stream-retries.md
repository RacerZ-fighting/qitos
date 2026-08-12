# Transactional model stream retries

## Goal

Prevent a stalled OpenAI-compatible response from consuming the full non-streaming
request timeout on every retry, without allowing partial text or tool calls from a
failed attempt to enter the Agent loop.

## Design

- Keep QitOS as the only retry owner and keep SDK retries disabled.
- Preserve the public live `stream()` / `astream()` APIs.
- Give the Engine a complete-attempt stream path that buffers one attempt until its
  terminal event.
- Retry retryable failures even after partial provider output; discard that attempt's
  buffered output before retrying.
- Treat a stream that closes without a terminal event as retryable transport failure.
- Use the stream idle timeout between provider events instead of imposing the ordinary
  request timeout on a healthy long-running response.
- Bound recovery after the first retryable failure by a configurable 300-second window
  as well as attempt count, so stalled attempts cannot multiply into an unbounded wait.

## Verification

- A mid-stream timeout retries and exposes only the successful attempt.
- Failed-attempt tool-call deltas are never executed.
- Non-retryable provider errors still fail immediately.
- The Engine selects the complete-attempt stream by default when the adapter supports
  it and preserves native tool calls.
