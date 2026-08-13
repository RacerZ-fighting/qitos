# Session Journal production boundary

## Goal

Keep one canonical JSON value across append, replay, stable-record-id settlement, JSONL,
and SQLite projection digests. Treat unsupported schemas as an upgrade concern rather
than corruption, and never leak a forked child if closing the source journal fails.

## Success conditions

- Append normalizes supported Python JSON containers before any in-memory state or I/O.
- Unsupported or lossy payload values fail before a record is written.
- Reopening a journal does not change stable-record-id comparison behavior.
- Unsupported old and new schemas fail closed with a dedicated public error.
- A failed source close after fork also closes the unreturned child.
- SQLite remains a disposable projection built only from validated JSONL.

## Verification

Add Journal and Engine contract tests first, then run the complete QitOS test, flake8,
and mypy gates. Update the session Journal docs, changelog, and both README news lists.
