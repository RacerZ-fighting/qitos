"""Process-scoped writer ownership for one Run journal."""

from __future__ import annotations

import errno
import json
import os
import sys
from pathlib import Path

from ...core.journal import JournalError, JournalOwnershipError


class JournalWriterLease:
    """Hold one non-blocking operating-system lock until explicit close."""

    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self._descriptor = descriptor
        self._released = False

    @classmethod
    def acquire(cls, run_directory: Path, run_id: str) -> "JournalWriterLease":
        path = run_directory / "journal.writer.lock"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError as exc:
            raise JournalError(f"failed to open writer lease for Run '{run_id}'") from exc
        try:
            _lock_descriptor(descriptor)
            owner = json.dumps(
                {"pid": os.getpid(), "run_id": run_id},
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, owner)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise JournalOwnershipError(
                f"Run '{run_id}' already has an active Journal writer"
            ) from exc
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                os.close(descriptor)
                raise JournalOwnershipError(
                    f"Run '{run_id}' already has an active Journal writer"
                ) from exc
            try:
                _unlock_descriptor(descriptor)
            finally:
                os.close(descriptor)
            raise JournalError(
                f"failed to acquire writer lease for Run '{run_id}'"
            ) from exc
        return cls(path, descriptor)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            _unlock_descriptor(self._descriptor)
        finally:
            os.close(self._descriptor)


def _lock_descriptor(descriptor: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_descriptor(descriptor: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


__all__ = ["JournalWriterLease"]
