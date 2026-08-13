"""Small synchronization helpers shared by filesystem backends."""

from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from typing import Iterator


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def normalize_expected_sha256(value: str | None) -> str | None:
    """Validate and normalize an optional lowercase SHA-256 revision."""

    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    return normalized


class FileMutationQueue:
    """Serialize mutations to one canonical path within a backend instance."""

    def __init__(self) -> None:
        self._registration_lock = threading.Lock()
        self._entries: dict[str, tuple[threading.Lock, int]] = {}

    @contextmanager
    def hold(self, key: str) -> Iterator[None]:
        """Wait for prior mutations of ``key`` and release the queue on exit."""

        with self._registration_lock:
            lock, users = self._entries.get(key, (threading.Lock(), 0))
            self._entries[key] = (lock, users + 1)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            with self._registration_lock:
                current_lock, current_users = self._entries[key]
                if current_lock is not lock:
                    raise RuntimeError("file mutation queue ownership changed")
                if current_users == 1:
                    del self._entries[key]
                else:
                    self._entries[key] = (lock, current_users - 1)


__all__ = ["FileMutationQueue", "normalize_expected_sha256"]
