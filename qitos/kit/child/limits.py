"""Root-Run admission limits shared by recursively nested Child supervisors."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass

from ...core.child import ChildRunLimitError


@dataclass(slots=True)
class _ChildRunLease:
    owner: "ChildRunLimiter"
    committed: bool = False
    released: bool = False

    def commit(self) -> None:
        """Make this launch consume the cumulative budget permanently."""

        if self.released:
            raise RuntimeError("cannot commit a released Child Run lease")
        self.committed = True

    async def release(self) -> None:
        """Return the active slot while retaining cumulative launch usage."""

        await self.owner._release(self, rollback=False)

    async def rollback(self) -> None:
        """Return an admission that never became a durable Child launch."""

        await self.owner._release(self, rollback=True)


class ChildRunLimiter:
    """Bound active and cumulative Child launches across one recursive Run tree.

    Each nested :class:`ChildSupervisor` receives the same instance. Admission is
    immediate rather than queued so a foreground Child cannot deadlock while it
    waits for a descendant to acquire the last active slot.
    """

    def __init__(self, *, max_active_children: int, max_children: int) -> None:
        for name, value in (
            ("max_active_children", max_active_children),
            ("max_children", max_children),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self._max_active_children = max_active_children
        self._max_children = max_children
        self._active_children = 0
        self._children_started = 0
        self._restored_launches: set[str] = set()
        self._lock = asyncio.Lock()

    @property
    def max_active_children(self) -> int:
        return self._max_active_children

    @property
    def max_children(self) -> int:
        return self._max_children

    @property
    def active_children(self) -> int:
        return self._active_children

    @property
    def children_started(self) -> int:
        return self._children_started

    async def reserve(self) -> _ChildRunLease:
        """Reserve one active slot and one provisional cumulative launch."""

        async with self._lock:
            if self._children_started >= self._max_children:
                raise ChildRunLimitError(
                    "Run child-agent budget exhausted: "
                    f"max_children={self._max_children}."
                )
            if self._active_children >= self._max_active_children:
                raise ChildRunLimitError(
                    "Run active child-agent limit exhausted: "
                    f"max_active_children={self._max_active_children}."
                )
            self._children_started += 1
            self._active_children += 1
        return _ChildRunLease(owner=self)

    async def restore_started(self, launch_ids: Iterable[str]) -> None:
        """Restore durable launches without treating terminal Children as active."""

        normalized_items: list[str] = []
        for item in launch_ids:
            if not isinstance(item, str):
                raise TypeError("launch_ids must contain strings")
            normalized = item.strip()
            if not normalized:
                raise ValueError("launch_ids must contain non-empty strings")
            normalized_items.append(normalized)
        normalized_launches = set(normalized_items)
        async with self._lock:
            unseen = normalized_launches.difference(self._restored_launches)
            restored_total = self._children_started + len(unseen)
            if restored_total > self._max_children:
                raise ChildRunLimitError(
                    "Restored Child launch history exceeds the Run budget: "
                    f"max_children={self._max_children}, "
                    f"children_started={restored_total}."
                )
            self._restored_launches.update(unseen)
            self._children_started += len(unseen)

    def reset_for_new_run(self) -> None:
        """Clear cumulative admission state at an owner-Run boundary.

        The composition root calls this only after every supervisor-owned task
        has settled.  Resetting a limiter with an active lease would detach
        that lease from its accounting owner, so it fails closed instead.
        """

        if self._active_children:
            raise RuntimeError("cannot reset Child Run limits with active children")
        self._children_started = 0
        self._restored_launches.clear()

    async def _release(
        self,
        lease: _ChildRunLease,
        *,
        rollback: bool,
    ) -> None:
        if lease.owner is not self:
            raise RuntimeError("Child Run lease belongs to another limiter")
        async with self._lock:
            if lease.released:
                return
            if rollback and lease.committed:
                raise RuntimeError("cannot roll back a durable Child launch")
            self._active_children -= 1
            if rollback:
                self._children_started -= 1
            lease.released = True


__all__ = ["ChildRunLimiter"]
