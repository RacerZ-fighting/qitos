"""Bounded daemon workers for runtime-owned blocking operations."""

from __future__ import annotations

import queue
import threading
from contextvars import Context, copy_context
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, TypeVar


ResultT = TypeVar("ResultT")


class DaemonTaskPool:
    """Run a bounded number of blocking callables without owning process exit."""

    _STOP = object()

    def __init__(
        self,
        max_workers: int,
        *,
        thread_name_prefix: str,
        propagate_context: bool = False,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._max_workers = max_workers
        self._thread_name_prefix = thread_name_prefix
        self._propagate_context = propagate_context
        self._tasks: queue.Queue[Any] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._closed = False

    def submit(
        self,
        function: Callable[..., ResultT],
        *args: Any,
        **kwargs: Any,
    ) -> Future[ResultT]:
        future: Future[ResultT] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("daemon task pool is closed")
            context = copy_context() if self._propagate_context else None
            self._tasks.put((future, context, function, args, kwargs))
            if len(self._threads) < self._max_workers:
                thread = threading.Thread(
                    target=self._worker,
                    name=(
                        f"{self._thread_name_prefix}-{id(self):x}-"
                        f"{len(self._threads) + 1}"
                    ),
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()
        return future

    def shutdown(self, *, wait_for_workers: bool, cancel_futures: bool) -> None:
        """Stop accepting work and optionally cancel queued calls."""

        with self._lock:
            if not self._closed:
                self._closed = True
                if cancel_futures:
                    while True:
                        try:
                            item = self._tasks.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            if item is not self._STOP:
                                item[0].cancel()
                        finally:
                            self._tasks.task_done()
                for _ in self._threads:
                    self._tasks.put(self._STOP)
            threads = list(self._threads)
        if wait_for_workers:
            for thread in threads:
                thread.join()

    def _worker(self) -> None:
        while True:
            item = self._tasks.get()
            try:
                if item is self._STOP:
                    return
                future, context, function, args, kwargs = item
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    result = self._invoke(context, function, args, kwargs)
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(result)
            finally:
                self._tasks.task_done()

    @staticmethod
    def _invoke(
        context: Context | None,
        function: Callable[..., ResultT],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ResultT:
        if context is None:
            return function(*args, **kwargs)
        return context.run(function, *args, **kwargs)


__all__ = ["DaemonTaskPool"]
