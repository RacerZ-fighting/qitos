"""@task decorator — wrap functions as composable task units.

A ``@task`` wraps a plain function so it can be:
1. Called directly like a regular function
2. Composed within an ``@agent`` for parallel execution

Unlike LangGraph's ``@task``, QitOS tasks are simpler: they don't
require a Pregel graph and work directly with the Engine.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, cast, Generic, Optional, TypeVar, overload

P = TypeVar("P")
T = TypeVar("T")
F = TypeVar("F", bound=Callable)


class TaskFunction(Generic[P, T]):
    """A function wrapped by @task.

    Can be called directly (sync) or via ``.submit()`` for parallel execution.
    """

    def __init__(
        self,
        func: Callable[..., T],
        *,
        name: Optional[str] = None,
    ) -> None:
        self._func = func
        self._name = str(name or getattr(func, "__name__", "task"))
        self._is_async = asyncio.iscoroutinefunction(func)
        functools.update_wrapper(self, func, updated=())

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_async(self) -> bool:
        return self._is_async

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Call the task directly (blocking)."""
        return self._func(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> T:
        """Call the task asynchronously."""
        if self._is_async:
            awaitable = cast(Awaitable[T], self._func(*args, **kwargs))
            return await awaitable
        return await asyncio.to_thread(self._func, *args, **kwargs)

    def submit(self, executor: Optional[ThreadPoolExecutor] = None, *args: Any, **kwargs: Any) -> Future:
        """Submit the task for parallel execution.

        Returns a concurrent.futures.Future that can be collected later.
        Only works for sync functions. For async, use acall() with asyncio.gather.
        """
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1)
        return executor.submit(self._func, *args, **kwargs)

    def __repr__(self) -> str:
        return f"TaskFunction({self._name!r})"


@overload
def task(
    __func_or_none__: None = None,
    *,
    name: Optional[str] = None,
) -> Callable[[F], TaskFunction]:
    ...


@overload
def task(
    __func_or_none__: F,
    *,
    name: Optional[str] = None,
) -> TaskFunction:
    ...


def task(
    __func_or_none__: Any = None,
    *,
    name: Optional[str] = None,
) -> Any:
    """Decorator to wrap a function as a composable task unit.

    Can be used bare (``@task``) or with an explicit name
    (``@task(name="search")``).

    Parameters
    ----------
    name : str, optional
        Name for the task (defaults to function name).
    """

    def decorator(func: F) -> TaskFunction:
        return TaskFunction(
            func,
            name=name,
        )

    if __func_or_none__ is not None:
        # Bare usage: @task
        return decorator(__func_or_none__)

    # With arguments: @task(name=...)
    return decorator


__all__ = ["task", "TaskFunction"]
