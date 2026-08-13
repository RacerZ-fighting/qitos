"""@function_tool decorator for QitOS — creates a FunctionTool from a plain function."""

from __future__ import annotations

from typing import Any, Callable, Optional

from .tool import FunctionTool, ToolMeta, RetryPolicy


def function_tool(
    func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    group: str = "default",
    timeout_s: Optional[float] = None,
    retry_policy: Optional[RetryPolicy] = None,
    on_failure: Optional[Callable] = None,
    read_only: bool = False,
    concurrency_safe: Optional[bool] = None,
    needs_approval: bool = False,
    **extra_meta: Any,
) -> Any:
    """Decorator that creates a :class:`FunctionTool` from a plain function.

    Can be used with or without parentheses::

        @function_tool
        def greet(name: str) -> str: ...

        @function_tool(name="custom", needs_approval=True)
        def greet(name: str) -> str: ...

    Returns a :class:`FunctionTool` instance.
    """

    def _make_tool(fn: Callable[..., Any]) -> FunctionTool:
        meta = ToolMeta(
            name=name,
            description=description,
            group=group,
            timeout_s=timeout_s,
            retry_policy=retry_policy,
            on_failure=on_failure,
            read_only=read_only,
            concurrency_safe=concurrency_safe,
            needs_approval=needs_approval,
        )
        # Store extra_meta as attributes on meta for future extensibility
        for key, value in extra_meta.items():
            setattr(meta, key, value)

        import inspect

        tool_instance = FunctionTool(fn, meta)
        # Description: explicit meta.description overrides docstring
        if meta.description:
            tool_instance.spec.description = inspect.cleandoc(meta.description)

        return tool_instance

    if func is not None:
        # Used as @function_tool without parentheses
        return _make_tool(func)

    # Used as @function_tool(...) with parentheses
    return _make_tool
