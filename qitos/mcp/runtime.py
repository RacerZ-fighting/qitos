"""Engine-owned event-loop runtime for long-lived MCP transports.

MCP transports bind subprocess streams and async HTTP clients to the loop where
they connect.  QitOS tools execute synchronously in worker threads, so MCP calls
must be submitted back to that same loop instead of creating a new loop for each
call.  This small runtime owns one daemon loop for one Engine run.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Coroutine, Optional


class MCPEventLoopRuntime:
    """Run MCP transport coroutines on one dedicated event-loop thread."""

    # This run-scoped variant directly adapts the proven dedicated-loop shape
    # from hermes:tools/mcp_tool.py without its process-global registry.

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    @property
    def running(self) -> bool:
        loop = self._loop
        thread = self._thread
        return bool(loop is not None and loop.is_running() and thread is not None)

    def start(self) -> None:
        """Start the owned loop once and wait until it accepts submissions."""
        if self.running:
            return
        if self._thread is not None:
            raise RuntimeError("MCP event-loop runtime cannot be restarted")

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._ready.set()
            try:
                loop.run_forever()
            finally:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()

        self._thread = threading.Thread(
            target=_run,
            name="qitos-mcp-loop",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0) or not self.running:
            raise RuntimeError("MCP event-loop runtime failed to start")

    def submit(self, coroutine: Coroutine[Any, Any, Any]) -> Future[Any]:
        """Submit a coroutine to the owned loop from any other thread."""
        loop = self._loop
        if loop is None or not loop.is_running():
            coroutine.close()
            raise RuntimeError("MCP event-loop runtime is not running")
        if threading.current_thread() is self._thread:
            coroutine.close()
            raise RuntimeError("cannot synchronously wait on the MCP event-loop thread")
        return asyncio.run_coroutine_threadsafe(coroutine, loop)

    async def run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        """Await a coroutine on the owned loop without blocking the caller loop."""
        return await asyncio.wrap_future(self.submit(coroutine))

    def run_sync(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        """Run a transport coroutine from a synchronous tool worker."""
        return self.submit(coroutine).result()

    async def close(self) -> None:
        """Stop the loop after server cleanup and join its daemon thread."""
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            return
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        await asyncio.to_thread(thread.join, 5.0)
        if thread.is_alive():
            raise RuntimeError("MCP event-loop runtime did not stop")
        self._loop = None


__all__ = ["MCPEventLoopRuntime"]
