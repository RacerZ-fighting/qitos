"""Behavior tests for model transaction caching and cache backends."""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from qitos import Action, AgentModule, Decision, Engine, StateSchema, ToolRegistry, tool
from qitos.cache import CachedModel, DiskCache, InMemoryCache
from qitos.core import ModelUsage, ModelUsageSource
from qitos.engine import RuntimeBudget
from qitos.models import Model, ModelStreamChunk


@dataclass
class DemoState(StateSchema):
    logs: list[str] = field(default_factory=list)


class DemoAgent(AgentModule[DemoState, dict[str, Any], Action]):
    def __init__(self, answer: str = "42", with_llm: bool = False) -> None:
        registry = ToolRegistry()

        @tool(name="add")
        def add(a: int, b: int) -> int:
            return a + b

        registry.register(add)
        self._answer = answer
        super().__init__(tool_registry=registry)
        if with_llm:
            self.llm = _StubModel(answer=answer)

    def init_state(self, task: str, **kwargs: Any) -> DemoState:
        _ = kwargs
        return DemoState(task=task, max_steps=3)

    def decide(self, state: DemoState, observation: dict[str, Any]) -> Decision[Action]:
        _ = observation
        if state.current_step == 0:
            return Decision.act(
                actions=[Action(name="add", args={"a": 1, "b": 2})],
                rationale="use tool",
            )
        return Decision.final(self._answer)

    def reduce(
        self,
        state: DemoState,
        observation: dict[str, Any],
        decision: Decision[Action],
    ) -> DemoState:
        _ = observation, decision
        return state


class _StubModel(Model):
    def __init__(self, answer: str = "done", model: str = "stub") -> None:
        super().__init__(model=model)
        self.answer = answer
        self.call_count = 0

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        deadline_monotonic: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelStreamChunk]:
        _ = messages, deadline_monotonic, kwargs
        self.call_count += 1
        yield ModelStreamChunk(
            text=f"Final Answer: {self.answer}",
            reasoning_content="checked",
            event_type="text.delta",
        )
        yield ModelStreamChunk(
            done=True,
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            native_items=[{"type": "message", "id": "msg_1"}],
            event_type="model.completed",
            event_metadata={"provider": "stub", "model": self.model},
            finish_reason="stop",
        )


async def _collect(
    model: Model,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> list[ModelStreamChunk]:
    return [chunk async for chunk in model.stream(messages, **kwargs)]


class TestInMemoryCache:
    def test_get_set_delete_and_clear(self) -> None:
        cache = InMemoryCache()
        cache.set("one", b"1")
        cache.set("two", b"2")
        assert cache.get("one") == b"1"
        assert cache.contains("two") is True
        cache.delete("one")
        assert cache.get("one") is None
        cache.clear()
        assert cache.get("two") is None

    def test_ttl_expiry(self) -> None:
        cache = InMemoryCache()
        cache.set("key", b"value", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("key") is None

    def test_lru_evicts_oldest_entry(self) -> None:
        cache = InMemoryCache(max_entries=2)
        cache.set("a", b"1")
        cache.set("b", b"2")
        cache.set("c", b"3")
        assert cache.get("a") is None
        assert cache.get("b") == b"2"
        assert cache.get("c") == b"3"


class TestDiskCache:
    def test_round_trip_delete_clear_and_new_instance(self, tmp_path: Path) -> None:
        cache = DiskCache(str(tmp_path))
        cache.set("one", b"1")
        cache.set("two", b"2")
        assert DiskCache(str(tmp_path)).get("one") == b"1"
        cache.delete("one")
        assert cache.get("one") is None
        cache.clear()
        assert cache.get("two") is None

    def test_ttl_expiry(self, tmp_path: Path) -> None:
        cache = DiskCache(str(tmp_path))
        cache.set("key", b"value", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("key") is None


class TestCachedModel:
    @pytest.mark.asyncio
    async def test_complete_transaction_is_cached_without_provider_objects(
        self,
    ) -> None:
        wrapped = _StubModel(answer="hello")
        cached = CachedModel(wrapped, InMemoryCache())
        messages = [{"role": "user", "content": "test"}]

        first = await _collect(cached, messages)
        second = await _collect(cached, messages)

        assert first == second
        assert wrapped.call_count == 1
        assert second[-1].usage == {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
        }
        assert isinstance(second[-1].usage, ModelUsage)
        assert second[-1].usage.source is ModelUsageSource.PROVIDER
        assert second[-1].native_items == [{"type": "message", "id": "msg_1"}]
        assert cached.stats == {"hits": 1, "misses": 1}

    @pytest.mark.asyncio
    async def test_different_requests_are_independent_misses(self) -> None:
        wrapped = _StubModel()
        cached = CachedModel(wrapped, InMemoryCache())

        await _collect(cached, [{"role": "user", "content": "one"}])
        await _collect(cached, [{"role": "user", "content": "two"}])

        assert wrapped.call_count == 2
        assert cached.stats == {"hits": 0, "misses": 2}

    def test_key_is_deterministic_and_includes_provider(self) -> None:
        first = CachedModel(_StubModel(), InMemoryCache())
        second_model = _StubModel()
        second_model.provider_name = "different"
        second = CachedModel(second_model, InMemoryCache())
        messages = [{"role": "user", "content": "test"}]

        assert first._cache_key(messages, {}) == first._cache_key(messages, {})
        assert first._cache_key(messages, {}) != second._cache_key(messages, {})

    @pytest.mark.asyncio
    async def test_disabled_cache_bypasses_backend(self) -> None:
        wrapped = _StubModel()
        cached = CachedModel(wrapped, InMemoryCache(), enabled=False)
        messages = [{"role": "user", "content": "test"}]

        await _collect(cached, messages)
        await _collect(cached, messages)

        assert wrapped.call_count == 2
        assert cached.stats == {"hits": 0, "misses": 0}

    @pytest.mark.asyncio
    async def test_failed_transaction_is_not_published_or_cached(self) -> None:
        class FailOnceModel(_StubModel):
            async def stream(
                self,
                messages: list[dict[str, Any]],
                *,
                deadline_monotonic: float | None = None,
                **kwargs: Any,
            ) -> AsyncIterator[ModelStreamChunk]:
                self.call_count += 1
                if self.call_count == 1:
                    yield ModelStreamChunk(text="discarded")
                    raise TimeoutError("stream failed")
                async for chunk in super().stream(
                    messages,
                    deadline_monotonic=deadline_monotonic,
                    **kwargs,
                ):
                    yield chunk

        wrapped = FailOnceModel(answer="recovered")
        backend = InMemoryCache()
        cached = CachedModel(wrapped, backend)
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(TimeoutError, match="stream failed"):
            await _collect(cached, messages)
        assert backend.get(cached._cache_key(messages, {})) is None

        chunks = await _collect(cached, messages)
        assert "".join(chunk.text for chunk in chunks) == "Final Answer: recovered"
        assert cached.stats == {"hits": 0, "misses": 2}

    @pytest.mark.asyncio
    async def test_backend_io_runs_off_the_event_loop_thread(self) -> None:
        event_loop_thread = threading.get_ident()
        observed: dict[str, int] = {}

        class RecordingBackend(InMemoryCache):
            def get(self, key: str) -> bytes | None:
                observed["get"] = threading.get_ident()
                return super().get(key)

            def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
                observed["set"] = threading.get_ident()
                super().set(key, value, ttl)

        cached = CachedModel(_StubModel(), RecordingBackend())
        await _collect(cached, [{"role": "user", "content": "test"}])

        assert observed["get"] != event_loop_thread
        assert observed["set"] != event_loop_thread

    def test_forwards_stable_model_capabilities(self) -> None:
        wrapped = _StubModel()
        cached = CachedModel(wrapped, InMemoryCache())

        assert cached.model == "stub"
        assert cached.temperature == 0.7
        assert cached.max_tokens == 2048
        assert cached.capabilities == wrapped.capabilities
        assert cached.count_tokens("one two") == wrapped.count_tokens("one two")


class TestEngineCacheIntegration:
    def test_engine_wraps_the_canonical_model(self) -> None:
        agent = DemoAgent(answer="cached", with_llm=True)
        engine = Engine(
            agent=agent,
            budget=RuntimeBudget(max_steps=5),
            cache_backend=InMemoryCache(),
        )

        assert isinstance(engine.agent.llm, CachedModel)

    def test_engine_without_cache_does_not_wrap(self) -> None:
        agent = DemoAgent(with_llm=True)
        model = agent.llm
        engine = Engine(agent=agent, budget=RuntimeBudget(max_steps=5))

        assert engine.cache_backend is None
        assert engine.agent.llm is model

    def test_engine_with_cache_preserves_non_model_agent_behavior(self) -> None:
        agent = DemoAgent(answer="from_cache")
        result = Engine(
            agent=agent,
            budget=RuntimeBudget(max_steps=5),
            cache_backend=InMemoryCache(),
        ).run("test task")

        assert result.state.final_result == "from_cache"
