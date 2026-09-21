"""Tests for the TTL cache used to absorb cloud retrieval latency."""

import time

from app.core.cache import TTLCache


class TestTTLCache:
    def test_second_call_hits_cache(self) -> None:
        cache: TTLCache[int] = TTLCache(ttl_seconds=60.0)
        calls = []

        def compute() -> int:
            calls.append(1)
            return 42

        assert cache.get_or_compute("key", compute) == 42
        assert cache.get_or_compute("key", compute) == 42
        assert len(calls) == 1

    def test_different_keys_do_not_share_entries(self) -> None:
        cache: TTLCache[int] = TTLCache(ttl_seconds=60.0)

        assert cache.get_or_compute("a", lambda: 1) == 1
        assert cache.get_or_compute("b", lambda: 2) == 2

    def test_expired_entry_is_recomputed(self) -> None:
        cache: TTLCache[int] = TTLCache(ttl_seconds=0.01)
        calls = []

        def compute() -> int:
            calls.append(1)
            return len(calls)

        assert cache.get_or_compute("key", compute) == 1
        time.sleep(0.02)
        assert cache.get_or_compute("key", compute) == 2

    def test_evicts_oldest_entry_when_full(self) -> None:
        cache: TTLCache[int] = TTLCache(ttl_seconds=60.0, max_size=2)

        cache.get_or_compute("a", lambda: 1)
        cache.get_or_compute("b", lambda: 2)
        cache.get_or_compute("c", lambda: 3)

        calls = []
        cache.get_or_compute("a", lambda: calls.append("a") or 99)
        assert calls == ["a"]  # "a" was evicted, so it had to be recomputed
