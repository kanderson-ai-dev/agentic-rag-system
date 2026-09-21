"""Minimal thread-safe TTL cache for expensive, idempotent read calls.

Not a general-purpose cache: it exists specifically to absorb the network
latency of repeated identical queries against cloud backends (Neo4j Aura,
Pinecone) that dominate end-to-end request latency, at the cost of a small,
bounded staleness window. Writes/mutations are never cached.
"""

import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """A tiny in-memory cache with per-entry expiry and a bounded size."""

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 256) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._lock = threading.Lock()
        self._store: dict[str, tuple[float, T]] = {}

    def get_or_compute(self, key: str, compute: Callable[[], T]) -> T:
        """Return the cached value for `key`, computing and storing it on a miss."""
        now = time.monotonic()
        with self._lock:
            cached = self._store.get(key)
            if cached is not None and cached[0] > now:
                return cached[1]

        value = compute()

        with self._lock:
            if len(self._store) >= self._max_size and key not in self._store:
                oldest_key = min(self._store, key=lambda k: self._store[k][0])
                self._store.pop(oldest_key, None)
            self._store[key] = (now + self._ttl, value)
        return value
