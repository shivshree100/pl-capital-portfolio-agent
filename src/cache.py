"""
src/cache.py — Lightweight in-memory TTL cache.

Provides a simple key/value store with per-entry time-to-live expiry.
No external dependencies — uses only stdlib (time, threading).

Design decisions
----------------
- Thread-safe using a single RLock (reentrant so get() can call itself).
- Entries are evicted lazily on access, not on a background timer.
- Max size enforced via LRU eviction to cap memory usage.
- Used for: SQL query results, exposure calculations, portfolio metadata.
"""

import time
import threading
from typing import Any, Optional

_NOT_FOUND = object()  # sentinel distinct from None


class TTLCache:
    """
    Thread-safe in-memory TTL cache with LRU eviction.

    Parameters
    ----------
    ttl : float
        Time-to-live in seconds.  Entries older than this are treated as
        expired and will not be returned.
    maxsize : int
        Maximum number of entries to hold.  When exceeded, the oldest
        entry (by insertion time) is evicted.
    """

    def __init__(self, ttl: float = 60.0, maxsize: int = 256):
        self._ttl = ttl
        self._maxsize = maxsize
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------

    def get(self, key: str) -> Any:
        """Return the cached value, or _NOT_FOUND if missing or expired."""
        with self._lock:
            entry = self._store.get(key, None)
            if entry is None:
                return _NOT_FOUND
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return _NOT_FOUND
            return value

    def set(self, key: str, value: Any) -> None:
        """Store a value under key with the configured TTL."""
        with self._lock:
            if len(self._store) >= self._maxsize and key not in self._store:
                # Evict the oldest entry
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[key] = (value, time.monotonic() + self._ttl)

    def invalidate(self, key: str) -> None:
        """Remove a specific key immediately."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Flush all entries."""
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        """Return number of non-expired entries currently in the cache."""
        now = time.monotonic()
        with self._lock:
            return sum(
                1 for _, (_, exp) in self._store.items() if now <= exp
            )

    def stats(self) -> dict:
        """Return cache statistics for monitoring/health endpoints."""
        with self._lock:
            return {
                "entries": self.size(),
                "maxsize": self._maxsize,
                "ttl_seconds": self._ttl,
            }

    # ------------------------------------------------------------------
    # Decorator interface

    def cached(self, key_fn=None):
        """
        Decorator that caches the return value of a function.

        Parameters
        ----------
        key_fn : callable, optional
            Called with the same args/kwargs as the decorated function to
            produce a cache key string.  Defaults to repr of all arguments.

        Usage
        -----
            @cache.cached(key_fn=lambda name: f"exposure:{name}")
            def get_exposure(name):
                ...
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                key = (
                    key_fn(*args, **kwargs)
                    if key_fn
                    else f"{func.__qualname__}:{repr(args)}:{repr(kwargs)}"
                )
                result = self.get(key)
                if result is not _NOT_FOUND:
                    return result
                result = func(*args, **kwargs)
                self.set(key, result)
                return result
            wrapper.__wrapped__ = func
            return wrapper
        return decorator


# ---------------------------------------------------------------------------
# Global cache instances (imported by api.py and other modules)
# ---------------------------------------------------------------------------

# Short-lived cache for database read queries (SELECT results, metadata).
# TTL 120s: data in our SQLite database doesn't change during a session.
query_cache = TTLCache(ttl=120.0, maxsize=512)

# Medium-lived cache for exposure calculations.
# TTL 300s: deterministic given fixed prices, can cache longer.
exposure_cache = TTLCache(ttl=300.0, maxsize=64)
