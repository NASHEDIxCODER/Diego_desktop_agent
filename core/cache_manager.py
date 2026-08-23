"""
CacheManager — Multi-domain caching for Diego.

Reduces unnecessary LLM calls, vision processing, and search queries
by caching results across multiple domains with TTL-based expiration.

Domains:
  - desktop: window title, app type, git branch, terminal path
  - vision: screen analysis results, UI tree, OCR text
  - search: web search results
  - memory: context composer results
  - project: project graph, file structure
  - music: current track, provider status

Usage:
    from core.cache_manager import cache_manager

    # Check cache before expensive operation
    cached = cache_manager.get("vision", "screen_context")
    if cached and not cached.is_expired:
        return cached.value

    # Store result after computation
    cache_manager.set("vision", "screen_context", result, ttl_s=2.0)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with TTL."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl_s: float = 5.0  # default 5 second TTL
    access_count: int = 0
    last_accessed: float = 0.0

    @property
    def age_s(self) -> float:
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        return self.age_s > self.ttl_s

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = time.time()


class CacheManager:
    """
    Multi-domain cache with per-domain TTL configuration.

    Each domain has its own TTL and max entries. Entries are
    automatically evicted when expired or when the domain is full.
    """

    def __init__(self):
        self._stores: Dict[str, Dict[str, CacheEntry]] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

        # Per-domain TTL configuration (seconds)
        self._domain_ttls: Dict[str, float] = {
            "desktop": 1.0,     # Desktop state changes fast
            "vision": 2.0,      # Screen analysis — moderate TTL
            "search": 300.0,    # Web search — long TTL
            "memory": 10.0,     # Memory context — moderate TTL
            "project": 60.0,    # Project graph — long TTL
            "music": 5.0,       # Music status — moderate TTL
            "llm": 0.5,         # LLM responses — very short (for dedup)
            "plan": 300.0,      # Action plans — long TTL (reuse across sessions)
            "response": 3600.0, # Cached responses — 1 hour
        }

        # Per-domain max entries
        self._domain_max: Dict[str, int] = {
            "desktop": 5,
            "vision": 10,
            "search": 50,
            "memory": 20,
            "project": 10,
            "music": 5,
            "llm": 3,
            "plan": 20,       # Reusable action plans
            "response": 50,   # Cached LLM responses
        }

    # ── Public API ────────────────────────────────────────

    def get(self, domain: str, key: str) -> Optional[CacheEntry]:
        """
        Get a cached value if it exists and hasn't expired.

        Args:
            domain: Cache domain (e.g. "vision", "search", "desktop")
            key: Cache key within the domain

        Returns:
            CacheEntry if found and valid, None otherwise.
        """
        store = self._stores.get(domain, {})
        entry = store.get(key)

        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired:
            del store[key]
            self._evictions += 1
            self._misses += 1
            return None

        entry.touch()
        self._hits += 1
        return entry

    def set(self, domain: str, key: str, value: Any,
            ttl_s: Optional[float] = None) -> None:
        """
        Store a value in the cache.

        Args:
            domain: Cache domain
            key: Cache key
            value: Value to store (must be JSON-serializable for some domains)
            ttl_s: Override the domain's default TTL
        """
        if domain not in self._stores:
            self._stores[domain] = {}

        store = self._stores[domain]
        effective_ttl = ttl_s if ttl_s is not None else self._domain_ttls.get(domain, 5.0)

        # Evict expired entries
        expired = [k for k, v in store.items() if v.is_expired]
        for k in expired:
            del store[k]
            self._evictions += 1

        # Evict oldest if at capacity
        max_entries = self._domain_max.get(domain, 20)
        while len(store) >= max_entries:
            oldest_key = min(store.keys(), key=lambda k: store[k].created_at)
            del store[oldest_key]
            self._evictions += 1

        store[key] = CacheEntry(
            key=key,
            value=value,
            ttl_s=effective_ttl,
        )

    def invalidate(self, domain: str, key: Optional[str] = None) -> None:
        """
        Invalidate cache entries.

        Args:
            domain: Cache domain to invalidate
            key: Specific key to invalidate, or None to clear entire domain
        """
        if key is not None:
            store = self._stores.get(domain, {})
            if key in store:
                del store[key]
                logger.debug("[CACHE] Invalidated %s:%s", domain, key)
        else:
            if domain in self._stores:
                count = len(self._stores[domain])
                del self._stores[domain]
                logger.debug("[CACHE] Invalidated domain %s (%d entries)", domain, count)

    def invalidate_all(self) -> None:
        """Clear all caches."""
        total = sum(len(s) for s in self._stores.values())
        self._stores.clear()
        logger.info("[CACHE] Invalidated all caches (%d entries)", total)

    # ── Convenience methods ───────────────────────────────

    def get_or_compute(self, domain: str, key: str,
                        compute_fn, ttl_s: Optional[float] = None) -> Any:
        """
        Get from cache or compute and store.

        Args:
            domain: Cache domain
            key: Cache key
            compute_fn: Callable that returns the value if not cached
            ttl_s: Optional TTL override

        Returns:
            Cached or computed value.
        """
        entry = self.get(domain, key)
        if entry is not None:
            return entry.value

        value = compute_fn()
        self.set(domain, key, value, ttl_s=ttl_s)
        return value

    @staticmethod
    def make_key(*parts: str) -> str:
        """
        Create a deterministic cache key from parts.

        Args:
            parts: String parts to combine into a key

        Returns:
            A hash-based cache key.
        """
        combined = "|".join(str(p) for p in parts)
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def desktop_key(self, attribute: str) -> str:
        """Create a key for desktop state cache."""
        return self.make_key("desktop", attribute)

    def vision_key(self, window_title: str, frame_hash: str = "") -> str:
        """Create a key for vision cache."""
        return self.make_key("vision", window_title, frame_hash)

    def search_key(self, query: str) -> str:
        """Create a key for search cache."""
        normalized = " ".join(query.lower().split())
        return self.make_key("search", normalized)

    def memory_key(self, user_text: str) -> str:
        """Create a key for memory context cache."""
        return self.make_key("memory", user_text[:100])

    # ── Stats ─────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(total, 1)

    def report(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": f"{self.hit_rate:.1%}",
            "domains": {
                domain: {
                    "entries": len(store),
                    "ttl_s": self._domain_ttls.get(domain, 5.0),
                    "max": self._domain_max.get(domain, 20),
                }
                for domain, store in self._stores.items()
            },
        }


# Global singleton
cache_manager = CacheManager()