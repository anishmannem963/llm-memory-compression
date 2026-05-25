"""
Cache Layer — Phase 3
========================
Redis-backed cache for retrieval results.

Why caching matters:
  Production LLM apps see 30–40% repeated or near-identical queries.
  Without cache: every query → ChromaDB vector search → 80–150ms
  With cache hit: query → Redis lookup → <5ms

  At 1000 req/s with 35% cache hit rate:
  - 650 queries go to ChromaDB (full latency)
  - 350 queries hit Redis (<5ms)
  - Average latency drops from 120ms to ~82ms

Architecture:
  - Query text is hashed (SHA-256) → cache key
  - Results serialized to JSON → stored in Redis with TTL
  - On miss: run full hybrid retrieval, store result
  - Falls back to in-memory LRU if Redis is unavailable

Cache key design:
  memory:query:{sha256(query_text)}:{top_k}:{tier_filter}
"""

import json
import hashlib
import time
from typing import List, Optional, Any
from collections import OrderedDict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InMemoryLRUCache:
    """
    Simple LRU (Least Recently Used) cache.
    Used as fallback when Redis is not available.
    Thread-safe for single-process use.
    """

    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 300):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: dict = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            self.misses += 1
            return None

        # Check TTL
        if time.time() - self._timestamps[key] > self.ttl:
            del self._cache[key]
            del self._timestamps[key]
            self.misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self.hits += 1
        return self._cache[key]

    def set(self, key: str, value: Any):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        self._timestamps[key] = time.time()

        # Evict LRU if over capacity
        if len(self._cache) > self.maxsize:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            del self._timestamps[oldest]

    def delete(self, key: str):
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)

    def clear(self):
        self._cache.clear()
        self._timestamps.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        return {
            "backend": "in-memory LRU",
            "size": self.size,
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate * 100, 1),
            "ttl_seconds": self.ttl,
        }


class RedisCache:
    """
    Redis-backed cache for retrieval results.
    Automatically falls back to InMemoryLRUCache if Redis unavailable.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl_seconds: int = 300,
        key_prefix: str = "memory:query:",
        fallback_maxsize: int = 1000,
    ):
        self.ttl = ttl_seconds
        self.key_prefix = key_prefix
        self._redis = None
        self._fallback = InMemoryLRUCache(maxsize=fallback_maxsize, ttl_seconds=ttl_seconds)
        self._using_redis = False

        self._connect(host, port, db)

    def _connect(self, host: str, port: int, db: int):
        try:
            import redis
            client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=2)
            client.ping()
            self._redis = client
            self._using_redis = True
            logger.info(f"Redis connected: {host}:{port}/db{db}")
        except ImportError:
            logger.info("redis-py not installed. Using in-memory LRU cache. Install: pip install redis")
        except Exception as e:
            logger.info(f"Redis not available ({e}). Using in-memory LRU fallback.")

    def _make_key(self, query: str, top_k: int, tier_filter: str = "all") -> str:
        """Create a deterministic cache key from query parameters."""
        raw = f"{query}|{top_k}|{tier_filter}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"{self.key_prefix}{digest}"

    def get(self, query: str, top_k: int = 5, tier_filter: str = "all") -> Optional[List[dict]]:
        """Get cached results. Returns None on miss."""
        key = self._make_key(query, top_k, tier_filter)

        if self._using_redis:
            try:
                raw = self._redis.get(key)
                if raw:
                    return json.loads(raw)
                return None
            except Exception as e:
                logger.warning(f"Redis get failed: {e}. Falling back to LRU.")
                return self._fallback.get(key)
        else:
            return self._fallback.get(key)

    def set(self, query: str, results: List[dict], top_k: int = 5, tier_filter: str = "all"):
        """Cache results with TTL."""
        key = self._make_key(query, top_k, tier_filter)

        if self._using_redis:
            try:
                self._redis.setex(key, self.ttl, json.dumps(results))
                return
            except Exception as e:
                logger.warning(f"Redis set failed: {e}. Falling back to LRU.")

        self._fallback.set(key, results)

    def invalidate(self, query: str, top_k: int = 5, tier_filter: str = "all"):
        """Remove a specific cache entry."""
        key = self._make_key(query, top_k, tier_filter)
        if self._using_redis:
            try:
                self._redis.delete(key)
                return
            except Exception:
                pass
        self._fallback.delete(key)

    def flush(self):
        """Clear all cache entries."""
        if self._using_redis:
            try:
                keys = self._redis.keys(f"{self.key_prefix}*")
                if keys:
                    self._redis.delete(*keys)
                return
            except Exception:
                pass
        self._fallback.clear()

    def stats(self) -> dict:
        if self._using_redis:
            try:
                info = self._redis.info("stats")
                keys = self._redis.keys(f"{self.key_prefix}*")
                return {
                    "backend": "redis",
                    "cached_queries": len(keys),
                    "redis_hits": info.get("keyspace_hits", 0),
                    "redis_misses": info.get("keyspace_misses", 0),
                    "ttl_seconds": self.ttl,
                }
            except Exception:
                pass
        return self._fallback.stats()

    @property
    def is_redis(self) -> bool:
        return self._using_redis
