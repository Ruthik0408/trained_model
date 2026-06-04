"""
Caching utilities for the application.

Provides TTL-based caching helpers for expensive database queries and computations.
"""
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Tuple

from app.core.valkey import delete as valkey_delete
from app.core.valkey import delete_prefix as valkey_delete_prefix
from app.core.valkey import count_prefix as valkey_count_prefix
from app.core.valkey import get_json as valkey_get_json
from app.core.valkey import set_json as valkey_set_json

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TTLCache:
    """Thread-safe TTL-based cache for expensive operations."""

    def __init__(self, ttl_seconds: float = 60.0, namespace: str | None = None) -> None:
        """
        Initialize TTL cache.
        
        Args:
            ttl_seconds: Time-to-live in seconds. Cache entries expire after this duration.
        """
        self.ttl = ttl_seconds
        self.namespace = namespace
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def _valkey_key(self, key: str) -> str:
        if not self.namespace:
            return key
        return f"{self.namespace}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if it exists and hasn't expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        # Namespaced caches are Valkey-backed only. We intentionally avoid
        # keeping a parallel in-process copy so Valkey is the single temporary
        # storage layer for shared backend caches/artifacts.
        if self.namespace:
            shared_value = valkey_get_json(self._valkey_key(key))
            if shared_value is not None:
                logger.debug(f"Valkey cache hit: {self._valkey_key(key)}")
                return shared_value
            logger.debug(f"Valkey cache miss: {self._valkey_key(key)}")
            return None
        with self._lock:
            if key not in self._cache:
                logger.debug(f"Cache miss: {key}")
                return None
            
            value, timestamp = self._cache[key]
            if time.monotonic() - timestamp > self.ttl:
                logger.debug(f"Cache expired: {key}")
                del self._cache[key]
                return None
            
            logger.debug(f"Cache hit: {key}")
            return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache with current timestamp.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if self.namespace:
            valkey_set_json(self._valkey_key(key), value, self.ttl)
            logger.debug(f"Valkey cache set: {self._valkey_key(key)}")
            return
        with self._lock:
            self._cache[key] = (value, time.monotonic())
            logger.debug(f"Cache set: {key}")
    
    def invalidate(self, key: Optional[str] = None) -> None:
        """
        Invalidate cache entry or entire cache.
        
        Args:
            key: Specific key to invalidate. If None, clears entire cache.
        """
        if self.namespace:
            if key is None:
                logger.info("Valkey cache cleared (all entries) for namespace=%s", self.namespace)
                valkey_delete_prefix(f"{self.namespace}:")
            else:
                logger.debug("Valkey cache invalidated: %s", self._valkey_key(key))
                valkey_delete(self._valkey_key(key))
            return
        with self._lock:
            if key is None:
                self._cache.clear()
                logger.info("Cache cleared (all entries)")
            else:
                if key in self._cache:
                    del self._cache[key]
                    logger.debug(f"Cache invalidated: {key}")
    
    def invalidate_prefix(self, prefix: str) -> None:
        """
        Invalidate all cache entries with given prefix.
        
        Args:
            prefix: Key prefix to match for invalidation
        """
        if self.namespace:
            logger.info(
                "Valkey cache invalidated by prefix for namespace=%s prefix=%s",
                self.namespace,
                prefix,
            )
            valkey_delete_prefix(f"{self.namespace}:{prefix}")
            return
        with self._lock:
            keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_delete:
                del self._cache[k]
            if keys_to_delete:
                logger.info(f"Cache invalidated {len(keys_to_delete)} entries with prefix: {prefix}")
    
    def size(self) -> int:
        """Get current cache size."""
        if self.namespace:
            shared_count = valkey_count_prefix(f"{self.namespace}:")
            return int(shared_count or 0)
        with self._lock:
            return len(self._cache)


def cached(cache: TTLCache, key_fn: Callable[..., str]) -> Callable:
    """
    Decorator for caching function results using TTLCache.
    
    Args:
        cache: TTLCache instance to use
        key_fn: Function to generate cache key from args/kwargs
        
    Example:
        table_cache = TTLCache(ttl_seconds=300)
        
        def make_key(*args, **kwargs):
            return f"table:{args[0]}"
        
        @cached(table_cache, make_key)
        def get_table_columns(table_name: str) -> list:
            # expensive operation
            return fetch_columns(table_name)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            key = key_fn(*args, **kwargs)
            cached_value = cache.get(key)
            
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}:{key}")
                return cached_value
            
            result = func(*args, **kwargs)
            cache.set(key, result)
            logger.debug(f"Cache set for {func.__name__}:{key}")
            return result
        
        wrapper.cache = cache  # Expose cache for manual management
        return wrapper
    
    return decorator


# Shared cache instances for common operations
TABLE_METADATA_CACHE = TTLCache(ttl_seconds=300.0, namespace="table_metadata")  # 5 minutes for table structures
QUERY_RESULT_CACHE = TTLCache(ttl_seconds=60.0, namespace="query_result")     # 1 minute for query results
DATASET_SUMMARY_CACHE = TTLCache(ttl_seconds=30.0, namespace="dataset_summary")  # 30 seconds for dataset summaries


def invalidate_all_caches() -> None:
    """Invalidate all shared cache instances."""
    TABLE_METADATA_CACHE.invalidate()
    QUERY_RESULT_CACHE.invalidate()
    DATASET_SUMMARY_CACHE.invalidate()
    from app.services.workbench.sql_runtime import _join_sql_cache
    _join_sql_cache.invalidate()
    logger.info("All caches invalidated")


def get_cache_stats() -> Dict[str, int]:
    """Get statistics about all cache instances."""
    return {
        "table_metadata_cache": TABLE_METADATA_CACHE.size(),
        "query_result_cache": QUERY_RESULT_CACHE.size(),
        "dataset_summary_cache": DATASET_SUMMARY_CACHE.size(),
    }
