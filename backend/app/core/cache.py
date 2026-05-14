"""
Caching utilities for the application.

Provides TTL-based caching helpers for expensive database queries and computations.
"""
import logging
import threading
import time
from typing import Any, Callable, Dict, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TTLCache:
    """Thread-safe TTL-based cache for expensive operations."""
    
    def __init__(self, ttl_seconds: float = 60.0):
        """
        Initialize TTL cache.
        
        Args:
            ttl_seconds: Time-to-live in seconds. Cache entries expire after this duration.
        """
        self.ttl = ttl_seconds
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Any | None:
        """
        Get value from cache if it exists and hasn't expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp = self._cache[key]
            if time.monotonic() - timestamp > self.ttl:
                del self._cache[key]
                return None
            
            return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache with current timestamp.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            self._cache[key] = (value, time.monotonic())
    
    def invalidate(self, key: str | None = None) -> None:
        """
        Invalidate cache entry or entire cache.
        
        Args:
            key: Specific key to invalidate. If None, clears entire cache.
        """
        with self._lock:
            if key is None:
                self._cache.clear()
            elif key in self._cache:
                del self._cache[key]
    
    def invalidate_prefix(self, prefix: str) -> None:
        """
        Invalidate all cache entries with given prefix.
        
        Args:
            prefix: Key prefix to match for invalidation
        """
        with self._lock:
            keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_delete:
                del self._cache[k]
    
    def size(self) -> int:
        """Get current cache size."""
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
TABLE_METADATA_CACHE = TTLCache(ttl_seconds=300.0)  # 5 minutes for table structures
QUERY_RESULT_CACHE = TTLCache(ttl_seconds=60.0)     # 1 minute for query results
DATASET_SUMMARY_CACHE = TTLCache(ttl_seconds=30.0)  # 30 seconds for dataset summaries


def invalidate_all_caches() -> None:
    """Invalidate all shared cache instances."""
    TABLE_METADATA_CACHE.invalidate()
    QUERY_RESULT_CACHE.invalidate()
    DATASET_SUMMARY_CACHE.invalidate()
    logger.info("All caches invalidated")


def get_cache_stats() -> Dict[str, int]:
    """Get statistics about all cache instances."""
    return {
        "table_metadata_cache": TABLE_METADATA_CACHE.size(),
        "query_result_cache": QUERY_RESULT_CACHE.size(),
        "dataset_summary_cache": DATASET_SUMMARY_CACHE.size(),
    }
