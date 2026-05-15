import logging
import threading
import time
from collections import deque
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """Fixed-window rate limiter keyed by a caller identifier with observability."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, int(window_seconds))
        self._entries: Dict[str, deque] = {}
        self._lock = threading.Lock()
        logger.debug(
            f"RateLimiter initialized: limit={self.limit} per {self.window_seconds}s"
        )

    def allow(self, key: str, now: Optional[float] = None) -> Tuple[bool, int, int]:
        current = float(now if now is not None else time.monotonic())
        cutoff = current - self.window_seconds
        with self._lock:
            window = self._entries.setdefault(key, deque())
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (current - window[0])))
                remaining_requests = self.limit - len(window)
                logger.warning(
                    f"Rate limit exceeded for {key}: {len(window)}/{self.limit}, "
                    f"retry_after={retry_after}s"
                )
                return False, remaining_requests, retry_after
            window.append(current)
            remaining_requests = self.limit - len(window)
            return True, remaining_requests, self.window_seconds
