import threading
import time
from collections import deque


class InMemoryRateLimiter:
    """Fixed-window rate limiter keyed by a caller identifier."""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, int(window_seconds))
        self._entries: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int, int]:
        current = float(now if now is not None else time.monotonic())
        cutoff = current - self.window_seconds
        with self._lock:
            window = self._entries.setdefault(key, deque())
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (current - window[0])))
                return False, self.limit - len(window), retry_after
            window.append(current)
            return True, self.limit - len(window), self.window_seconds
