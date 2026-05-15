import threading
import time


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a circuit breaker is open and rejects a call."""


class CircuitBreaker:
    """Small in-memory circuit breaker for unreliable upstream calls."""

    def __init__(self, fail_threshold: int, reset_timeout_seconds: float):
        self.fail_threshold = max(1, int(fail_threshold))
        self.reset_timeout_seconds = max(1.0, float(reset_timeout_seconds))
        self._failure_count = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self._failure_count < self.fail_threshold:
                return True
            if time.monotonic() - self._opened_at >= self.reset_timeout_seconds:
                self._failure_count = 0
                self._opened_at = 0.0
                return True
            return False

    def assert_request_allowed(self) -> None:
        if not self.allow_request():
            raise CircuitBreakerOpenError("Upstream circuit breaker is open.")

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._opened_at = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.fail_threshold and self._opened_at == 0.0:
                self._opened_at = time.monotonic()

    def snapshot(self) -> dict[str, float | int | bool]:
        with self._lock:
            is_open = self._failure_count >= self.fail_threshold and (
                time.monotonic() - self._opened_at < self.reset_timeout_seconds
            )
            return {
                "failure_count": self._failure_count,
                "is_open": is_open,
                "opened_at": self._opened_at,
                "reset_timeout_seconds": self.reset_timeout_seconds,
            }
