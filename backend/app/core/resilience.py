import logging
import threading
import time
from enum import Enum
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Explicit circuit breaker states for clear state management."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a circuit breaker is open and rejects a call."""


class CircuitBreaker:
    """
    Thread-safe circuit breaker pattern for resilient upstream calls.
    
    Prevents cascading failures by failing fast when downstream services fail.
    Uses monotonic time (immune to system clock adjustments) for reliability.
    """

    _NOT_OPENED: Optional[float] = None  # Sentinel for "never opened"

    def __init__(self, fail_threshold: int, reset_timeout_seconds: float) -> None:
        self.fail_threshold = max(1, int(fail_threshold))
        self.reset_timeout_seconds = max(1.0, float(reset_timeout_seconds))
        self._failure_count = 0
        self._last_failure_time: Optional[float] = self._NOT_OPENED
        self._lock = threading.Lock()
        logger.debug(
            f"CircuitBreaker initialized: threshold={self.fail_threshold}, "
            f"reset_timeout={self.reset_timeout_seconds}s"
        )

    def _get_state(self) -> CircuitBreakerState:
        """Determine current state (must hold lock)."""
        if self._failure_count < self.fail_threshold:
            return CircuitBreakerState.CLOSED
        if self._last_failure_time is None:
            return CircuitBreakerState.CLOSED
        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= self.reset_timeout_seconds:
            return CircuitBreakerState.HALF_OPEN
        return CircuitBreakerState.OPEN

    def allow_request(self) -> bool:
        with self._lock:
            state = self._get_state()
            if state == CircuitBreakerState.CLOSED or state == CircuitBreakerState.HALF_OPEN:
                return True
            return False

    def assert_request_allowed(self) -> None:
        if not self.allow_request():
            logger.warning("Request rejected: circuit breaker is OPEN")
            raise CircuitBreakerOpenError("Circuit breaker is OPEN, rejecting request.")

    def record_success(self) -> None:
        with self._lock:
            old_count = self._failure_count
            self._failure_count = 0
            self._last_failure_time = self._NOT_OPENED
            if old_count > 0:
                logger.info(f"Request succeeded, reset failures from {old_count}")

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._failure_count == self.fail_threshold and self._last_failure_time is None:
                self._last_failure_time = time.monotonic()
                logger.warning(f"Circuit breaker OPENED: failure_count={self._failure_count}")
            elif self._failure_count > self.fail_threshold:
                self._last_failure_time = time.monotonic()
                logger.debug(f"Failure recorded: count={self._failure_count}")

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._get_state()

    def snapshot(self) -> Dict[str, Union[str, float, int, bool]]:
        with self._lock:
            state = self._get_state()
            time_in_open = 0.0
            if self._last_failure_time is not None and state == CircuitBreakerState.OPEN:
                time_in_open = time.monotonic() - self._last_failure_time
            return {
                "state": state.value,
                "failure_count": self._failure_count,
                "fail_threshold": self.fail_threshold,
                "reset_timeout_seconds": self.reset_timeout_seconds,
                "time_in_open_seconds": round(time_in_open, 2),
                "is_open": state == CircuitBreakerState.OPEN,
            }
