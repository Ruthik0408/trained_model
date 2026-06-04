from types import SimpleNamespace

from app.core import cache as cache_module
from app.core.cache import TTLCache
from app.core import resilience
from app.core.resilience import CircuitBreaker, CircuitBreakerState
from app.services import dashboard_service


def test_namespaced_cache_size_counts_valkey_prefix(monkeypatch) -> None:
    seen_prefixes = []

    def fake_count_prefix(prefix: str) -> int:
        seen_prefixes.append(prefix)
        return 7

    monkeypatch.setattr(cache_module, "valkey_count_prefix", fake_count_prefix)

    cache = TTLCache(ttl_seconds=30, namespace="query_result")

    assert cache.size() == 7
    assert seen_prefixes == ["query_result:"]


def test_circuit_breaker_refreshes_open_timer_on_repeated_failures(monkeypatch) -> None:
    current_time = 100.0

    def fake_monotonic() -> float:
        return current_time

    monkeypatch.setattr(resilience.time, "monotonic", fake_monotonic)
    breaker = CircuitBreaker(fail_threshold=2, reset_timeout_seconds=10)

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.get_state() == CircuitBreakerState.OPEN

    current_time = 109.0
    breaker.record_failure()

    current_time = 111.0
    assert breaker.get_state() == CircuitBreakerState.OPEN


def test_latest_run_id_cache_stores_no_run_sentinel(monkeypatch) -> None:
    calls = 0

    def fake_iter_recent_runs(_db):
        nonlocal calls
        calls += 1
        return []

    monkeypatch.setattr(dashboard_service, "_latest_run_id_cache", TTLCache(ttl_seconds=30))
    monkeypatch.setattr(dashboard_service, "_iter_recent_runs", fake_iter_recent_runs)

    db = SimpleNamespace()

    assert dashboard_service._latest_run_id_for_dataset(db, "missing_table") is None
    assert dashboard_service._latest_run_id_for_dataset(db, "missing_table") is None
    assert calls == 1
