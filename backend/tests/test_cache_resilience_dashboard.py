from types import SimpleNamespace

from app.core import cache as cache_module
from app.core.cache import TTLCache
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


def test_namespaced_cache_falls_back_to_local_memory_when_valkey_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(cache_module, "valkey_available", lambda: False)
    monkeypatch.setattr(cache_module, "valkey_get_json", lambda _key: None)
    monkeypatch.setattr(cache_module, "valkey_count_prefix", lambda _prefix: None)

    cache = TTLCache(ttl_seconds=30, namespace="query_result")

    cache.set("expensive-report", {"rows": 3})

    assert cache.get("expensive-report") == {"rows": 3}
    assert cache.size() == 1


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


def test_ml_run_id_lookup_is_cached(monkeypatch) -> None:
    calls = 0

    class FakeQuery:
        def filter(self, *_args, **_kwargs):
            return self

        def first(self):
            nonlocal calls
            calls += 1
            return SimpleNamespace(metrics_json={"ml_run_id": 77})

    db = SimpleNamespace(query=lambda _model: FakeQuery())
    monkeypatch.setattr(dashboard_service, "_ml_run_id_cache", TTLCache(ttl_seconds=30))

    assert dashboard_service._ml_run_id_for_app_run(db, 12) == 77
    assert dashboard_service._ml_run_id_for_app_run(db, 12) == 77
    assert calls == 1
