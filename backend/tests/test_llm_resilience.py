import httpx

from app.core.rate_limit import InMemoryRateLimiter
from app.schemas.workbench_schema import IsolationReasonRequest
from app.services import llm_reason_service


class _DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_explain_isolation_anomaly_uses_cache(monkeypatch) -> None:
    calls = {"count": 0, "timeout": None}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        calls["timeout"] = kwargs.get("timeout")
        return _DummyResponse({"response": '{"reason":"Unusual amount pattern."}'})

    monkeypatch.setattr(httpx, "post", fake_post)
    llm_reason_service.ANOMALY_REASON_CACHE.invalidate()
    llm_reason_service.ANOMALY_REASON_CIRCUIT.record_success()

    payload = IsolationReasonRequest(
        prediction_id=7,
        review_key="row-7",
        row_payload={"amount": 1200, "vendor_name": "ABC"},
    )

    first = llm_reason_service.explain_isolation_anomaly(payload)
    second = llm_reason_service.explain_isolation_anomaly(payload)

    assert first["fallback"] is False
    assert second["fallback"] is False
    assert calls["count"] == 1
    assert calls["timeout"] is not None
    assert calls["timeout"] >= llm_reason_service.settings.anomaly_reason_timeout_min_seconds
    assert calls["timeout"] <= llm_reason_service.settings.anomaly_reason_timeout_seconds


def test_explain_isolation_anomaly_short_circuits_when_breaker_open(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("httpx.post should not be called while circuit is open")

    monkeypatch.setattr(httpx, "post", fake_post)
    llm_reason_service.ANOMALY_REASON_CACHE.invalidate()
    llm_reason_service.ANOMALY_REASON_CIRCUIT.record_success()
    for _ in range(llm_reason_service.ANOMALY_REASON_CIRCUIT.fail_threshold):
        llm_reason_service.ANOMALY_REASON_CIRCUIT.record_failure()

    payload = IsolationReasonRequest(review_key="row-open", row_payload={"amount": 99})
    result = llm_reason_service.explain_isolation_anomaly(payload)

    assert result["fallback"] is True
    assert calls["count"] == 0
    llm_reason_service.ANOMALY_REASON_CIRCUIT.record_success()


def test_in_memory_rate_limiter_blocks_after_limit() -> None:
    limiter = InMemoryRateLimiter(limit=2, window_seconds=60)

    first = limiter.allow("client-1", now=100.0)
    second = limiter.allow("client-1", now=101.0)
    third = limiter.allow("client-1", now=102.0)

    assert first[0] is True
    assert second[0] is True
    assert third[0] is False


def test_adaptive_timeout_grows_with_prompt_length_and_failure_count() -> None:
    llm_reason_service.ANOMALY_REASON_CIRCUIT.record_success()
    short_timeout = llm_reason_service._adaptive_reason_timeout_seconds("short prompt")

    for _ in range(2):
        llm_reason_service.ANOMALY_REASON_CIRCUIT.record_failure()
    long_prompt = "x" * 4000
    loaded_timeout = llm_reason_service._adaptive_reason_timeout_seconds(long_prompt)

    assert loaded_timeout > short_timeout
    assert loaded_timeout <= llm_reason_service.settings.anomaly_reason_timeout_seconds
    llm_reason_service.ANOMALY_REASON_CIRCUIT.record_success()


def test_clean_reason_rejects_prompt_instruction_echo() -> None:
    cleaned = llm_reason_service._clean_reason(
        "We are given a bill/invoice anomaly to explain in one business sentence (max 32 words)."
    )

    assert cleaned == ""
