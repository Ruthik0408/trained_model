from app.core.rate_limit import InMemoryRateLimiter


def test_in_memory_rate_limiter_blocks_after_limit() -> None:
    limiter = InMemoryRateLimiter(limit=2, window_seconds=60)

    first = limiter.allow("client-1", now=100.0)
    second = limiter.allow("client-1", now=101.0)
    third = limiter.allow("client-1", now=102.0)

    assert first[0] is True
    assert second[0] is True
    assert third[0] is False
