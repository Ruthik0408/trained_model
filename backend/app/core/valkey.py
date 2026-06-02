from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

try:
    from redis import Redis
    from redis.exceptions import RedisError
except ModuleNotFoundError:
    Redis = None

    class RedisError(Exception):
        pass

from app.core.config import settings

logger = logging.getLogger(__name__)


def _prefixed_key(key: str) -> str:
    return f"{settings.valkey_key_prefix}:{key}"


@lru_cache(maxsize=1)
def get_valkey_client() -> Redis | None:
    if not settings.valkey_enabled:
        return None
    if Redis is None:
        logger.warning("redis package is not installed; using local memory cache fallback.")
        return None

    client = Redis(
        host=settings.valkey_host,
        port=settings.valkey_port,
        db=settings.valkey_db,
        password=settings.valkey_password or None,
        decode_responses=True,
        socket_connect_timeout=settings.valkey_socket_timeout_seconds,
        socket_timeout=settings.valkey_socket_timeout_seconds,
    )
    try:
        client.ping()
        logger.info(
            "Connected to Valkey at %s:%s db=%s",
            settings.valkey_host,
            settings.valkey_port,
            settings.valkey_db,
        )
        return client
    except RedisError as exc:
        logger.warning("Valkey unavailable, falling back to local memory cache: %s", exc)
        return None


def valkey_available() -> bool:
    return get_valkey_client() is not None


def get_json(key: str) -> Any | None:
    client = get_valkey_client()
    if client is None:
        return None
    try:
        raw = client.get(_prefixed_key(key))
        if raw is None:
            return None
        return json.loads(raw)
    except (RedisError, json.JSONDecodeError) as exc:
        logger.warning("Valkey get failed for key=%s: %s", key, exc)
        return None


def set_json(key: str, value: Any, ttl_seconds: float) -> None:
    client = get_valkey_client()
    if client is None:
        return
    try:
        client.setex(
            _prefixed_key(key),
            max(1, int(ttl_seconds)),
            json.dumps(value, ensure_ascii=True, default=str),
        )
    except (RedisError, TypeError, ValueError) as exc:
        logger.warning("Valkey set failed for key=%s: %s", key, exc)


def delete(key: str) -> None:
    client = get_valkey_client()
    if client is None:
        return
    try:
        client.delete(_prefixed_key(key))
    except RedisError as exc:
        logger.warning("Valkey delete failed for key=%s: %s", key, exc)


def delete_prefix(prefix: str) -> None:
    client = get_valkey_client()
    if client is None:
        return
    match = _prefixed_key(f"{prefix}*")
    try:
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor=cursor, match=match, count=200)
            if keys:
                client.delete(*keys)
            if cursor == 0:
                break
    except RedisError as exc:
        logger.warning("Valkey prefix delete failed for prefix=%s: %s", prefix, exc)


def incr(key: str, ttl_seconds: int) -> int | None:
    client = get_valkey_client()
    if client is None:
        return None
    redis_key = _prefixed_key(key)
    try:
        value = int(client.incr(redis_key))
        if value == 1:
            client.expire(redis_key, max(1, int(ttl_seconds)))
        return value
    except RedisError as exc:
        logger.warning("Valkey incr failed for key=%s: %s", key, exc)
        return None


def ttl(key: str) -> int | None:
    client = get_valkey_client()
    if client is None:
        return None
    try:
        ttl_value = int(client.ttl(_prefixed_key(key)))
        return ttl_value if ttl_value >= 0 else None
    except RedisError as exc:
        logger.warning("Valkey ttl failed for key=%s: %s", key, exc)
        return None
