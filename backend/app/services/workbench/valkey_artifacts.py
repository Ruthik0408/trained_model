import base64
import gzip
import hashlib
import io
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.core.valkey import get_json as valkey_get_json
from app.core.valkey import set_json as valkey_set_json

_RUN_CACHE_TTL_SECONDS = 1800.0
_ISOLATION_CACHE_TTL_SECONDS = 1800.0
_REVIEW_PAYLOAD_CACHE_TTL_SECONDS = 604800.0


def _cache_payload_dict(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        payload_dict = payload.model_dump()
    elif isinstance(payload, dict):
        payload_dict = dict(payload)
    else:
        raise TypeError("Payload must be a Pydantic model or a plain dict.")

    payload_dict.pop("run_name", None)
    return payload_dict


def _cache_signature(payload: Any) -> str:
    canonical = json.dumps(
        _cache_payload_dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_execution_cache_key(payload: Any) -> str:
    return f"run_execution:{_cache_signature(payload)}"


def isolation_forest_cache_key(payload: Any) -> str:
    return f"isolation_forest:{_cache_signature(payload)}"


def get_run_execution_artifact(payload: Any) -> dict[str, Any] | None:
    return _get_artifact(run_execution_cache_key(payload))


def set_run_execution_artifact(payload: Any, artifact: dict[str, Any]) -> None:
    _set_artifact(run_execution_cache_key(payload), artifact, _RUN_CACHE_TTL_SECONDS)


def get_isolation_forest_artifact(payload: Any) -> dict[str, Any] | None:
    return _get_artifact(isolation_forest_cache_key(payload))


def set_isolation_forest_artifact(payload: Any, artifact: dict[str, Any]) -> None:
    _set_artifact(
        isolation_forest_cache_key(payload),
        artifact,
        _ISOLATION_CACHE_TTL_SECONDS,
    )


def review_payload_cache_key(run_id: int) -> str:
    return f"review_payload_rows:{int(run_id)}"


def get_review_payload_artifact(run_id: int) -> dict[str, Any] | None:
    return _get_artifact(review_payload_cache_key(run_id))


def set_review_payload_artifact(run_id: int, artifact: dict[str, Any]) -> None:
    _set_artifact(
        review_payload_cache_key(run_id),
        artifact,
        _REVIEW_PAYLOAD_CACHE_TTL_SECONDS,
    )


def _get_artifact(key: str) -> dict[str, Any] | None:
    cached = valkey_get_json(key)
    if cached is None or not isinstance(cached, dict):
        return None
    return cached


def _set_artifact(key: str, artifact: dict[str, Any], ttl_seconds: float) -> None:
    valkey_set_json(key, artifact, ttl_seconds)


def serialize_dataframe(df: pd.DataFrame) -> dict[str, str]:
    payload = df.to_json(orient="split", date_format="iso", default_handler=str)
    return {"encoding": "gzip+base64", "payload": _encode_text(payload)}


def deserialize_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    payload = _decode_text(str(data.get("payload") or ""))
    return pd.read_json(io.StringIO(payload), orient="split")


def serialize_series(series: pd.Series) -> dict[str, Any]:
    values = series.astype(object).where(series.notna(), None).tolist()
    return {
        "index": list(series.index),
        "values": values,
        "dtype": str(series.dtype),
    }


def deserialize_series(data: dict[str, Any]) -> pd.Series:
    values = list(data.get("values") or [])
    index = list(data.get("index") or [])
    dtype = str(data.get("dtype") or "object")
    series = pd.Series(values, index=index, dtype=dtype)
    if dtype in {"bool", "boolean"}:
        return series.astype(bool)
    return series


def serialize_ndarray(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    return {"values": array.tolist(), "dtype": str(array.dtype)}


def deserialize_ndarray(data: dict[str, Any]) -> np.ndarray:
    return np.asarray(data.get("values") or [], dtype=data.get("dtype") or None)


def serialize_pipeline(pipeline: Pipeline) -> dict[str, str]:
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    return {
        "encoding": "gzip+base64",
        "payload": _encode_bytes(buffer.getvalue()),
    }


def deserialize_pipeline(data: dict[str, Any]) -> Pipeline:
    payload = _decode_bytes(str(data.get("payload") or ""))
    return joblib.load(io.BytesIO(payload))


def normalize_explanation_signals(
    explanation_signals: dict[Any, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row_index, signals in explanation_signals.items():
        items.append({"row_index": row_index, "signals": list(signals or [])})
    return items


def denormalize_explanation_signals(
    items: list[dict[str, Any]] | None,
) -> dict[Any, list[dict[str, Any]]]:
    explanation_signals: dict[Any, list[dict[str, Any]]] = {}
    for item in items or []:
        row_index = item.get("row_index")
        if row_index is None:
            continue
        if isinstance(row_index, str) and row_index.lstrip("-").isdigit():
            row_index = int(row_index)
        explanation_signals[row_index] = [dict(signal) for signal in item.get("signals") or []]
    return explanation_signals


def _encode_text(value: str) -> str:
    return _encode_bytes(value.encode("utf-8"))


def _decode_text(value: str) -> str:
    return _decode_bytes(value).decode("utf-8")


def _encode_bytes(value: bytes) -> str:
    compressed = gzip.compress(value)
    return base64.b64encode(compressed).decode("ascii")


def _decode_bytes(value: str) -> bytes:
    compressed = base64.b64decode(value.encode("ascii"))
    return gzip.decompress(compressed)
