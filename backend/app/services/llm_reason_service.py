import logging
import json
import re
import time
from typing import Any

import httpx
import numpy as np
import pandas as pd
import shap

from app.core.cache import TTLCache
from app.core.config import settings
from app.core.resilience import CircuitBreaker, CircuitBreakerOpenError
from app.schemas.workbench_schema import IsolationReasonRequest

logger = logging.getLogger(__name__)

IMPORTANT_KEYWORDS = (
    "amount",
    "invoice",
    "bill",
    "date",
    "reference",
    "status",
    "vendor",
    "office",
    "feature_",
    "iqr_flag::",
    "days",
    "difference",
    "ratio",
)

ANOMALY_REASON_CACHE = TTLCache(ttl_seconds=settings.anomaly_reason_cache_ttl_seconds)
ANOMALY_REASON_CIRCUIT = CircuitBreaker(
    fail_threshold=settings.anomaly_reason_circuit_fail_threshold,
    reset_timeout_seconds=settings.anomaly_reason_circuit_reset_seconds,
)

def build_feature_explanation_signals(
    pipeline: Any,
    feature_frame: pd.DataFrame,
    transformed: np.ndarray,
    anomaly_indices: Any,
) -> dict[Any, list[dict[str, Any]]]:
    if feature_frame.empty or transformed is None:
        return {}

    column_labels = _transformed_feature_labels(pipeline, feature_frame)
    transformed_frame = pd.DataFrame(
        np.asarray(transformed),
        index=feature_frame.index,
        columns=column_labels,
    )
    selected_index = [index for index in anomaly_indices if index in transformed_frame.index]
    if not selected_index:
        return {}

    try:
        return _build_shap_feature_explanation_signals(
            pipeline,
            feature_frame,
            transformed_frame,
            selected_index,
        )
    except Exception as exc:
        logger.warning("Falling back to non-SHAP feature explanation signals: %s", exc)
        return _build_ranked_feature_explanation_signals(
            feature_frame,
            transformed_frame,
            selected_index,
        )

def explain_isolation_anomaly(payload: IsolationReasonRequest) -> dict[str, Any]:
    cache_key = _reason_cache_key(payload)
    cached_reason = ANOMALY_REASON_CACHE.get(cache_key)
    if cached_reason is not None:
        logger.info("Returning cached anomaly reason for review_key=%s", payload.review_key or "unknown")
        return cached_reason

    deterministic_reason = build_deterministic_isolation_reason(payload.feature_signals, payload.row_payload)
    if deterministic_reason:
        result = {
            "reason": deterministic_reason,
            "model": "deterministic",
            "fallback": False,
        }
        ANOMALY_REASON_CACHE.set(cache_key, result)
        return result

    prompt = _build_prompt(payload)
    fallback_reason = _fallback_reason(payload)

    try:
        ANOMALY_REASON_CIRCUIT.assert_request_allowed()
        reason = _generate_reason_with_retry(prompt)
        if reason:
            result = {
                "reason": reason,
                "model": settings.anomaly_reason_model,
                "fallback": False,
            }
            ANOMALY_REASON_CACHE.set(cache_key, result)
            return result
    except CircuitBreakerOpenError as exc:
        logger.warning("Ollama anomaly reason generation skipped because circuit breaker is open: %s", exc)
    except Exception as exc:
        logger.warning("Ollama anomaly reason generation failed: %s", exc)

    result = {
        "reason": fallback_reason,
        "model": settings.anomaly_reason_model,
        "fallback": True,
    }
    ANOMALY_REASON_CACHE.set(cache_key, result)
    return result

def _build_shap_feature_explanation_signals(
    pipeline: Any,
    feature_frame: pd.DataFrame,
    transformed_frame: pd.DataFrame,
    selected_index: list[Any],
) -> dict[Any, list[dict[str, Any]]]:
    model = pipeline.named_steps["model"]
    selected_rows = transformed_frame.loc[selected_index]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(selected_rows)
    shap_array = np.asarray(shap_values)
    if shap_array.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {shap_array.shape}")

    signals_by_index: dict[Any, list[dict[str, Any]]] = {}
    for row_position, row_index in enumerate(selected_rows.index):
        row_values = shap_array[row_position]
        ranked_positions = np.argsort(np.abs(row_values))[::-1]
        signals = _signals_from_ranked_positions(
            feature_frame=feature_frame,
            transformed_frame=transformed_frame,
            row_index=row_index,
            ranked_positions=ranked_positions,
            contribution_values=row_values,
            method="shap",
        )
        if signals:
            signals_by_index[row_index] = signals
    return signals_by_index

def _build_ranked_feature_explanation_signals(
    feature_frame: pd.DataFrame,
    transformed_frame: pd.DataFrame,
    selected_index: list[Any],
) -> dict[Any, list[dict[str, Any]]]:
    signals_by_index: dict[Any, list[dict[str, Any]]] = {}
    selected_rows = transformed_frame.loc[selected_index]

    for row_index, row in selected_rows.iterrows():
        ranked_positions = np.argsort(np.abs(row.to_numpy()))[::-1]
        signals = _signals_from_ranked_positions(
            feature_frame=feature_frame,
            transformed_frame=transformed_frame,
            row_index=row_index,
            ranked_positions=ranked_positions,
            contribution_values=row.to_numpy(),
            method="scaled_magnitude",
        )
        if signals:
            signals_by_index[row_index] = signals
    return signals_by_index

def _signals_from_ranked_positions(
    *,
    feature_frame: pd.DataFrame,
    transformed_frame: pd.DataFrame,
    row_index: Any,
    ranked_positions: np.ndarray,
    contribution_values: np.ndarray,
    method: str,
) -> list[dict[str, Any]]:
    row = transformed_frame.loc[row_index]
    seen_features: set[str] = set()
    signals: list[dict[str, Any]] = []

    for position in ranked_positions:
        feature_name = str(transformed_frame.columns[int(position)])
        source_feature_name = _source_feature_name(feature_name)
        if source_feature_name in seen_features:
            continue
        seen_features.add(source_feature_name)
        raw_value = (
            feature_frame.at[row_index, source_feature_name]
            if source_feature_name in feature_frame.columns
            else None
        )
        contribution = float(contribution_values[int(position)])
        strength = abs(contribution)
        signals.append(
            {
                "feature": str(source_feature_name),
                "value": _safe_float(raw_value),
                "strength": _safe_float(strength),
                "direction": "high" if float(row.iloc[int(position)]) >= 0 else "low",
                "attribution": _safe_float(contribution),
                "method": method,
            }
        )
        if len(signals) >= 5:
            break
    return signals


def _generate_reason_with_retry(prompt: str) -> str:
    attempts = max(1, settings.anomaly_reason_retry_attempts + 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            timeout_seconds = _adaptive_reason_timeout_seconds(prompt)
            response = httpx.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.anomaly_reason_model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 120,
                    },
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            ANOMALY_REASON_CIRCUIT.record_success()
            return _clean_reason(response.json().get("response"))
        except Exception as exc:
            last_error = exc
            ANOMALY_REASON_CIRCUIT.record_failure()
            logger.warning("Ollama attempt %d/%d failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(settings.anomaly_reason_retry_backoff_seconds * attempt)
    if last_error is not None:
        raise last_error
    return ""


def _adaptive_reason_timeout_seconds(prompt: str) -> float:
    prompt_length = max(0, len(prompt))
    prompt_penalty = (prompt_length / 1000.0) * settings.anomaly_reason_timeout_per_1k_chars_seconds
    circuit_snapshot = ANOMALY_REASON_CIRCUIT.snapshot()
    load_penalty = float(circuit_snapshot["failure_count"]) * settings.anomaly_reason_timeout_load_penalty_seconds
    timeout_seconds = settings.anomaly_reason_timeout_min_seconds + prompt_penalty + load_penalty
    clamped_timeout = max(
        settings.anomaly_reason_timeout_min_seconds,
        min(settings.anomaly_reason_timeout_seconds, timeout_seconds),
    )
    logger.debug(
        "Adaptive anomaly-reason timeout computed as %.2fs for prompt_length=%d failure_count=%s",
        clamped_timeout,
        prompt_length,
        circuit_snapshot["failure_count"],
    )
    return clamped_timeout


def _reason_cache_key(payload: IsolationReasonRequest) -> str:
    raw = payload.model_dump(mode="json")
    return json.dumps(raw, sort_keys=True, ensure_ascii=True)

def _build_prompt(payload: IsolationReasonRequest) -> str:
    facts = _compact_row_facts(payload.row_payload)
    existing_reasons = [str(item).strip() for item in payload.existing_reasons if str(item).strip()]
    feature_signals = _compact_feature_signals(payload.feature_signals)

    return (
        "/no_think\n"
        "You explain Isolation Forest anomalies for a bill/invoice review UI.\n"
        "Return only JSON like {\"reason\":\"...\"}.\n"
        "The reason must be one short human-readable sentence, maximum 50 words.\n"
        "Use only the facts provided. Do not invent duplicates, fraud, vendors, or dates.\n"
        "Explain the business/data signals that look unusual.\n"
        "Do not mention IF score, threshold, score gap, model score, or numeric anomaly cutoff.\n\n"
        f"Review key: {payload.review_key or 'unknown'}\n"
        f"IF score: {_format_number(payload.if_score)}\n"
        f"IF threshold: {_format_number(payload.ml_threshold)}\n"
        f"Rule anomaly: {'yes' if payload.rule_anomaly else 'no'}\n"
        f"Rule count: {payload.rule_count if payload.rule_count is not None else 'unknown'}\n"
        f"Existing rule reasons: {', '.join(existing_reasons) if existing_reasons else 'none'}\n"
        f"Isolation signals: {feature_signals if feature_signals else 'none'}\n"
        f"Relevant row signals: {facts if facts else 'none'}\n\n"
        "JSON:"
    )

def _compact_row_facts(row_payload: dict[str, Any]) -> str:
    if not isinstance(row_payload, dict):
        return ""

    selected: list[tuple[str, Any]] = []
    for key, value in row_payload.items():
        if value in (None, ""):
            continue
        key_text = str(key)
        lower_key = key_text.lower()
        if any(keyword in lower_key for keyword in IMPORTANT_KEYWORDS):
            selected.append((key_text, value))

    if not selected:
        selected = [(str(key), value) for key, value in list(row_payload.items())[:12] if value not in (None, "")]

    formatted = []
    for key, value in selected[:18]:
        formatted.append(f"{_readable_key(key)}={_short_value(value)}")
    return "; ".join(formatted)

def _compact_feature_signals(feature_signals: list[dict[str, Any]]) -> str:
    if not isinstance(feature_signals, list):
        return ""

    parts: list[str] = []
    for item in feature_signals[:5]:
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature") or "").strip()
        if not feature:
            continue
        direction = str(item.get("direction") or "").strip()
        value = _short_value(item.get("value"))
        text = feature
        if direction:
            text += f" ({direction})"
        if value:
            text += f"={value}"
        parts.append(text)
    return "; ".join(parts)

def _fallback_reason(payload: IsolationReasonRequest) -> str:
    deterministic_reason = build_deterministic_isolation_reason(payload.feature_signals, payload.row_payload)
    if deterministic_reason:
        return deterministic_reason
    facts = _compact_row_facts(payload.row_payload)
    if facts:
        first_facts = facts.split("; ")[:2]
        return f"Unusual row signals: {'; '.join(first_facts)}."
    return "This row has an unusual combination of values compared with the normal review records."

def build_deterministic_isolation_reason(
    feature_signals: list[dict[str, Any]] | None,
    row_payload: dict[str, Any] | None,
) -> str | None:
    payload = row_payload if isinstance(row_payload, dict) else {}
    if not isinstance(feature_signals, list):
        feature_signals = []

    for signal in feature_signals:
        if not isinstance(signal, dict):
            continue
        feature_name = str(signal.get("feature") or "").strip()
        if "_to_" not in feature_name:
            continue
        value = _safe_float(signal.get("value"))
        if value is None:
            continue
        left_stage, right_stage = feature_name.split("_to_", 1)
        if not left_stage or not right_stage:
            continue
        day_count = int(round(abs(value)))
        if day_count <= 0:
            continue
        left_label = _humanize_stage_name(left_stage)
        right_label = _humanize_stage_name(right_stage)
        left_date = _find_row_date_for_stage(payload, left_stage)
        right_date = _find_row_date_for_stage(payload, right_stage)
        if left_date and right_date:
            return (
                f"The {left_label} ({left_date}) is {day_count} days before the {right_label} ({right_date}), "
                "which is unusual for similar records."
            )
        return f"The gap from {left_label} to {right_label} is {day_count} days, which is unusual for similar records."

    fallback_pairs = [
        ("bill_date", "list_date"),
        ("bill_date", "disposal_date"),
        ("invoice_date", "list_date"),
        ("invoice_date", "disposal_date"),
        ("reference_date", "disposal_date"),
    ]
    for left_stage, right_stage in fallback_pairs:
        left_date = _find_row_date_for_stage(payload, left_stage)
        right_date = _find_row_date_for_stage(payload, right_stage)
        if not left_date or not right_date:
            continue
        day_count = _date_gap_days(left_date, right_date)
        if day_count is None or day_count <= 0:
            continue
        left_label = _humanize_stage_name(left_stage)
        right_label = _humanize_stage_name(right_stage)
        return (
            f"The {left_label} ({left_date}) is {day_count} days before the {right_label} ({right_date}), "
            "which is unusual for similar records."
        )
    return None

def _clean_reason(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip("\"'")
    if not text:
        return ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            text = str(parsed.get("reason") or "").strip()
    except json.JSONDecodeError:
        pass
    text = re.sub(r"^(reason\s*:\s*)", "", text, flags=re.IGNORECASE).strip()
    text = _remove_score_language(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = sentences[0].strip() if sentences else text
    words = cleaned.split()
    if len(words) > 28:
        cleaned = " ".join(words[:28]).rstrip(",;:")
    return cleaned

def _remove_score_language(text: str) -> str:
    cleaned = re.sub(
        r"\bIF\s+score\b[^.;,]*(?:[.;,]\s*)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:threshold|score gap|model score|numeric anomaly cutoff)\b[^.;,]*(?:[.;,]\s*)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,.")
    return cleaned

def _transformed_feature_labels(pipeline: Any, feature_frame: pd.DataFrame) -> list[str]:
    base_labels = [str(column) for column in feature_frame.columns]
    imputer = pipeline.named_steps.get("imputer") if hasattr(pipeline, "named_steps") else None
    indicator_features = getattr(getattr(imputer, "indicator_", None), "features_", None)
    indicator_labels = []
    if indicator_features is not None:
        for feature_index in indicator_features:
            if 0 <= int(feature_index) < len(base_labels):
                indicator_labels.append(f"{base_labels[int(feature_index)]}__missing")
            else:
                indicator_labels.append(f"feature_{int(feature_index)}__missing")
    labels = base_labels + indicator_labels
    transformed_width = int(np.asarray(pipeline.named_steps["scaler"].transform(
        pipeline.named_steps["imputer"].transform(feature_frame.iloc[:1])
    )).shape[1]) if len(feature_frame.index) > 0 else len(labels)
    if len(labels) < transformed_width:
        labels.extend([f"feature_{index}" for index in range(len(labels), transformed_width)])
    return labels[:transformed_width]

def _source_feature_name(feature_name: Any) -> str:
    text = str(feature_name)
    if text.endswith("__missing"):
        return text[: -len("__missing")]
    return text

def _format_number(value: float | None) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "unknown"

def _readable_key(key: str) -> str:
    raw = key.split(".")[-1].split("__")[-1]
    return re.sub(r"[_\s]+", " ", raw).strip()

def _humanize_stage_name(name: str) -> str:
    return re.sub(r"[_\s]+", " ", str(name)).strip()

def _find_row_date_for_stage(row_payload: dict[str, Any], stage_name: str) -> str | None:
    target = str(stage_name).strip().lower()
    for key, value in row_payload.items():
        if value in (None, ""):
            continue
        key_text = str(key).split(".")[-1].split("__")[-1].strip().lower()
        if key_text == target:
            return _short_value(value)
    return None

def _date_gap_days(left_date_text: str, right_date_text: str) -> int | None:
    left_date = _parse_iso_date(left_date_text)
    right_date = _parse_iso_date(right_date_text)
    if left_date is None or right_date is None:
        return None
    return abs((right_date - left_date).days)

def _parse_iso_date(value: Any):
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    try:
        return pd.Timestamp(text[:10]).date()
    except Exception:
        return None

def _short_value(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= 40:
        return text
    return f"{text[:37]}..."

def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric) or np.isinf(numeric):
        return None
    return numeric
