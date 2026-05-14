import logging
import json
import re
from typing import Any

import httpx
import numpy as np
import pandas as pd

from app.core.config import settings
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

    signals_by_index: dict[Any, list[dict[str, Any]]] = {}
    selected_rows = transformed_frame.loc[selected_index]

    for row_index, row in selected_rows.iterrows():
        ranked_features = row.abs().sort_values(ascending=False)
        signals: list[dict[str, Any]] = []
        for feature_name, strength in ranked_features.head(5).items():
            source_feature_name = _source_feature_name(feature_name)
            raw_value = (
                feature_frame.at[row_index, source_feature_name]
                if source_feature_name in feature_frame.columns
                else None
            )
            signals.append(
                {
                    "feature": str(source_feature_name),
                    "value": _safe_float(raw_value),
                    "strength": _safe_float(strength),
                    "direction": "high" if float(row[feature_name]) >= 0 else "low",
                }
            )
        if signals:
            signals_by_index[row_index] = signals

    return signals_by_index

def explain_isolation_anomaly(payload: IsolationReasonRequest) -> dict[str, Any]:
    prompt = _build_prompt(payload)
    fallback_reason = _fallback_reason(payload)

    try:
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
            timeout=settings.anomaly_reason_timeout_seconds,
        )
        response.raise_for_status()
        reason = _clean_reason(response.json().get("response"))
        if reason:
            return {
                "reason": reason,
                "model": settings.anomaly_reason_model,
                "fallback": False,
            }
    except Exception as exc:
        logger.warning("Ollama anomaly reason generation failed: %s", exc)

    return {
        "reason": fallback_reason,
        "model": settings.anomaly_reason_model,
        "fallback": True,
    }

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
    facts = _compact_row_facts(payload.row_payload)
    if facts:
        first_facts = facts.split("; ")[:2]
        return f"Unusual row signals: {'; '.join(first_facts)}."
    return "This row has an unusual combination of values compared with the normal review records."

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
