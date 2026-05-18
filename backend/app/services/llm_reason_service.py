import logging
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
import pandas as pd

from app.core.cache import TTLCache
from app.core.config import settings
from app.core.resilience import CircuitBreaker, CircuitBreakerOpenError
from app.schemas.workbench_schema import IsolationReasonRequest

logger = logging.getLogger(__name__)

# Keyword filter used when selecting which row fields to show the LLM

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

NOISY_TEXT_FEATURE_KEYWORDS = (
    "subject",
    "payment_detail",
    "payment detail",
    "remarks",
    "reason",
    "description",
    "narration",
    "batch",
    "name",
)


# Shared infrastructure
ANOMALY_REASON_CACHE = TTLCache(ttl_seconds=settings.anomaly_reason_cache_ttl_seconds)
ANOMALY_REASON_CIRCUIT = CircuitBreaker(
    fail_threshold=settings.anomaly_reason_circuit_fail_threshold,
    reset_timeout_seconds=settings.anomaly_reason_circuit_reset_seconds,
)


# Public API: build feature signals (called from orchestrator after IF fit)


def build_feature_explanation_signals(
    pipeline: Any,
    feature_frame: pd.DataFrame,
    transformed: np.ndarray,
    anomaly_indices: Any,
) -> dict[Any, list[dict[str, Any]]]:
    """
    Build per-row feature signals explaining why IF flagged each anomaly row.

    Strategy: rank the StandardScaler-transformed features by absolute value.
    A large absolute scaled value means that feature was far from the training
    median in standard-deviation units — exactly what causes IF to assign a
    short path length (= anomaly). No SHAP needed; the scaled matrix is already
    computed by the pipeline and passed in as `transformed`.

    Args:
        pipeline:        Fitted sklearn Pipeline (imputer → scaler → model).
        feature_frame:   Original (pre-transform) feature DataFrame.
        transformed:     Output of scaler.transform(imputer.transform(feature_frame)).
        anomaly_indices: Iterable of row indices that IF flagged as anomalies.

    Returns:
        Dict mapping row index → list of signal dicts (feature, value,
        scaled_value, direction, strength, method).
    """
    if feature_frame.empty or transformed is None:
        return {}

    column_labels = _transformed_feature_labels(pipeline, feature_frame)
    transformed_array = np.asarray(transformed)

    # Safety: trim labels to match actual column count
    n_cols = transformed_array.shape[1] if transformed_array.ndim == 2 else 0
    column_labels = column_labels[:n_cols]

    transformed_frame = pd.DataFrame(
        transformed_array,
        index=feature_frame.index,
        columns=column_labels,
    )

    selected_index = [idx for idx in anomaly_indices if idx in transformed_frame.index]
    if not selected_index:
        return {}

    return _build_magnitude_signals(feature_frame, transformed_frame, selected_index)



# Public API: generate explanation text (called from llm_prep per anomaly row)


def explain_isolation_anomaly(payload: IsolationReasonRequest) -> dict[str, Any]:
    """
    Return a short human-readable explanation for one anomaly row.

    Order of operations:
      1. Cache hit → return immediately.
      2. Build pre-translated signal clauses from feature_signals.
      3. Try Ollama (circuit breaker guards the call).
      4. On failure → assemble fallback from the same clauses.
    """
    cache_key = _reason_cache_key(payload)
    cached = ANOMALY_REASON_CACHE.get(cache_key)
    if cached is not None:
        logger.debug("Cache hit for anomaly reason, review_key=%s", payload.review_key or "unknown")
        return cached

    # Translate signals to plain-English clauses once; reuse in prompt AND fallback
    signal_clauses = _translate_signals_to_clauses(payload.feature_signals)
    row_facts = _compact_row_facts(payload.row_payload)
    rule_reasons = [str(r).strip() for r in payload.existing_reasons if str(r).strip()]

    prompt = _build_prompt(signal_clauses, row_facts, rule_reasons, payload)
    fallback_reason = _build_fallback_from_clauses(signal_clauses, row_facts)

    try:
        ANOMALY_REASON_CIRCUIT.assert_request_allowed()
        llm_reason = _generate_reason_with_retry(prompt)
        if llm_reason:
            result = {
                "reason": llm_reason,
                "model": settings.anomaly_reason_model,
                "fallback": False,
            }
            ANOMALY_REASON_CACHE.set(cache_key, result)
            return result
    except CircuitBreakerOpenError as exc:
        logger.warning("Ollama circuit breaker open, using fallback: %s", exc)
    except Exception as exc:
        logger.warning("Ollama call failed, using fallback: %s", exc)

    result = {
        "reason": fallback_reason,
        "model": settings.anomaly_reason_model,
        "fallback": True,
    }
    ANOMALY_REASON_CACHE.set(cache_key, result)
    return result


def build_deterministic_isolation_reason(
    feature_signals: list[dict[str, Any]],
    row_payload: dict[str, Any] | None = None,
) -> str | None:
    """
    Build a non-LLM explanation from translated feature signals.

    This is used by dashboard hydration paths that need a stable, cheap reason
    string without making an Ollama call.
    """
    signal_clauses = _translate_signals_to_clauses(feature_signals)
    row_facts = _compact_row_facts(row_payload or {})

    if not signal_clauses and not row_facts:
        return None

    reason = _build_fallback_from_clauses(signal_clauses, row_facts)
    cleaned = _clean_reason(reason)
    return cleaned or None


#Signal extraction: ranked by scaled magnitude (replaces SHAP)


def _build_magnitude_signals(
    feature_frame: pd.DataFrame,
    transformed_frame: pd.DataFrame,
    selected_index: list[Any],
) -> dict[Any, list[dict[str, Any]]]:
    """
    For each anomaly row, sort features by |scaled_value| descending.

    The scaled value is the StandardScaler output: (x - median) / std.
    A high absolute value = that feature is far from what is normal.
    This is a faithful proxy for IF's path-length contribution and runs
    in O(n_features * log n_features) per row — negligible on big datasets.
    """
    signals_by_index: dict[Any, list[dict[str, Any]]] = {}
    selected_rows = transformed_frame.loc[selected_index]

    for row_index, scaled_row in selected_rows.iterrows():
        scaled_values = scaled_row.to_numpy()
        ranked_positions = np.argsort(np.abs(scaled_values))[::-1]

        seen: set[str] = set()
        signals: list[dict[str, Any]] = []

        for position in ranked_positions:
            feature_name = str(transformed_frame.columns[int(position)])
            source_name = _source_feature_name(feature_name)
            feature_group = _signal_group_name(feature_name)
            if feature_group in seen:
                continue
            seen.add(feature_group)

            scaled_val = float(scaled_values[int(position)])
            raw_value = (
                feature_frame.at[row_index, source_name]
                if source_name in feature_frame.columns
                else None
            )

            signals.append({
                "feature": feature_name,
                "value": _safe_float(raw_value),
                "scaled_value": _safe_float(scaled_val),
                "strength": _safe_float(abs(scaled_val)),
                "direction": "high" if scaled_val >= 0 else "low",
                "method": "scaled_magnitude",
            })

            if len(signals) >= 5:
                break

        if signals:
            prioritized = sorted(
                signals,
                key=lambda item: (
                    _signal_priority(item),
                    -abs(_safe_float(item.get("scaled_value")) or 0.0),
                ),
            )
            signals_by_index[row_index] = prioritized[:5]

    return signals_by_index


# Signal translation: turn raw signal dicts into readable English clauses
#These clauses are passed to the LLM as structured facts, not raw numbers.
# The LLM job is ONLY to combine and phrase them naturally — not to interpret


def _translate_signals_to_clauses(
    feature_signals: list[dict[str, Any]],
) -> list[str]:
    """
    Convert each feature signal into a self-contained plain-English clause.

    Examples produced:
      "the bill amount (₹1,24,500) is unusually high compared to similar records"
      "the gap from invoice date to disposal date is 45 days, which is longer than normal"
      "the invoice amount field is missing"
      "the amount was flagged as a statistical outlier (IQR rule)"
      "the days between reference date and list date is 3, which is shorter than normal"
    """
    if not isinstance(feature_signals, list):
        return []

    clauses: list[str] = []
    for signal in feature_signals[:5]:
        if not isinstance(signal, dict):
            continue
        clause = _translate_one_signal(signal)
        if clause:
            clauses.append(clause)
    return clauses


def _translate_one_signal(signal: dict[str, Any]) -> str | None:
    feature = str(signal.get("feature") or "").strip()
    if not feature:
        return None

    direction = str(signal.get("direction") or "").strip().lower()
    raw_value = signal.get("value")
    scaled = _safe_float(signal.get("scaled_value"))

    # --- Missing-value flag (feature ends with __missing or is a missingflag rule) ---
    if feature.endswith("__missing") or "missingflag" in feature.lower():
        source = _readable_feature(feature.replace("__missing", "")).strip()
        if not source:
            return "a required field is missing"
        return f"the {source} field is missing"

    # --- IQR statistical outlier flag ---
    if feature.startswith("iqr_flag::"):
        source = _readable_feature(feature[len("iqr_flag::"):]).strip()
        if not source:
            return "an unusual statistical outlier was detected by IQR analysis"
        return f"the {source} is a statistical outlier by IQR analysis"

    # --- One-hot / encoded categorical feature (pattern: column::value) ---
    if "::" in feature:
        return _translate_categorical_signal(feature, raw_value, direction)

    # --- Date-gap feature (pattern: left_stage_to_right_stage) ---
    if "_to_" in feature:
        return _translate_date_gap_signal(feature, raw_value, direction)

    # --- Plain date/time feature stored as numeric timestamp ---
    if _is_datetime_like_feature(feature):
        return _translate_datetime_signal(feature, raw_value, direction)

    # --- IsWeekend / IsBusinessHour boolean flags ---
    if "isweekend" in feature.lower():
        source = _readable_feature(feature).strip()
        val_text = "on a weekend" if _truthy(raw_value) else "not on a weekend"
        return f"the transaction date is {val_text}"

    if "isbusinesshour" in feature.lower():
        val_text = "within business hours" if _truthy(raw_value) else "outside business hours"
        return f"the transaction time is {val_text}, which is unusual"

    # --- Generic numeric feature ---
    human_name = _readable_feature(feature).strip()
    if not human_name:
        return "an unusual value pattern was detected"
    
    if raw_value is None:
        if direction == "high":
            return f"the {human_name} is unusually high compared to similar records"
        if direction == "low":
            return f"the {human_name} is unusually low compared to similar records"
        return f"the {human_name} shows an unusual pattern"

    formatted_val = _format_raw_value(raw_value)
    if formatted_val == "unknown":
        # Don't display "unknown" in parentheses; use direction only
        if direction == "high":
            return f"the {human_name} is unusually high compared to similar records"
        if direction == "low":
            return f"the {human_name} is unusually low compared to similar records"
        return f"the {human_name} shows an unusual pattern"
    
    if direction == "high":
        return f"the {human_name} ({formatted_val}) is unusually high"
    if direction == "low":
        return f"the {human_name} ({formatted_val}) is unusually low"
    return f"the {human_name} ({formatted_val}) is unusual"


def _translate_datetime_signal(feature: str, raw_value: Any, direction: str) -> str | None:
    human_name = _readable_feature(feature).strip()
    if not human_name:
        return None
    formatted_date = _format_timestamp_value(raw_value)

    if formatted_date is None:
        if direction == "high":
            return f"the {human_name} is later than usual"
        if direction == "low":
            return f"the {human_name} is earlier than usual"
        return f"the {human_name} is unusual"

    if direction == "high":
        return f"the {human_name} ({formatted_date}) is later than usual"
    if direction == "low":
        return f"the {human_name} ({formatted_date}) is earlier than usual"
    return f"the {human_name} ({formatted_date}) is unusual"


def _translate_categorical_signal(feature: str, raw_value: Any, direction: str) -> str | None:
    parts = feature.split("::", 1)
    if len(parts) != 2:
        return None

    field_name = _readable_feature(parts[0]).strip()
    category_value = _readable_category_value(parts[1]).strip()
    if not field_name:
        return None

    if _is_noise_heavy_text_field(parts[0]):
        if category_value:
            return f"the {field_name} has an unusual value ({category_value}) compared to similar records"
        return f"the {field_name} has an unusual value compared to similar records"

    if category_value:
        if direction == "low":
            return f"the record is missing a usually common {field_name} value ({category_value})"
        return f"the {field_name} value ({category_value}) is unusual compared to similar records"

    return f"the {field_name} value is unusual compared to similar records"


def _translate_date_gap_signal(feature: str, raw_value: Any, direction: str) -> str | None:
    """
    Translate a date-gap feature like 'invoice_date_to_disposal_date' into a sentence.
    raw_value is the number of days between the two dates.
    """
    parts = feature.split("_to_", 1)
    if len(parts) != 2:
        return None
    left_label = _readable_feature(parts[0]).strip()
    right_label = _readable_feature(parts[1]).strip()
    if not left_label or not right_label:
        return "an unusual time gap was detected between key dates"

    day_count = _safe_float(raw_value)
    if day_count is None:
        if direction == "high":
            return f"the gap from {left_label} to {right_label} is longer than normal"
        if direction == "low":
            return f"the gap from {left_label} to {right_label} is shorter than normal"
        return f"the gap from {left_label} to {right_label} is unusual"

    days_int = int(round(abs(day_count)))
    day_word = "day" if days_int == 1 else "days"
    if direction == "high":
        return f"the gap from {left_label} to {right_label} is {days_int} {day_word}, which is longer than normal"
    if direction == "low":
        return f"the gap from {left_label} to {right_label} is {days_int} {day_word}, which is shorter than normal"
    return f"the gap from {left_label} to {right_label} is {days_int} {day_word}"


def _readable_feature(name: str) -> str:
    """Convert a raw feature column name into a human-readable phrase."""
    # Strip table prefix (e.g. "bill.amount" → "amount")
    plain = str(name).split(".")[-1]
    # Remove internal suffixes used by the pipeline
    plain = re.sub(r"__(missing|flag|ratio|diff)$", "", plain, flags=re.IGNORECASE)
    # Replace underscores/hyphens with spaces
    plain = re.sub(r"[_\-]+", " ", plain).strip()
    result = plain.lower()
    # Ensure we never return empty string; return generic fallback
    return result if result else "field"


def _readable_category_value(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip("_ -")
    if not text or text.lower() in {"missing", "blank", "null", "none"}:
        return ""
    text = re.sub(r"[_\-]+", " ", text).strip()
    if len(text) > 60:
        text = text[:57].rstrip() + "..."
    return text.lower()


def _is_datetime_like_feature(feature_name: str) -> bool:
    normalized = str(feature_name or "").strip().lower()
    plain = normalized.split("::", 1)[0].split(".")[-1]
    return any(
        token in plain
        for token in (
            "date",
            "time",
            "timestamp",
            "created_at",
            "updated_at",
            "disposed_at",
        )
    )


def _is_noise_heavy_text_field(feature_name: str) -> bool:
    normalized = str(feature_name or "").strip().lower().replace(".", " ")
    return any(token in normalized for token in NOISY_TEXT_FEATURE_KEYWORDS)


def _signal_priority(signal: dict[str, Any]) -> int:
    feature = str(signal.get("feature") or "").strip().lower()

    if feature.startswith("iqr_flag::"):
        return 0
    if feature.endswith("__missing") or "missingflag" in feature:
        return 1
    if "_to_" in feature or "isweekend" in feature or "isbusinesshour" in feature:
        return 2
    if "::" in feature and _is_noise_heavy_text_field(feature.split("::", 1)[0]):
        return 5
    if "::" in feature:
        return 4
    return 3


def _format_raw_value(value: Any) -> str:
    """Format a raw numeric or string value for display inside a clause."""
    if value is None:
        return "unknown"
    numeric = _safe_float(value)
    if numeric is not None:
        if abs(numeric) >= 1_000:
            return f"{numeric:,.0f}"
        if abs(numeric) < 1 and numeric != 0:
            return f"{numeric:.4f}"
        return f"{numeric:.2f}"
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:40] if len(text) <= 40 else f"{text[:37]}..."


def _format_timestamp_value(value: Any) -> str | None:
    numeric = _safe_float(value)
    if numeric is not None:
        # Feature engineering stores datetime columns as Unix seconds.
        if 100_000_000 <= abs(numeric) <= 50_000_000_000:
            try:
                dt = datetime.fromtimestamp(float(numeric), tz=timezone.utc)
                return dt.strftime("%Y-%m-%d")
            except (OverflowError, OSError, ValueError):
                return None

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return text[:10]
    return None


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    numeric = _safe_float(value)
    if numeric is not None:
        return numeric >= 0.5
    return str(value).strip().lower() in {"true", "1", "yes"}



# Prompt construction


def _build_prompt(
    signal_clauses: list[str],
    row_facts: str,
    rule_reasons: list[str],
    payload: IsolationReasonRequest,
) -> str:
    """
    Build the Ollama prompt.

    The prompt gives the LLM:
      - Pre-translated plain-English clauses (what actually looks unusual)
      - Any rule-based reasons already detected
      - A small selection of relevant raw row fields for extra context

    The LLM is told NOT to mention scores/thresholds and NOT to invent facts.
    Its only job is to combine the provided clauses into one clear sentence.
    """
    clauses_text = (
        "\n".join(f"- {c}" for c in signal_clauses)
        if signal_clauses
        else "- No specific signals available"
    )
    rules_text = (
        ", ".join(rule_reasons)
        if rule_reasons
        else "none"
    )

    return (
        "/no_think\n"
        "You explain anomalies in a bill/invoice review system.\n"
        "Return ONLY valid JSON: {\"reason\":\"...\"}\n"
        "Write exactly one sentence, maximum 40 words.\n"
        "Use only the signals listed below. Do not invent numbers, dates, vendors, or fraud.\n"
        "Do not mention 'anomaly score', 'threshold', 'model', or 'Isolation Forest'.\n"
        "Combine the signals into one plain business-language sentence.\n\n"
        f"Unusual signals detected for this record:\n{clauses_text}\n\n"
        f"Rule-based flags: {rules_text}\n"
        f"Supporting row data: {row_facts if row_facts else 'none'}\n\n"
        "JSON:"
    )



# Fallback reason (no LLM required)

def _build_fallback_from_clauses(
    signal_clauses: list[str],
    row_facts: str,
) -> str:
    """
    Build a fallback explanation directly from pre-translated clauses.
    Used when Ollama is unavailable. No hardcoded templates — the clauses
    themselves are already human-readable.
    """
    if signal_clauses:
        # Use the top 2 clauses, joined naturally
        top = signal_clauses[:2]
        if len(top) == 1:
            return f"This record is unusual because {top[0]}."
        return f"This record is unusual: {top[0]}, and {top[1]}."

    # Last resort: first two keyword-matched row fields
    if row_facts:
        first = row_facts.split("; ")[:2]
        return f"Unusual combination of values: {'; '.join(first)}."

    return "This record has an unusual combination of values compared to similar records."



# Ollama HTTP call with retry

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
    clamped = max(
        settings.anomaly_reason_timeout_min_seconds,
        min(settings.anomaly_reason_timeout_seconds, timeout_seconds),
    )
    logger.debug(
        "Anomaly-reason timeout=%.2fs prompt_len=%d circuit_failures=%s",
        clamped, prompt_length, circuit_snapshot["failure_count"],
    )
    return clamped


# Cache key


def _reason_cache_key(payload: IsolationReasonRequest) -> str:
    raw = payload.model_dump(mode="json")
    return json.dumps(raw, sort_keys=True, ensure_ascii=True)



# LLM output cleaning


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
    if len(words) > 35:
        cleaned = " ".join(words[:35]).rstrip(",;:")
    return cleaned


def _remove_score_language(text: str) -> str:
    cleaned = re.sub(
        r"\bIF\s+score\b[^.;,]*(?:[.;,]\s*)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:isolation forest|anomaly score|threshold|score gap|model score|numeric anomaly cutoff)\b[^.;,]*(?:[.;,]\s*)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,.")
    return cleaned



# Pipeline label helpers (unchanged from previous version)


def _transformed_feature_labels(pipeline: Any, feature_frame: pd.DataFrame) -> list[str]:
    base_labels = [str(c) for c in feature_frame.columns]
    imputer = pipeline.named_steps.get("imputer") if hasattr(pipeline, "named_steps") else None
    indicator_features = getattr(getattr(imputer, "indicator_", None), "features_", None)
    indicator_labels: list[str] = []
    if indicator_features is not None:
        for fi in indicator_features:
            if 0 <= int(fi) < len(base_labels):
                indicator_labels.append(f"{base_labels[int(fi)]}__missing")
            else:
                indicator_labels.append(f"feature_{int(fi)}__missing")
    labels = base_labels + indicator_labels
    if len(feature_frame.index) > 0:
        transformed_width = int(
            np.asarray(
                pipeline.named_steps["scaler"].transform(
                    pipeline.named_steps["imputer"].transform(feature_frame.iloc[:1])
                )
            ).shape[1]
        )
    else:
        transformed_width = len(labels)
    if len(labels) < transformed_width:
        labels.extend([f"feature_{i}" for i in range(len(labels), transformed_width)])
    return labels[:transformed_width]


def _source_feature_name(feature_name: Any) -> str:
    text = str(feature_name)
    return text[: -len("__missing")] if text.endswith("__missing") else text


def _signal_group_name(feature_name: Any) -> str:
    text = str(feature_name)
    # Keep missing indicators distinct so we can explain "field is missing"
    # instead of collapsing them back into the base column.
    if text.endswith("__missing"):
        return text
    return _source_feature_name(text)



# General helpers

def _compact_row_facts(row_payload: dict[str, Any]) -> str:
    """Select the most informative fields from the raw row for the prompt context."""
    if not isinstance(row_payload, dict):
        return ""
    selected: list[tuple[str, Any]] = []
    for key, value in row_payload.items():
        if value in (None, ""):
            continue
        lower_key = str(key).lower()
        if any(kw in lower_key for kw in IMPORTANT_KEYWORDS):
            selected.append((str(key), value))
    if not selected:
        selected = [
            (str(k), v)
            for k, v in list(row_payload.items())[:12]
            if v not in (None, "")
        ]
    parts: list[str] = []
    for key, value in selected[:16]:
        plain_key = _readable_feature(key).strip()
        if plain_key:  # Only include if key name is meaningful
            parts.append(f"{plain_key}={_format_raw_value(value)}")
    return "; ".join(parts)


def _format_number(value: float | None) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "unknown"


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric) or np.isinf(numeric):
        return None
    return numeric
