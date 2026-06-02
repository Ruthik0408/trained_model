from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.schemas.workbench_schema import IsolationReasonRequest
from app.services.workbench.utils import _select_series_column


# Constants
IMPORTANT_KEYWORDS = (
    "amount",
    "invoice",
    "bill",
    "approved",
    "passed",
    "record_status",
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

# Connectors used to join multiple clauses so sentences don't sound repetitive
_CLAUSE_CONNECTORS = [
    "and also",
    "additionally",
    "while",
    "along with",
    "combined with",
    "and",
]

# Helper function to extract rule reasons (DRY principle)
def _extract_rule_reasons(existing_reasons: Any) -> list[str]:
    """Extract non-empty rule reasons from payload."""
    return [
        str(r).strip()
        for r in (existing_reasons or [])
        if str(r).strip()
    ]


# Public API — build signals during orchestration (unchanged interface)

def build_feature_explanation_signals(
    pipeline: Any,
    feature_frame: pd.DataFrame,
    transformed: np.ndarray,
    anomaly_indices: Any,
    transformed_feature_labels: list[str] | None = None,
) -> dict[Any, list[dict[str, Any]]]:
    """
    Build per-row feature signals explaining why IF flagged each anomaly row.

    Ranks transformed IF features by absolute value. Continuous numeric
    features are standardized, while indicator-style features are preserved
    as raw 0/1 values so the explanation signals still mirror model input.

    Returns:
        Dict mapping row-index → list of signal dicts.
    """
    if feature_frame.empty or transformed is None:
        return {}

    column_labels = list(transformed_feature_labels or _transformed_feature_labels(pipeline, feature_frame))
    transformed_array = np.asarray(transformed)

    n_cols = transformed_array.shape[1] if transformed_array.ndim == 2 else 0
    column_labels = column_labels[:n_cols]

    transformed_frame = pd.DataFrame(
        transformed_array,
        index=feature_frame.index,
        columns=column_labels,
    )

    selected_index = [
        idx for idx in anomaly_indices if idx in transformed_frame.index
    ]
    if not selected_index:
        return {}

    return _build_magnitude_signals(feature_frame, transformed_frame, selected_index)


def build_deterministic_isolation_reason(
    feature_signals: list[dict[str, Any]],
    row_payload: dict[str, Any] | None = None,
) -> str | None:
    """
    Build a non-LLM explanation from translated feature signals.

    Convenience wrapper for dashboard hydration paths that only need
    a plain string, not the full result dict.
    """
    mock_payload = IsolationReasonRequest(
        feature_signals=feature_signals,
        row_payload=row_payload or {},
    )
    result = _build_deterministic_explanation(mock_payload)
    cleaned = _clean_reason(result["reason"])
    if cleaned and not cleaned.endswith("."):
        cleaned = f"{cleaned}."
    return cleaned or None

# Deterministic explanation builder — the real improvement

def _build_deterministic_explanation(
    payload: IsolationReasonRequest,
) -> dict[str, Any]:
    """
    Build a high-quality explanation without any LLM call.

    Strategy:
      1. Translate each IF signal into a self-contained English clause.
      2. Prioritise clauses (IQR flags > missing fields > date gaps >
         categoricals > generic numeric).
      3. Select the top 3 clauses and assemble them with natural connectors.
      4. Append rule-based reasons if present and not already covered.
      5. Clean and return.
    """
    signal_clauses = [
        clause
        for clause in _translate_signals_to_clauses(payload.feature_signals)
        if not _is_missing_reason_clause(clause)
    ]
    rule_reasons = [
        reason
        for reason in _extract_rule_reasons(payload.existing_reasons)
        if not _is_missing_reason_clause(reason)
    ]
    row_facts = _compact_row_facts(payload.row_payload)

    reason = _assemble_deterministic_sentence(signal_clauses, rule_reasons, row_facts)
    reason = _clean_reason(reason)

    if not reason:
        reason = "This record shows an unusual combination of values compared to similar records."

    return {
        "reason": reason,
        "model": "deterministic",
        "fallback": True,
    }


def _is_missing_reason_clause(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return (
        "missing field" in text
        or "field is missing" in text
        or "required field is missing" in text
        or "record is missing" in text
        or "missing a usually-common" in text
        or "__missing" in text
        or "missingflag" in text
    )


def _assemble_deterministic_sentence(
    signal_clauses: list[str],
    rule_reasons: list[str],
    row_facts: str,
) -> str:
    """
    Assemble a coherent sentence from pre-translated clauses.

    Linguistic rules applied here:
    • First clause is always a complete sentence starter.
    • Second clause uses a connector word chosen by position so sentences
      don't all sound identical.
    • Third clause is appended only if it adds genuinely new information
      (different feature group from clause 1 and 2).
    • Rule-based reasons are appended at the end if not already implied
      by the signal clauses.
    """
    parts: list[str] = []

    if signal_clauses:
        # Clause 1 — lead with "This record is flagged because …"
        parts.append(_capitalise(signal_clauses[0]))

    if len(signal_clauses) >= 2:
        connector = _CLAUSE_CONNECTORS[1 % len(_CLAUSE_CONNECTORS)]  # "additionally"
        parts.append(f"{connector} {signal_clauses[1]}")

    if len(signal_clauses) >= 3:
        # Only add a 3rd clause if it belongs to a different feature group
        if _different_feature_group(signal_clauses[0], signal_clauses[2]):
            connector = _CLAUSE_CONNECTORS[4 % len(_CLAUSE_CONNECTORS)]  # "combined with"
            parts.append(f"{connector} {signal_clauses[2]}")

    # Append rule reasons that are not already paraphrased in the clauses
    for rule in rule_reasons[:2]:
        rule_clean = re.sub(r"^(?:OUTLIER|RULE)::", "", rule).replace("_", " ").strip().lower()
        if rule_clean and not any(rule_clean[:20] in p.lower() for p in parts):
            parts.append(f"(rule: {rule_clean})")

    if not parts:
        # Last resort: use raw row facts
        if row_facts:
            snippets = row_facts.split("; ")[:2]
            return f"Unusual combination of values: {'; '.join(snippets)}."
        return ""

    sentence = ", ".join(parts)

    # Ensure the sentence ends with a period
    if not sentence.rstrip().endswith("."):
        sentence = sentence.rstrip() + "."

    return sentence


def _capitalise(text: str) -> str:
    """Capitalise just the first character without lowercasing the rest."""
    if not text:
        return text
    return text[0].upper() + text[1:]


def _different_feature_group(clause_a: str, clause_b: str) -> bool:
    """
    Heuristic: two clauses are 'different' if their first content word differs.
    Prevents assembling "X is high, additionally X is high, combined with X is high".
    """
    def _first_content_word(text: str) -> str:
        stop = {"the", "a", "an", "this", "that", "record", "is", "are", "was"}
        for word in re.findall(r"[a-z]+", text.lower()):
            if word not in stop:
                return word
        return text[:10]

    return _first_content_word(clause_a) != _first_content_word(clause_b)


# Signal extraction — ranked by transformed magnitude

def _build_magnitude_signals(
    feature_frame: pd.DataFrame,
    transformed_frame: pd.DataFrame,
    selected_index: list[Any],
) -> dict[Any, list[dict[str, Any]]]:
    """
    For each anomaly row, sort features by |transformed_value| descending.

    Continuous numeric features use standardized values.
    Indicator-style features keep their raw 0/1 representation.
    """
    signals_by_index: dict[Any, list[dict[str, Any]]] = {}
    selected_rows = transformed_frame.loc[selected_index]

    for row_index, scaled_row in selected_rows.iterrows():
        scaled_values = scaled_row.to_numpy()
        ranked_positions = np.argsort(np.abs(scaled_values))[::-1]

        seen: set[str] = set()
        signals: list[dict[str, Any]] = []

        for position in ranked_positions:
            raw_feature_name = str(transformed_frame.columns[int(position)])
            feature_name = _display_feature_name(raw_feature_name, feature_frame.columns)
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

            signals.append(
                {
                    "feature": feature_name,
                    "value": _safe_signal_value(raw_value),
                    "scaled_value": _safe_float(scaled_val),
                    "strength": _safe_float(abs(scaled_val)),
                    "direction": "high" if scaled_val >= 0 else "low",
                    "method": "transformed_magnitude",
                    "comparison": _build_signal_comparison(
                        feature_frame,
                        source_name,
                        feature_name,
                    ),
                }
            )

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


# Signal translation — raw signal dicts → English clauses
def _translate_signals_to_clauses(
    feature_signals: list[dict[str, Any]],
) -> list[str]:
    """
    Convert each feature signal into a self-contained plain-English clause.

    Improved over previous version:
    • Amounts formatted with locale-style separators (1,24,500 → 1,24,500)
    • Date-gap direction explicitly stated ("longer/shorter than normal")
    • Missing fields phrased more naturally
    """
    if not isinstance(feature_signals, list):
        return []

    clauses: list[str] = []
    for signal in feature_signals[: settings.anomaly_reason_max_signals]:
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
    strength = _safe_float(signal.get("strength")) or 0.0

    # ── Missing-value flag ────────────────────────────────────────────────
    if feature.endswith("__missing") or "missingflag" in feature.lower():
        source = _readable_feature(feature.replace("__missing", "")).strip()
        if not source:
            return "a required field is missing"
        return f"the {source} field is missing"


    # ── One-hot / encoded categorical feature (column::value) ─────────────
    if "::" in feature:
        return _translate_categorical_signal(feature, raw_value, direction)

    # ── Date-gap feature (left_stage_to_right_stage) ──────────────────────
    if "_to_" in feature:
        return _translate_date_gap_signal(feature, raw_value, direction)

    if feature.lower().endswith("_date") or feature.lower().endswith("date"):
        date_clause = _translate_datetime_signal(feature, raw_value, direction)
        if date_clause:
            return date_clause

    # ── IsWeekend / IsBusinessHour boolean flags ──────────────────────────
    if "isweekend" in feature.lower():
        val_text = "on a weekend" if _truthy(raw_value) else "not on a weekend"
        return f"the transaction was posted {val_text}, which is unusual"

    if "isbusinesshour" in feature.lower():
        val_text = (
            "within normal business hours" if _truthy(raw_value)
            else "outside business hours"
        )
        return f"the transaction time is {val_text}, which is unusual"
    # ── Generic numeric / normal feature fallback ─────────────────────────
    readable = _readable_feature(feature)
    value_text = _format_raw_value(raw_value)

    if direction == "high":
        return f"the {readable} value ({value_text}) is higher than normal"

    if direction == "low":
        return f"the {readable} value ({value_text}) is lower than normal"

    if strength >= 2:
        return f"the {readable} value ({value_text}) is unusual"

    return None

def _translate_categorical_signal(
    feature: str,
    raw_value: Any,
    direction: str,
) -> str | None:
    parts = feature.split("::", 1)
    if len(parts) != 2:
        return None

    field_name = _readable_feature(parts[0]).strip()
    category_value = _readable_category_value(parts[1]).strip()
    if not field_name:
        return None

    if category_value:
        if direction == "low":
            return (
                f"the record is missing a usually-common "
                f"{field_name} value ({category_value})"
            )
        return (
            f"the {field_name} value ({category_value}) "
            f"is an unusual value compared to similar records"
        )

    return f"the {field_name} value is unusual compared to similar records"


def _translate_date_gap_signal(
    feature: str,
    raw_value: Any,
    direction: str,
) -> str | None:
    """
    Translate a date-gap feature like 'invoice_date_to_disposal_date'.

    Improved: raw_value is the number of days; we now always state the
    direction explicitly and use "significantly" for large gaps.
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

    # Qualify very large gaps
    qualifier = "significantly " if days_int > 90 else ""

    if direction == "high":
        return (
            f"the gap from {left_label} to {right_label} is "
            f"{days_int} {day_word}, which is {qualifier}longer than normal"
        )
    if direction == "low":
        return (
            f"the gap from {left_label} to {right_label} is "
            f"{days_int} {day_word}, which is {qualifier}shorter than normal"
        )
    return (
        f"the gap from {left_label} to {right_label} is {days_int} {day_word}"
    )


def _translate_datetime_signal(
    feature: str,
    raw_value: Any,
    direction: str,
) -> str | None:
    readable = _readable_feature(feature).strip()
    if not readable:
        readable = "date field"

    date_text = _format_datetime_value(raw_value)
    if date_text:
        if direction == "low":
            return f"the {readable} ({date_text}) is earlier than usual"
        return f"the {readable} ({date_text}) is later than usual"

    if direction == "low":
        return f"the {readable} is earlier than usual"
    return f"the {readable} is later than usual"


def _format_datetime_value(value: Any) -> str | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None

    timestamp = abs(numeric)
    if timestamp >= 1e12:
        seconds = numeric / 1000.0
    elif timestamp >= 1e9:
        seconds = numeric
    else:
        return None

    try:
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None

    return dt.strftime("%Y-%m-%d")


# Feature name helpers
def _readable_feature(name: str) -> str:
    """Convert a raw feature column name into a human-readable phrase."""
    plain = str(name).split(".")[-1]
    plain = re.sub(r"__(missing|flag|ratio|diff)$", "", plain, flags=re.IGNORECASE)
    plain = re.sub(r"[_\-]+", " ", plain).strip()
    result = plain.lower()
    return result if result else "field"


def _readable_category_value(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip("_ -")
    if not text or text.lower() in {"missing", "blank", "null", "none"}:
        return ""
    text = re.sub(r"[_\-]+", " ", text).strip()
    if len(text) > 60:
        text = text[:57].rstrip() + "..."
    return text.lower()



def _signal_priority(signal: dict[str, Any]) -> int:
    feature = str(signal.get("feature") or "").strip().lower()
    if feature.startswith("iqr_flag::"):
        return 0
    if feature.endswith("__missing") or "missingflag" in feature:
        return 1
    if "_to_" in feature or "isweekend" in feature or "isbusinesshour" in feature:
        return 2
    if "::" in feature:
        return 3
    return 4



# Value formatting
def _format_raw_value(value: Any) -> str:
    """Format a raw numeric or string value for display inside a clause."""
    if value is None:
        return "unknown"
    numeric = _safe_float(value)
    if numeric is not None:
        if abs(numeric) >= 1_000:
            # Locale-style grouping makes large numbers more readable
            return f"{numeric:,.0f}"
        if abs(numeric) < 1 and numeric != 0:
            return f"{numeric:.4f}"
        return f"{numeric:.2f}"
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:40] if len(text) <= 40 else f"{text[:37]}..."


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    numeric = _safe_float(value)
    if numeric is not None:
        return numeric >= 0.5
    return str(value).strip().lower() in {"true", "1", "yes"}


# LLM output cleaning (improved: no mid-word truncation)
def _clean_reason(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip("\"'")
    if not text:
        return ""

    # Try to extract from JSON wrapper
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            text = str(parsed.get("reason") or "").strip()
    except json.JSONDecodeError:
        pass

    text = re.sub(r"^(reason\s*:\s*)", "", text, flags=re.IGNORECASE).strip()
    text = _remove_score_language(text)

    # Take only the first sentence
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = sentences[0].strip() if sentences else text

    # Truncate at word boundary — never mid-word
    words = cleaned.split()
    if len(words) > 35:
        cleaned = " ".join(words[:35])
        # Remove trailing punctuation fragments before adding period
        cleaned = cleaned.rstrip(",;:") + "."

    return cleaned


def _remove_score_language(text: str) -> str:
    cleaned = re.sub(
        r"\bIF\s+score\b[^.;,]*(?:[.;,]\s*)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:isolation forest|anomaly score|threshold|score gap"
        r"|model score|numeric anomaly cutoff)\b[^.;,]*(?:[.;,]\s*)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ;,.")


# Pipeline label helpers

def _transformed_feature_labels(
    pipeline: Any,
    feature_frame: pd.DataFrame,
) -> list[str]:
    base_labels = [str(c) for c in feature_frame.columns]
    imputer = (
        pipeline.named_steps.get("imputer")
        if hasattr(pipeline, "named_steps")
        else None
    )
    indicator_features = getattr(
        getattr(imputer, "indicator_", None), "features_", None
    )
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
                    pipeline.named_steps["imputer"].transform(
                        feature_frame.iloc[:1]
                    )
                )
            ).shape[1]
        )
    else:
        transformed_width = len(labels)
    if len(labels) < transformed_width:
        labels.extend(
            [f"feature_{i}" for i in range(len(labels), transformed_width)]
        )
    return labels[:transformed_width]


def _source_feature_name(feature_name: Any) -> str:
    text = str(feature_name)
    if "::" in text:
        return text.split("::", 1)[0]
    return text[: -len("__missing")] if text.endswith("__missing") else text


def _display_feature_name(
    feature_name: Any,
    source_columns: Any,
) -> str:
    text = str(feature_name)
    if text in source_columns or "::" in text or text.endswith("__missing"):
        return text

    matches = [
        str(column)
        for column in source_columns
        if text.startswith(f"{column}_") and len(text) > len(str(column)) + 1
    ]
    if not matches:
        return text

    source_name = max(matches, key=len)
    category_value = text[len(source_name) + 1 :]
    if not category_value:
        return text
    return f"{source_name}::{category_value}"


def _build_signal_comparison(
    feature_frame: pd.DataFrame,
    source_name: str,
    feature_name: str,
) -> dict[str, Any] | None:
    series_name = feature_name if feature_name in feature_frame.columns else source_name
    if series_name not in feature_frame.columns:
        return None

    series = _select_series_column(feature_frame, series_name)
    non_missing = series.dropna()
    if len(non_missing.index) == 0:
        return None

    if feature_name.endswith("__missing"):
        total_count = int(len(series.index))
        missing_count = int(series.isna().sum())
        present_count = int(total_count - missing_count)
        return {
            "kind": "missing",
            "total_count": total_count,
            "present_count": present_count,
            "missing_count": missing_count,
            "present_ratio": _safe_float(present_count / total_count) if total_count else None,
        }

    if "::" in feature_name:
        category_value = feature_name.split("::", 1)[1]
        normalized_category = str(category_value).strip().lower()
        active_count = int(
            series.astype("string")
            .str.strip()
            .str.lower()
            .eq(normalized_category)
            .sum()
        )
        total_count = int(len(series.index))
        return {
            "kind": "category",
            "total_count": total_count,
            "match_count": active_count,
            "match_ratio": _safe_float(active_count / total_count) if total_count else None,
        }

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric_series.index) == 0:
        return None

    return {
        "kind": "numeric",
        "count": int(len(numeric_series.index)),
        "median": _safe_float(numeric_series.median()),
        "p25": _safe_float(numeric_series.quantile(0.25)),
        "p75": _safe_float(numeric_series.quantile(0.75)),
        "min": _safe_float(numeric_series.min()),
        "max": _safe_float(numeric_series.max()),
        "mean": _safe_float(numeric_series.mean()),
    }


def _signal_group_name(feature_name: Any) -> str:
    text = str(feature_name)
    if text.endswith("__missing"):
        return text
    return _source_feature_name(text)

# Row fact extraction

def _compact_row_facts(row_payload: dict[str, Any]) -> str:
    """Select the most informative fields from the raw row for prompt context."""
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
    for key, value in selected[: settings.anomaly_reason_max_row_facts]:
        plain_key = _readable_feature(key).strip()
        if plain_key:
            parts.append(f"{plain_key}={_format_raw_value(value)}")
    return "; ".join(parts)



# General helpers

def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric) or np.isinf(numeric):
        return None
    return numeric


def _safe_signal_value(value: Any) -> Any:
    numeric = _safe_float(value)
    if numeric is not None:
        return numeric
    if value is None or pd.isna(value):
        return None
    return str(value)
