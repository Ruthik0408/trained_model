import math
import re
from datetime import date, datetime
from typing import Any

import pandas as pd


def _roundoff_score(value: Any, digits: int = 3) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return round(float(numeric), digits)


def _safe_rule_name(name: str | None, prefix: str) -> str:
    text_value = (name or "").strip()
    if not text_value:
        return prefix

    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", text_value)
    if sanitized and not (sanitized[0].isalpha() or sanitized[0] == "_"):
        sanitized = f"col_{sanitized}"
    return sanitized[:63] if len(sanitized) > 63 else sanitized


def _is_date_like_column_name(column_name: str, data_type: str | None = None) -> bool:
    lower_name = str(column_name).strip().lower()
    lower_type = str(data_type or "").strip().lower()

    return any(token in lower_type for token in ["date", "time"]) or any(
        token in lower_name
        for token in ["date", "time", "created_at", "updated_at", "timestamp"]
    )


def _safe_json(value):
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _safe_numeric_scalar(
    value: Any,
    default: Any = 0.0,
) -> float | None:
    try:
        if value is None:
            return None if default is None else float(default)

        return float(value)

    except (TypeError, ValueError):
        return None if default is None else float(default)


def _select_series_column(frame: pd.DataFrame, column_name: str) -> pd.Series:
    selected = frame.loc[:, column_name]

    if isinstance(selected, pd.Series):
        return selected

    if selected.shape[1] == 0:
        raise ValueError(f"Column not found: {column_name}")

    if selected.shape[1] > 1:
        from app.services.workbench.constants import logger

        logger.warning(
            "Multiple columns matched '%s' while selecting a scalar series; using the first match.",
            column_name,
        )

    return selected.iloc[:, 0]


def _slug_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return token or "table"
