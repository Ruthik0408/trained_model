import math
import re
from datetime import date, datetime
from hashlib import blake2s
from typing import Any

import numpy as np
import pandas as pd

def _round_storage_score(value: Any, digits: int = 3) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return round(float(numeric), digits)

def _is_amount_like_column(column_name: str) -> bool:
    normalized = str(column_name).strip().lower()
    return "amount" in normalized or normalized.endswith("_amt") or ".amt" in normalized

def _is_identifier_like_column(column_name: str) -> bool:
    plain_name = str(column_name).strip().lower().split(".")[-1]
    return (
        plain_name == "id"
        or plain_name.endswith("_id")
        or plain_name.startswith("fk_")
        or plain_name.endswith("_no")
        or plain_name.endswith("_number")
    )

def _safe_rule_name(name: str | None, prefix: str) -> str:
    text_value = (name or "").strip()
    if not text_value:
        return prefix

    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", text_value)
    if sanitized and not (sanitized[0].isalpha() or sanitized[0] == "_"):
        sanitized = f"col_{sanitized}"
    return sanitized[:63] if len(sanitized) > 63 else sanitized

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

def _safe_numeric_scalar(value: Any, default: Any = 0.0) -> float | None:
    try:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            if default is None:
                return None
            return float(default)
        return float(numeric)
    except Exception:
        if default is None:
            return None
        return float(default)

def _slug_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return token or "table"
