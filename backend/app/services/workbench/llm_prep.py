import re
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.workbench_schema import IsolationReasonRequest, WorkbenchRunRequest
from app.services.llm_reason_service import explain_isolation_anomaly
from app.services.workbench.constants import SYSTEM_COLUMNS
from app.services.workbench.sql_runtime import _feature_rule_aliases
from app.services.workbench.utils import _safe_json, _safe_numeric_scalar

def _feature_name_for_tables(selected_tables: list[str]) -> str:
    parts = [str(table) for table in selected_tables[:3]]
    parts.extend(["null"] * (3 - len(parts)))
    return ".".join(parts)

def _presentable_reason_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    return re.sub(r"\s+", " ", text_value.replace("OUTLIER::", "").replace("_", " ")).strip()

def _review_payload_for_row(row: pd.Series, feature_aliases: set[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for column, value in row.items():
        column_name = str(column)
        if column_name in SYSTEM_COLUMNS or column_name in feature_aliases:
            continue
        payload[column_name] = _safe_json(value)
    return payload

def _build_persisted_llm_if_reasons(
    payload: WorkbenchRunRequest,
    filtered_feature_frame: pd.DataFrame,
    filtered_joined: pd.DataFrame,
    filtered_ml_flag: pd.Series,
    filtered_human_outlier_flag: pd.Series,
    filtered_human_reasons: pd.Series,
    filtered_builtin_reasons: pd.Series,
    filtered_isolation_scores: np.ndarray,
    ml_threshold: float,
    explanation_signals: dict[Any, list[dict[str, Any]]],
) -> dict[Any, dict[str, Any]]:
    stored_reasons: dict[Any, dict[str, Any]] = {}
    feature_aliases = set(_feature_rule_aliases(payload.feature_rules))
    review_key = _feature_name_for_tables(payload.selected_tables)

    for position, row_index in enumerate(filtered_feature_frame.index):
        if not bool(filtered_ml_flag.get(row_index, False)):
            continue

        reason_list: list[str] = []
        human_reason = _presentable_reason_text(filtered_human_reasons.get(row_index))
        builtin_reason = _presentable_reason_text(filtered_builtin_reasons.get(row_index))
        if human_reason:
            reason_list.append(human_reason)
        if builtin_reason and builtin_reason not in reason_list:
            reason_list.append(builtin_reason)

        row_payload = _review_payload_for_row(filtered_joined.loc[row_index], feature_aliases)
        request_payload = IsolationReasonRequest(
            prediction_id=None,
            review_key=review_key,
            if_score=_safe_numeric_scalar(filtered_isolation_scores[position], default=None),
            ml_threshold=ml_threshold,
            rule_anomaly=bool(filtered_human_outlier_flag.get(row_index, False)),
            rule_count=len(reason_list),
            existing_reasons=reason_list,
            feature_signals=explanation_signals.get(row_index, []),
            row_payload=row_payload,
        )
        stored_reasons[row_index] = explain_isolation_anomaly(request_payload)

    return stored_reasons
