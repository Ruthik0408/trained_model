import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.orm import Session

from app.core.models import WorkbenchRun
from app.schemas.workbench_schema import WorkbenchRunRequest
from app.services.workbench.constants import (
    FEATURE_NAME_COLUMN,
    FEATURE_VALUES_COLUMN,
    FEEDBACK_SCORE_COLUMN,
    FEEDBACK_TO_SCORE,
    HUMAN_RULE_COLUMN,
    HUMAN_RULE_NAME_COLUMN,
    IF_SCORE_COLUMN,
    ISOLATION_RULE_COLUMN,
    ML_THRESHOLD_COLUMN,
    RESULT_SCHEMA,
    REVIEW_PAYLOAD_COLUMN,
    RUN_ID_COLUMN,
    SCORE_TO_FEEDBACK,
    SERIAL_COLUMN,
    SYSTEM_COLUMNS,
    logger,
)
from app.services.workbench.source_db import (
    _clear_result_table_columns_cache,
    _clear_result_table_exists_cache,
    _quote,
    _result_table_columns,
    _result_table_exists,
    _result_table_ref,
    _source_begin,
    _source_engine,
    _previous_dataset_row_count,
)
from app.services.workbench.sql_runtime import _feature_rule_aliases
from app.services.workbench.source_db import _normalize_storage_columns
from app.services.workbench.utils import _round_storage_score, _safe_json, _safe_numeric_scalar


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


@dataclass(frozen=True)
class DatasetBuildInputs:
    joined: pd.DataFrame
    feature_frame: pd.DataFrame
    payload: WorkbenchRunRequest
    dataset_table: str
    dataset_run_id: int
    human_outlier_flag: pd.Series
    human_reasons: list[str]
    human_reason_series: pd.Series | None
    builtin_reason_series: pd.Series | None
    isolation_scores: np.ndarray
    ml_flag: pd.Series
    ml_threshold: float
    final_flag: pd.Series
    filtered_joined_override: pd.DataFrame | None = None
    explanation_signals_override: dict[Any, list[dict[str, Any]]] | None = None

def list_saved_datasets(db: Session) -> list[dict[str, Any]]:
    runs = db.query(WorkbenchRun).order_by(WorkbenchRun.run_id.desc()).all()
    latest_by_table: dict[str, WorkbenchRun] = {}
    for run in runs:
        metrics = run.metrics_json or {}
        dataset_table = metrics.get("dataset_table")
        if dataset_table and dataset_table not in latest_by_table:
            latest_by_table[dataset_table] = run

    items: list[dict[str, Any]] = []
    for dataset_table, run in latest_by_table.items():
        if not _result_table_exists(dataset_table):
            continue
        items.append(
            {
                "dataset_table": dataset_table,
                "run_id": run.run_id,
                "run_name": run.run_name,
                "selected_tables": run.source_tables_json or [],
                "total_rows": run.total_rows,
                "final_anomaly_count": run.final_anomaly_count,
                "selected_model": run.selected_model,
                "amount_field": run.amount_field,
            }
        )
    return items

def update_dataset_feedback(db: Session, payload) -> dict[str, Any]:
    del db
    feedback = str(payload.feedback).strip().lower()
    feedback_score = _feedback_to_score(feedback)
    _update_dataset_row(
        payload.dataset_table,
        payload.record_id,
        {FEEDBACK_SCORE_COLUMN: feedback_score},
    )
    return {
        "status": "ok",
        "dataset_table": payload.dataset_table,
        "record_id": payload.record_id,
        "feedback": feedback,
        "feedback_score": feedback_score,
    }

def _feature_values_payload(
    row: pd.Series,
    signals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        str(column): _safe_json(value)
        for column, value in row.items()
    }
    if signals:
        payload["__ml_explanation_signals"] = signals
    return payload

def _coerce_insert_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value

def _feedback_to_score(feedback: str) -> float:
    normalized = str(feedback or "").strip().lower()
    if normalized not in FEEDBACK_TO_SCORE:
        raise ValueError("Feedback must be accept, reject, or maybe.")
    return FEEDBACK_TO_SCORE[normalized]

def _score_to_feedback(value: Any) -> str | None:
    numeric = _safe_numeric_scalar(value, default=float("nan"))
    if pd.isna(numeric):
        return None
    for score, label in SCORE_TO_FEEDBACK.items():
        if abs(numeric - score) < 0.001:
            return label
    return None

def _build_dataset_frame(inputs: DatasetBuildInputs) -> pd.DataFrame:
    anomaly_mask = inputs.final_flag.astype(bool)
    filtered_feature_frame = inputs.feature_frame.loc[anomaly_mask]
    filtered_human_outlier_flag = inputs.human_outlier_flag.loc[anomaly_mask]
    filtered_ml_flag = inputs.ml_flag.loc[anomaly_mask]
    filtered_human_reasons = (
        inputs.human_reason_series.loc[anomaly_mask]
        if inputs.human_reason_series is not None
        else pd.Series(None, index=filtered_human_outlier_flag.index)
    )
    filtered_builtin_reasons = (
        inputs.builtin_reason_series.loc[anomaly_mask]
        if inputs.builtin_reason_series is not None
        else pd.Series(None, index=filtered_human_outlier_flag.index)
    )
    filtered_joined = (
        inputs.filtered_joined_override
        if inputs.filtered_joined_override is not None
        else inputs.joined.loc[anomaly_mask]
    )
    feature_aliases = set(_feature_rule_aliases(inputs.payload.feature_rules))

    logger.info(
        "Joined has %s rows; saving %s anomaly rows to %s.%s",
        len(inputs.joined),
        len(filtered_feature_frame),
        RESULT_SCHEMA,
        inputs.dataset_table,
    )

    default_rule_reason = inputs.human_reasons or ["Human-defined outlier rule matched"]
    explanation_signals = inputs.explanation_signals_override or {}

    dataset = pd.DataFrame(index=filtered_feature_frame.index)
    dataset.insert(0, FEATURE_NAME_COLUMN, _feature_name_for_tables(inputs.payload.selected_tables))
    dataset[FEATURE_VALUES_COLUMN] = [
        _feature_values_payload(
            row,
            explanation_signals.get(row_index),
        )
        for row_index, row in filtered_feature_frame.iterrows()
    ]
    dataset[HUMAN_RULE_NAME_COLUMN] = [
        _presentable_reason_text(reason) if bool(flag) and _presentable_reason_text(reason)
        else _presentable_reason_text(builtin_reason) if bool(flag) and _presentable_reason_text(builtin_reason)
        else ", ".join(default_rule_reason) if bool(flag)
        else None
        for flag, reason, builtin_reason in zip(
            filtered_human_outlier_flag,
            filtered_human_reasons,
            filtered_builtin_reasons,
        )
    ]
    dataset[HUMAN_RULE_COLUMN] = [bool(flag) for flag in filtered_human_outlier_flag]
    dataset[ISOLATION_RULE_COLUMN] = [bool(flag) for flag in filtered_ml_flag]
    dataset[IF_SCORE_COLUMN] = [
        _round_storage_score(value)
        for value in np.asarray(inputs.isolation_scores)[anomaly_mask.to_numpy()]
    ]
    dataset[ML_THRESHOLD_COLUMN] = _round_storage_score(inputs.ml_threshold)
    dataset[RUN_ID_COLUMN] = int(inputs.dataset_run_id)
    dataset[REVIEW_PAYLOAD_COLUMN] = [
        _review_payload_for_row(row, feature_aliases)
        for _, row in filtered_joined.iterrows()
    ]

    ordered_columns = (
        [FEATURE_NAME_COLUMN]
        + [
            FEATURE_VALUES_COLUMN,
            HUMAN_RULE_NAME_COLUMN,
            HUMAN_RULE_COLUMN,
            ISOLATION_RULE_COLUMN,
            IF_SCORE_COLUMN,
            ML_THRESHOLD_COLUMN,
            RUN_ID_COLUMN,
            REVIEW_PAYLOAD_COLUMN,
        ]
    )
    dataset = dataset.loc[:, ordered_columns]
    logger.info("Final ML_Features dataset has %s columns", len(dataset.columns))
    return dataset.replace({np.nan: None})

def _write_dataset_to_result(df: pd.DataFrame, dataset_table: str) -> dict[str, Any]:
    df = _normalize_storage_columns(df)
    engine = _source_engine()
    logger.info("Writing %s rows to %s.%s", len(df), RESULT_SCHEMA, dataset_table)
    if df.empty:
        total_rows = _previous_dataset_row_count(dataset_table)
        return {
            "schema": RESULT_SCHEMA,
            "table_name": dataset_table,
            "row_count": total_rows,
            "appended_row_count": 0,
            "column_count": int(len(df.columns)),
        }

    if not _result_table_exists(dataset_table):
        raise ValueError(
            f"Target table {RESULT_SCHEMA}.{dataset_table} does not exist. "
            "Use the existing PostgreSQL table before running the workbench."
        )

    available_columns = _result_table_columns(dataset_table)
    missing_columns = [str(column) for column in df.columns if str(column) not in available_columns]
    if missing_columns:
        raise ValueError(
            f"Target table {RESULT_SCHEMA}.{dataset_table} is missing required columns: {missing_columns}"
        )

    metadata = MetaData()
    result_table = Table(dataset_table, metadata, schema=RESULT_SCHEMA, autoload_with=engine)
    records = [
        {str(column): _coerce_insert_value(value) for column, value in row.items()}
        for row in df.to_dict(orient="records")
    ]
    inserted_ids: list[int] = []
    with engine.begin() as conn:
        result = conn.execute(
            result_table.insert().returning(result_table.c[SERIAL_COLUMN]),
            records,
        )
        inserted_ids = [int(row[0]) for row in result]
    logger.info("Successfully wrote to %s.%s", RESULT_SCHEMA, dataset_table)
    total_rows = _previous_dataset_row_count(dataset_table)

    # Clear the caches after the table has been written to
    _clear_result_table_columns_cache(dataset_table)
    _clear_result_table_exists_cache(dataset_table)

    return {
        "schema": RESULT_SCHEMA,
        "table_name": dataset_table,
        "row_count": int(total_rows or 0),
        "appended_row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "inserted_ids": inserted_ids,
    }

def _update_dataset_row(dataset_table: str, record_id: int, assignments: dict[str, Any]) -> None:
    if not _result_table_exists(dataset_table):
        raise ValueError(f"Dataset table {dataset_table} does not exist in {RESULT_SCHEMA}.")
    set_parts = [f"{_quote(column)} = :{column}" for column in assignments]
    params = {**assignments, "record_id": int(record_id)}
    sql = text(
        f"UPDATE {_result_table_ref(dataset_table)} "
        f"SET {', '.join(set_parts)} "
        f"WHERE {_quote(SERIAL_COLUMN)} = :record_id"
    )
    with _source_begin() as conn:
        result = conn.execute(sql, params)
        if result.rowcount == 0:
            raise ValueError(f"Record {record_id} was not found in dataset {dataset_table}.")
