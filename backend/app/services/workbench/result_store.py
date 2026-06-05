"""Transform scored anomalies into persisted dataset rows and feedback updates."""
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.orm import Session

from app.core.cache import invalidate_all_caches
from app.core.models import WorkbenchRun
from app.schemas.workbench_schema import WorkbenchRunRequest
from app.services.workbench.constants import (
    FEATURE_VALUES_COLUMN,
    FEEDBACK_SCORE_COLUMN,
    FEEDBACK_TO_SCORE,
    FK_DAK_COLUMN,
    IF_SCORE_COLUMN,
    ISOLATION_RULE_COLUMN,
    ML_THRESHOLD_COLUMN,
    RESULT_SCHEMA,
    RUN_ID_COLUMN,
    SCORE_TO_FEEDBACK,
    SELECTED_TABLES_COLUMN,
    SERIAL_COLUMN,
    SYSTEM_COLUMNS,
    USER_RULE_COLUMN,
    USER_RULE_NAME_COLUMN,
    logger,
)
from app.services.workbench.source_db import (
    _clear_result_table_columns_cache,
    _clear_result_table_exists_cache,
    _index_name,
    _normalize_storage_columns,
    _previous_dataset_row_count,
    _quote,
    _result_table_columns,
    _result_table_exists,
    _result_table_ref,
    _source_begin,
    _source_engine,
)
from app.services.workbench.utils import (
    _roundoff_score,
    _safe_json,
    _safe_numeric_scalar,
)


def _feature_name_for_tables(selected_tables: list[str]) -> str:
    """Collapse the selected source tables into the canonical dataset label."""
    return ".".join(str(table) for table in selected_tables if str(table).strip())


def _presentable_reason_text(value: Any) -> str | None:
    """Normalize internal rule or outlier reason text into a UI-friendly label."""
    if value is None or pd.isna(value):
        return None

    text_value = str(value).strip()

    if not text_value:
        return None

    return re.sub(
        r"\s+",
        " ",
        text_value.replace("RULE::", "").replace("OUTLIER::", "").replace("_", " "),
    ).strip()


def _review_payload_for_row(
    row: pd.Series,
    feature_aliases: set[str],
    selected_tables: list[str],
) -> dict[str, Any]:
    """Extract business columns for one anomaly row into the review payload artifact."""
    payload: dict[str, Any] = {}

    for column, value in row.items():
        column_name = str(column)

        if column_name in SYSTEM_COLUMNS or column_name in feature_aliases:
            continue

        payload[_canonical_payload_column_name(column_name, selected_tables)] = _safe_json(value)

    return payload

def _canonical_payload_column_name(column_name: str, selected_tables: list[str]) -> str:
    """Rewrite stored column aliases back into `table.column` style payload keys."""
    if "." in column_name:
        return column_name
    for table_name in sorted((str(table) for table in selected_tables), key=len, reverse=True):
        prefix = f"{table_name}_"
        if column_name.startswith(prefix) and len(column_name) > len(prefix):
            return f"{table_name}.{column_name[len(prefix):]}"
    return column_name


def _result_fk_dak_for_row(
    row: pd.Series,
    selected_tables: list[str],
) -> int | None:
    """Resolve the best available `fk_dak` identifier for one persisted anomaly row."""
    if "dak" in selected_tables:
        numeric = _safe_numeric_scalar(row.get("dak.id"), default=None)
        return int(numeric) if numeric is not None else None

    for table_name in selected_tables:
        numeric = _safe_numeric_scalar(row.get(f"{table_name}.fk_dak"), default=None)
        if numeric is not None:
            return int(numeric)

    numeric = _safe_numeric_scalar(row.get("fk_dak"), default=None)
    return int(numeric) if numeric is not None else None


def _review_payload_cache_entries(
    rows: pd.DataFrame,
    feature_aliases: set[str],
    inserted_ids: list[int],
    selected_tables: list[str],
) -> dict[str, dict[str, Any]]:
    """Build the run-level review payload artifact keyed by inserted dataset record id."""
    payloads = [
        _review_payload_for_row(row, feature_aliases, selected_tables)
        for _, row in rows.iterrows()
    ]
    return {
        str(int(record_id)): payload
        for record_id, payload in zip(inserted_ids, payloads)
    }


@dataclass(frozen=True)
class DatasetBuildInputs:
    """Inputs required to turn scored model outputs into a persisted anomaly dataset frame."""
    joined: pd.DataFrame
    feature_frame: pd.DataFrame
    payload: WorkbenchRunRequest
    dataset_table: str
    dataset_run_id: int
    combined_rule_flag: pd.Series
    user_reasons: list[str]
    user_reason_series: pd.Series | None
    default_reason_series: pd.Series | None
    default_evidence_series: pd.Series | None
    isolation_scores: np.ndarray
    ml_flag: pd.Series
    ml_threshold: float
    final_flag: pd.Series
    filtered_joined_override: pd.DataFrame | None = None
    explanation_signals_override: dict[Any, list[dict[str, Any]]] | None = None


def list_saved_datasets(db: Session) -> list[dict[str, Any]]:
    """Return the latest persisted run for each dataset table exposed to the UI."""
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
    """Persist reviewer feedback for one anomaly row and invalidate downstream caches."""
    del db

    feedback = str(payload.feedback).strip().lower()
    feedback_score = _feedback_to_score(feedback)

    _update_dataset_row(
        payload.dataset_table,
        int(payload.record_id),
        {FEEDBACK_SCORE_COLUMN: feedback_score},
    )
    invalidate_all_caches()
    from app.services.dashboard_service import invalidate_dashboard_caches
    invalidate_dashboard_caches(payload.dataset_table)

    return {
        "status": "ok",
        "dataset_table": payload.dataset_table,
        "record_id": payload.record_id,
        "feedback": feedback,
        "feedback_score": feedback_score,
    }


def _feature_values_payload(
    signals: list[dict[str, Any]] | None = None,
    rule_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pack ML explanation signals and rule evidence into one persisted JSON payload."""
    return {
        "__ml_explanation_signals": signals or [],
        "__rule_evidence": rule_evidence or [],
    }


def _rule_evidence_payload(value: Any) -> list[dict[str, Any]]:
    """Parse newline-delimited rule evidence JSON into a normalized list."""
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    evidence_items: list[dict[str, Any]] = []
    for line in str(value).splitlines():
        text_value = line.strip()
        if not text_value:
            continue
        try:
            parsed = json.loads(text_value)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        similar_fk_daks = parsed.get("similar_fk_daks")
        if isinstance(similar_fk_daks, str):
            parsed["similar_fk_daks"] = [
                item.strip()
                for item in similar_fk_daks.split(",")
                if item.strip()
            ]
        evidence_items.append(parsed)
    return evidence_items


def _coerce_insert_value(value: Any) -> Any:
    """Convert NumPy scalars into plain Python values before SQLAlchemy inserts them."""
    if isinstance(value, np.generic):
        return value.item()

    return value


def _feedback_to_score(feedback: str) -> float:
    """Translate the review label into the numeric score stored in PostgreSQL."""
    normalized = str(feedback or "").strip().lower()
    normalized_scores = {
        str(label).strip().lower(): float(score)
        for label, score in FEEDBACK_TO_SCORE.items()
    }

    if normalized not in normalized_scores:
        raise ValueError("Feedback must be Accept, Reject, or Maybe.")

    return float(normalized_scores[normalized])


def _score_to_feedback(value: Any) -> str | None:
    """Translate a stored numeric feedback score back into the UI label."""
    numeric = _safe_numeric_scalar(value, default=float("nan"))

    if numeric is None or pd.isna(numeric):
        return None

    for score, label in SCORE_TO_FEEDBACK.items():
        if abs(float(numeric) - float(score)) < 0.001:
            return label

    return None


def _aligned_series(
    series: pd.Series | None,
    index: pd.Index,
    default: Any = None,
) -> pd.Series:
    """Reindex an optional series onto the feature frame index with a default fill value."""
    if series is None:
        return pd.Series(default, index=index)

    return series.reindex(index, fill_value=default)


def _build_dataset_frame(inputs: DatasetBuildInputs) -> pd.DataFrame:
    """Build the persisted anomaly-row DataFrame from the scoring and rule outputs."""
    base_index = inputs.feature_frame.index

    anomaly_mask = inputs.final_flag.reindex(base_index, fill_value=False).astype(bool)

    filtered_feature_frame = inputs.feature_frame.loc[anomaly_mask]
    anomaly_index = filtered_feature_frame.index

    filtered_combined_rule_flag = _aligned_series(
        inputs.combined_rule_flag,
        base_index,
        False,
    ).loc[anomaly_index].astype(bool)

    filtered_ml_flag = _aligned_series(
        inputs.ml_flag,
        base_index,
        False,
    ).loc[anomaly_index].astype(bool)

    filtered_user_reasons = _aligned_series(
        inputs.user_reason_series,
        base_index,
        None,
    ).loc[anomaly_index]

    filtered_default_reasons = _aligned_series(
        inputs.default_reason_series,
        base_index,
        None,
    ).loc[anomaly_index]

    filtered_default_evidence = _aligned_series(
        inputs.default_evidence_series,
        base_index,
        None,
    ).loc[anomaly_index]

    if inputs.filtered_joined_override is not None:
        filtered_joined = inputs.filtered_joined_override.reindex(anomaly_index)
    else:
        filtered_joined = inputs.joined.reindex(anomaly_index)

    logger.info(
        "Joined has %s rows; saving %s anomaly rows to %s.%s",
        len(inputs.joined),
        len(filtered_feature_frame),
        RESULT_SCHEMA,
        inputs.dataset_table,
    )

    default_rule_reason = inputs.user_reasons or ["User-defined rule matched"]
    explanation_signals = inputs.explanation_signals_override or {}

    dataset = pd.DataFrame(index=anomaly_index)

    dataset.insert(
        0,
        SELECTED_TABLES_COLUMN,
        _feature_name_for_tables(inputs.payload.selected_tables),
    )

    dataset[FEATURE_VALUES_COLUMN] = [
        _feature_values_payload(
            explanation_signals.get(row_index),
            _rule_evidence_payload(evidence),
        )
        for row_index, evidence in zip(
            filtered_feature_frame.index,
            filtered_default_evidence,
        )
    ]

    user_rule_names: list[str | None] = []

    for flag, reason, default_reason in zip(
        filtered_combined_rule_flag,
        filtered_user_reasons,
        filtered_default_reasons,
    ):
        clean_reason = _presentable_reason_text(reason)
        clean_default_reason = _presentable_reason_text(default_reason)

        if bool(flag) and clean_reason:
            user_rule_names.append(clean_reason)
        elif bool(flag) and clean_default_reason:
            user_rule_names.append(clean_default_reason)
        elif bool(flag):
            user_rule_names.append(", ".join(default_rule_reason))
        else:
            user_rule_names.append(None)

    dataset[USER_RULE_NAME_COLUMN] = user_rule_names
    dataset[USER_RULE_COLUMN] = [bool(flag) for flag in filtered_combined_rule_flag]
    dataset[ISOLATION_RULE_COLUMN] = [bool(flag) for flag in filtered_ml_flag]

    isolation_scores = np.asarray(inputs.isolation_scores)

    if len(isolation_scores) != len(base_index):
        raise ValueError(
            "isolation_scores length must match feature_frame length. "
            f"Got isolation_scores={len(isolation_scores)}, feature_frame={len(base_index)}."
        )

    dataset[IF_SCORE_COLUMN] = [
        _roundoff_score(value)
        for value in isolation_scores[anomaly_mask.to_numpy()]
    ]

    dataset[ML_THRESHOLD_COLUMN] = _roundoff_score(inputs.ml_threshold)
    dataset[RUN_ID_COLUMN] = int(inputs.dataset_run_id)

    dataset[FK_DAK_COLUMN] = [
        _result_fk_dak_for_row(row, inputs.payload.selected_tables)
        for _, row in filtered_joined.iterrows()
    ]

    ordered_columns = [
        SELECTED_TABLES_COLUMN,
        FEATURE_VALUES_COLUMN,
        USER_RULE_NAME_COLUMN,
        USER_RULE_COLUMN,
        ISOLATION_RULE_COLUMN,
        IF_SCORE_COLUMN,
        ML_THRESHOLD_COLUMN,
        RUN_ID_COLUMN,
        FK_DAK_COLUMN,
    ]

    dataset = dataset.loc[:, ordered_columns]

    logger.info("Final ML_Features dataset has %s columns", len(dataset.columns))

    return dataset.replace({np.nan: None})


def _write_dataset_to_result(
    df: pd.DataFrame,
    dataset_table: str,
) -> dict[str, Any]:
    """Append anomaly rows into the existing PostgreSQL result table."""
    df = _normalize_storage_columns(df)
    engine = _source_engine()

    logger.info("Writing %s rows to %s.%s", len(df), RESULT_SCHEMA, dataset_table)

    if df.empty:
        total_rows = _previous_dataset_row_count(dataset_table)

        return {
            "schema": RESULT_SCHEMA,
            "table_name": dataset_table,
            "row_count": int(total_rows or 0),
            "appended_row_count": 0,
            "column_count": int(len(df.columns)),
            "inserted_ids": [],
        }

    if not _result_table_exists(dataset_table):
        raise ValueError(
            f"Target table {RESULT_SCHEMA}.{dataset_table} does not exist. "
            "Use the existing PostgreSQL table before running the workbench."
        )

    available_columns = _result_table_columns(dataset_table)

    missing_columns = [
        str(column)
        for column in df.columns
        if str(column) not in available_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Target table {RESULT_SCHEMA}.{dataset_table} is missing required columns: "
            f"{missing_columns}"
        )

    metadata = MetaData()

    result_table = Table(
        dataset_table,
        metadata,
        schema=RESULT_SCHEMA,
        autoload_with=engine,
    )

    if SERIAL_COLUMN not in result_table.c:
        raise ValueError(
            f"Target table {RESULT_SCHEMA}.{dataset_table} is missing serial column "
            f"{SERIAL_COLUMN!r}."
        )

    records = [
        {
            str(column): _coerce_insert_value(value)
            for column, value in row.items()
        }
        for row in df.to_dict(orient="records")
    ]

    inserted_ids: list[int] = []

    with engine.begin() as conn:
        _normalize_result_table_feature_names(conn, dataset_table)
        _ensure_result_table_indexes(conn, dataset_table, available_columns)
        result = conn.execute(
            result_table.insert().returning(result_table.c[SERIAL_COLUMN]),
            records,
        )
        inserted_ids = [int(row[0]) for row in result]

    _clear_result_table_columns_cache(dataset_table)
    _clear_result_table_exists_cache(dataset_table)
    invalidate_all_caches()
    from app.services.dashboard_service import invalidate_dashboard_caches
    invalidate_dashboard_caches(dataset_table)

    total_rows = _previous_dataset_row_count(dataset_table)

    logger.info("Successfully wrote to %s.%s", RESULT_SCHEMA, dataset_table)

    return {
        "schema": RESULT_SCHEMA,
        "table_name": dataset_table,
        "row_count": int(total_rows or 0),
        "appended_row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "inserted_ids": inserted_ids,
    }


def _ensure_result_table_indexes(
    conn,
    dataset_table: str,
    available_columns: frozenset[str] | set[str],
) -> None:
    """Create indexes used by dashboard/review queries on the persisted result table."""
    indexed_columns = {
        SERIAL_COLUMN,
        RUN_ID_COLUMN,
        SELECTED_TABLES_COLUMN,
        FEEDBACK_SCORE_COLUMN,
        FK_DAK_COLUMN,
    }
    for column_name in sorted(indexed_columns.intersection(available_columns)):
        index_name = _index_name(RESULT_SCHEMA, dataset_table, column_name)
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {_quote(index_name)} "
                f"ON {_result_table_ref(dataset_table)} ({_quote(column_name)})"
            )
        )

    flag_columns = {
        USER_RULE_COLUMN,
        ISOLATION_RULE_COLUMN,
    }.intersection(available_columns)
    for column_name in sorted(flag_columns):
        index_name = _index_name(RESULT_SCHEMA, dataset_table, column_name, "truthy")
        column_ref = _quote(column_name)
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {_quote(index_name)} "
                f"ON {_result_table_ref(dataset_table)} "
                f"(LOWER(BTRIM({column_ref}::text)))"
            )
        )

    if RUN_ID_COLUMN in available_columns and SERIAL_COLUMN in available_columns:
        index_name = _index_name(RESULT_SCHEMA, dataset_table, RUN_ID_COLUMN, SERIAL_COLUMN)
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {_quote(index_name)} "
                f"ON {_result_table_ref(dataset_table)} "
                f"({_quote(RUN_ID_COLUMN)}, {_quote(SERIAL_COLUMN)} DESC)"
            )
        )


def _normalize_result_table_feature_names(conn, dataset_table: str) -> None:
    """Clean malformed selected-table labels left from earlier result-table writes."""
    if SELECTED_TABLES_COLUMN not in _result_table_columns(dataset_table):
        return
    selected_column = _quote(SELECTED_TABLES_COLUMN)
    conn.execute(
        text(
            f"""
            UPDATE {_result_table_ref(dataset_table)}
            SET {selected_column} = array_to_string(
                ARRAY(
                    SELECT part
                    FROM unnest(string_to_array({selected_column}, '.')) AS part
                    WHERE part <> '' AND lower(part) <> 'null'
                ),
                '.'
            )
            WHERE {selected_column} IS NOT NULL
              AND (
                  {selected_column} LIKE '%.null%'
                  OR {selected_column} LIKE 'null.%'
                  OR lower({selected_column}) = 'null'
              )
            """
        )
    )

def _update_dataset_row(dataset_table: str, record_id: int, assignments: dict[str, Any]) -> None:
    """Apply an in-place update to one persisted anomaly row in the result table."""
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
