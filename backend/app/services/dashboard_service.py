import json
import math
import logging
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.cache import DATASET_SUMMARY_CACHE, QUERY_RESULT_CACHE, TTLCache
from app.core.models import WorkbenchRun
from app.services.reason_service import build_deterministic_isolation_reason
from app.services.workbench.constants import (
    FEATURE_VALUES_COLUMN,
    FEEDBACK_SCORE_COLUMN,
    FK_DAK_COLUMN,
    IF_SCORE_COLUMN,
    ISOLATION_RULE_COLUMN,
    ML_FEATURES_TABLE,
    ML_THRESHOLD_COLUMN,
    RESULT_SCHEMA,
    REVIEW_PAYLOAD_COLUMN,
    RUN_ID_COLUMN,
    SELECTED_TABLES_COLUMN,
    SERIAL_COLUMN,
    SYSTEM_COLUMNS,
    USER_RULE_COLUMN,
    USER_RULE_NAME_COLUMN,
)
from app.services.workbench.result_store import _score_to_feedback
from app.services.workbench.source_db import (
    _quote,
    _result_table_columns,
    _result_table_exists,
    _result_table_ref,
    _source_connect,
    _source_columns_map,
    _source_table_ref,
)
from app.services.workbench.utils import (
    _safe_json,
    _safe_numeric_scalar,
)
from app.services.workbench.valkey_artifacts import get_review_payload_artifact

logger = logging.getLogger(__name__)
DEFAULT_REVIEW_PAGE_SIZE = 50
_latest_dataset_cache = TTLCache(ttl_seconds=5.0, namespace="latest_dataset")
_latest_run_id_cache = TTLCache(ttl_seconds=5.0, namespace="latest_run_id")

def _sql_truthy(column_name: str) -> str:
    column_ref = _quote(column_name)
    return f"LOWER(BTRIM({column_ref}::text)) IN ('true', 't', '1', 'yes', 'y')"


_ANOMALY_FILTER_CLAUSES: dict[str, str] = {
    "rule": f"WHERE {_sql_truthy(USER_RULE_COLUMN)}",
    "ml": f"WHERE {_sql_truthy(ISOLATION_RULE_COLUMN)}",
    "reviewed": f"WHERE {_quote(FEEDBACK_SCORE_COLUMN)} IS NOT NULL",
    "not_reviewed": f"WHERE {_quote(FEEDBACK_SCORE_COLUMN)} IS NULL",
    "all": (
        "WHERE ("
        f"{_sql_truthy(USER_RULE_COLUMN)} OR "
        f"{_sql_truthy(ISOLATION_RULE_COLUMN)}"
        ")"
    ),
}

def review_table_data(
    db: Session,
    dataset_table: str | None = None,
    *,
    anomaly_filter: str = "all",
    limit: int | None = None,
    offset: int = 0,
    run_id: int | None = None,
) -> dict[str, Any]:
    cache_key = (
        f"review_table:{dataset_table or 'latest'}:{anomaly_filter}:"
        f"{limit or DEFAULT_REVIEW_PAGE_SIZE}:{offset}:{run_id or 'latest'}"
    )
    cached = QUERY_RESULT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    selected_dataset = dataset_table or _latest_dataset_table(db)
    if not selected_dataset:
        return {
            "rows": [],
            "total_amount": 0.0,
            "total_rows": 0,
            "dataset_table": None,
        }

    selected_run_id = run_id if run_id is not None else _latest_run_id_for_dataset(db, selected_dataset)
    selected_ml_run_id = _ml_run_id_for_app_run(db, selected_run_id)
    summary = _dataset_summary(
        db,
        selected_dataset,
        anomaly_filter=anomaly_filter,
        run_id=selected_ml_run_id,
        app_run_id=selected_run_id,
    )
    review_data = review_rows_data(
        db,
        dataset_table=selected_dataset,
        anomaly_filter=anomaly_filter,
        limit=limit,
        offset=offset,
        run_id=selected_run_id,
    )
    rows = []
    running_total = 0.0
    for index, item in enumerate(review_data["rows"], start=1):
        payload = item.get("row_payload_json") or {}
        amount = _payload_amount(payload)
        running_total += amount
        reasons = item.get("reasons_json") or {}
        risk_score = _safe_numeric_scalar(item.get("ml_score"), default=0.0) * 100.0
        detected_on = _payload_text(
            payload,
            [
                "reference_date",
                "bill_date",
                "invoice_date",
                "list_date",
                "created_at",
                "cheque_slip_date",
                "dp_sheet_date",
            ],
        )
        vendor_name = _payload_text(
            payload,
            ["resolved_vendor_name", "vendor_name", "sender_name", "beneficiary_name"],
        )
        office_name = _payload_text(
            payload,
            ["resolved_office_name", "office_name", "resolved_section_name", "section_name"],
        )
        bill_no = _payload_text(
            payload,
            ["bill_no", "reference_no", "invoice_no", "invoice_number", "payment_detail"],
            default=f"ID {item['prediction_id']}",
        )
        anomaly_type = ", ".join(reasons.get("reason_list", [])) or item.get("rule_codes") or "No reason"
        rows.append(
            {
                "serial_no": index,
                "prediction_id": item["prediction_id"],
                "anomaly": "Yes" if item.get("final_label") else "No",
                "reason": anomaly_type,
                "amount": amount,
                "total_amount": running_total,
                "review_status": item.get("review_status"),
                "feedback": item.get("feedback"),
                "bill_no": bill_no,
                "vendor_name": vendor_name or "Unknown Vendor",
                "office_name": office_name or "Unknown Office",
                "detected_on": detected_on,
                "anomaly_type": anomaly_type,
                "risk_score": round(risk_score, 2),
            }
        )
    result = {
        "rows": rows,
        "total_amount": float(summary["total_amount"]),
        "total_rows": int(summary["total_rows"]),
        "dataset_table": selected_dataset,
        "run_id": selected_run_id,
        "pagination": review_data.get("pagination", {}),
    }
    QUERY_RESULT_CACHE.set(cache_key, result)
    return result

def review_rows_data(
    db: Session,
    dataset_table: str | None = None,
    *,
    anomaly_filter: str = "all",
    limit: int | None = None,
    offset: int = 0,
    run_id: int | None = None,
) -> dict[str, Any]:
    cache_key = (
        f"review_rows:{dataset_table or 'latest'}:{anomaly_filter}:"
        f"{limit or DEFAULT_REVIEW_PAGE_SIZE}:{offset}:{run_id or 'latest'}"
    )
    cached = QUERY_RESULT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    selected_dataset = dataset_table or _latest_dataset_table(db)
    if not selected_dataset:
        return {
            "dataset_table": None,
            "rows": [],
            "summary": {"total_rows": 0, "total_amount": 0.0},
            "pagination": {"limit": 0, "offset": 0, "page_count": 0},
        }

    selected_run_id = run_id if run_id is not None else _latest_run_id_for_dataset(db, selected_dataset)
    selected_ml_run_id = _ml_run_id_for_app_run(db, selected_run_id)
    page_limit = max(1, int(limit or DEFAULT_REVIEW_PAGE_SIZE))
    page_offset = max(0, int(offset or 0))
    builtin_reason_by_record_id: dict[str, str] = {}
    run: WorkbenchRun | None = None
    if selected_run_id is not None:
        run = db.query(WorkbenchRun).filter(WorkbenchRun.run_id == selected_run_id).first()
        if run and isinstance(run.metrics_json, dict):
            raw_map = run.metrics_json.get("builtin_reason_by_record_id") or {}
            if isinstance(raw_map, dict):
                builtin_reason_by_record_id = {str(key): str(value) for key, value in raw_map.items() if value is not None}

    summary = _dataset_summary(
        db,
        selected_dataset,
        anomaly_filter=anomaly_filter,
        run_id=selected_ml_run_id,
        app_run_id=selected_run_id,
    )
    df = _dataset_frame(
        selected_dataset,
        anomaly_filter=anomaly_filter,
        limit=page_limit,
        offset=page_offset,
        run_id=selected_ml_run_id,
    )
    rows = [_dataset_row_to_prediction(row, builtin_reason_by_record_id) for _, row in df.iterrows()]
    for row in rows:
        row["dataset_table"] = selected_dataset
    rows = _rehydrate_prediction_payloads_with_run(run if selected_run_id is not None else None, rows)
    rows = _enrich_payload_rows(rows)
    total_amount = sum(_payload_amount(row["row_payload_json"]) for row in rows)
    logger.info(
        "Loaded review rows page for %s: offset=%s limit=%s page_count=%s total_rows=%s",
        selected_dataset,
        page_offset,
        page_limit,
        len(rows),
        summary["total_rows"],
    )
    result = {
        "dataset_table": selected_dataset,
        "run_id": selected_run_id,
        "rows": rows,
        "summary": {
            "total_rows": int(summary["total_rows"]),
            "total_amount": float(summary["total_amount"]),
            "anomaly_filter": anomaly_filter,
            "page_total_amount": float(total_amount),
        },
        "pagination": {
            "limit": page_limit,
            "offset": page_offset,
            "page_count": int(len(rows)),
        },
    }
    QUERY_RESULT_CACHE.set(cache_key, result)
    return result

def report_data(
    db: Session,
    dataset_table: str | None = None,
    *,
    run_id: int | None = None,
) -> dict[str, Any]:
    cache_key = f"report:{dataset_table or 'latest'}:{run_id or 'latest'}"
    cached = QUERY_RESULT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    selected_run: WorkbenchRun | None = None

    if run_id is not None:
        selected_run = db.query(WorkbenchRun).filter(WorkbenchRun.run_id == run_id).first()

    if selected_run is None and dataset_table:
        selected_run_id = _latest_run_id_for_dataset(db, dataset_table)
        if selected_run_id is not None:
            selected_run = db.query(WorkbenchRun).filter(WorkbenchRun.run_id == selected_run_id).first()

    if selected_run is None:
        selected_run = db.query(WorkbenchRun).order_by(WorkbenchRun.run_id.desc()).first()

    if not selected_run:
        return {
            "run_id": None,
            "dataset_table": dataset_table,
            "run_name": None,
            "selected_tables": [],
            "total_rows": 0,
            "anomaly_count": 0,
            "reviewed_count": 0,
            "pending_count": 0,
            "accepted_count": 0,
            "amount": 0.0,
            "from_date": None,
            "to_date": None,
            "selected_model": None,
        }

    metrics = selected_run.metrics_json or {}
    selected_dataset = dataset_table or metrics.get("dataset_table")
    selected_run_id = int(selected_run.run_id)
    selected_ml_run_id = _ml_run_id_for_app_run(db, selected_run_id)
    summary = (
        _dataset_summary(
            db,
            selected_dataset,
            anomaly_filter="all",
            run_id=selected_ml_run_id,
            app_run_id=selected_run_id,
        )
        if selected_dataset
        else {"total_rows": 0, "total_amount": 0.0}
    )
    feedback_summary = (
        _feedback_summary(selected_dataset, selected_ml_run_id)
        if selected_dataset
        else {"reviewed_count": 0, "pending_count": 0, "accepted_count": 0}
    )
    anomaly_count = int(selected_run.final_anomaly_count or summary.get("total_rows") or 0)

    result = {
        "run_id": selected_run_id,
        "dataset_table": selected_dataset,
        "run_name": selected_run.run_name,
        "selected_tables": selected_run.source_tables_json or metrics.get("selected_tables") or [],
        "total_rows": int(selected_run.total_rows or 0),
        "anomaly_count": anomaly_count,
        "reviewed_count": int(feedback_summary.get("reviewed_count") or 0),
        "pending_count": int(feedback_summary.get("pending_count") or 0),
        "accepted_count": int(feedback_summary.get("accepted_count") or 0),
        "amount": float(summary.get("total_amount") or 0.0),
        "from_date": metrics.get("from_date"),
        "to_date": metrics.get("to_date"),
        "selected_model": selected_run.selected_model,
    }
    QUERY_RESULT_CACHE.set(cache_key, result)
    return result


def invalidate_dashboard_caches(dataset_table: str | None = None) -> None:
    QUERY_RESULT_CACHE.invalidate()
    DATASET_SUMMARY_CACHE.invalidate()
    _latest_dataset_cache.invalidate()
    if dataset_table:
        _latest_run_id_cache.invalidate(dataset_table)
    else:
        _latest_run_id_cache.invalidate()

def _latest_dataset_table(db: Session) -> str | None:
    cached = _latest_dataset_cache.get("value")
    if cached is not None:
        return cached
    for latest_run in _iter_recent_runs(db):
        metrics = latest_run.metrics_json or {}
        dataset_table = metrics.get("dataset_table")
        if dataset_table and _result_table_exists(dataset_table):
            _latest_dataset_cache.set("value", dataset_table)
            return dataset_table
    return None

def _latest_run_id_for_dataset(db: Session, dataset_table: str) -> int | None:
    cached = _latest_run_id_cache.get(dataset_table)
    if cached is not None:
        return int(cached) if cached is not None else None
    for run in _iter_recent_runs(db):
        metrics = run.metrics_json or {}
        if metrics.get("dataset_table") == dataset_table:
            value = int(run.run_id)
            _latest_run_id_cache.set(dataset_table, value)
            return value
    return None

def _ml_run_id_for_app_run(db: Session, app_run_id: int | None) -> int | None:
    if app_run_id is None:
        return None
    run = db.query(WorkbenchRun).filter(WorkbenchRun.run_id == int(app_run_id)).first()
    if not run or not isinstance(run.metrics_json, dict):
        return int(app_run_id)
    ml_run_id = run.metrics_json.get("ml_run_id")
    return int(ml_run_id) if ml_run_id is not None else int(app_run_id)

def _iter_recent_runs(db: Session, batch_size: int = 100):
    offset = 0
    while True:
        batch = (
            db.query(WorkbenchRun)
            .order_by(WorkbenchRun.run_id.desc())
            .limit(batch_size)
            .offset(offset)
            .all()
        )
        if not batch:
            break
        for run in batch:
            yield run
        if len(batch) < batch_size:
            break
        offset += batch_size

def _dataset_summary(
    db: Session,
    dataset_table: str,
    *,
    anomaly_filter: str = "all",
    run_id: int | None = None,
    app_run_id: int | None = None,
) -> dict[str, Any]:
    cache_key = (
        f"dataset_summary:{dataset_table}:{anomaly_filter}:"
        f"{run_id or 'all'}:{app_run_id or 'latest'}"
    )
    cached = DATASET_SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not _result_table_exists(dataset_table):
        return {"total_rows": 0, "total_amount": 0.0}

    if app_run_id is not None:
        run = db.query(WorkbenchRun).filter(WorkbenchRun.run_id == int(app_run_id)).first()
        payload_rows = _review_payload_rows_for_run(run)
        if payload_rows:
            df = _dataset_frame(
                dataset_table,
                anomaly_filter=anomaly_filter,
                run_id=run_id,
            )
            total_amount = 0.0
            for _, row in df.iterrows():
                record_id = row.get(SERIAL_COLUMN)
                numeric_record_id = _safe_numeric_scalar(record_id, default=None)
                if numeric_record_id is None:
                    continue
                payload = payload_rows.get(str(int(numeric_record_id)))
                if isinstance(payload, dict):
                    total_amount += _payload_amount(payload)
            result = {
                "total_rows": int(len(df.index)),
                "total_amount": float(total_amount),
            }
            DATASET_SUMMARY_CACHE.set(cache_key, result)
            return result

    normalized_summary_filter = (anomaly_filter or "all").strip().lower()
    where_clause = _add_run_filter(
        _dataset_query_filter(normalized_summary_filter),
        run_id,
        dataset_table=dataset_table,
    )
    sql = text(
        f"""
        SELECT
            COUNT(*) AS total_rows,
            COALESCE(SUM({_dashboard_amount_sql(dataset_table)}), 0.0) AS total_amount
        FROM {_result_table_ref(dataset_table)}
        {where_clause}
        """
    )
    try:
        with _source_connect() as conn:
            row = conn.execute(sql).mappings().first()
    except Exception as exc:
        logger.warning(
            "Dataset summary amount query failed for %s; falling back to count-only summary: %s",
            dataset_table,
            exc,
        )
        count_sql = text(
            f"""
            SELECT COUNT(*) AS total_rows
            FROM {_result_table_ref(dataset_table)}
            {where_clause}
            """
        )
        with _source_connect() as conn:
            count_row = conn.execute(count_sql).mappings().first()
        result = {
            "total_rows": int(count_row.get("total_rows") or 0) if count_row else 0,
            "total_amount": 0.0,
        }
        DATASET_SUMMARY_CACHE.set(cache_key, result)
        return result

    if not row:
        return {"total_rows": 0, "total_amount": 0.0}

    result = {
        "total_rows": int(row.get("total_rows") or 0),
        "total_amount": float(row.get("total_amount") or 0.0),
    }
    DATASET_SUMMARY_CACHE.set(cache_key, result)
    return result

def _feedback_summary(dataset_table: str, run_id: int | None = None) -> dict[str, int]:
    cache_key = f"feedback_summary:{dataset_table}:{run_id or 'all'}"
    cached = DATASET_SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not _result_table_exists(dataset_table):
        return {"reviewed_count": 0, "pending_count": 0, "accepted_count": 0}

    where_clause = _add_run_filter(
        "",
        run_id,
        dataset_table=dataset_table,
    )
    feedback_column = _quote(FEEDBACK_SCORE_COLUMN)
    numeric_feedback = (
        f"CASE "
        f"WHEN {feedback_column} IS NULL THEN NULL "
        f"WHEN btrim({feedback_column}::text) ~ '^[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)$' "
        f"THEN btrim({feedback_column}::text)::double precision "
        f"ELSE NULL "
        f"END"
    )
    sql = text(
        f"""
        SELECT
            COALESCE(SUM(CASE WHEN {feedback_column} IS NOT NULL THEN 1 ELSE 0 END), 0) AS reviewed_count,
            COALESCE(SUM(CASE WHEN {feedback_column} IS NULL THEN 1 ELSE 0 END), 0) AS pending_count,
            COALESCE(SUM(CASE WHEN COALESCE({numeric_feedback}, -1.0) = 1.0 THEN 1 ELSE 0 END), 0) AS accepted_count
        FROM {_result_table_ref(dataset_table)}
        {where_clause if where_clause else 'WHERE 1 = 1'}
        """
    )
    with _source_connect() as conn:
        row = conn.execute(sql).mappings().first()
    if not row:
        return {"reviewed_count": 0, "pending_count": 0, "accepted_count": 0}
    result = {
        "reviewed_count": int(row.get("reviewed_count") or 0),
        "pending_count": int(row.get("pending_count") or 0),
        "accepted_count": int(row.get("accepted_count") or 0),
    }
    DATASET_SUMMARY_CACHE.set(cache_key, result)
    return result

def _dataset_query_filter(anomaly_filter: str) -> str:
    normalized = (anomaly_filter or "all").strip().lower()
    return _ANOMALY_FILTER_CLAUSES.get(normalized, _ANOMALY_FILTER_CLAUSES["all"])

def _add_run_filter(
    where_clause: str,
    run_id: int | None,
    *,
    dataset_table: str,
) -> str:
    if run_id is None:
        return where_clause
    if RUN_ID_COLUMN not in _result_table_columns(dataset_table):
        return where_clause

    run_filter = f"WHERE {_quote(RUN_ID_COLUMN)} = {int(run_id)}"
    if not where_clause.strip():
        return run_filter
    return f"{where_clause}\nAND {_quote(RUN_ID_COLUMN)} = {int(run_id)}"

def _dataset_frame(
    dataset_table: str,
    *,
    anomaly_filter: str = "all",
    limit: int | None = None,
    offset: int | None = None,
    run_id: int | None = None,
) -> pd.DataFrame:
    if not _result_table_exists(dataset_table):
        return pd.DataFrame()
    available_columns = _result_table_columns(dataset_table)
    where_clause = _add_run_filter(
        _dataset_query_filter(anomaly_filter),
        run_id,
        dataset_table=dataset_table,
    )
    selected_columns = "*"
    compact_columns = [
        column_name
        for column_name in (
            SERIAL_COLUMN,
            SELECTED_TABLES_COLUMN,
            USER_RULE_NAME_COLUMN,
            USER_RULE_COLUMN,
            ISOLATION_RULE_COLUMN,
            IF_SCORE_COLUMN,
            ML_THRESHOLD_COLUMN,
            FEEDBACK_SCORE_COLUMN,
            RUN_ID_COLUMN,
            FK_DAK_COLUMN,
            FEATURE_VALUES_COLUMN,
        )
        if column_name in available_columns
    ]
    if compact_columns:
        selected_columns = ", ".join(_quote(column_name) for column_name in compact_columns)
    sql = (
        f"SELECT {selected_columns} FROM {_result_table_ref(dataset_table)} "
        f"{where_clause} "
        f"ORDER BY {_quote(SERIAL_COLUMN)} ASC"
    )
    if limit and limit > 0:
        sql += f" LIMIT {int(limit)}"
    if offset and offset > 0:
        sql += f" OFFSET {int(offset)}"
    with _source_connect() as conn:
        return pd.read_sql_query(text(sql), conn)

def _dataset_row_to_prediction(
    row: pd.Series,
    builtin_reason_by_record_id: dict[str, str] | None = None,
) -> dict[str, Any]:
    rule_reason = row.get(USER_RULE_NAME_COLUMN)
    reason_list = [rule_reason] if rule_reason else []
    payload = _dataset_payload(row)
    feature_payload = _parse_json_text(row.get(FEATURE_VALUES_COLUMN), {})
    ml_feature_signals = []
    if isinstance(feature_payload, dict):
        raw_signals = feature_payload.get("__ml_explanation_signals")
        if isinstance(raw_signals, list):
            ml_feature_signals = [item for item in raw_signals if isinstance(item, dict)]
    if_reason = None
    if_reason_model = None
    if_reason_fallback = False
    payload["review_key"] = row.get(SELECTED_TABLES_COLUMN)
    user_rule = _safe_bool(row.get(USER_RULE_COLUMN))
    isolation_rule = _safe_bool(row.get(ISOLATION_RULE_COLUMN))
    if not isolation_rule:
        ml_feature_signals = []
        if_reason = None
        if_reason_model = None
        if_reason_fallback = False
    if_score = _safe_json(row.get(IF_SCORE_COLUMN))
    ml_threshold = _safe_json(row.get(ML_THRESHOLD_COLUMN))
    feedback = _score_to_feedback(row.get(FEEDBACK_SCORE_COLUMN))
    record_id = row.get(SERIAL_COLUMN)
    if record_id is None:
        record_id = row.get("id")
    if record_id is None:
        record_id = row.get("s.no")
    numeric_record_id = _safe_numeric_scalar(record_id, default=None)
    if numeric_record_id is None:
        raise ValueError(f"Review row is missing a valid record id: {record_id!r}")
    record_id_int = int(numeric_record_id)
    builtin_reason = None
    if builtin_reason_by_record_id:
        builtin_reason = builtin_reason_by_record_id.get(str(record_id_int))
        if builtin_reason and builtin_reason not in reason_list:
            reason_list.append(builtin_reason)
    ml_reason = (
        build_deterministic_isolation_reason(ml_feature_signals, payload)
        if isolation_rule and ml_feature_signals
        else None
    )
    if ml_reason and ml_reason not in reason_list:
        reason_list.append(ml_reason)
    return {
        "prediction_id": record_id_int,
        "source_record_id": record_id_int,
        "dataset_table": ML_FEATURES_TABLE,
        "batch_id": None,
        "rule_flag": user_rule,
        "rule_codes": rule_reason,
        "rule_score": 1.0 if user_rule else 0.0,
        "ml_score": if_score,
        "raw_ml_score": if_score,
        "ml_threshold": ml_threshold,
        "final_score": if_score if if_score is not None else 1.0,
        "final_label": 1,
        "review_status": "REVIEWED" if feedback else "PENDING_REVIEW",
        "feedback": feedback,
        "reasons_json": {
            "user_rule_flag": user_rule,
            "default_rule_flag": bool(builtin_reason),
            "ml_anomaly_flag": isolation_rule,
            "reason_list": reason_list,
            "if_score": if_score,
            "ml_threshold": ml_threshold,
            "ml_feature_signals": ml_feature_signals,
            "if_reason": if_reason,
            "if_reason_model": if_reason_model,
            "if_reason_fallback": if_reason_fallback,
            "feedback_score": _safe_json(row.get(FEEDBACK_SCORE_COLUMN)),
        },
        "row_payload_json": payload,
    }


def _safe_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    return bool(value)


def _review_payload_rows_for_run(run: WorkbenchRun | None) -> dict[str, dict[str, Any]]:
    if run is None:
        return {}
    artifact = get_review_payload_artifact(int(run.run_id))
    if not isinstance(artifact, dict):
        return {}
    raw_rows = artifact.get("rows") or {}
    if not isinstance(raw_rows, dict):
        return {}
    return {
        str(key): dict(value)
        for key, value in raw_rows.items()
        if isinstance(value, dict)
    }


def _rehydrate_prediction_payloads(
    app_run_id: int | None,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if app_run_id is None or not rows:
        return rows

    artifact = get_review_payload_artifact(int(app_run_id))
    if not isinstance(artifact, dict):
        return rows

    raw_rows = artifact.get("rows") or {}
    if not isinstance(raw_rows, dict):
        return rows

    for row in rows:
        record_id = row.get("prediction_id")
        if record_id is None:
            continue
        payload = raw_rows.get(str(int(record_id)))
        if isinstance(payload, dict):
            row["row_payload_json"] = payload
    return rows


def _rehydrate_prediction_payloads_with_run(
    run: WorkbenchRun | None,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = _rehydrate_prediction_payloads(int(run.run_id), rows) if run is not None else rows
    if run is None or not rows:
        return rows

    selected_tables = list(run.source_tables_json or [])
    if not selected_tables:
        metrics = run.metrics_json or {}
        selected_tables = list(metrics.get("selected_tables") or [])

    for row in rows:
        payload = row.get("row_payload_json") or {}
        if _payload_has_business_data(payload):
            continue
        fk_dak_value = _safe_numeric_scalar(payload.get(FK_DAK_COLUMN), default=None)
        if fk_dak_value is None:
            continue
        rebuilt_payload = _source_payload_for_fk_dak(int(fk_dak_value), selected_tables)
        if rebuilt_payload:
            rebuilt_payload["review_key"] = payload.get("review_key")
            row["row_payload_json"] = rebuilt_payload
    return rows


def _payload_has_business_data(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(
        key not in {FK_DAK_COLUMN, "review_key"}
        for key in payload.keys()
    )


def _source_payload_for_fk_dak(
    fk_dak: int,
    selected_tables: list[str],
) -> dict[str, Any]:
    if fk_dak <= 0 or not selected_tables:
        return {}

    payload: dict[str, Any] = {}

    try:
        source_columns = _source_columns_map(selected_tables)
        with _source_connect() as conn:
            for table_name in selected_tables:
                table_columns = source_columns.get(table_name) or []
                if not table_columns:
                    continue

                available_columns = {str(item.get("column_name")) for item in table_columns}
                if table_name == "dak" and "id" in available_columns:
                    where_column = "id"
                elif "fk_dak" in available_columns:
                    where_column = "fk_dak"
                else:
                    continue

                order_by = ' ORDER BY "id" ASC' if "id" in available_columns else ""
                sql = text(
                    f"SELECT * FROM {_source_table_ref(table_name)} "
                    f"WHERE {_quote(where_column)} = :fk_dak"
                    f"{order_by} LIMIT 1"
                )
                row = conn.execute(sql, {"fk_dak": int(fk_dak)}).mappings().first()
                if not row:
                    continue

                for column_name, value in row.items():
                    payload[f"{table_name}.{column_name}"] = _safe_json(value)
    except Exception as exc:
        logger.warning("Could not rebuild source payload for fk_dak=%s: %s", fk_dak, exc)

    return payload

def _dataset_payload(row: pd.Series) -> dict[str, Any]:
    fk_dak_value = row.get(FK_DAK_COLUMN)
    numeric_fk_dak = _safe_numeric_scalar(fk_dak_value, default=None)
    if numeric_fk_dak is not None:
        return {FK_DAK_COLUMN: int(numeric_fk_dak)}

    feature_payload = _parse_json_text(row.get(FEATURE_VALUES_COLUMN), None)
    if isinstance(feature_payload, dict):
        business_payload = {
            str(column): _safe_json(value)
            for column, value in feature_payload.items()
            if str(column) != "__ml_explanation_signals"
        }
        if business_payload:
            return business_payload

    payload = {}
    for column, value in row.items():
        if column in SYSTEM_COLUMNS:
            continue
        payload[str(column)] = _safe_json(value)
    return payload

def _parse_json_text(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default

def _payload_amount(payload: dict[str, Any]) -> float:
    for aliases in (
        ["amount"],
        ["amount_passed"],
        ["amount_claimed"],
        ["invoice_amount"],
        ["schedule3_amount"],
    ):
        value = _payload_value(payload, aliases)
        if value not in (None, ""):
            numeric = _safe_numeric_scalar(value, default=0.0)
            if numeric:
                return numeric
    return 0.0

def _payload_text(payload: dict[str, Any], aliases: list[str], default: str = "") -> str:
    value = _payload_value(payload, aliases)
    if value in (None, ""):
        return default
    return str(value).strip()

def _payload_value(payload: dict[str, Any], aliases: list[str]) -> Any:
    for alias in aliases:
        direct = payload.get(alias)
        if direct not in (None, ""):
            return direct

        normalized_alias = alias.lower()
        for key, value in payload.items():
            if value in (None, ""):
                continue
            plain_key = str(key).split(".")[-1].lower()
            if plain_key == normalized_alias or str(key).lower() == normalized_alias:
                return value
    return None

def _enrich_payload_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    office_ids: set[int] = set()
    vendor_ids: set[int] = set()
    central_vendor_ids: set[int] = set()
    section_ids: set[int] = set()

    for row in rows:
        payload = row.get("row_payload_json") or {}
        office_ids.update(_payload_lookup_ids(payload, ["fk_office_id"]))
        vendor_ids.update(_payload_lookup_ids(payload, ["fk_vendor"]))
        central_vendor_ids.update(_payload_lookup_ids(payload, ["fk_central_vendor"]))
        section_ids.update(_payload_lookup_ids(payload, ["fk_section"]))

    office_map = _fetch_name_map("dad_office", "id", "office_name", office_ids)
    vendor_map = _fetch_name_map("vendor", "id", "vendor_name", vendor_ids)
    central_vendor_map = _fetch_name_map(
        "aaa_central_vendor",
        "id",
        "vendor_name",
        central_vendor_ids,
    )
    section_map = _fetch_name_map("section", "id", "section_name", section_ids)

    for row in rows:
        payload = dict(row.get("row_payload_json") or {})

        office_id = int(_safe_numeric_scalar(_payload_value(payload, ["fk_office_id"]), default=0.0) or 0)
        vendor_id = int(_safe_numeric_scalar(_payload_value(payload, ["fk_vendor"]), default=0.0) or 0)
        central_vendor_id = int(_safe_numeric_scalar(_payload_value(payload, ["fk_central_vendor"]), default=0.0) or 0)
        section_id = int(_safe_numeric_scalar(_payload_value(payload, ["fk_section"]), default=0.0) or 0)

        office_name = office_map.get(office_id, "")
        vendor_name = vendor_map.get(vendor_id, "") or central_vendor_map.get(central_vendor_id, "")
        section_name = section_map.get(section_id, "")

        if office_name:
            payload["resolved_office_name"] = office_name
        if vendor_name:
            payload["resolved_vendor_name"] = vendor_name
        if section_name:
            payload["resolved_section_name"] = section_name

        row["row_payload_json"] = payload
    return rows

def _fetch_name_map(
    table_name: str,
    id_column: str,
    value_column: str,
    ids: set[int],
) -> dict[int, str]:
    if not ids:
        return {}
    id_list = ",".join(str(int(value)) for value in sorted(ids))
    sql = text(
        f"SELECT {_quote(id_column)} AS row_id, {_quote(value_column)} AS row_value "
        f"FROM {_source_table_ref(table_name)} "
        f"WHERE {_quote(id_column)} IN ({id_list})"
    )
    try:
        with _source_connect() as conn:
            rows = conn.execute(sql).mappings().all()
    except Exception as exc:
        logger.warning("Skipping optional lookup table %s: %s", table_name, exc)
        return {}
    return {
        int(row["row_id"]): str(row["row_value"]).strip()
        for row in rows
        if row.get("row_id") is not None and row.get("row_value") not in (None, "")
    }

def _payload_lookup_ids(payload: dict[str, Any], aliases: list[str]) -> set[int]:
    ids: set[int] = set()
    for alias in aliases:
        value = _payload_value(payload, [alias])
        numeric = _safe_numeric_scalar(value, default=0.0)
        if numeric > 0:
            ids.add(int(numeric))
    return ids

def _dashboard_amount_sql(dataset_table: str) -> str:
    if REVIEW_PAYLOAD_COLUMN not in _result_table_columns(dataset_table):
        return "0.0"
    payload_column = _quote(REVIEW_PAYLOAD_COLUMN)
    candidates = [
        "amount",
        "amount_passed",
        "amount_claimed",
        "invoice_amount",
        "schedule3_amount",
        "dak.amount",
        "bill.amount",
        "bill.amount_claimed",
        "bill.amount_passed",
        "gem_bill.invoice_amount",
        "civ_medical_bill.amount_claimed",
        "civ_medical_bill.amount_passed",
        "civ_paybill.amount_claimed",
        "civ_paybill.amount_passed",
        "civ_tada_ltc_bill.amount_claimed",
        "civ_tada_ltc_bill.amount_passed",
        "cheque_slip.amount",
        "schedule3.schedule3_amount",
    ]
    numeric_parts = []
    for key in candidates:
        escaped_key = key.replace("'", "''")
        value_expr = f"NULLIF({payload_column}::jsonb ->> '{escaped_key}', '')"
        numeric_parts.append(
            "CASE "
            f"WHEN {value_expr} ~ '^[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)$' "
            f"THEN {value_expr}::double precision "
            "ELSE NULL END"
        )
    return f"COALESCE({', '.join(numeric_parts)}, 0.0)"
