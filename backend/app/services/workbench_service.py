import json
import logging
import re
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
from functools import lru_cache
from hashlib import blake2s
from pathlib import Path
from typing import Any
import time
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.cache import TABLE_METADATA_CACHE
from app.core.models import WorkbenchRun
from app.schemas.workbench_schema import BuiltinRuleRequest, FeatureRuleInput, OutlierRuleInput, WorkbenchRunRequest
from app.services.llm_reason_service import explain_isolation_anomaly, build_feature_explanation_signals

logger = logging.getLogger(__name__)

PREVIEW_ROW_LIMIT = 50
ENABLE_EXPENSIVE_JOIN_DEBUG = False
DATE_SEQUENCE_STAGE_ALIASES = [
    ("invoice_date", ["invoice_date"]),
    ("bill_date", ["bill_date"]),
    ("reference_date", ["reference_date"]),
    ("list_date", ["list_date"]),
    ("auditor_stage", ["auditor_date", "aud_date", "auditor_disposal_date"]),
    ("aao_stage", ["aao_date", "aao_disposal_date"]),
    ("ao_stage", ["ao_date", "ao_disposal_date"]),
    ("go_date", ["go_date"]),
    ("dp_sheet_date", ["dp_sheet_date"]),
    ("cmp_date", ["cmp_date", "cmp_batch_date"]),
    ("disposal_date", ["disposal_date"]),
    ("utr_date", ["utr_date"]),
]
SAME_TABLE_DATE_SEQUENCE_STAGES = {
    "auditor_stage",
    "aao_stage",
    "ao_stage",
    "go_date",
    "disposal_date",
}
SINGLE_FEATURE_TYPES = [
    "isweekend",
    "isbusinesshour",
]
MIN_FEATURE_COLUMN_PRESENT_RATIO = 0.70
DATE_FILTER_COLUMN_PRIORITY = [
    "list_date",
    "created_at",
]
SYSTEM_COLUMN_PREFIX = "ml_"
RESULT_SCHEMA = "public"
RESULT_TABLE = "ML_Features"
ML_FEATURES_TABLE = RESULT_TABLE
SERIAL_COLUMN = "id"

BUILTIN_FEATURE_RULES_CACHE_TTL = 60.0
_builtin_feature_rules_cache: dict[tuple[Any, ...], tuple[float, list[dict]]] = {}
FEATURE_NAME_COLUMN = "feature_name"
HUMAN_RULE_NAME_COLUMN = "human_rule_name"
HUMAN_RULE_COLUMN = "human_rule"
ISOLATION_RULE_COLUMN = "isolation_rule"
IF_SCORE_COLUMN = "ml_if_score"
ML_THRESHOLD_COLUMN = "ml_threshold"
FEEDBACK_SCORE_COLUMN = "feedback_score"
RUN_ID_COLUMN = "ml_run_id"
REVIEW_PAYLOAD_COLUMN = "review_payload_json"
FEATURE_VALUES_COLUMN = "feature_values_json"
SYSTEM_COLUMNS = {
    SERIAL_COLUMN,
    FEATURE_NAME_COLUMN,
    HUMAN_RULE_NAME_COLUMN,
    HUMAN_RULE_COLUMN,
    ISOLATION_RULE_COLUMN,
    IF_SCORE_COLUMN,
    ML_THRESHOLD_COLUMN,
    FEEDBACK_SCORE_COLUMN,
    RUN_ID_COLUMN,
    REVIEW_PAYLOAD_COLUMN,
    FEATURE_VALUES_COLUMN,
}
TEMP_ROW_ID_COLUMN = "__ml_row_number"
SQL_RULE_FLAG_COLUMN = "sql_rule_flag"
SQL_RULE_REASONS_COLUMN = "sql_rule_reasons"
USER_RULE_FLAG_COLUMN = "__ml_sql_rule_flag"
USER_RULE_REASONS_COLUMN = "__ml_sql_rule_reasons"

FEEDBACK_TO_SCORE = {
    "accept": 1.0,
    "reject": 0.0,
    "maybe": 0.5,
}
SCORE_TO_FEEDBACK = {v: k for k, v in FEEDBACK_TO_SCORE.items()}

# API flow order: source discovery -> defaults -> preview/run -> dataset listing -> feedback.
def list_source_tables() -> list[dict]:
    """
    Get list of tables from source database.
    
    Results are cached for 5 minutes to avoid repeated information_schema queries.
    """
    # Try to get from cache first
    cached_result = TABLE_METADATA_CACHE.get("source_tables")
    if cached_result is not None:
        logger.debug("Returning cached source tables list")
        return cached_result
    
    schema = settings.source_db_schema
    query = text(
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = :schema
        ORDER BY table_name, ordinal_position
        """
    )
    grouped: dict[str, list[dict]] = {}
    with _source_connect() as conn:
        for row in conn.execute(query, {"schema": schema}):
            grouped.setdefault(row.table_name, []).append(
                {
                    "table_name": row.table_name,
                    "column_name": row.column_name,
                    "data_type": row.data_type,
                }
            )
    
    result = [{"table_name": table, "columns": columns} for table, columns in grouped.items()]
    TABLE_METADATA_CACHE.set("source_tables", result)
    return result

def source_connection_status() -> dict:
    resolved_database = settings.source_db_name
    try:
        tables = list_source_tables()
        return {
            "connected": True,
            "host": settings.source_db_host,
            "database": resolved_database,
            "table_count": len(tables),
        }
    except Exception as exc:
        return {
            "connected": False,
            "host": settings.source_db_host,
            "database": resolved_database,
            "table_count": 0,
            "error": _friendly_source_db_error(exc),
        }

def builtin_feature_rules(
    payload: BuiltinRuleRequest | WorkbenchRunRequest | list[str],
) -> list[dict]:
    if isinstance(payload, list):
        selected_tables = payload
        request_payload: BuiltinRuleRequest | WorkbenchRunRequest | None = None
    else:
        selected_tables = payload.selected_tables
        request_payload = payload

    join_signature: tuple[tuple[str, str, str, str, str], ...] = ()
    from_date: str | None = None
    to_date: str | None = None
    if request_payload is not None:
        join_signature = tuple(
            sorted(
                (
                    str(join.left_table),
                    str(join.left_column),
                    str(join.right_table),
                    str(join.right_column),
                    str(join.join_type),
                )
                for join in request_payload.joins
            )
        )
        from_date = _safe_date_literal(getattr(request_payload, "from_date", None))
        to_date = _safe_date_literal(getattr(request_payload, "to_date", None))

    cache_key = (tuple(sorted(selected_tables)), join_signature, from_date, to_date)
    now = time.monotonic()
    cached = _builtin_feature_rules_cache.get(cache_key)
    if cached is not None and now - cached[0] < BUILTIN_FEATURE_RULES_CACHE_TTL:
        return cached[1]

    table_frames = _source_columns_map(selected_tables)
    use_joined_presence = bool(
        request_payload
        and (
            request_payload.joins
            or _safe_date_literal(getattr(request_payload, "from_date", None))
            or _safe_date_literal(getattr(request_payload, "to_date", None))
        )
    )
    present_ratios = (
        _joined_feature_column_presence_ratios(request_payload, table_frames)
        if use_joined_presence and request_payload is not None
        else _feature_column_presence_ratios(table_frames)
    )
    available = {
        f"{table_name}.{column['column_name']}": column.get("data_type", "")
        for table_name, columns in table_frames.items()
        for column in columns
        if _has_enough_values_for_builtin_features(
            present_ratios.get(f"{table_name}.{column['column_name']}", 0.0)
        )
    }
    rules: list[dict] = []
    rules.extend(_build_dynamic_date_gap_rules(available))
    for column_name, data_type in available.items():
        if not _is_date_like_column_name(column_name, data_type):
            continue
        for feature_type in SINGLE_FEATURE_TYPES:
            pretty = feature_type.title()
            rules.append(
                {
                    "name": f"{pretty}-{column_name}",
                    "feature_type": feature_type,
                    "first_column": column_name,
                    "second_column": "",
                    "operator": "",
                }
            )
    deduped_rules: list[dict] = []
    seen_rule_keys: set[tuple[str, str, str, str]] = set()
    for rule in rules:
        rule_key = (
            str(rule.get("feature_type") or ""),
            str(rule.get("first_column") or ""),
            str(rule.get("second_column") or ""),
            str(rule.get("operator") or ""),
        )
        if rule_key in seen_rule_keys:
            continue
        seen_rule_keys.add(rule_key)
        deduped_rules.append(rule)

    _builtin_feature_rules_cache[cache_key] = (now, deduped_rules)
    return deduped_rules

def preview_workbench(payload: WorkbenchRunRequest) -> dict:
    joined, source_row_counts, join_debug, warnings, executed_sql = _execute_sql_joined_frame(payload, for_preview=True)
    dataset_table = _dataset_table_name(payload.selected_tables)
    preview_rows = joined.head(PREVIEW_ROW_LIMIT).replace({np.nan: None}).to_dict(orient="records")
    return {
        "mode": "preview",
        "total_rows_previewed": int(len(joined)),
        "preview_limit": PREVIEW_ROW_LIMIT,
        "selected_tables": payload.selected_tables,
        "columns": [str(column) for column in joined.columns],
        "rows": preview_rows,
        "metrics": {
            "source_row_counts_estimate": source_row_counts,
            "join_debug": join_debug,
            "executed_join_sql": executed_sql,
            "dataset_table": dataset_table,
            "warnings": warnings,
        },
    }

def run_workbench(db: Session, payload: WorkbenchRunRequest) -> dict:
    logger.info(
        "Starting workbench run for tables: %s, feature_rules: %s, outlier_rules: %s",
        payload.selected_tables,
        len(payload.feature_rules),
        len(payload.outlier_rules),
    )
    warnings: list[str] = []

    (
        joined,
        source_row_counts,
        join_debug,
        join_warnings,
        executed_sql,
        human_reasons,
        applied_feature_rule_count,
        sql_pushdown_warnings,
        applied_outlier_rule_count,
        staging_table,
    ) = _execute_sql_workbench_frame(payload)
    warnings.extend(join_warnings)
    warnings.extend(sql_pushdown_warnings)
    logger.info("Joined %d rows from %d tables", len(joined), len(payload.selected_tables))

    batch_id = f"workbench_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    dataset_table = _dataset_table_name(payload.selected_tables)
    dataset_run_id = _next_dataset_run_id(dataset_table)
    if joined.empty:
        run = WorkbenchRun(
            run_name=payload.run_name,
            source_tables_json=payload.selected_tables,
            join_config_json=[item.model_dump() for item in payload.joins],
            outlier_rules_json=[item.model_dump() for item in payload.outlier_rules],
            feature_rules_json=[item.model_dump() for item in payload.feature_rules],
            amount_field=payload.amount_field,
            total_rows=0,
            human_outlier_count=0,
            ml_anomaly_count=0,
            final_anomaly_count=0,
            selected_model="IsolationForest",
            metrics_json={
                "batch_id": batch_id,
                "selected_tables": payload.selected_tables,
                "source_row_counts": source_row_counts,
                "join_debug": join_debug,
                "executed_join_sql": executed_sql,
                "dataset_table": dataset_table,
                "ml_run_id": dataset_run_id,
                "from_date": _safe_date_literal(payload.from_date),
                "to_date": _safe_date_literal(payload.to_date),
                "requested_outlier_rule_count": int(len(payload.outlier_rules)),
                "requested_feature_rule_count": int(len(payload.feature_rules)),
                "applied_outlier_rule_count": int(applied_outlier_rule_count),
                "applied_feature_rule_count": int(applied_feature_rule_count),
                "warnings": warnings,
            },
            status="COMPLETED",
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return {
            "run_id": run.run_id,
            "run_name": run.run_name,
            "total_rows": 0,
            "human_outlier_count": 0,
            "ml_anomaly_count": 0,
            "final_anomaly_count": 0,
            "amount_total": 0.0,
            "selected_model": "IsolationForest",
            "metrics": run.metrics_json or {},
        }

    if "__ml_sql_rule_flag" in joined.columns:
        user_rule_flag = joined.pop("__ml_sql_rule_flag").map(lambda value: bool(value) if pd.notna(value) else False)
    else:
        user_rule_flag = pd.Series(False, index=joined.index, dtype=bool)
    if "sql_rule_flag" in joined.columns:
        builtin_rule_flag = joined.pop("sql_rule_flag").map(lambda value: bool(value) if pd.notna(value) else False)
    else:
        builtin_rule_flag = pd.Series(False, index=joined.index, dtype=bool)
    human_outlier_flag = user_rule_flag | builtin_rule_flag
    if "__ml_sql_rule_reasons" in joined.columns:
        human_reason_series = joined.pop("__ml_sql_rule_reasons")
    else:
        human_reason_series = None
    if "sql_rule_reasons" in joined.columns:
        builtin_reason_series = joined.pop("sql_rule_reasons")
    else:
        builtin_reason_series = None
    if builtin_rule_flag.any():
        if builtin_reason_series is None:
            builtin_reason_series = builtin_rule_flag.map(
                lambda flag: "OUTLIER::Built-in SQL anomaly rule" if bool(flag) else None
            )

    features = pd.DataFrame(index=joined.index)
    sql_feature_cols = [column for column in _feature_rule_aliases(payload.feature_rules) if column in joined.columns]
    if sql_feature_cols:
        extra = joined[sql_feature_cols].apply(pd.to_numeric, errors="coerce")
        features = pd.concat([features, extra], axis=1)
        logger.info("Added %d explicit feature-rule columns to IF input", len(sql_feature_cols))

    features = _add_statistical_outlier_signals(features)
    logger.info("Feature set has %d columns after statistical signals", len(features.columns))

    feature_frame = _prepare_isolation_forest_feature_frame(features)
    logger.info("Feature frame has %d rows and %d columns", len(feature_frame), len(feature_frame.columns))

    if feature_frame.empty:
        _drop_workbench_temp_table(staging_table)
        raise ValueError(
            "No usable feature columns were produced from the selected SQL-joined dataset and feature rules. "
            "Choose at least one numeric/date-like column or add valid feature rules."
        )

    contamination = payload.contamination
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", IsolationForest(contamination=contamination, random_state=settings.random_state)),
        ]
    )

    try:
        pipeline.fit(feature_frame)
    except Exception as exc:
        _drop_workbench_temp_table(staging_table)
        raise ValueError(f"Isolation Forest training failed on the engineered feature set: {exc}") from exc

    try:
        transformed = pipeline.named_steps["scaler"].transform(
            pipeline.named_steps["imputer"].transform(feature_frame)
        )
        isolation_scores = -pipeline.named_steps["model"].score_samples(transformed)
        ml_flag = pd.Series(
            pipeline.named_steps["model"].predict(transformed) == -1,
            index=joined.index,
        )
    except Exception as exc:
        _drop_workbench_temp_table(staging_table)
        raise ValueError(f"Isolation Forest scoring failed: {exc}") from exc

    model = pipeline.named_steps["model"]
    if contamination == "auto":
        ml_threshold = float(-model.offset_)
    else:
        ml_threshold = float(np.quantile(isolation_scores, max(0.0, min(1.0, 1.0 - contamination))))

    final_flag = human_outlier_flag | ml_flag
    explanation_signals = build_feature_explanation_signals(
        pipeline,
        feature_frame,
        transformed,
        feature_frame.loc[final_flag].index,
    )

    anomaly_row_ids = [int(row_id) for row_id in final_flag[final_flag].index.tolist()]
    try:
        with _source_connect() as conn:
            filtered_joined = _read_temp_anomaly_payload_frame(conn, staging_table, anomaly_row_ids, payload)
    finally:
        _drop_workbench_temp_table(staging_table)

    amount_total = 0.0
    if payload.amount_field and not filtered_joined.empty:
        amount_column = _resolve_column(filtered_joined, payload.amount_field)
        amount_total = float(pd.to_numeric(filtered_joined[amount_column], errors="coerce").fillna(0).sum())

    filtered_feature_frame = feature_frame.loc[final_flag]
    filtered_human_outlier_flag = human_outlier_flag.loc[final_flag]
    filtered_ml_flag = ml_flag.loc[final_flag]
    filtered_human_reasons = (
        human_reason_series.loc[final_flag]
        if human_reason_series is not None
        else pd.Series(None, index=filtered_human_outlier_flag.index)
    )
    filtered_builtin_reasons = (
        builtin_reason_series.loc[final_flag]
        if builtin_reason_series is not None
        else pd.Series(None, index=filtered_human_outlier_flag.index)
    )
    filtered_isolation_scores = np.asarray(isolation_scores)[final_flag.to_numpy()]
    llm_if_reasons = _build_persisted_llm_if_reasons(
        payload,
        filtered_feature_frame,
        filtered_joined,
        filtered_ml_flag,
        filtered_human_outlier_flag,
        filtered_human_reasons,
        filtered_builtin_reasons,
        filtered_isolation_scores,
        ml_threshold,
        explanation_signals,
    )

    run = WorkbenchRun(
        run_name=payload.run_name,
        source_tables_json=payload.selected_tables,
        join_config_json=[item.model_dump() for item in payload.joins],
        outlier_rules_json=[item.model_dump() for item in payload.outlier_rules],
        feature_rules_json=[item.model_dump() for item in payload.feature_rules],
        amount_field=payload.amount_field,
        total_rows=int(len(joined)),
        human_outlier_count=int(human_outlier_flag.sum()),
        ml_anomaly_count=int(ml_flag.sum()),
        final_anomaly_count=int(final_flag.sum()),
        selected_model="IsolationForest",
        metrics_json={
            "contamination": contamination,
            "feature_count": int(feature_frame.shape[1]),
            "batch_id": batch_id,
            "selected_tables": payload.selected_tables,
            "join_debug": join_debug,
            "executed_join_sql": executed_sql,
            "dataset_table": dataset_table,
            "ml_run_id": dataset_run_id,
            "from_date": _safe_date_literal(payload.from_date),
            "to_date": _safe_date_literal(payload.to_date),
        },
        status="COMPLETED",
    )
    db.add(run)
    db.flush()

    dataset_frame = _build_dataset_frame(
        joined,
        feature_frame,
        payload,
        dataset_table=dataset_table,
        dataset_run_id=dataset_run_id,
        human_outlier_flag=human_outlier_flag,
        human_reasons=human_reasons,
        human_reason_series=human_reason_series,
        builtin_reason_series=builtin_reason_series,
        isolation_scores=isolation_scores,
        ml_flag=ml_flag,
        ml_threshold=ml_threshold,
        final_flag=final_flag,
        filtered_joined_override=filtered_joined,
        explanation_signals_override=explanation_signals,
        llm_if_reasons_override=llm_if_reasons,
    )
    dataset_storage = _write_dataset_to_result(dataset_frame, dataset_table)
    builtin_reason_by_record_id = {}
    if builtin_reason_series is not None:
        filtered_builtin_reasons = builtin_reason_series.loc[final_flag]
        inserted_ids = dataset_storage.get("inserted_ids") or []
        builtin_reason_by_record_id = {
            str(record_id): str(reason).strip()
            for record_id, reason in zip(inserted_ids, filtered_builtin_reasons)
            if pd.notna(reason) and str(reason).strip()
        }
    logger.info("Appended rows into dataset table %s; total rows now %s", dataset_table, dataset_storage["row_count"])

    run.metrics_json = {
        **(run.metrics_json or {}),
        "source_row_counts": source_row_counts,
        "requested_outlier_rule_count": int(len(payload.outlier_rules)),
        "requested_feature_rule_count": int(len(payload.feature_rules)),
        "applied_outlier_rule_count": int(applied_outlier_rule_count),
        "applied_feature_rule_count": int(applied_feature_rule_count),
        "ml_run_id": dataset_run_id,
        "warnings": warnings,
        "join_execution_mode": "postgres_sql",
        "new_rows_written": int(len(dataset_frame)),
        "builtin_reason_by_record_id": builtin_reason_by_record_id,
        "joined_result_table": dataset_storage["table_name"],
        "joined_result_row_count": dataset_storage["row_count"],
        "joined_result_column_count": dataset_storage["column_count"],
    }

    db.commit()
    db.refresh(run)

    model_path = Path(settings.active_model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return {
        "run_id": run.run_id,
        "run_name": run.run_name,
        "total_rows": run.total_rows,
        "human_outlier_count": run.human_outlier_count,
        "ml_anomaly_count": run.ml_anomaly_count,
        "final_anomaly_count": run.final_anomaly_count,
        "amount_total": amount_total,
        "selected_model": "IsolationForest",
        "metrics": run.metrics_json or {},
    }

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

# Helper functions below are grouped to support the same top-to-bottom lifecycle.
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


@lru_cache(maxsize=16)

def _source_engine():
    resolved_database = settings.source_db_name
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=settings.source_db_user,
        password=settings.source_db_password,
        host=settings.source_db_host,
        port=settings.source_db_port,
        database=resolved_database,
    )
    return create_engine(
        url,
        future=True,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )

def _dispose_source_engine() -> None:
    try:
        _source_engine().dispose()
        _source_engine.cache_clear()
    except Exception:
        logger.exception("Failed to dispose source engine")

def _friendly_source_db_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    missing_database_match = re.search(r'database "([^"]+)" does not exist', message, flags=re.IGNORECASE)
    if missing_database_match:
        missing_database = missing_database_match.group(1)
        return (
            f'PostgreSQL database "{missing_database}" does not exist on '
            f'{settings.source_db_host}:{settings.source_db_port}. '
            "Update TULIP_SOURCE_DB_NAME in your .env to the exact database name."
        )
    if "too many clients already" in lowered:
        return (
            "PostgreSQL has no free client connections right now. "
            "Reduce concurrent workbench/scheduler jobs and use a smaller SQLAlchemy pool."
        )
    if "timeout" in lowered and "queuepool" in lowered:
        return (
            "Timed out while waiting for a PostgreSQL connection from the SQLAlchemy pool. "
            "The pool is saturated or queries are holding connections too long."
        )
    return message


@contextmanager

def _source_connect():
    engine = _source_engine()
    try:
        with engine.connect() as conn:
            yield conn
    except OperationalError as exc:
        _dispose_source_engine()
        raise ValueError(_friendly_source_db_error(exc)) from exc


@contextmanager

def _source_begin():
    engine = _source_engine()
    try:
        with engine.begin() as conn:
            yield conn
    except OperationalError as exc:
        _dispose_source_engine()
        raise ValueError(_friendly_source_db_error(exc)) from exc

def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'

def _index_name(*parts: str) -> str:
    base = "idx_" + "_".join(_slug_token(part) for part in parts)
    if len(base) <= 55:
        return base
    digest = blake2s(base.encode("utf-8")).hexdigest()[:8]
    return f"{base[:46].rstrip('_')}_{digest}"

def _storage_column_name(column_name: Any, used_names: set[str]) -> str:
    raw_name = str(column_name)
    if len(raw_name) <= 55 and raw_name not in used_names:
        used_names.add(raw_name)
        return raw_name

    digest = blake2s(raw_name.encode("utf-8")).hexdigest()[:8]
    candidate = f"{raw_name[:46].rstrip('_')}_{digest}"
    suffix = 1
    while candidate in used_names:
        suffix_text = f"_{suffix}"
        candidate = f"{raw_name[:46 - len(suffix_text)].rstrip('_')}_{digest}{suffix_text}"
        suffix += 1
    used_names.add(candidate)
    return candidate

def _normalize_storage_columns(df: pd.DataFrame) -> pd.DataFrame:
    used_names: set[str] = set()
    renamed = [_storage_column_name(column, used_names) for column in df.columns]
    if renamed == [str(column) for column in df.columns]:
        return df
    normalized = df.copy()
    normalized.columns = renamed
    return normalized

def _source_table_ref(table_name: str) -> str:
    return f"{_quote(settings.source_db_schema)}.{_quote(table_name)}"

def _source_columns_map(
    selected_tables: list[str] | None = None,
) -> dict[str, list[dict]]:
    tables = list_source_tables()
    out: dict[str, list[dict]] = {}
    for item in tables:
        table_name = item["table_name"]
        if selected_tables and table_name not in selected_tables:
            continue
        out[table_name] = item["columns"]
    return out

def _is_date_like_column_name(column_name: str, data_type: str | None = None) -> bool:
    lower_name = str(column_name).strip().lower()
    lower_type = str(data_type or "").strip().lower()
    return any(token in lower_type for token in ["date", "time"]) or any(
        token in lower_name for token in ["date", "time", "created_at", "updated_at", "timestamp"]
    )

def _feature_column_presence_ratios(
    table_frames: dict[str, list[dict]],
) -> dict[str, float]:
    ratios: dict[str, float] = {}
    with _source_connect() as conn:
        for table_name, columns in table_frames.items():
            date_columns = [
                str(column["column_name"])
                for column in columns
                if _is_date_like_column_name(str(column["column_name"]), column.get("data_type"))
            ]
            if not date_columns:
                continue
            select_parts = ["COUNT(*) AS total_rows"]
            alias_by_column: dict[str, str] = {}
            for index, column_name in enumerate(date_columns):
                alias = f"present_{index}"
                alias_by_column[column_name] = alias
                select_parts.append(
                    f"COUNT({_quote(column_name)}) AS {_quote(alias)}"
                )
            sql = text(f"SELECT {', '.join(select_parts)} FROM {_source_table_ref(table_name)}")
            row = conn.execute(sql).mappings().first()
            total_rows = int((row or {}).get("total_rows") or 0)
            for column_name in date_columns:
                qualified_name = f"{table_name}.{column_name}"
                if total_rows == 0:
                    ratios[qualified_name] = 0.0
                    continue
                present_count = int((row or {}).get(alias_by_column[column_name]) or 0)
                ratios[qualified_name] = present_count / total_rows
    return ratios

def _joined_feature_column_presence_ratios(
    payload: BuiltinRuleRequest | WorkbenchRunRequest,
    table_frames: dict[str, list[dict]],
) -> dict[str, float]:
    sql, _join_debug, _warnings = _build_join_sql(payload, table_frames, row_limit=None)
    date_columns = [
        f"{table_name}.{column['column_name']}"
        for table_name, columns in table_frames.items()
        for column in columns
        if _is_date_like_column_name(str(column["column_name"]), column.get("data_type"))
    ]
    if not date_columns:
        return {}

    select_parts = ["COUNT(*) AS total_rows"]
    alias_by_column: dict[str, str] = {}
    for index, column_name in enumerate(date_columns):
        alias = f"present_{index}"
        alias_by_column[column_name] = alias
        select_parts.append(f"COUNT(joined_source.{_quote(column_name)}) AS {_quote(alias)}")

    query = text(
        f"""
        SELECT {', '.join(select_parts)}
        FROM ({sql}) AS joined_source
        """
    )

    ratios: dict[str, float] = {}
    with _source_connect() as conn:
        row = conn.execute(query).mappings().first()
        total_rows = int((row or {}).get("total_rows") or 0)
        for column_name in date_columns:
            if total_rows == 0:
                ratios[column_name] = 0.0
                continue
            present_count = int((row or {}).get(alias_by_column[column_name]) or 0)
            ratios[column_name] = present_count / total_rows
    return ratios

def _has_enough_values_for_builtin_features(present_ratio: float) -> bool:
    return present_ratio >= MIN_FEATURE_COLUMN_PRESENT_RATIO

def _builtin_stage_column_matches(available: dict[str, str]) -> dict[str, list[str]]:
    matches_by_alias: dict[str, list[str]] = {}
    for qualified_name, data_type in available.items():
        _, plain_column = qualified_name.split(".", 1)
        normalized_plain = plain_column.strip().lower()
        if not _is_date_like_column_name(normalized_plain, data_type):
            continue
        matches_by_alias.setdefault(normalized_plain, []).append(qualified_name)
    return matches_by_alias

def _same_table_date_sequence_pair(left_stage_name: str, right_stage_name: str) -> bool:
    return (
        left_stage_name in SAME_TABLE_DATE_SEQUENCE_STAGES
        and right_stage_name in SAME_TABLE_DATE_SEQUENCE_STAGES
    )

def _build_dynamic_date_gap_rules(available: dict[str, str]) -> list[dict]:
    matches_by_alias = _builtin_stage_column_matches(available)
    stage_columns: list[tuple[str, list[str]]] = []
    for stage_name, aliases in DATE_SEQUENCE_STAGE_ALIASES:
        matched_columns: list[str] = []
        for alias in aliases:
            matched_columns.extend(matches_by_alias.get(alias, []))
        if matched_columns:
            stage_columns.append((stage_name, matched_columns))

    rules: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index in range(len(stage_columns) - 1):
        left_stage_name, left_columns = stage_columns[index]
        right_stage_name, right_columns = stage_columns[index + 1]
        for left_column in left_columns:
            for right_column in right_columns:
                if left_column == right_column:
                    continue
                if _same_table_date_sequence_pair(left_stage_name, right_stage_name):
                    left_table = left_column.split(".", 1)[0]
                    right_table = right_column.split(".", 1)[0]
                    if left_table != right_table:
                        continue
                pair_key = (right_column, left_column)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                rules.append(
                    {
                        "name": f"{left_stage_name}_to_{right_stage_name}",
                        "feature_type": "daysbetween",
                        "first_column": right_column,
                        "second_column": left_column,
                    }
                )
    return rules

def _approx_table_row_count(conn, table_name: str) -> int:
    sql = text(
        """
        SELECT COALESCE(c.reltuples, 0)::bigint
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema_name
          AND c.relname = :table_name
        """
    )
    return int(conn.execute(sql, {"schema_name": settings.source_db_schema, "table_name": table_name}).scalar() or 0)

def _validate_selected_tables(selected_tables: list[str], source_columns: dict[str, list[dict]]) -> None:
    if not selected_tables:
        raise ValueError("At least one source table must be selected.")

    missing = [table_name for table_name in selected_tables if table_name not in source_columns]
    if missing:
        raise ValueError(f"Selected source tables not found in schema '{settings.source_db_schema}': {missing}")

    counts = Counter(selected_tables)
    duplicates = [table for table, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(
            "Duplicate source tables are not supported in the current SQL join builder. "
            f"Remove duplicates from selected_tables: {sorted(set(duplicates))}"
        )

def _resolve_source_column_name(source_columns: dict[str, list[dict]], table_name: str, column_name: str) -> str:
    if not column_name:
        raise ValueError(f"Join column missing for table: {table_name}")
    if table_name not in source_columns:
        raise ValueError(f"Table '{table_name}' is not available in source schema '{settings.source_db_schema}'.")

    plain_column = column_name.split(".", 1)[1] if column_name.startswith(f"{table_name}.") else column_name
    available = {item["column_name"] for item in source_columns[table_name]}
    if plain_column not in available:
        raise ValueError(
            f"Column '{column_name}' was not found in source table '{table_name}'. "
            f"Available columns sample: {sorted(list(available))[:20]}"
        )
    return plain_column

def _source_column_meta(source_columns: dict[str, list[dict]], table_name: str, column_name: str) -> dict[str, Any]:
    plain_column = _resolve_source_column_name(source_columns, table_name, column_name)
    for item in source_columns[table_name]:
        if item["column_name"] == plain_column:
            return item
    raise ValueError(f"Column metadata not found for {table_name}.{plain_column}")

def _type_family(data_type: str | None) -> str:
    value = str(data_type or "").strip().lower()
    if value in {"smallint", "integer", "bigint", "decimal", "numeric", "real", "double precision"}:
        return "numeric"
    if value in {"date", "timestamp without time zone", "timestamp with time zone", "time without time zone", "time with time zone"}:
        return "datetime"
    if value in {"character varying", "character", "text", "uuid"}:
        return "text"
    if value in {"boolean"}:
        return "boolean"
    return value

def _sql_join_keyword(join_type: str) -> str:
    normalized = (join_type or "inner").strip().lower()
    mapping = {
        "inner": "INNER JOIN",
        "left": "LEFT JOIN",
        "left outer": "LEFT JOIN",
        "right": "RIGHT JOIN",
        "right outer": "RIGHT JOIN",
        "outer": "FULL OUTER JOIN",
        "full": "FULL OUTER JOIN",
        "full outer": "FULL OUTER JOIN",
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported join type '{join_type}'. Supported types: inner, left, right, outer/full."
        )
    return mapping[normalized]


_SAFE_IDENTIFIER_RE = re.compile(r'^[\w.\- ]+$')

def _validate_identifier(value: str, label: str) -> None:
    if not _SAFE_IDENTIFIER_RE.match(value):
        raise ValueError(f"Unsafe SQL identifier for {label}: {value!r}")

def _sql_table_column_samples(conn, table_name: str, column_name: str, size: int = 10) -> list[Any]:
    _validate_identifier(table_name, "table_name")
    _validate_identifier(column_name, "column_name")
    schema = _quote(settings.source_db_schema)
    table_q = _quote(table_name)
    column_q = _quote(column_name)
    sql = text(
        f"SELECT {column_q} FROM {schema}.{table_q} WHERE {column_q} IS NOT NULL LIMIT {int(size)}"
    )
    return [_safe_json(row[0]) for row in conn.execute(sql).fetchall()]

def _sql_common_key_count(conn, left_table: str, left_column: str, right_table: str, right_column: str) -> int:
    _validate_identifier(left_table, "left_table")
    _validate_identifier(left_column, "left_column")
    _validate_identifier(right_table, "right_table")
    _validate_identifier(right_column, "right_column")
    schema = _quote(settings.source_db_schema)
    left_table_q = _quote(left_table)
    right_table_q = _quote(right_table)
    left_key = _quote(left_column)
    right_key = _quote(right_column)
    sql = text(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT {left_key} AS join_key
            FROM {schema}.{left_table_q}
            WHERE {left_key} IS NOT NULL
        ) l
        INNER JOIN (
            SELECT DISTINCT {right_key} AS join_key
            FROM {schema}.{right_table_q}
            WHERE {right_key} IS NOT NULL
        ) r
        ON l.join_key = r.join_key
        """
    )
    return int(conn.execute(sql).scalar() or 0)

def _sql_common_key_sample(
    conn,
    left_table: str,
    left_column: str,
    right_table: str,
    right_column: str,
    size: int = 20,
) -> list[Any]:
    schema = _quote(settings.source_db_schema)
    left_table_q = _quote(left_table)
    right_table_q = _quote(right_table)
    left_key = _quote(left_column)
    right_key = _quote(right_column)
    safe_size = max(1, min(int(size), 100))
    sql = text(
        f"""
        SELECT l.join_key
        FROM (
            SELECT DISTINCT {left_key} AS join_key
            FROM {schema}.{left_table_q}
            WHERE {left_key} IS NOT NULL
        ) l
        INNER JOIN (
            SELECT DISTINCT {right_key} AS join_key
            FROM {schema}.{right_table_q}
            WHERE {right_key} IS NOT NULL
        ) r
        ON l.join_key = r.join_key
        ORDER BY l.join_key
        LIMIT {safe_size}
        """
    )
    return [_safe_json(row[0]) for row in conn.execute(sql).fetchall()]

def _build_join_select_list(selected_tables: list[str], source_columns: dict[str, list[dict]]) -> str:
    parts: list[str] = []
    for table_name in selected_tables:
        for column in source_columns[table_name]:
            column_name = column["column_name"]
            source_expr = f"{_quote(table_name)}.{_quote(column_name)}"
            if _type_family(column.get("data_type")) == "datetime":
                source_expr = f"CAST({source_expr} AS text)"
            parts.append(f"{source_expr} AS {_quote(f'{table_name}.{column_name}')}")
    if not parts:
        raise ValueError("No source columns were found for the selected tables.")
    return ",\n    ".join(parts)

def _joined_column_meta(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    column_name: str,
) -> tuple[str, dict[str, Any]]:
    if "." in str(column_name):
        table_name, plain_column = str(column_name).split(".", 1)
        if table_name not in source_columns:
            raise ValueError(f"Table not found for column '{column_name}'.")
        meta = _source_column_meta(source_columns, table_name, plain_column)
        return f"{table_name}.{meta['column_name']}", meta

    matches: list[tuple[str, dict[str, Any]]] = []
    for table_name in selected_tables:
        for item in source_columns[table_name]:
            if item["column_name"] == column_name:
                matches.append((f"{table_name}.{item['column_name']}", item))
    if not matches:
        raise ValueError(f"Column not found: {column_name}")
    if len(matches) > 1:
        raise ValueError(f"Column is ambiguous: {column_name}. Matches: {[name for name, _ in matches[:10]]}")
    return matches[0]

def _joined_column_expr(joined_column_name: str, alias: str = "src") -> str:
    return f"{_quote(alias)}.{_quote(joined_column_name)}"

def _sql_safe_numeric(expr: str) -> str:
    trimmed = f"btrim(CAST({expr} AS text))"
    numeric_regex = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$"
    return (
        "CASE "
        f"WHEN {expr} IS NULL THEN NULL "
        f"WHEN lower({trimmed}) IN ('', 'nan', 'none', 'null', '<na>', 'nat') THEN NULL "
        f"WHEN {trimmed} ~ '{numeric_regex}' THEN ({trimmed})::double precision "
        "ELSE NULL "
        "END"
    )

def _sql_numeric_or_text_comparison(first_expr: str, second_expr: str, operator: str) -> str:
    comparator = "=" if operator == "=" else "<>"
    left_num = _sql_safe_numeric(first_expr)
    right_num = _sql_safe_numeric(second_expr)
    text_comparison = f"CAST({first_expr} AS text) {comparator} CAST({second_expr} AS text)"
    numeric_comparison = f"{left_num} {comparator} {right_num}"
    return (
        "("
        f"CASE WHEN {left_num} IS NOT NULL AND {right_num} IS NOT NULL "
        f"THEN {numeric_comparison} "
        f"ELSE {text_comparison} "
        "END"
        ")"
    )

def _sql_safe_timestamp(expr: str, data_type: str | None = None) -> str:
    text_expr = f"btrim(CAST({expr} AS text))"
    return (
        "CASE "
        f"WHEN {expr} IS NULL THEN NULL::timestamp "
        f"WHEN {text_expr} ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}' THEN left({text_expr}, 10)::date::timestamp "
        "ELSE NULL::timestamp "
        "END"
    )

def _sql_boolean_as_double(predicate: str) -> str:
    return f"(CASE WHEN {predicate} THEN 1.0 ELSE 0.0 END)"

def _next_param_name(params: dict[str, Any], prefix: str) -> str:
    return f"{prefix}_{len(params) + 1}"

def _build_sql_feature_expression(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    rule: FeatureRuleInput,
    *,
    alias: str,
) -> str:
    first_name, first_meta = _joined_column_meta(selected_tables, source_columns, rule.first_column)
    first_expr = _joined_column_expr(first_name, alias)
    second_meta: dict[str, Any] | None = None
    second_expr: str | None = None
    if rule.second_column:
        second_name, second_meta = _joined_column_meta(selected_tables, source_columns, rule.second_column)
        second_expr = _joined_column_expr(second_name, alias)

    feature_type = str(rule.feature_type or "").strip().lower()

    if feature_type == "numeric":
        return _sql_safe_numeric(first_expr)
    if feature_type == "difference" and second_expr is not None:
        return f"({_sql_safe_numeric(first_expr)} - {_sql_safe_numeric(second_expr)})"
    if feature_type == "ratio" and second_expr is not None:
        denominator = _sql_safe_numeric(second_expr)
        return f"({_sql_safe_numeric(first_expr)} / NULLIF({denominator}, 0))"
    if feature_type == "sum" and second_expr is not None:
        return f"(COALESCE({_sql_safe_numeric(first_expr)}, 0) + COALESCE({_sql_safe_numeric(second_expr)}, 0))"
    if feature_type == "comparisonflag" and second_expr is not None and rule.operator:
        left_num = _sql_safe_numeric(first_expr)
        right_num = _sql_safe_numeric(second_expr)
        if rule.operator == ">":
            return _sql_boolean_as_double(f"{left_num} > {right_num}")
        if rule.operator == ">=":
            return _sql_boolean_as_double(f"{left_num} >= {right_num}")
        if rule.operator == "<":
            return _sql_boolean_as_double(f"{left_num} < {right_num}")
        if rule.operator == "<=":
            return _sql_boolean_as_double(f"{left_num} <= {right_num}")
        if rule.operator == "=":
            return _sql_boolean_as_double(f"{left_num} = {right_num}")
        if rule.operator == "!=":
            return _sql_boolean_as_double(f"{left_num} <> {right_num}")
    if feature_type == "missingflag":
        return _sql_boolean_as_double(f"{first_expr} IS NULL")
    if feature_type == "comparisonflag" and rule.operator == "null":
        return _sql_boolean_as_double(f"{first_expr} IS NULL")
    if feature_type == "comparisonflag" and rule.operator == "not null":
        return _sql_boolean_as_double(f"{first_expr} IS NOT NULL")
    if feature_type == "daysbetween" and second_expr is not None and second_meta is not None:
        first_ts = _sql_safe_timestamp(first_expr, first_meta.get("data_type"))
        second_ts = _sql_safe_timestamp(second_expr, second_meta.get("data_type"))
        return f"(EXTRACT(EPOCH FROM ({first_ts} - {second_ts})) / 86400.0)"
    if feature_type == "isweekend":
        first_ts = _sql_safe_timestamp(first_expr, first_meta.get("data_type"))
        return _sql_boolean_as_double(f"EXTRACT(DOW FROM {first_ts}) IN (0, 6)")
    if feature_type == "isbusinesshour":
        first_ts = _sql_safe_timestamp(first_expr, first_meta.get("data_type"))
        return _sql_boolean_as_double(f"EXTRACT(HOUR FROM {first_ts}) BETWEEN 8 AND 21")
    raise ValueError(f"Unsupported feature type: {rule.feature_type}")

def _build_sql_feature_selects(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    rules: list[FeatureRuleInput],
    *,
    alias: str = "src",
) -> tuple[list[str], list[str], int]:
    warnings: list[str] = []
    selects: list[str] = []
    applied_count = 0
    used_aliases: set[str] = set()

    for index, rule in enumerate(rules, start=1):
        rule_name = _safe_rule_name(rule.name, f"Feature {index}")
        column_alias = f"{rule_name}"
        while column_alias in used_aliases:
            column_alias = f"{column_alias}_{len(used_aliases) + 1}"
        try:
            feature_columns = [rule.first_column, rule.second_column]
            if any(_is_identifier_like_column(column) for column in feature_columns if column):
                warnings.append(f"Skipped feature rule '{rule_name}': identifier columns are not used as ML features.")
                continue
            expr = _build_sql_feature_expression(selected_tables, source_columns, rule, alias=alias)
            selects.append(f"{expr} AS {_quote(column_alias)}")
            used_aliases.add(column_alias)
            applied_count += 1
        except Exception as exc:
            warnings.append(f"Skipped feature rule '{rule_name}': {exc}")
    return selects, warnings, applied_count

def _feature_rule_aliases(rules: list[FeatureRuleInput]) -> list[str]:
    aliases: list[str] = []
    used_aliases: set[str] = set()
    for index, rule in enumerate(rules, start=1):
        column_alias = _safe_rule_name(rule.name, f"Feature {index}")
        while column_alias in used_aliases:
            column_alias = f"{column_alias}_{len(used_aliases) + 1}"
        used_aliases.add(column_alias)
        aliases.append(column_alias)
    return aliases

def _build_sql_outlier_predicate(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    rule: OutlierRuleInput,
    params: dict[str, Any],
    *,
    alias: str,
) -> str:
    first_name, _ = _joined_column_meta(selected_tables, source_columns, rule.first_column)
    first_expr = _joined_column_expr(first_name, alias)
    second_expr: str | None = None
    if rule.second_column:
        second_name, _ = _joined_column_meta(selected_tables, source_columns, rule.second_column)
        second_expr = _joined_column_expr(second_name, alias)

    operator = str(rule.operator or "").strip().lower()
    if operator in {">", ">=", "<", "<="}:
        left_num = _sql_safe_numeric(first_expr)
        if second_expr is not None:
            right_num = _sql_safe_numeric(second_expr)
            return f"({left_num} {operator} {right_num})"
        numeric_value = pd.to_numeric(pd.Series([rule.value]), errors="coerce").iloc[0]
        if pd.isna(numeric_value):
            raise ValueError(f"Rule value '{rule.value}' is not numeric.")
        param_name = _next_param_name(params, "rule_num")
        params[param_name] = float(numeric_value)
        return f"({left_num} {operator} :{param_name})"
    if operator == "null":
        return f"({first_expr} IS NULL)"
    if operator == "not null":
        return f"({first_expr} IS NOT NULL)"
    if operator in {"=", "!="}:
        if second_expr is not None:
            return _sql_numeric_or_text_comparison(first_expr, second_expr, operator)
        param_name = _next_param_name(params, "rule_value")
        params[param_name] = None if rule.value is None else str(rule.value)
        comparator = "=" if operator == "=" else "<>"
        return f"(CAST({first_expr} AS text) {comparator} :{param_name})"
    raise ValueError(f"Unsupported outlier operator: {rule.operator}")

def _build_sql_outlier_flag(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    rules: list[OutlierRuleInput],
    *,
    alias: str = "src",
) -> tuple[str, str, dict[str, Any], list[str], list[str], int]:
    if not rules:
        return "FALSE", "NULL::text", {}, [], [], 0

    predicates: list[str] = []
    reason_parts: list[str] = []
    warnings: list[str] = []
    labels: list[str] = []
    params: dict[str, Any] = {}
    applied_count = 0

    for index, rule in enumerate(rules, start=1):
        rule_name = _safe_rule_name(rule.name, f"Outlier rule {index}")
        try:
            predicate = _build_sql_outlier_predicate(selected_tables, source_columns, rule, params, alias=alias)
            label = f"OUTLIER::{rule_name}"
            label_param = _next_param_name(params, "rule_label")
            params[label_param] = label
            predicates.append(predicate)
            reason_parts.append(f"CASE WHEN {predicate} THEN :{label_param} ELSE NULL END")
            labels.append(label)
            applied_count += 1
        except Exception as exc:
            warnings.append(f"Skipped outlier rule '{rule_name}': {exc}")

    if not predicates:
        return "FALSE", "NULL::text", params, [], warnings, applied_count
    flag_expr = "(" + " OR ".join(predicates) + ")"
    reason_expr = "array_to_string(array_remove(ARRAY[" + ", ".join(reason_parts) + "]::text[], NULL), ', ')"
    return flag_expr, reason_expr, params, labels, warnings, applied_count

def _build_sql_workbench_query(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
    joined_sql: str,
) -> tuple[str, dict[str, Any], list[str], int, list[str], list[str], int]:
    feature_selects, feature_warnings, applied_feature_rule_count = _build_sql_feature_selects(
        payload.selected_tables,
        source_columns,
        payload.feature_rules,
        alias="src",
    )
    (
        outlier_expr,
        outlier_reason_expr,
        params,
        outlier_labels,
        outlier_warnings,
        applied_outlier_rule_count,
    ) = _build_sql_outlier_flag(
        payload.selected_tables,
        source_columns,
        payload.outlier_rules,
        alias="src",
    )

    select_parts = ["src.*"]
    select_parts.extend(feature_selects)
    select_parts.append(f"{outlier_expr} AS {_quote('__ml_sql_rule_flag')}")
    select_parts.append(f"{outlier_reason_expr} AS {_quote('__ml_sql_rule_reasons')}")
    sql = "WITH src AS (\n" + joined_sql + "\n)\nSELECT\n    " + ",\n    ".join(select_parts) + "\nFROM src"
    return (
        sql,
        params,
        outlier_labels,
        applied_feature_rule_count,
        feature_warnings,
        outlier_warnings,
        applied_outlier_rule_count,
    )

def _build_source_table_ref(table_name: str) -> str:
    schema = _quote(settings.source_db_schema)
    quoted_table = _quote(table_name)
    return f"{schema}.{quoted_table} AS {quoted_table}"

def _unqualified_source_table_ref(table_name: str) -> str:
    _validate_identifier(table_name, "table_name")
    return table_name

def _safe_date_literal(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        parsed = date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError):
        return None
    if parsed.year < 1 or parsed.year > 9999:
        return None
    return parsed.isoformat()

def _safe_sql_date_expr(expr: str) -> str:
    text_expr = f"btrim(CAST({expr} AS text))"
    return (
        "CASE "
        f"WHEN {expr} IS NULL THEN NULL::date "
        f"WHEN {text_expr} ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}' THEN left({text_expr}, 10)::date "
        "ELSE NULL::date "
        "END"
    )

def _date_range_filter_sql(
    table_name: str,
    column: dict[str, Any],
    *,
    from_date: str | None,
    to_date: str | None,
) -> str:
    column_name = str(column["column_name"])
    raw_expr = f"{_quote(table_name)}.{_quote(column_name)}"

    if _type_family(column.get("data_type")) == "datetime":
        parts = [f"{raw_expr} IS NOT NULL"]
        if from_date:
            parts.append(f"{raw_expr} >= DATE '{from_date}'")
        if to_date:
            parts.append(f"{raw_expr} < (DATE '{to_date}' + INTERVAL '1 day')")
    else:
        safe_date = _safe_sql_date_expr(raw_expr)
        parts = [f"{safe_date} IS NOT NULL"]
        if from_date:
            parts.append(f"{safe_date} >= DATE '{from_date}'")
        if to_date:
            parts.append(f"{safe_date} <= DATE '{to_date}'")

    return "(" + " AND ".join(parts) + ")"

def _preferred_date_filter_target(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
) -> tuple[str, dict[str, Any]] | None:
    preferred_columns = DATE_FILTER_COLUMN_PRIORITY
    selected_lookup = {
        table_name: {
            str(column.get("column_name", "")).strip().lower(): column
            for column in source_columns.get(table_name, [])
        }
        for table_name in payload.selected_tables
    }

    for preferred_name in preferred_columns:
        if "dak" in payload.selected_tables:
            dak_column = selected_lookup.get("dak", {}).get(preferred_name)
            if dak_column is not None:
                return "dak", dak_column

        for table_name in payload.selected_tables:
            selected_column = selected_lookup.get(table_name, {}).get(preferred_name)
            if selected_column is not None:
                return table_name, selected_column

    return None

def _list_date_filter_sql(payload: WorkbenchRunRequest, source_columns: dict[str, list[dict]]) -> str | None:
    from_date = _safe_date_literal(getattr(payload, "from_date", None))
    to_date = _safe_date_literal(getattr(payload, "to_date", None))
    if not from_date and not to_date:
        return None
    if from_date and to_date and from_date > to_date:
        return None

    target = _preferred_date_filter_target(payload, source_columns)
    if target is None:
        return None
    table_name, column = target
    return _date_range_filter_sql(
        table_name,
        column,
        from_date=from_date,
        to_date=to_date,
    )

def _validate_join_payload_tables(payload: WorkbenchRunRequest, source_columns: dict[str, list[dict]]) -> None:
    _validate_selected_tables(payload.selected_tables, source_columns)

    if len(payload.selected_tables) > 1 and len(payload.joins) == 0:
        raise ValueError(
            "When selecting 2 or more tables, at least 1 join configuration is required to connect them."
        )

    selected = set(payload.selected_tables)
    for index, join in enumerate(payload.joins, start=1):
        if join.left_table not in selected:
            raise ValueError(
                f"Join step {index}: left table '{join.left_table}' is not present in selected_tables."
            )
        if join.right_table not in selected:
            raise ValueError(
                f"Join step {index}: right table '{join.right_table}' is not present in selected_tables."
            )
        if join.left_table == join.right_table:
            raise ValueError(
                f"Join step {index}: self-joins are not supported by the current SQL join builder: "
                f"'{join.left_table}' -> '{join.right_table}'."
            )

def _ensure_source_column_index(conn, table_name: str, column_name: str) -> None:
    index_name = _index_name(settings.source_db_schema, table_name, column_name, "join")
    conn.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS {_quote(index_name)} "
            f"ON {_source_table_ref(table_name)} ({_quote(column_name)})"
        )
    )

def _ensure_join_indexes(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
) -> list[str]:
    if not payload.joins:
        return []

    warnings: list[str] = []
    indexed_columns: set[tuple[str, str]] = set()
    try:
        with _source_begin() as conn:
            for join in payload.joins:
                left_meta = _source_column_meta(source_columns, join.left_table, join.left_column)
                right_meta = _source_column_meta(source_columns, join.right_table, join.right_column)
                for table_name, column_name in (
                    (join.left_table, left_meta["column_name"]),
                    (join.right_table, right_meta["column_name"]),
                ):
                    key = (table_name, column_name)
                    if key in indexed_columns:
                        continue
                    _ensure_source_column_index(conn, table_name, column_name)
                    indexed_columns.add(key)
    except (SQLAlchemyError, ValueError) as exc:
        warnings.append(
            "Could not automatically create indexes for selected join columns. "
            f"The join will still run, but it may be slower: {exc}"
        )

    if indexed_columns:
        warnings.append(
            "Ensured PostgreSQL indexes exist for selected join columns: "
            + ", ".join(f"{table}.{column}" for table, column in sorted(indexed_columns))
            + "."
        )
    return warnings

def _ensure_date_filter_indexes(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
) -> list[str]:
    if not _safe_date_literal(getattr(payload, "from_date", None)) and not _safe_date_literal(getattr(payload, "to_date", None)):
        return []

    warnings: list[str] = []
    indexed_columns: set[tuple[str, str]] = set()
    try:
        with _source_begin() as conn:
            target = _preferred_date_filter_target(payload, source_columns)
            if target is None:
                warnings.append("No supported date filter column was found for the selected tables.")
            else:
                table_name, date_column = target
                column_name = str(date_column["column_name"])
                key = (table_name, column_name)
                if key not in indexed_columns:
                    _ensure_source_column_index(conn, table_name, column_name)
                    indexed_columns.add(key)
    except (SQLAlchemyError, ValueError) as exc:
        warnings.append(
            "Could not automatically create indexes for selected date filters. "
            f"The run will still continue, but date filtering may be slower: {exc}"
        )

    if indexed_columns:
        warnings.append(
            "Ensured PostgreSQL indexes exist for selected date filters: "
            + ", ".join(f"{table}.{column}" for table, column in sorted(indexed_columns))
            + "."
        )
    return warnings

def _build_date_sequence_anomaly_conditions(
    joined_tables: list[str],
    source_columns: dict[str, list[dict]],
) -> list[tuple[str, str]]:
    stage_exprs: list[tuple[str, list[tuple[str, str]]]] = []

    for stage_name, aliases in DATE_SEQUENCE_STAGE_ALIASES:
        exprs: list[tuple[str, str]] = []

        for table_name in joined_tables:
            table_columns = {
                str(column.get("column_name"))
                for column in source_columns.get(table_name, [])
            }

            for alias in aliases:
                if alias in table_columns:
                    exprs.append(
                        (
                            table_name,
                            _safe_sql_date_expr(f'base."{table_name}.{alias}"'),
                        )
                    )

        if not exprs:
            continue

        stage_exprs.append((stage_name, exprs))

    if len(stage_exprs) < 2:
        return []

    comparisons: list[tuple[str, str]] = []

    for index in range(len(stage_exprs) - 1):
        left_label, left_exprs = stage_exprs[index]
        right_label, right_exprs = stage_exprs[index + 1]
        predicate_parts: list[str] = []

        for left_table_name, left_expr in left_exprs:
            for right_table_name, right_expr in right_exprs:
                if _same_table_date_sequence_pair(left_label, right_label) and left_table_name != right_table_name:
                    continue
                predicate_parts.append(
                    f"""
                    (
                        ({left_expr}) IS NOT NULL
                        AND ({right_expr}) IS NOT NULL
                        AND ({left_expr}) > ({right_expr})
                    )
                    """
                )

        if not predicate_parts:
            continue

        comparisons.append(
            (
                "(" + " OR ".join(predicate_parts) + ")",
                f"Date sequence violated across processing stages: {left_label} after {right_label}",
            )
        )
    return comparisons

def _build_sql_anomaly_expressions(
    joined_tables: list[str],
    source_columns: dict[str, list[dict]],
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    conditions: list[tuple[str, str]] = []
    ctes: list[str] = []
    outer_joins: list[str] = []

    joined_set = set(joined_tables)

    def has_table(table: str) -> bool:
        return table in joined_set

    def table_cols(table: str) -> set[str]:
        return {
            str(column.get("column_name"))
            for column in source_columns.get(table, [])
        }

    # Duplicate invoice check: bill / gem_bill
    for table_name in ("bill", "gem_bill"):
        if not has_table(table_name):
            continue

        cols = table_cols(table_name)
        invoice_column = next(
            (col for col in ("invoice_no", "invoice_number") if col in cols),
            None,
        )

        if not invoice_column:
            continue

        required_cols = {"invoice_date", "record_status"}
        if not required_cols.issubset(cols):
            continue

        conditions.append((
            f"""
            (
                base."{table_name}.record_status" = 'V'
                AND base."{table_name}.{invoice_column}" IS NOT NULL
                AND base."{table_name}.invoice_date" IS NOT NULL
                AND (
                    SELECT COUNT(*)
                    FROM {_unqualified_source_table_ref(table_name)} b2
                    WHERE b2.record_status = 'V'
                      AND b2.{invoice_column} = base."{table_name}.{invoice_column}"
                      AND {_safe_sql_date_expr(f'b2.invoice_date')}
                          = {_safe_sql_date_expr(f'base."{table_name}.invoice_date"')}
                ) > 1
            )
            """,
            f"{table_name} has duplicate valid invoice number + invoice_date",
        ))

    # CMP scroll payment_reference_no must exist in ECS
    if has_table("cmp_scroll") and has_table("ecs"):
        conditions.append((
            f"""
            (
                base."cmp_scroll.payment_reference_no" IS NOT NULL
                AND base."cmp_scroll.cda_name" = 'CDA- Main Office Jabalpur'
                AND NOT EXISTS (
                    SELECT 1
                    FROM {_unqualified_source_table_ref("ecs")} e
                    WHERE e.payment_reference_no = base."cmp_scroll.payment_reference_no"
                )
            )
            """,
            "CMP scroll has payment_reference_no but not found in ECS",
        ))

    # cheque_slip ECS mode = 1 but ECS record exists
    if has_table("cheque_slip") and has_table("ecs"):
        conditions.append((
            f"""
            (
                base."cheque_slip.fk_ecs_payment_mode" = 1
                AND base."cheque_slip.fk_dak" IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM {_unqualified_source_table_ref("ecs")} e
                    WHERE e.fk_dak = base."cheque_slip.fk_dak"
                )
            )
            """,
            "Cheque slip ECS mode=1 but ECS record exists",
        ))

    # Cheque slip + schedule3 rules
    if has_table("cheque_slip") and has_table("schedule3"):
        ctes.append(
            f"""
            schedule3_by_dak AS (
                SELECT
                    fk_dak,
                    COUNT(*) AS schedule3_total_count,
                    COUNT(*) FILTER (WHERE record_status IN ('P', 'V')) AS schedule3_pv_count
                FROM {_unqualified_source_table_ref("schedule3")}
                WHERE fk_dak IS NOT NULL
                GROUP BY fk_dak
            )
            """
        )
        ctes.append(
            f"""
            cheque_slip_approved_by_dak AS (
                SELECT
                    fk_dak,
                    COUNT(*) FILTER (WHERE record_status = 'V' AND approved = true) AS cheque_slip_v_approved_count
                FROM {_unqualified_source_table_ref("cheque_slip")}
                WHERE fk_dak IS NOT NULL
                GROUP BY fk_dak
            )
            """
        )
        outer_joins.append(
            """
            LEFT JOIN schedule3_by_dak
                ON schedule3_by_dak.fk_dak = base."cheque_slip.fk_dak"
            """
        )
        outer_joins.append(
            """
            LEFT JOIN cheque_slip_approved_by_dak
                ON cheque_slip_approved_by_dak.fk_dak = base."cheque_slip.fk_dak"
            """
        )

        # cheque_slip V + approved false should not have schedule3 for same fk_dak
        conditions.append((
            f"""
            (
                base."cheque_slip.record_status" = 'V'
                AND base."cheque_slip.approved" = false
                AND base."cheque_slip.fk_dak" IS NOT NULL
                AND COALESCE(schedule3_by_dak.schedule3_total_count, 0) > 0
            )
            """,
            "Cheque slip record_status V and approved false but schedule3 exists for same fk_dak",
        ))

        # approved V cheque_slip count should match schedule3 P/V count for same fk_dak
        conditions.append((
            f"""
            (
                base."cheque_slip.record_status" = 'V'
                AND base."cheque_slip.approved" = true
                AND base."cheque_slip.fk_dak" IS NOT NULL
                AND COALESCE(cheque_slip_approved_by_dak.cheque_slip_v_approved_count, 0)
                    <> COALESCE(schedule3_by_dak.schedule3_pv_count, 0)
            )
            """,
            "Approved V cheque_slip count does not match schedule3 P/V count for same fk_dak",
        ))

    # approved cheque slip must have at least one approval officer/user
    if has_table("cheque_slip"):
        cheque_cols = table_cols("cheque_slip")

        officer_columns = [
            col for col in ("fk_aao", "fk_ao", "fk_go", "fk_auditor")
            if col in cheque_cols
        ]

        if officer_columns:
            all_officers_null = "\n                AND ".join(
                f'base."cheque_slip.{col}" IS NULL'
                for col in officer_columns
            )

            conditions.append((
                f"""
                (
                    base."cheque_slip.approved" = true
                    AND {all_officers_null}
                )
                """,
                f"Approved cheque slip but all approval columns are null: {', '.join(officer_columns)}",
            ))

    # Date sequence rule
    date_sequence_conditions = _build_date_sequence_anomaly_conditions(
        joined_tables,
        source_columns,
    )

    conditions.extend(date_sequence_conditions)

    return conditions, ctes, outer_joins

def _build_join_sql(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
    *,
    row_limit: int | None,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    _validate_join_payload_tables(payload, source_columns)

    warnings: list[str] = []
    selected_tables = payload.selected_tables
    first_table = selected_tables[0]
    joined_aliases: set[str] = {first_table}
    used_tables: set[str] = {first_table}
    select_clause = _build_join_select_list(selected_tables, source_columns)
    first_source_ref = _build_source_table_ref(first_table)
    from_clause = f"FROM {first_source_ref}"

    join_clauses: list[str] = []
    join_debug: list[dict[str, Any]] = []

    for index, join in enumerate(payload.joins, start=1):
        left_table = join.left_table
        right_table = join.right_table

        if left_table not in joined_aliases:
            raise ValueError(
                f"Join step {index} is invalid. Left table '{left_table}' is not yet part of the SQL join chain. "
                f"Start from '{first_table}' and chain joins in order."
            )
        if right_table in joined_aliases:
            raise ValueError(
                f"Join step {index} is invalid. Right table '{right_table}' is already part of the SQL join chain. "
                "Duplicate/repeated table joins are not supported by the current builder."
            )
        if right_table not in source_columns:
            raise ValueError(f"Join step {index}: right table '{right_table}' is not present in source schema.")

        left_meta = _source_column_meta(source_columns, left_table, join.left_column)
        right_meta = _source_column_meta(source_columns, right_table, join.right_column)
        left_column = left_meta["column_name"]
        right_column = right_meta["column_name"]
    
        join_condition=(
            f'{_quote(left_table)}.{_quote(left_column)} = ' 
            f'{_quote(right_table)}.{_quote(right_column)}'
        )
        join_mode = "raw_equal"

        right_source_ref = _build_source_table_ref(right_table)
        join_keyword = _sql_join_keyword(join.join_type)
        join_clause = f"{join_keyword} {right_source_ref} ON {join_condition}"
        join_clauses.append(join_clause)

        joined_aliases.add(right_table)
        used_tables.add(right_table)
        join_debug.append(
            {
                "step": index,
                "left_table": left_table,
                "right_table": right_table,
                "left_key": f"{left_table}.{left_column}",
                "right_key": f"{right_table}.{right_column}",
                "left_data_type": left_meta.get("data_type"),
                "right_data_type": right_meta.get("data_type"),
                "join_condition_mode": join_mode,
                "join_sql": join_clause,
            }
        )

    unjoined_tables = [table_name for table_name in selected_tables if table_name not in used_tables]
    if unjoined_tables:
        raise ValueError(
            "Some selected tables are not connected by the join configuration: "
            f"{unjoined_tables}. Add join steps for them or remove them from selected_tables."
        )

    base_sql = f"SELECT\n    {select_clause}\n{from_clause}"
    if join_clauses:
        base_sql += "\n" + "\n".join(join_clauses)

    filters = []
    list_date_filter = _list_date_filter_sql(payload, source_columns)
    if list_date_filter:
        filters.append(list_date_filter)

    sql = base_sql
    if filters:
        sql += "\nWHERE " + "\n  AND ".join(f"({item})" for item in filters)

    if row_limit and row_limit > 0:
        safe_limit = max(1, int(row_limit))
        sql += f"\nLIMIT {safe_limit}"
        warnings.append(
            "The final joined result is capped with LIMIT "
            f"{safe_limit}. This is a final-result limit, not a per-table source limit."
        )

    logger.info("Workbench SQL join query built: %s", sql)

    anomaly_conditions, anomaly_ctes, anomaly_outer_joins = _build_sql_anomaly_expressions(list(used_tables), source_columns)
    if anomaly_conditions:
        anomaly_sql = " OR ".join([f"({condition})" for condition, _reason in anomaly_conditions])
        anomaly_reason_parts: list[str] = []
        for condition, reason in anomaly_conditions:
            escaped_reason = str(reason).replace("'", "''")
            anomaly_reason_parts.append(
                f"CASE WHEN ({condition}) THEN '{escaped_reason}' ELSE NULL END"
            )
        anomaly_reason_sql = (
            "array_to_string(array_remove(ARRAY["
            + ", ".join(anomaly_reason_parts)
            + "]::text[], NULL), ', ')"
        )
        cte_sql = [f"joined_base AS (\n{sql}\n)"]
        cte_sql.extend(anomaly_ctes)
        cte_sql_text = ",\n".join(cte_sql)
        outer_join_sql = "\n".join(anomaly_outer_joins)
        sql = f"""WITH {cte_sql_text}
SELECT base.*,
       CASE WHEN {anomaly_sql} THEN TRUE ELSE FALSE END AS sql_rule_flag,
       {anomaly_reason_sql} AS sql_rule_reasons
FROM joined_base base"""
        if outer_join_sql:
            sql += "\n" + outer_join_sql

    return sql, join_debug, warnings

def _enrich_sql_join_debug(
    conn,
    join_debug: list[dict[str, Any]],
    source_row_counts: dict[str, int],
) -> list[dict[str, Any]]:
    if not ENABLE_EXPENSIVE_JOIN_DEBUG:
        return [
            {
                **item,
                "left_table_rows_estimate": source_row_counts.get(item["left_table"]),
                "right_table_rows_estimate": source_row_counts.get(item["right_table"]),
                "debug_mode": "light",
            }
            for item in join_debug
        ]

    enriched = []
    for item in join_debug:
        left_table = item["left_table"]
        right_table = item["right_table"]
        left_col = item["left_key"].split(".", 1)[1]
        right_col = item["right_key"].split(".", 1)[1]

        try:
            common_count = _sql_common_key_count(conn, left_table, left_col, right_table, right_col)
            common_sample = _sql_common_key_sample(conn, left_table, left_col, right_table, right_col)
        except Exception as exc:
            common_count = -1
            common_sample = [f"debug_error: {exc}"]

        try:
            left_raw_sample = _sql_table_column_samples(conn, left_table, left_col)
        except Exception as exc:
            left_raw_sample = [f"debug_error: {exc}"]

        try:
            right_raw_sample = _sql_table_column_samples(conn, right_table, right_col)
        except Exception as exc:
            right_raw_sample = [f"debug_error: {exc}"]

        enriched.append(
            {
                **item,
                "left_table_rows_estimate": source_row_counts.get(left_table),
                "right_table_rows_estimate": source_row_counts.get(right_table),
                "left_raw_sample": left_raw_sample,
                "right_raw_sample": right_raw_sample,
                "common_raw_key_count_table_level": common_count,
                "common_raw_key_sample_table_level": common_sample,
                "debug_mode": "full",
            }
        )
    return enriched

def _execute_sql_joined_frame(
    payload: WorkbenchRunRequest,
    *,
    for_preview: bool = False,
) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, Any]], list[str], str]:
    row_limit = PREVIEW_ROW_LIMIT if for_preview else None
    source_columns = _source_columns_map(payload.selected_tables)
    sql, join_debug, warnings = _build_join_sql(payload, source_columns, row_limit=row_limit)
    warnings.extend(_ensure_join_indexes(payload, source_columns))
    warnings.extend(_ensure_date_filter_indexes(payload, source_columns))

    source_row_counts: dict[str, int] = {}
    with _source_connect() as conn:
        for table_name in payload.selected_tables:
            source_row_counts[table_name] = _approx_table_row_count(conn, table_name)

        joined = pd.read_sql_query(text(sql), conn)
        join_debug = _enrich_sql_join_debug(conn, join_debug, source_row_counts)

    if joined.empty:
        previous_count = _previous_dataset_row_count(_dataset_table_name(payload.selected_tables))
        if previous_count > 0:
            warnings.append(
                "No new rows were available after skipping previously joined rows. "
                f"Previously saved rows: {previous_count}."
            )
        else:
            raise ValueError("The selected SQL join returned no rows. Review the chosen join keys and join type.")

    return joined, source_row_counts, join_debug, warnings, sql

def _execute_sql_workbench_frame(
    payload: WorkbenchRunRequest,
) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, Any]], list[str], str, list[str], int, list[str], int, str]:
    source_columns = _source_columns_map(payload.selected_tables)
    joined_sql, join_debug, warnings = _build_join_sql(payload, source_columns, row_limit=None)
    warnings.extend(_ensure_join_indexes(payload, source_columns))
    warnings.extend(_ensure_date_filter_indexes(payload, source_columns))
    (
        workbench_sql,
        params,
        outlier_labels,
        applied_feature_rule_count,
        feature_warnings,
        outlier_warnings,
        applied_outlier_rule_count,
    ) = _build_sql_workbench_query(payload, source_columns, joined_sql)

    source_row_counts: dict[str, int] = {}
    staging_table: str | None = None
    with _source_begin() as conn:
        for table_name in payload.selected_tables:
            source_row_counts[table_name] = _approx_table_row_count(conn, table_name)

        _log_workbench_query_plan(conn, workbench_sql, params)
        staging_table, staged_row_count = _materialize_workbench_temp_table(conn, workbench_sql, params)
        joined = _read_temp_scoring_frame(conn, staging_table, payload)
        join_debug = _enrich_sql_join_debug(conn, join_debug, source_row_counts)

    if staged_row_count == 0:
        _drop_workbench_temp_table(staging_table)
        previous_count = _previous_dataset_row_count(_dataset_table_name(payload.selected_tables))
        if previous_count > 0:
            warnings.append(
                "No new rows were available after skipping previously joined rows. "
                f"Previously saved rows: {previous_count}."
            )
        else:
            raise ValueError("The selected SQL join returned no rows. Review the chosen join keys and join type.")

    return (
        joined,
        source_row_counts,
        join_debug,
        warnings,
        joined_sql,
        outlier_labels,
        applied_feature_rule_count,
        feature_warnings + outlier_warnings,
        applied_outlier_rule_count,
        staging_table,
    )

def _workbench_temp_table_name() -> str:
    return f"tmp_ml_join_{uuid4().hex[:12]}"

def _workbench_temp_table_ref(temp_table: str) -> str:
    return f"{_quote(RESULT_SCHEMA)}.{_quote(temp_table)}"

def _materialize_workbench_temp_table(conn, workbench_sql: str, params: dict[str, Any]) -> tuple[str, int]:
    temp_table = _workbench_temp_table_name()
    temp_ref = _workbench_temp_table_ref(temp_table)
    started_at = time.monotonic()
    logger.info("Materializing workbench join into PostgreSQL staging table %s.%s", RESULT_SCHEMA, temp_table)
    conn.execute(
        text(
            f"""
            CREATE UNLOGGED TABLE {temp_ref} AS
            SELECT row_number() OVER () AS {_quote(TEMP_ROW_ID_COLUMN)}, src.*
            FROM (
            {workbench_sql}
            ) src
            """
        ),
        params,
    )
    conn.execute(text(f"CREATE INDEX ON {temp_ref} ({_quote(TEMP_ROW_ID_COLUMN)})"))
    conn.execute(text(f"ANALYZE {temp_ref}"))
    row_count = int(conn.execute(text(f"SELECT COUNT(*) FROM {temp_ref}")).scalar() or 0)
    logger.info(
        "Materialized %s rows into PostgreSQL staging table %s.%s in %.2fs",
        row_count,
        RESULT_SCHEMA,
        temp_table,
        time.monotonic() - started_at,
    )
    return temp_table, row_count

def _drop_workbench_temp_table(temp_table: str | None) -> None:
    if not temp_table:
        return
    try:
        with _source_begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {_workbench_temp_table_ref(temp_table)}"))
    except Exception:
        logger.warning("Unable to drop workbench staging table %s.%s", RESULT_SCHEMA, temp_table, exc_info=True)

def _temp_table_columns(conn, temp_table: str) -> list[str]:
    result = conn.execute(text(f"SELECT * FROM {_workbench_temp_table_ref(temp_table)} LIMIT 0"))
    return [str(column) for column in result.keys()]

def _log_workbench_query_plan(conn, workbench_sql: str, params: dict[str, Any]) -> None:
    explain_mode = "ANALYZE, BUFFERS, FORMAT TEXT" if settings.workbench_explain_analyze else "FORMAT TEXT"
    started_at = time.monotonic()
    try:
        rows = conn.execute(text(f"EXPLAIN ({explain_mode})\n{workbench_sql}"), params).fetchall()
    except Exception as exc:
        logger.warning("Unable to collect workbench query plan: %s", exc)
        return

    plan_text = "\n".join(str(row[0]) for row in rows)
    logger.info(
        "Workbench query plan collected in %.2fs using EXPLAIN (%s):\n%s",
        time.monotonic() - started_at,
        explain_mode,
        plan_text[:12000],
    )

def _read_temp_scoring_frame(
    conn,
    temp_table: str,
    payload: WorkbenchRunRequest,
) -> pd.DataFrame:
    available = set(_temp_table_columns(conn, temp_table))
    requested_columns = [
        TEMP_ROW_ID_COLUMN,
        USER_RULE_FLAG_COLUMN,
        USER_RULE_REASONS_COLUMN,
        SQL_RULE_FLAG_COLUMN,
        SQL_RULE_REASONS_COLUMN,
    ]
    requested_columns.extend(_feature_rule_aliases(payload.feature_rules))
    selected_columns = [column for column in requested_columns if column in available]
    if TEMP_ROW_ID_COLUMN not in selected_columns:
        selected_columns.insert(0, TEMP_ROW_ID_COLUMN)
    select_sql = ", ".join(_quote(column) for column in selected_columns)
    df = pd.read_sql_query(
        text(
            f"""
            SELECT {select_sql}
            FROM {_workbench_temp_table_ref(temp_table)}
            ORDER BY {_quote(TEMP_ROW_ID_COLUMN)}
            """
        ),
        conn,
    )
    if TEMP_ROW_ID_COLUMN in df.columns:
        df = df.set_index(TEMP_ROW_ID_COLUMN, drop=True)
    logger.info("Loaded %s scoring columns from temp table %s", len(df.columns), temp_table)
    return df

def _read_temp_anomaly_payload_frame(
    conn,
    temp_table: str,
    row_ids: list[int],
    payload: WorkbenchRunRequest,
) -> pd.DataFrame:
    if not row_ids:
        return pd.DataFrame()
    started_at = time.monotonic()
    available_columns = _temp_table_columns(conn, temp_table)
    excluded = {
        TEMP_ROW_ID_COLUMN,
        USER_RULE_FLAG_COLUMN,
        USER_RULE_REASONS_COLUMN,
        SQL_RULE_FLAG_COLUMN,
        SQL_RULE_REASONS_COLUMN,
        *_feature_rule_aliases(payload.feature_rules),
    }
    payload_columns = [column for column in available_columns if column not in excluded]
    select_columns = [_quote(TEMP_ROW_ID_COLUMN), *[_quote(column) for column in payload_columns]]
    frames: list[pd.DataFrame] = []
    chunk_size = 5000
    for start in range(0, len(row_ids), chunk_size):
        chunk = row_ids[start:start + chunk_size]
        rows_sql = ", ".join(f"({int(row_id)})" for row_id in chunk)
        frames.append(
            pd.read_sql_query(
                text(
                    f"""
                    SELECT {", ".join(select_columns)}
                    FROM {_workbench_temp_table_ref(temp_table)}
                    INNER JOIN (VALUES {rows_sql}) AS wanted({_quote(TEMP_ROW_ID_COLUMN)})
                      USING ({_quote(TEMP_ROW_ID_COLUMN)})
                    ORDER BY {_quote(TEMP_ROW_ID_COLUMN)}
                    """
                ),
                conn,
            )
        )
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if TEMP_ROW_ID_COLUMN in df.columns:
        df = df.set_index(TEMP_ROW_ID_COLUMN, drop=True)
    logger.info(
        "Loaded %s anomaly payload rows from temp table %s in %.2fs",
        len(df),
        temp_table,
        time.monotonic() - started_at,
    )
    return df

def _resolve_column(df: pd.DataFrame, column_name: str) -> str:
    if column_name in df.columns:
        return column_name
    matches = [column for column in df.columns if column.endswith(f".{column_name}")]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Column not found: {column_name}")
    raise ValueError(f"Column is ambiguous: {column_name}. Matches: {matches[:10]}")

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

def _safe_numeric_scalar(value: Any, default: float = 0.0) -> float:
    try:
        series = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(default)
        return float(series.iloc[0])
    except Exception:
        return float(default)

def _add_statistical_outlier_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [column for column in out.columns if _is_amount_like_column(str(column))]
    for column in numeric_cols[:20]:
        series = pd.to_numeric(out[column], errors="coerce")
        if pd.api.types.is_bool_dtype(series):
            series = series.astype(float)
        if series.notna().sum() < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out[f"iqr_flag::{column}"] = ((series < lower) | (series > upper)).astype(float)
    return out

def _prepare_isolation_forest_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    feature_frame = features.replace([np.inf, -np.inf], np.nan).copy()
    if feature_frame.empty:
        return feature_frame

    all_missing_cols = feature_frame.columns[~feature_frame.notna().any(axis=0)].tolist()
    if all_missing_cols:
        logger.info(
            "Retaining %d all-missing IF feature columns for imputation/indicator handling: %s",
            len(all_missing_cols),
            all_missing_cols[:10],
        )

    variance = feature_frame.var(numeric_only=True).fillna(0.0)
    zero_var_cols = variance[variance == 0].index.tolist()
    if zero_var_cols:
        logger.info(
            "Retaining %d zero-variance IF feature columns instead of dropping them: %s",
            len(zero_var_cols),
            zero_var_cols[:10],
        )

    return feature_frame

def _slug_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return token or "table"

def _dataset_table_name(_selected_tables: list[str]) -> str:
    # All runs currently write to the single shared result table.
    # The parameter is kept for API compatibility should per-run tables be added later.
    return RESULT_TABLE

def _result_table_ref(table_name: str) -> str:
    return f"{_quote(RESULT_SCHEMA)}.{_quote(table_name)}"

def _previous_dataset_row_count(dataset_table: str) -> int:
    _validate_identifier(dataset_table, "dataset_table")
    if not _result_table_exists(dataset_table, ):
        return 0
    with _source_connect() as conn:
        total_rows = conn.execute(text(f"SELECT COUNT(*) FROM {_result_table_ref(dataset_table)}")).scalar()
    return int(total_rows or 0)

def _next_dataset_run_id(dataset_table: str) -> int:
    _validate_identifier(dataset_table, "dataset_table")
    if not _result_table_exists(dataset_table):
        return 1
    if RUN_ID_COLUMN not in _result_table_columns(dataset_table):
        return 1
    sql = text(
        f"""
        SELECT COALESCE(MAX({_quote(RUN_ID_COLUMN)}), 0)
        FROM {_result_table_ref(dataset_table)}
        """
    )
    with _source_connect() as conn:
        current_max = conn.execute(sql).scalar()
    return int(current_max or 0) + 1


@lru_cache(maxsize=16)

def _result_table_exists(dataset_table: str) -> bool:
    if not dataset_table:
        return False
    sql = text(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = :schema_name
          AND table_name = :table_name
        LIMIT 1
        """
    )
    try:
        with _source_connect() as conn:
            return conn.execute(sql, {"schema_name": RESULT_SCHEMA, "table_name": dataset_table}).scalar() == 1
    except (SQLAlchemyError, ValueError) as exc:
        logger.warning("Unable to verify result table %s.%s: %s", RESULT_SCHEMA, dataset_table, exc)
        return False

def _clear_result_table_exists_cache(dataset_table: str | None = None) -> None:
    """Clear the cache for _result_table_exists. If dataset_table is provided, clear all caches."""
    try:
        _result_table_exists.cache_clear()
    except Exception:
        pass


@lru_cache(maxsize=16)

def _result_table_columns(dataset_table: str) -> frozenset[str]:
    if not dataset_table:
        return frozenset()
    sql = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema_name
          AND table_name = :table_name
        """
    )
    with _source_connect() as conn:
        return frozenset(
            str(row.column_name)
            for row in conn.execute(sql, {"schema_name": RESULT_SCHEMA, "table_name": dataset_table})
        )

def _clear_result_table_columns_cache(dataset_table: str | None = None) -> None:
    """Clear the cache for _result_table_columns. If dataset_table is provided, only that table's cache is cleared."""
    if dataset_table is None:
        _result_table_columns.cache_clear()
    else:
        try:
            _result_table_columns.cache_clear()
        except Exception:
            pass

def _feature_values_payload(
    row: pd.Series,
    signals: list[dict[str, Any]] | None = None,
    llm_if_reason: str | None = None,
    llm_if_reason_model: str | None = None,
    llm_if_reason_fallback: bool | None = None,
) -> dict[str, Any]:
    payload = {
        str(column): _safe_json(value)
        for column, value in row.items()
    }
    if signals:
        payload["__ml_explanation_signals"] = signals
    if llm_if_reason:
        payload["__ml_llm_if_reason"] = llm_if_reason
        payload["__ml_llm_if_reason_model"] = llm_if_reason_model
        payload["__ml_llm_if_reason_fallback"] = bool(llm_if_reason_fallback)
    return payload

def _coerce_insert_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value

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

def _build_dataset_frame(
    joined: pd.DataFrame,
    feature_frame: pd.DataFrame,
    payload: WorkbenchRunRequest,
    *,
    dataset_table: str,
    dataset_run_id: int,
    human_outlier_flag: pd.Series,
    human_reasons: list[str],
    human_reason_series: pd.Series | None,
    builtin_reason_series: pd.Series | None,
    isolation_scores: np.ndarray,
    ml_flag: pd.Series,
    ml_threshold: float,
    final_flag: pd.Series,
    filtered_joined_override: pd.DataFrame | None = None,
    explanation_signals_override: dict[Any, list[dict[str, Any]]] | None = None,
    llm_if_reasons_override: dict[Any, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    anomaly_mask = final_flag.astype(bool)
    filtered_feature_frame = feature_frame.loc[anomaly_mask]
    filtered_human_outlier_flag = human_outlier_flag.loc[anomaly_mask]
    filtered_ml_flag = ml_flag.loc[anomaly_mask]
    filtered_human_reasons = (
        human_reason_series.loc[anomaly_mask]
        if human_reason_series is not None
        else pd.Series(None, index=filtered_human_outlier_flag.index)
    )
    filtered_builtin_reasons = (
        builtin_reason_series.loc[anomaly_mask]
        if builtin_reason_series is not None
        else pd.Series(None, index=filtered_human_outlier_flag.index)
    )
    filtered_joined = filtered_joined_override if filtered_joined_override is not None else joined.loc[anomaly_mask]
    feature_aliases = set(_feature_rule_aliases(payload.feature_rules))

    logger.info(
        "Joined has %s rows; saving %s anomaly rows to %s.%s",
        len(joined),
        len(filtered_feature_frame),
        RESULT_SCHEMA,
        dataset_table,
    )

    default_rule_reason = human_reasons or ["Human-defined outlier rule matched"]
    explanation_signals = explanation_signals_override or {}
    llm_if_reasons = llm_if_reasons_override or {}

    dataset = pd.DataFrame(index=filtered_feature_frame.index)
    dataset.insert(0, FEATURE_NAME_COLUMN, _feature_name_for_tables(payload.selected_tables))
    dataset[FEATURE_VALUES_COLUMN] = [
        _feature_values_payload(
            row,
            explanation_signals.get(row_index),
            llm_if_reasons.get(row_index, {}).get("reason"),
            llm_if_reasons.get(row_index, {}).get("model"),
            llm_if_reasons.get(row_index, {}).get("fallback"),
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
    dataset[IF_SCORE_COLUMN] = [_round_storage_score(value) for value in np.asarray(isolation_scores)[anomaly_mask.to_numpy()]]
    dataset[ML_THRESHOLD_COLUMN] = _round_storage_score(ml_threshold)
    dataset[RUN_ID_COLUMN] = int(dataset_run_id)
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
