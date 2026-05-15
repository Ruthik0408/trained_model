from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import WorkbenchValidationError
from app.core.models import WorkbenchRun
from app.schemas.workbench_schema import WorkbenchRunRequest
from app.services.llm_reason_service import build_feature_explanation_signals
from app.services.workbench.constants import PREVIEW_ROW_LIMIT, RESULT_TABLE, logger
from app.services.workbench.ml_pipeline import (
    FeatureSelectionResult,
    _add_statistical_outlier_signals,
    _prepare_isolation_forest_feature_frame,
    _validate_isolation_forest_feature_frame,
)
from app.services.workbench.result_store import DatasetBuildInputs, _build_dataset_frame, _write_dataset_to_result
from app.services.workbench.source_db import _next_dataset_run_id, _source_begin
from app.services.workbench.sql_runtime import (
    _execute_sql_joined_frame,
    _execute_sql_workbench_frame,
    _feature_rule_aliases,
    _read_temp_anomaly_payload_frame,
    _resolve_column,
    _safe_date_literal,
)


@dataclass(frozen=True)
class WorkbenchExecutionState:
    joined: pd.DataFrame
    source_row_counts: dict[str, Any]
    join_debug: dict[str, Any]
    warnings: list[str]
    executed_sql: str
    human_reasons: list[str]
    applied_feature_rule_count: int
    applied_outlier_rule_count: int
    staging_table: str | None
    batch_id: str
    dataset_table: str
    dataset_run_id: int


@dataclass(frozen=True)
class RuleFlagState:
    user_rule_flag: pd.Series
    builtin_rule_flag: pd.Series
    human_outlier_flag: pd.Series
    human_reason_series: pd.Series | None
    builtin_reason_series: pd.Series | None


@dataclass(frozen=True)
class IsolationForestState:
    feature_frame: pd.DataFrame
    feature_selection: FeatureSelectionResult
    pipeline: Pipeline
    transformed: np.ndarray
    isolation_scores: np.ndarray
    ml_flag: pd.Series
    ml_threshold: float
    final_flag: pd.Series
    explanation_signals: dict[Any, list[dict[str, Any]]]
    filtered_joined: pd.DataFrame


def preview_workbench(payload: WorkbenchRunRequest) -> dict:
    joined, source_row_counts, join_debug, warnings, executed_sql = _execute_sql_joined_frame(payload, for_preview=True)
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
            "dataset_table": RESULT_TABLE,
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
    with _source_begin() as source_conn:
        execution = _execute_workbench(payload, source_conn)
        if execution.joined.empty:
            return _persist_empty_run(db, payload, execution)

        rule_flags = _extract_rule_flags(execution.joined)
        model_state = _run_isolation_forest(payload, execution, source_conn, rule_flags)

    amount_total = _calculate_amount_total(payload, model_state.filtered_joined)
    run = _create_run_record(db, payload, execution, rule_flags, model_state)
    dataset_frame = _build_dataset_frame(
        DatasetBuildInputs(
            joined=execution.joined,
            feature_frame=model_state.feature_frame,
            payload=payload,
            dataset_table=execution.dataset_table,
            dataset_run_id=execution.dataset_run_id,
            human_outlier_flag=rule_flags.human_outlier_flag,
            human_reasons=execution.human_reasons,
            human_reason_series=rule_flags.human_reason_series,
            builtin_reason_series=rule_flags.builtin_reason_series,
            isolation_scores=model_state.isolation_scores,
            ml_flag=model_state.ml_flag,
            ml_threshold=model_state.ml_threshold,
            final_flag=model_state.final_flag,
            filtered_joined_override=model_state.filtered_joined,
            explanation_signals_override=model_state.explanation_signals,
        )
    )
    dataset_storage = _write_dataset_to_result(dataset_frame, execution.dataset_table)
    builtin_reason_by_record_id = _build_builtin_reason_lookup(rule_flags, model_state, dataset_storage)
    logger.info(
        "Appended rows into dataset table %s; total rows now %s",
        execution.dataset_table,
        dataset_storage["row_count"],
    )

    run.metrics_json = _build_persisted_metrics(payload, execution, model_state, dataset_frame, dataset_storage, builtin_reason_by_record_id)

    db.commit()
    db.refresh(run)

    model_path = Path(settings.active_model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_state.pipeline, model_path)

    return _build_run_response(run, amount_total=amount_total)


def _execute_workbench(payload: WorkbenchRunRequest, source_conn) -> WorkbenchExecutionState:
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
    ) = _execute_sql_workbench_frame(payload, source_conn)
    warnings = [*join_warnings, *sql_pushdown_warnings]
    logger.info("Joined %d rows from %d tables", len(joined), len(payload.selected_tables))
    return WorkbenchExecutionState(
        joined=joined,
        source_row_counts=source_row_counts,
        join_debug=join_debug,
        warnings=warnings,
        executed_sql=executed_sql,
        human_reasons=human_reasons,
        applied_feature_rule_count=int(applied_feature_rule_count),
        applied_outlier_rule_count=int(applied_outlier_rule_count),
        staging_table=staging_table,
        batch_id=f"workbench_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        dataset_table=RESULT_TABLE,
        dataset_run_id=_next_dataset_run_id(RESULT_TABLE),
    )


def _persist_empty_run(db: Session, payload: WorkbenchRunRequest, execution: WorkbenchExecutionState) -> dict:
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
        metrics_json=_base_metrics(payload, execution),
        status="COMPLETED",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return _build_run_response(run, amount_total=0.0)


def _extract_rule_flags(joined: pd.DataFrame) -> RuleFlagState:
    user_rule_flag = _pop_rule_flag(joined, "__ml_sql_rule_flag")
    builtin_rule_flag = _pop_rule_flag(joined, "sql_rule_flag")
    human_reason_series = joined.pop("__ml_sql_rule_reasons") if "__ml_sql_rule_reasons" in joined.columns else None
    builtin_reason_series = joined.pop("sql_rule_reasons") if "sql_rule_reasons" in joined.columns else None
    if builtin_rule_flag.any() and builtin_reason_series is None:
        builtin_reason_series = builtin_rule_flag.map(
            lambda flag: "OUTLIER::Built-in SQL anomaly rule" if bool(flag) else None
        )
    return RuleFlagState(
        user_rule_flag=user_rule_flag,
        builtin_rule_flag=builtin_rule_flag,
        human_outlier_flag=user_rule_flag | builtin_rule_flag,
        human_reason_series=human_reason_series,
        builtin_reason_series=builtin_reason_series,
    )


def _pop_rule_flag(joined: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in joined.columns:
        return pd.Series(False, index=joined.index, dtype=bool)
    return joined.pop(column_name).map(lambda value: bool(value) if pd.notna(value) else False)


def _build_feature_frame(joined: pd.DataFrame, payload: WorkbenchRunRequest) -> tuple[pd.DataFrame, FeatureSelectionResult]:
    features = pd.DataFrame(index=joined.index)
    sql_feature_cols = [column for column in _feature_rule_aliases(payload.feature_rules) if column in joined.columns]
    if sql_feature_cols:
        extra = joined[sql_feature_cols].apply(pd.to_numeric, errors="coerce")
        features = pd.concat([features, extra], axis=1)
        logger.info("Added %d explicit feature-rule columns to IF input", len(sql_feature_cols))
    features = _add_statistical_outlier_signals(features)
    logger.info("Feature set has %d columns after statistical signals", len(features.columns))
    selection_result = _prepare_isolation_forest_feature_frame(features)
    feature_frame = _validate_isolation_forest_feature_frame(selection_result.feature_frame)
    logger.info("Feature frame has %d rows and %d columns", len(feature_frame), len(feature_frame.columns))
    return feature_frame, selection_result


def _build_isolation_forest_pipeline(contamination: float | str) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", IsolationForest(contamination=contamination, random_state=settings.random_state)),
        ]
    )


def _fit_and_score_isolation_forest(
    feature_frame: pd.DataFrame,
    pipeline: Pipeline,
    joined_index: pd.Index,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    try:
        pipeline.fit(feature_frame)
    except Exception as exc:
        raise WorkbenchValidationError(
            "Isolation Forest training failed on the engineered feature set.",
            suggestion="Reduce unstable feature rules, inspect missing-value-heavy columns, and retry with a simpler feature set.",
            details={
                "feature_count": int(feature_frame.shape[1]),
                "row_count": int(len(feature_frame.index)),
                "original_error": str(exc),
            },
        ) from exc

    try:
        transformed = pipeline.named_steps["scaler"].transform(
            pipeline.named_steps["imputer"].transform(feature_frame)
        )
        isolation_scores = -pipeline.named_steps["model"].score_samples(transformed)
        ml_flag = pd.Series(
            pipeline.named_steps["model"].predict(transformed) == -1,
            index=joined_index,
        )
    except Exception as exc:
        raise WorkbenchValidationError(
            "Isolation Forest scoring failed.",
            suggestion="Verify the engineered feature set still has consistent numeric columns after preprocessing.",
            details={
                "feature_count": int(feature_frame.shape[1]),
                "row_count": int(len(feature_frame.index)),
                "original_error": str(exc),
            },
        ) from exc
    return transformed, isolation_scores, ml_flag


def _resolve_ml_threshold(contamination: float | str, model: IsolationForest, isolation_scores: np.ndarray) -> float:
    if contamination == "auto":
        return float(-model.offset_)
    return float(np.quantile(isolation_scores, max(0.0, min(1.0, 1.0 - contamination))))


def _run_isolation_forest(
    payload: WorkbenchRunRequest,
    execution: WorkbenchExecutionState,
    source_conn,
    rule_flags: RuleFlagState,
) -> IsolationForestState:
    feature_frame, feature_selection = _build_feature_frame(execution.joined, payload)
    pipeline = _build_isolation_forest_pipeline(payload.contamination)
    transformed, isolation_scores, ml_flag = _fit_and_score_isolation_forest(
        feature_frame,
        pipeline,
        execution.joined.index,
    )
    ml_threshold = _resolve_ml_threshold(payload.contamination, pipeline.named_steps["model"], isolation_scores)
    final_flag = rule_flags.human_outlier_flag | ml_flag
    explanation_signals = build_feature_explanation_signals(
        pipeline,
        feature_frame,
        transformed,
        feature_frame.loc[ml_flag].index,
    )
    anomaly_row_ids = [int(row_id) for row_id in final_flag[final_flag].index.tolist()]
    filtered_joined = _read_temp_anomaly_payload_frame(source_conn, execution.staging_table, anomaly_row_ids, payload)
    return IsolationForestState(
        feature_frame=feature_frame,
        feature_selection=feature_selection,
        pipeline=pipeline,
        transformed=transformed,
        isolation_scores=isolation_scores,
        ml_flag=ml_flag,
        ml_threshold=ml_threshold,
        final_flag=final_flag,
        explanation_signals=explanation_signals,
        filtered_joined=filtered_joined,
    )


def _calculate_amount_total(payload: WorkbenchRunRequest, filtered_joined: pd.DataFrame) -> float:
    if not payload.amount_field or filtered_joined.empty:
        return 0.0
    amount_column = _resolve_column(filtered_joined, payload.amount_field)
    return float(pd.to_numeric(filtered_joined[amount_column], errors="coerce").fillna(0).sum())



def _create_run_record(
    db: Session,
    payload: WorkbenchRunRequest,
    execution: WorkbenchExecutionState,
    rule_flags: RuleFlagState,
    model_state: IsolationForestState,
) -> WorkbenchRun:
    run = WorkbenchRun(
        run_name=payload.run_name,
        source_tables_json=payload.selected_tables,
        join_config_json=[item.model_dump() for item in payload.joins],
        outlier_rules_json=[item.model_dump() for item in payload.outlier_rules],
        feature_rules_json=[item.model_dump() for item in payload.feature_rules],
        amount_field=payload.amount_field,
        total_rows=int(len(execution.joined)),
        human_outlier_count=int(rule_flags.human_outlier_flag.sum()),
        ml_anomaly_count=int(model_state.ml_flag.sum()),
        final_anomaly_count=int(model_state.final_flag.sum()),
        selected_model="IsolationForest",
        metrics_json={
            **_base_metrics(payload, execution),
            "contamination": payload.contamination,
            "feature_count": int(model_state.feature_frame.shape[1]),
            "selected_feature_columns": model_state.feature_selection.selected_columns,
            "dropped_all_missing_feature_columns": model_state.feature_selection.dropped_all_missing_columns,
            "dropped_constant_feature_columns": model_state.feature_selection.dropped_constant_columns,
            "dropped_low_score_feature_columns": model_state.feature_selection.dropped_low_score_columns,
            "feature_scores": model_state.feature_selection.feature_scores,
        },
        status="COMPLETED",
    )
    db.add(run)
    db.flush()
    return run


def _base_metrics(payload: WorkbenchRunRequest, execution: WorkbenchExecutionState) -> dict[str, Any]:
    return {
        "batch_id": execution.batch_id,
        "selected_tables": payload.selected_tables,
        "source_row_counts": execution.source_row_counts,
        "join_debug": execution.join_debug,
        "executed_join_sql": execution.executed_sql,
        "dataset_table": execution.dataset_table,
        "ml_run_id": execution.dataset_run_id,
        "from_date": _safe_date_literal(payload.from_date),
        "to_date": _safe_date_literal(payload.to_date),
        "requested_outlier_rule_count": int(len(payload.outlier_rules)),
        "requested_feature_rule_count": int(len(payload.feature_rules)),
        "applied_outlier_rule_count": execution.applied_outlier_rule_count,
        "applied_feature_rule_count": execution.applied_feature_rule_count,
        "warnings": execution.warnings,
    }


def _build_builtin_reason_lookup(
    rule_flags: RuleFlagState,
    model_state: IsolationForestState,
    dataset_storage: dict[str, Any],
) -> dict[str, str]:
    if rule_flags.builtin_reason_series is None:
        return {}
    filtered_builtin_reasons = rule_flags.builtin_reason_series.loc[model_state.final_flag]
    inserted_ids = dataset_storage.get("inserted_ids") or []
    return {
        str(record_id): str(reason).strip()
        for record_id, reason in zip(inserted_ids, filtered_builtin_reasons)
        if pd.notna(reason) and str(reason).strip()
    }


def _build_persisted_metrics(
    payload: WorkbenchRunRequest,
    execution: WorkbenchExecutionState,
    model_state: IsolationForestState,
    dataset_frame: pd.DataFrame,
    dataset_storage: dict[str, Any],
    builtin_reason_by_record_id: dict[str, str],
) -> dict[str, Any]:
    return {
        **_base_metrics(payload, execution),
        "contamination": payload.contamination,
        "feature_count": int(model_state.feature_frame.shape[1]),
        "selected_feature_columns": model_state.feature_selection.selected_columns,
        "dropped_all_missing_feature_columns": model_state.feature_selection.dropped_all_missing_columns,
        "dropped_constant_feature_columns": model_state.feature_selection.dropped_constant_columns,
        "dropped_low_score_feature_columns": model_state.feature_selection.dropped_low_score_columns,
        "feature_scores": model_state.feature_selection.feature_scores,
        "join_execution_mode": "postgres_sql",
        "new_rows_written": int(len(dataset_frame)),
        "builtin_reason_by_record_id": builtin_reason_by_record_id,
        "joined_result_table": dataset_storage["table_name"],
        "joined_result_row_count": dataset_storage["row_count"],
        "joined_result_column_count": dataset_storage["column_count"],
    }


def _build_run_response(run: WorkbenchRun, amount_total: float) -> dict[str, Any]:
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
