"""End-to-end run execution for the anomaly workbench."""
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from app.core.errors import WorkbenchValidationError
from app.core.models import WorkbenchRun
from app.schemas.workbench_schema import WorkbenchRunRequest
from app.services.reason_service import build_feature_explanation_signals
from app.services.workbench.constants import (
    RESULT_TABLE,
    SQL_RULE_EVIDENCE_COLUMN,
    logger,
)
from app.services.workbench.result_store import (
    DatasetBuildInputs,
    _build_dataset_frame,
    _review_payload_cache_entries,
    _write_dataset_to_result,
)
from app.services.workbench.source_db import _next_dataset_run_id, _source_begin
from app.services.workbench.sql_runtime import (
    _execute_sql_workbench_frame,
    _feature_rule_aliases,
    _resolve_column,
    _safe_date_literal,
)
from app.services.workbench.saved_model_inference import (
    FeatureSelectionResult,
    build_saved_model_feature_frame,
    load_saved_model_artifact,
    score_with_saved_model,
)
from app.services.workbench.valkey_artifacts import (
    denormalize_explanation_signals,
    deserialize_dataframe,
    deserialize_ndarray,
    deserialize_pipeline,
    deserialize_series,
    get_isolation_forest_artifact,
    get_run_execution_artifact,
    normalize_explanation_signals,
    serialize_dataframe,
    serialize_ndarray,
    serialize_pipeline,
    serialize_series,
    set_review_payload_artifact,
    set_isolation_forest_artifact,
    set_run_execution_artifact,
)
from app.services.workbench.utils import _select_series_column


@dataclass(frozen=True)
class WorkbenchExecutionState:
    """Carries the SQL execution outputs needed by later scoring and persistence steps."""
    joined: pd.DataFrame
    source_row_counts: dict[str, Any]
    join_debug: dict[str, Any]
    warnings: list[str]
    executed_sql: str
    user_reasons: list[str]
    applied_feature_rule_count: int
    applied_user_rule_count: int
    batch_id: str
    dataset_table: str
    dataset_run_id: int


@dataclass(frozen=True)
class RuleFlagState:
    """Carries rule-trigger flags and optional reason/evidence series for a joined frame."""
    user_rule_flag: pd.Series
    default_rule_flag: pd.Series
    combined_rule_flag: pd.Series
    user_reason_series: pd.Series | None
    default_reason_series: pd.Series | None
    default_evidence_series: pd.Series | None = None


@dataclass(frozen=True)
class IsolationForestState:
    """Carries model outputs and filtered anomaly rows after scoring completes."""
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

def run_workbench(db: Session, payload: WorkbenchRunRequest) -> dict:
    """Execute the full anomaly run, persist outputs, and return the run summary."""
    run_started_at = time.monotonic()
    logger.info(
        "Starting workbench run for tables: %s, feature_rules: %s, user_rules: %s",
        payload.selected_tables,
        len(payload.feature_rules),
        len(payload.user_rules),
    )

    cached_execution_artifact = get_run_execution_artifact(payload)
    cached_if_artifact = get_isolation_forest_artifact(payload)

    if cached_execution_artifact is not None and cached_if_artifact is not None:
        cache_reuse_started_at = time.monotonic()
        logger.info(
            "Run cache HIT: reusing Valkey execution and IF artifacts for tables=%s",
            payload.selected_tables,
        )
        execution = _execution_state_from_artifact(payload, cached_execution_artifact)
        if execution.joined.empty:
            return _persist_empty_run(db, payload, execution)
        rule_flags = _extract_rule_flags(execution.joined)
        model_state = _isolation_forest_state_from_artifact(cached_if_artifact)
        logger.info(
            "Run cache reuse completed in %.2fs for tables=%s",
            time.monotonic() - cache_reuse_started_at,
            payload.selected_tables,
        )
    else:
        fresh_run_started_at = time.monotonic()
        logger.info(
            "Run cache MISS: executing fresh SQL workbench pipeline for tables=%s",
            payload.selected_tables,
        )
        with _source_begin() as source_conn:
            execution = _execute_workbench(payload, source_conn)

            if execution.joined.empty:
                return _persist_empty_run(db, payload, execution)

            rule_flags = _extract_rule_flags(execution.joined)
            model_state = _run_isolation_forest(payload, execution, rule_flags)

        set_run_execution_artifact(payload, _execution_artifact_from_state(execution))
        set_isolation_forest_artifact(payload, _isolation_forest_artifact_from_state(model_state))
        logger.info(
            "Fresh SQL workbench pipeline completed in %.2fs for tables=%s",
            time.monotonic() - fresh_run_started_at,
            payload.selected_tables,
        )

    amount_started_at = time.monotonic()
    amount_total = _calculate_amount_total(payload, model_state.filtered_joined)
    logger.info(
        "Amount total calculation completed in %.2fs for tables=%s",
        time.monotonic() - amount_started_at,
        payload.selected_tables,
    )

    dataset_build_started_at = time.monotonic()
    dataset_frame = _build_dataset_frame(
        DatasetBuildInputs(
            joined=execution.joined,
            feature_frame=model_state.feature_frame,
            payload=payload,
            dataset_table=execution.dataset_table,
            dataset_run_id=execution.dataset_run_id,
            combined_rule_flag=rule_flags.combined_rule_flag,
            user_reasons=execution.user_reasons,
            user_reason_series=rule_flags.user_reason_series,
            default_reason_series=rule_flags.default_reason_series,
            default_evidence_series=rule_flags.default_evidence_series,
            isolation_scores=model_state.isolation_scores,
            ml_flag=model_state.ml_flag,
            ml_threshold=model_state.ml_threshold,
            final_flag=model_state.final_flag,
            filtered_joined_override=model_state.filtered_joined,
            explanation_signals_override=model_state.explanation_signals,
        )
    )
    logger.info(
        "ML_Features dataset frame built in %.2fs with %d rows and %d columns",
        time.monotonic() - dataset_build_started_at,
        len(dataset_frame),
        len(dataset_frame.columns),
    )

    persist_run_started_at = time.monotonic()
    run = _create_run_record(db, payload, execution, rule_flags, model_state)
    logger.info(
        "WorkbenchRun record persisted in %.2fs with run_id=%s",
        time.monotonic() - persist_run_started_at,
        run.run_id,
    )

    write_started_at = time.monotonic()
    dataset_storage = _write_dataset_to_result(dataset_frame, execution.dataset_table)
    logger.info(
        "ML_Features write completed in %.2fs with %d inserted ids",
        time.monotonic() - write_started_at,
        len(dataset_storage.get("inserted_ids") or []),
    )

    builtin_reason_started_at = time.monotonic()
    builtin_reason_by_record_id = _build_builtin_reason_lookup(
        rule_flags,
        model_state,
        dataset_storage,
    )
    logger.info(
        "Built builtin reason lookup in %.2fs with %d mapped rows",
        time.monotonic() - builtin_reason_started_at,
        len(builtin_reason_by_record_id),
    )

    logger.info(
        "Appended rows into dataset table %s; total rows now %s",
        execution.dataset_table,
        dataset_storage["row_count"],
    )

    feature_aliases = set(_feature_rule_aliases(payload.feature_rules))
    review_payload_started_at = time.monotonic()
    set_review_payload_artifact(
        int(run.run_id),
        {
            "dataset_table": execution.dataset_table,
            "rows": _review_payload_cache_entries(
                model_state.filtered_joined,
                feature_aliases,
                dataset_storage.get("inserted_ids") or [],
                payload.selected_tables,
            ),
        },
    )
    logger.info(
        "Stored review payload artifact in Valkey in %.2fs for run_id=%s",
        time.monotonic() - review_payload_started_at,
        run.run_id,
    )

    metrics_started_at = time.monotonic()
    run.metrics_json = _build_persisted_metrics(
        payload,
        execution,
        model_state,
        dataset_frame,
        dataset_storage,
        builtin_reason_by_record_id,
    )
    logger.info(
        "Metrics payload prepared in %.2fs for run_id=%s",
        time.monotonic() - metrics_started_at,
        run.run_id,
    )

    commit_started_at = time.monotonic()
    db.commit()
    db.refresh(run)
    logger.info(
        "Run commit/refresh completed in %.2fs for run_id=%s",
        time.monotonic() - commit_started_at,
        run.run_id,
    )

    logger.info(
        "Workbench run FINISHED in %.2fs for run_id=%s tables=%s final_anomalies=%d",
        time.monotonic() - run_started_at,
        run.run_id,
        payload.selected_tables,
        int(model_state.final_flag.sum()) if hasattr(model_state.final_flag, "sum") else 0,
    )

    return _build_run_response(run, amount_total=amount_total)


def _execute_workbench(payload: WorkbenchRunRequest, source_conn) -> WorkbenchExecutionState:
    """Execute the SQL workbench query inside the source transaction and normalize outputs."""
    execute_started_at = time.monotonic()
    (
        joined,
        source_row_counts,
        join_debug,
        join_warnings,
        executed_sql,
        user_reasons,
        applied_feature_rule_count,
        sql_pushdown_warnings,
        applied_user_rule_count,
    ) = _execute_sql_workbench_frame(payload, source_conn)

    warnings = [*join_warnings, *sql_pushdown_warnings]

    logger.info(
        "SQL workbench execution completed in %.2fs; joined %d rows from %d tables; applied_feature_rules=%d applied_user_rules=%d",
        time.monotonic() - execute_started_at,
        len(joined),
        len(payload.selected_tables),
        int(applied_feature_rule_count),
        int(applied_user_rule_count),
    )

    return WorkbenchExecutionState(
        joined=joined,
        source_row_counts=source_row_counts,
        join_debug=join_debug,
        warnings=warnings,
        executed_sql=executed_sql,
        user_reasons=user_reasons,
        applied_feature_rule_count=int(applied_feature_rule_count),
        applied_user_rule_count=int(applied_user_rule_count),
        batch_id=f"workbench_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        dataset_table=RESULT_TABLE,
        dataset_run_id=_next_dataset_run_id(RESULT_TABLE),
    )


def _execution_artifact_from_state(
    execution: WorkbenchExecutionState,
) -> dict[str, Any]:
    """Serialize execution state into a Valkey-safe artifact payload."""
    return {
        "joined": serialize_dataframe(execution.joined),
        "source_row_counts": execution.source_row_counts,
        "join_debug": [dict(item) for item in execution.join_debug],
        "warnings": list(execution.warnings),
        "executed_sql": execution.executed_sql,
        "user_reasons": list(execution.user_reasons),
        "applied_feature_rule_count": execution.applied_feature_rule_count,
        "applied_user_rule_count": execution.applied_user_rule_count,
        "dataset_table": execution.dataset_table,
    }


def _execution_state_from_artifact(
    payload: WorkbenchRunRequest,
    artifact: dict[str, Any],
) -> WorkbenchExecutionState:
    """Rehydrate cached execution state so a run can skip the expensive SQL phase."""
    return WorkbenchExecutionState(
        joined=deserialize_dataframe(artifact["joined"]),
        source_row_counts={
            str(key): int(value)
            for key, value in (artifact.get("source_row_counts") or {}).items()
        },
        join_debug=[dict(item) for item in artifact.get("join_debug") or []],
        warnings=list(artifact.get("warnings") or []),
        executed_sql=str(artifact.get("executed_sql") or ""),
        user_reasons=[str(item) for item in artifact.get("user_reasons") or []],
        applied_feature_rule_count=int(artifact.get("applied_feature_rule_count") or 0),
        applied_user_rule_count=int(artifact.get("applied_user_rule_count") or 0),
        batch_id=f"workbench_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        dataset_table=str(artifact.get("dataset_table") or RESULT_TABLE),
        dataset_run_id=_next_dataset_run_id(RESULT_TABLE),
    )


def _persist_empty_run(
    db: Session,
    payload: WorkbenchRunRequest,
    execution: WorkbenchExecutionState,
) -> dict:
    """Persist a completed run record when the SQL stage produced zero rows."""
    run = WorkbenchRun(
        run_name=payload.run_name,
        source_tables_json=payload.selected_tables,
        join_config_json=[item.model_dump() for item in payload.joins],
        feature_rules_json=[item.model_dump() for item in payload.feature_rules],
        amount_field=payload.amount_field,
        total_rows=0,
        user_rule_count=0,
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
    """Split SQL-produced rule columns away from the joined frame and normalize them."""
    user_rule_flag = _pop_rule_flag(joined, "__ml_sql_rule_flag")
    default_rule_flag = _pop_rule_flag(joined, "sql_rule_flag")

    user_reason_series = (
        joined.pop("__ml_sql_rule_reasons")
        if "__ml_sql_rule_reasons" in joined.columns
        else None
    )

    default_reason_series = (
        joined.pop("sql_rule_reasons")
        if "sql_rule_reasons" in joined.columns
        else None
    )

    default_evidence_series = (
        joined.pop(SQL_RULE_EVIDENCE_COLUMN)
        if SQL_RULE_EVIDENCE_COLUMN in joined.columns
        else None
    )

    if default_rule_flag.any() and default_reason_series is None:
        default_reason_series = default_rule_flag.map(
            lambda flag: "RULE::Default SQL rule" if bool(flag) else None
        )

    return RuleFlagState(
        user_rule_flag=user_rule_flag,
        default_rule_flag=default_rule_flag,
        combined_rule_flag=user_rule_flag | default_rule_flag,
        user_reason_series=user_reason_series,
        default_reason_series=default_reason_series,
        default_evidence_series=default_evidence_series,
    )


def _pop_rule_flag(joined: pd.DataFrame, column_name: str) -> pd.Series:
    """Remove one boolean-like rule flag column from the joined frame."""
    if column_name not in joined.columns:
        return pd.Series(False, index=joined.index, dtype=bool)

    return joined.pop(column_name).map(
        lambda value: bool(value) if pd.notna(value) else False
    )


def _run_isolation_forest(
    payload: WorkbenchRunRequest,
    execution: WorkbenchExecutionState,
    rule_flags: RuleFlagState,
) -> IsolationForestState:
    """Load the saved model, score candidate rows, and build explanation signals."""
    if_started_at = time.monotonic()
    logger.info("Saved Isolation Forest inference START for tables=%s", payload.selected_tables)

    artifact = load_saved_model_artifact(payload)
    feature_frame, feature_selection = build_saved_model_feature_frame(
        execution.joined,
        artifact,
    )
    combined_rule_flag = rule_flags.combined_rule_flag.reindex(
        feature_frame.index,
        fill_value=False,
    ).astype(bool)
    eligible_feature_frame = feature_frame.loc[~combined_rule_flag]
    logger.info(
        "Saved model feature preparation completed for dataset=%s: %d selected columns; scored_rows=%d rule_rows_skipped=%d; sample=%s",
        artifact.get("dataset_name"),
        len(feature_selection.selected_columns),
        len(eligible_feature_frame.index),
        int(combined_rule_flag.sum()),
        feature_selection.selected_columns[:20],
    )

    transformed, eligible_scores, eligible_ml_flag, ml_threshold = score_with_saved_model(
        eligible_feature_frame,
        artifact,
    )
    pipeline = artifact["pipeline"]
    isolation_scores = np.full(len(feature_frame.index), np.nan, dtype=float)
    if len(eligible_scores):
        isolation_scores[feature_frame.index.get_indexer(eligible_feature_frame.index)] = eligible_scores
    ml_flag = pd.Series(False, index=feature_frame.index, dtype=bool)
    if not eligible_ml_flag.empty:
        ml_flag.loc[eligible_ml_flag.index] = eligible_ml_flag.astype(bool)
    logger.info(
        "Saved Isolation Forest scoring completed: eligible_rows=%d columns=%d ml_anomalies=%d",
        len(eligible_feature_frame.index),
        len(eligible_feature_frame.columns),
        int(ml_flag.sum()),
    )

    final_flag = combined_rule_flag | ml_flag

    ml_feature_index = eligible_feature_frame.index[eligible_ml_flag]

    explanation_signals = build_feature_explanation_signals(
        pipeline,
        eligible_feature_frame,
        transformed,
        ml_feature_index,
        transformed_feature_labels=[
            str(label)
            for label in artifact.get("transformed_feature_names") or []
        ],
    )
    logger.info(
        "Saved model inference skipped retraining and scored %d ML anomaly rows with %d explanation signal sets",
        len(ml_feature_index),
        len(explanation_signals),
    )

    filtered_joined = execution.joined.reindex(final_flag[final_flag].index).copy()

    logger.info(
        "Saved Isolation Forest inference FINISHED in %.2fs; total_final_anomalies=%d payload_rows=%d threshold=%.4f",
        time.monotonic() - if_started_at,
        int(final_flag.sum()),
        len(filtered_joined),
        float(ml_threshold),
    )

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


def _isolation_forest_artifact_from_state(
    model_state: IsolationForestState,
) -> dict[str, Any]:
    """Serialize model outputs so repeated runs can reuse scoring artifacts."""
    return {
        "feature_frame": serialize_dataframe(model_state.feature_frame),
        "feature_selection": {
            "selected_columns": list(model_state.feature_selection.selected_columns),
            "dropped_all_missing_columns": list(
                model_state.feature_selection.dropped_all_missing_columns
            ),
            "dropped_constant_columns": list(
                model_state.feature_selection.dropped_constant_columns
            ),
        },
        "pipeline": serialize_pipeline(model_state.pipeline),
        "transformed": serialize_ndarray(model_state.transformed),
        "isolation_scores": serialize_ndarray(model_state.isolation_scores),
        "ml_flag": serialize_series(model_state.ml_flag),
        "ml_threshold": float(model_state.ml_threshold),
        "final_flag": serialize_series(model_state.final_flag),
        "explanation_signals": normalize_explanation_signals(
            model_state.explanation_signals
        ),
        "filtered_joined": serialize_dataframe(model_state.filtered_joined),
    }


def _isolation_forest_state_from_artifact(
    artifact: dict[str, Any],
) -> IsolationForestState:
    """Rehydrate cached model outputs back into the in-process scoring state."""
    feature_selection = artifact.get("feature_selection") or {}
    feature_frame = deserialize_dataframe(artifact["feature_frame"])
    return IsolationForestState(
        feature_frame=feature_frame,
        feature_selection=FeatureSelectionResult(
            feature_frame=feature_frame,
            selected_columns=[
                str(column)
                for column in feature_selection.get("selected_columns") or []
            ],
            dropped_all_missing_columns=[
                str(column)
                for column in feature_selection.get("dropped_all_missing_columns") or []
            ],
            dropped_constant_columns=[
                str(column)
                for column in feature_selection.get("dropped_constant_columns") or []
            ],
        ),
        pipeline=deserialize_pipeline(artifact["pipeline"]),
        transformed=deserialize_ndarray(artifact["transformed"]),
        isolation_scores=deserialize_ndarray(artifact["isolation_scores"]),
        ml_flag=deserialize_series(artifact["ml_flag"]).astype(bool),
        ml_threshold=float(artifact.get("ml_threshold") or 0.0),
        final_flag=deserialize_series(artifact["final_flag"]).astype(bool),
        explanation_signals=denormalize_explanation_signals(
            artifact.get("explanation_signals")
        ),
        filtered_joined=deserialize_dataframe(artifact["filtered_joined"]),
    )


def _calculate_amount_total(
    payload: WorkbenchRunRequest,
    filtered_joined: pd.DataFrame,
) -> float:
    """Sum the configured amount field across the filtered anomaly rows."""
    if not payload.amount_field or filtered_joined.empty:
        return 0.0

    if payload.amount_field in filtered_joined.columns:
        amount_column = payload.amount_field
    else:
        amount_column = _resolve_column(filtered_joined, payload.amount_field)
    if amount_column is None or amount_column not in filtered_joined.columns:
        logger.warning(
            "Amount field '%s' not found in filtered results. Returning 0.0",
            payload.amount_field,
        )
        return 0.0

    amount_series = _select_series_column(filtered_joined, amount_column)

    return float(
        pd.to_numeric(amount_series, errors="coerce")
        .fillna(0)
        .sum()
    )


def _create_run_record(
    db: Session,
    payload: WorkbenchRunRequest,
    execution: WorkbenchExecutionState,
    rule_flags: RuleFlagState,
    model_state: IsolationForestState,
) -> WorkbenchRun:
    """Create and flush the app-database metadata row for a completed workbench run."""
    reviewable_rule_count = int(
        rule_flags.combined_rule_flag.reindex(
            model_state.final_flag.index,
            fill_value=False,
        ).sum()
    )
    run = WorkbenchRun(
        run_name=payload.run_name,
        source_tables_json=payload.selected_tables,
        join_config_json=[item.model_dump() for item in payload.joins],
        feature_rules_json=[item.model_dump() for item in payload.feature_rules],
        amount_field=payload.amount_field,
        total_rows=int(len(execution.joined)),
        user_rule_count=reviewable_rule_count,
        ml_anomaly_count=int(model_state.ml_flag.sum()),
        final_anomaly_count=int(model_state.final_flag.sum()),
        selected_model="IsolationForest",
        metrics_json={
            **_base_metrics(payload, execution),
            "contamination": payload.contamination,
            "feature_count": int(model_state.feature_frame.shape[1]),
            "selected_feature_columns": model_state.feature_selection.selected_columns,
            "dropped_all_missing_feature_columns": (
                model_state.feature_selection.dropped_all_missing_columns
            ),
            "dropped_constant_feature_columns": (
                model_state.feature_selection.dropped_constant_columns
            ),
        },
        status="COMPLETED",
    )

    db.add(run)
    db.flush()

    return run


def _base_metrics(
    payload: WorkbenchRunRequest,
    execution: WorkbenchExecutionState,
) -> dict[str, Any]:
    """Build the base metrics payload shared by empty and non-empty run records."""
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
        "requested_user_rule_count": int(len(payload.user_rules)),
        "requested_feature_rule_count": int(len(payload.feature_rules)),
        "applied_user_rule_count": execution.applied_user_rule_count,
        "applied_feature_rule_count": execution.applied_feature_rule_count,
        "warnings": execution.warnings,
    }


def _build_builtin_reason_lookup(
    rule_flags: RuleFlagState,
    model_state: IsolationForestState,
    dataset_storage: dict[str, Any],
) -> dict[str, str]:
    """Map inserted dataset record ids to the generated built-in reason text."""
    if rule_flags.default_reason_series is None:
        return {}

    # Get reasons aligned to the final_flag rows (preserving original index)
    filtered_builtin_reasons = rule_flags.default_reason_series.reindex(
        model_state.final_flag.index
    ).loc[model_state.final_flag]

    inserted_ids = dataset_storage.get("inserted_ids") or []

    # Validate that lengths match to avoid silent data loss
    if len(inserted_ids) != len(filtered_builtin_reasons):
        raise ValueError(
            f"Mismatch: {len(inserted_ids)} inserted IDs but "
            f"{len(filtered_builtin_reasons)} reason entries. "
            "Cannot map builtin anomaly reasons safely."
        )

    # Pair inserted_ids with reasons using their aligned indices
    # filtered_builtin_reasons is indexed by original row indices
    result = {}
    for db_id, (row_index, reason) in zip(
        inserted_ids,
        filtered_builtin_reasons.items(),
    ):
        if pd.notna(reason) and str(reason).strip():
            result[str(db_id)] = str(reason).strip()

    return result


def _build_persisted_metrics(
    payload: WorkbenchRunRequest,
    execution: WorkbenchExecutionState,
    model_state: IsolationForestState,
    dataset_frame: pd.DataFrame,
    dataset_storage: dict[str, Any],
    builtin_reason_by_record_id: dict[str, str],
) -> dict[str, Any]:
    """Assemble the final metrics JSON stored with the app-database run record."""
    return {
        **_base_metrics(payload, execution),
        "contamination": payload.contamination,
        "feature_count": int(model_state.feature_frame.shape[1]),
        "selected_feature_columns": model_state.feature_selection.selected_columns,
        "dropped_all_missing_feature_columns": (
            model_state.feature_selection.dropped_all_missing_columns
        ),
        "dropped_constant_feature_columns": (
            model_state.feature_selection.dropped_constant_columns
        ),
        "join_execution_mode": "postgres_sql",
        "new_rows_written": int(len(dataset_frame)),
        "builtin_reason_by_record_id": builtin_reason_by_record_id,
        "joined_result_table": dataset_storage["table_name"],
        "joined_result_row_count": dataset_storage["row_count"],
        "joined_result_column_count": dataset_storage["column_count"],
    }


def _build_run_response(run: WorkbenchRun, amount_total: float) -> dict[str, Any]:
    """Convert the ORM run record into the API response payload."""
    return {
        "run_id": run.run_id,
        "run_name": run.run_name,
        "total_rows": run.total_rows,
        "user_rule_count": run.user_rule_count,
        "ml_anomaly_count": run.ml_anomaly_count,
        "final_anomaly_count": run.final_anomaly_count,
        "amount_total": amount_total,
        "selected_model": "IsolationForest",
        "metrics": run.metrics_json or {},
    }
