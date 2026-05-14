from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.models import WorkbenchRun
from app.schemas.workbench_schema import WorkbenchRunRequest
from app.services.llm_reason_service import build_feature_explanation_signals
from app.services.workbench.constants import PREVIEW_ROW_LIMIT, RESULT_TABLE, logger
from app.services.workbench.llm_prep import _build_persisted_llm_if_reasons
from app.services.workbench.ml_pipeline import (
    _add_statistical_outlier_signals,
    _prepare_isolation_forest_feature_frame,
)
from app.services.workbench.result_store import _build_dataset_frame, _write_dataset_to_result
from app.services.workbench.source_db import _next_dataset_run_id, _source_begin
from app.services.workbench.sql_runtime import (
    _execute_sql_joined_frame,
    _execute_sql_workbench_frame,
    _feature_rule_aliases,
    _read_temp_anomaly_payload_frame,
    _resolve_column,
    _safe_date_literal,
)


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
    warnings: list[str] = []

    with _source_begin() as source_conn:
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
        warnings.extend(join_warnings)
        warnings.extend(sql_pushdown_warnings)
        logger.info("Joined %d rows from %d tables", len(joined), len(payload.selected_tables))

        batch_id = f"workbench_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        dataset_table = RESULT_TABLE
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
        if builtin_rule_flag.any() and builtin_reason_series is None:
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
        filtered_joined = _read_temp_anomaly_payload_frame(source_conn, staging_table, anomaly_row_ids, payload)

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
