from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from app.core.config import settings
from app.core.errors import WorkbenchValidationError
from app.schemas.workbench_schema import WorkbenchRunRequest
from app.services.workbench.constants import TEMP_ROW_ID_COLUMN
from app.services.workbench.sql_runtime import _temp_table_columns, _workbench_temp_table_ref
from app.services.workbench.source_db import _quote
from app.services.workbench.trained_datasets import resolve_dataset_name

KNOWN_MODEL_TABLE_PREFIXES = (
    "civ_tada_ltc_bill",
    "civ_medical_bill",
    "echs_medical_bill",
    "cheque_slip",
    "gem_bill",
    "dak_info",
    "civ_paybill",
    "schedule3",
    "bill",
    "dak",
    "ecs",
)


@dataclass(frozen=True)
class FeatureSelectionResult:
    feature_frame: pd.DataFrame
    selected_columns: list[str]
    dropped_all_missing_columns: list[str]
    dropped_constant_columns: list[str]


def load_saved_model_artifact(payload: WorkbenchRunRequest) -> dict[str, Any]:
    dataset_name = resolve_dataset_name(payload.selected_tables)
    model_path = Path(settings.trained_model_dir) / f"{dataset_name}_pipeline.joblib"
    if not model_path.exists():
        raise WorkbenchValidationError(
            "The saved trained model could not be found.",
            suggestion="Run backend/train_models.py once, then rerun anomaly detection without retraining.",
            details={"model_path": str(model_path), "dataset_name": dataset_name},
        )

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise WorkbenchValidationError(
            "The saved model artifact is not in the expected format.",
            details={"model_path": str(model_path), "dataset_name": dataset_name},
        )
    return artifact


def build_saved_model_feature_frame(
    conn,
    temp_table: str,
    artifact: dict[str, Any],
) -> tuple[pd.DataFrame, FeatureSelectionResult]:
    raw_df = _read_model_raw_frame(conn, temp_table, artifact)
    cleaned_df = _apply_saved_training_cleaning(raw_df, artifact)
    feature_input_columns = [str(column) for column in artifact.get("feature_input_columns") or []]

    if not feature_input_columns:
        raise WorkbenchValidationError(
            "The saved model artifact does not list any feature input columns.",
            details={"dataset_name": artifact.get("dataset_name")},
        )

    feature_df = cleaned_df.reindex(columns=feature_input_columns)
    if feature_df.empty:
        raise WorkbenchValidationError(
            "No rows remain after applying the saved model preprocessing steps.",
            suggestion="Use a date range and selected tables that match a trained pipeline.",
            details={"dataset_name": artifact.get("dataset_name")},
        )

    selection = FeatureSelectionResult(
        feature_frame=feature_df,
        selected_columns=feature_input_columns,
        dropped_all_missing_columns=[],
        dropped_constant_columns=[],
    )
    return feature_df, selection


def score_with_saved_model(
    feature_frame: pd.DataFrame,
    artifact: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, pd.Series, float]:
    pipeline = artifact["pipeline"]

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        transformed = preprocessor.transform(feature_frame)
        isolation_scores = -model.score_samples(transformed)
        predictions = model.predict(transformed)
        ml_flag = pd.Series(predictions == -1, index=feature_frame.index, dtype=bool)
        ml_threshold = float(-getattr(model, "offset_", np.nan))
    except Exception as exc:
        raise WorkbenchValidationError(
            "Saved model scoring failed on the selected Postgres data.",
            suggestion=(
                "Confirm the selected tables match the trained dataset and that the saved "
                "pipeline was produced by backend/train_models.py."
            ),
            details={
                "dataset_name": artifact.get("dataset_name"),
                "feature_count": int(feature_frame.shape[1]),
                "row_count": int(len(feature_frame.index)),
                "original_error": str(exc),
            },
        ) from exc

    return np.asarray(transformed), np.asarray(isolation_scores), ml_flag, ml_threshold


def _read_model_raw_frame(conn, temp_table: str, artifact: dict[str, Any]) -> pd.DataFrame:
    available_columns = set(_temp_table_columns(conn, temp_table))
    raw_columns = [str(column) for column in artifact.get("raw_columns") or []]
    raw_joined_columns = _raw_joined_columns_for_artifact(raw_columns, artifact)
    select_columns = [TEMP_ROW_ID_COLUMN]
    select_columns.extend(column for column in raw_joined_columns if column in available_columns)

    if len(select_columns) == 1:
        raise WorkbenchValidationError(
            "The selected data does not contain any columns expected by the saved model.",
            details={
                "dataset_name": artifact.get("dataset_name"),
                "expected_sample": raw_columns[:20],
                "available_sample": sorted(available_columns)[:20],
            },
        )

    sql = text(
        f"""
        SELECT {", ".join(_quote(column) for column in select_columns)}
        FROM {_workbench_temp_table_ref(temp_table)}
        ORDER BY {_quote(TEMP_ROW_ID_COLUMN)}
        """
    )
    df = pd.read_sql_query(sql, conn)
    df = df.rename(
        columns={
            joined_column: raw_column
            for raw_column, joined_column in zip(raw_columns, raw_joined_columns)
        }
    )
    if TEMP_ROW_ID_COLUMN in df.columns:
        df = df.set_index(TEMP_ROW_ID_COLUMN, drop=True)

    for column in raw_columns:
        if column not in df.columns:
            df[column] = np.nan

    return df.reindex(columns=raw_columns)


def _raw_joined_columns_for_artifact(
    raw_columns: list[str],
    artifact: dict[str, Any],
) -> list[str]:
    raw_joined_columns = [
        str(column)
        for column in artifact.get("raw_joined_columns") or []
    ]
    if len(raw_joined_columns) == len(raw_columns):
        return raw_joined_columns
    return [_model_column_to_joined_column(column) for column in raw_columns]


def _apply_saved_training_cleaning(
    raw_df: pd.DataFrame,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    # Training removes duplicate voided invoices before fitting. During review/scoring we keep
    # those rows so the workbench SQL rules can show them as anomalies.
    working = raw_df.copy()
    cleaned_columns = [str(column) for column in artifact.get("cleaned_columns") or []]
    if cleaned_columns:
        working = working.reindex(columns=cleaned_columns)

    for column in artifact.get("date_sequence_checked_columns") or []:
        column_name = str(column)
        if column_name in working.columns and _looks_like_date_column(column_name):
            working[column_name] = pd.to_datetime(working[column_name], errors="coerce")

    working = _drop_saved_invalid_date_sequence_rows(working, artifact)
    working = _add_saved_date_gap_features(working, artifact)
    working = working.where(pd.notna(working), np.nan)
    return working


def _drop_saved_void_invoice_rows(
    df: pd.DataFrame,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    summary = artifact.get("invoice_row_filter_summary") or {}
    rows_to_drop: set[Any] = set()

    for table_summary in summary.get("tables_applied") or []:
        invoice_column = table_summary.get("invoice_column")
        invoice_date_column = table_summary.get("invoice_date_column")
        record_status_column = table_summary.get("record_status_column")
        required = [invoice_column, invoice_date_column, record_status_column]
        if not all(column in df.columns for column in required):
            continue

        group_keys = [invoice_column, invoice_date_column]
        non_null_invoice_mask = df[group_keys].notna().all(axis=1)
        duplicate_group_mask = (
            df.loc[non_null_invoice_mask, group_keys]
            .duplicated(keep=False)
            .reindex(df.index, fill_value=False)
        )
        void_status_mask = (
            df[record_status_column]
            .astype("string")
            .str.strip()
            .str.lower()
            .eq("v")
            .fillna(False)
        )
        rows_to_drop.update(df.index[non_null_invoice_mask & duplicate_group_mask & void_status_mask])

    if rows_to_drop:
        return df.drop(index=sorted(rows_to_drop), errors="ignore")
    return df


def _drop_saved_invalid_date_sequence_rows(
    df: pd.DataFrame,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    summary = artifact.get("date_sequence_summary") or {}
    rows_to_drop: set[Any] = set()

    for check in summary.get("checks") or []:
        previous_column = check.get("previous_column")
        next_column = check.get("next_column")
        if not previous_column or not next_column:
            continue
        if previous_column not in df.columns or next_column not in df.columns:
            continue
        comparable = df[previous_column].notna() & df[next_column].notna()
        invalid = comparable & (df[next_column] < df[previous_column])
        rows_to_drop.update(df.index[invalid])

    if rows_to_drop:
        return df.drop(index=sorted(rows_to_drop), errors="ignore")
    return df


def _add_saved_date_gap_features(
    df: pd.DataFrame,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    working = df.copy()
    original_date_columns = set()

    for gap in artifact.get("date_gap_features") or []:
        feature_name = str(gap.get("feature_name") or "")
        previous_column = str(gap.get("previous_column") or "")
        next_column = str(gap.get("next_column") or "")
        if not feature_name or previous_column not in working.columns or next_column not in working.columns:
            continue
        previous = pd.to_datetime(working[previous_column], errors="coerce")
        next_value = pd.to_datetime(working[next_column], errors="coerce")
        working[feature_name] = (next_value - previous).dt.total_seconds() / 86400.0
        original_date_columns.update([previous_column, next_column])

    configured_dropped_dates = {
        str(column)
        for column in artifact.get("sequence_filtered_columns") or []
        if str(column) not in (artifact.get("feature_input_columns") or [])
        and _looks_like_date_column(str(column))
    }
    working = working.drop(columns=sorted(original_date_columns | configured_dropped_dates), errors="ignore")
    return working


def _model_column_to_joined_column(column_name: str) -> str:
    text = str(column_name)
    if "." in text:
        return text
    for table_name in KNOWN_MODEL_TABLE_PREFIXES:
        prefix = f"{table_name}_"
        if text.startswith(prefix):
            return f"{table_name}.{text[len(prefix):]}"
    table_name, separator, plain_column = text.partition("_")
    if not separator:
        return text
    return f"{table_name}.{plain_column}"


def _joined_column_to_model_column(column_name: str) -> str:
    return str(column_name).replace(".", "_", 1)


def _looks_like_date_column(column_name: str) -> bool:
    lowered = column_name.lower()
    return "date" in lowered or "time" in lowered
