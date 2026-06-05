from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from app.services.workbench.anomaly_data_policy import (
    SavedModelPreprocessingPolicy,
    apply_saved_model_preprocessing_policy,
)

from app.core.config import settings
from app.core.errors import WorkbenchValidationError
from app.schemas.workbench_schema import WorkbenchRunRequest
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

ML_SCORE_THRESHOLD_MARGIN = 0.07


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
    joined_df: pd.DataFrame,
    artifact: dict[str, Any],
) -> tuple[pd.DataFrame, FeatureSelectionResult]:
    raw_df = _read_model_raw_frame(joined_df, artifact)
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
        ml_threshold = float(-getattr(model, "offset_", np.nan))
        if feature_frame.empty:
            empty_flag = pd.Series(index=feature_frame.index, dtype=bool)
            return np.empty((0, 0)), np.asarray([], dtype=float), empty_flag, ml_threshold
        transformed = preprocessor.transform(feature_frame)
        isolation_scores = -model.score_samples(transformed)
        effective_threshold = ml_threshold + ML_SCORE_THRESHOLD_MARGIN
        ml_flag = pd.Series(
            isolation_scores >= effective_threshold,
            index=feature_frame.index,
            dtype=bool,
        )
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

    return _as_2d_numpy(transformed), np.asarray(isolation_scores), ml_flag, ml_threshold


def _as_2d_numpy(values: Any) -> np.ndarray:
    if sparse.issparse(values):
        return values.toarray()
    array = np.asarray(values)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _read_model_raw_frame(joined_df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    available_columns = set(str(column) for column in joined_df.columns)
    model_raw_columns = _model_raw_columns_for_artifact(artifact)
    raw_joined_columns = _raw_joined_columns_for_artifact(model_raw_columns, artifact)
    select_columns = [column for column in raw_joined_columns if column in available_columns]

    if not select_columns:
        raise WorkbenchValidationError(
            "The selected data does not contain any columns expected by the saved model.",
            details={
                "dataset_name": artifact.get("dataset_name"),
                "expected_sample": raw_joined_columns[:20],
                "available_sample": sorted(available_columns)[:20],
            },
        )

    df = joined_df.loc[:, select_columns].copy()
    df = df.rename(
        columns={
            joined_column: raw_column
            for raw_column, joined_column in zip(model_raw_columns, raw_joined_columns)
        }
    )

    for column in model_raw_columns:
        if column not in df.columns:
            df[column] = np.nan

    return df.reindex(columns=model_raw_columns)


def _model_raw_columns_for_artifact(artifact: dict[str, Any]) -> list[str]:
    model_raw_columns = [str(column) for column in artifact.get("model_raw_columns") or []]
    if model_raw_columns:
        return model_raw_columns
    return [str(column) for column in artifact.get("raw_columns") or []]


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
    policy = SavedModelPreprocessingPolicy.from_artifact(artifact)
    return apply_saved_model_preprocessing_policy(raw_df, policy)


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
