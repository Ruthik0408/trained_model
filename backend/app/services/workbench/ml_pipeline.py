from dataclasses import dataclass

import pandas as pd
import numpy as np

from app.core.config import settings
from app.core.errors import WorkbenchValidationError
from app.services.workbench.constants import logger
from app.services.workbench.utils import _is_amount_like_column


@dataclass(frozen=True)
class FeatureSelectionResult:
    feature_frame: pd.DataFrame
    selected_columns: list[str]
    dropped_all_missing_columns: list[str]
    dropped_constant_columns: list[str]
    dropped_low_score_columns: list[str]
    feature_scores: dict[str, float]

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

def _prepare_isolation_forest_feature_frame(features: pd.DataFrame) -> FeatureSelectionResult:
    feature_frame = features.replace([np.inf, -np.inf], np.nan).copy()
    if feature_frame.empty:
        return FeatureSelectionResult(
            feature_frame=feature_frame,
            selected_columns=[],
            dropped_all_missing_columns=[],
            dropped_constant_columns=[],
            dropped_low_score_columns=[],
            feature_scores={},
        )

    all_missing_cols = feature_frame.columns[~feature_frame.notna().any(axis=0)].tolist()
    if all_missing_cols:
        logger.info(
            "Dropping %d all-missing IF feature columns: %s",
            len(all_missing_cols),
            all_missing_cols[:10],
        )
        feature_frame = feature_frame.drop(columns=all_missing_cols, errors="ignore")

    if feature_frame.empty:
        return FeatureSelectionResult(
            feature_frame=feature_frame,
            selected_columns=[],
            dropped_all_missing_columns=[str(column) for column in all_missing_cols],
            dropped_constant_columns=[],
            dropped_low_score_columns=[],
            feature_scores={},
        )

    variance = feature_frame.var(numeric_only=True).fillna(0.0)
    zero_var_cols = variance[variance == 0].index.tolist()
    if zero_var_cols:
        logger.info(
            "Dropping %d zero-variance IF feature columns: %s",
            len(zero_var_cols),
            zero_var_cols[:10],
        )
        feature_frame = feature_frame.drop(columns=zero_var_cols, errors="ignore")

    if feature_frame.empty:
        return FeatureSelectionResult(
            feature_frame=feature_frame,
            selected_columns=[],
            dropped_all_missing_columns=[str(column) for column in all_missing_cols],
            dropped_constant_columns=[str(column) for column in zero_var_cols],
            dropped_low_score_columns=[],
            feature_scores={},
        )

    feature_scores = _score_feature_columns(feature_frame)
    selected_columns, dropped_low_score_columns = _select_feature_columns(feature_scores)
    selected_frame = feature_frame.loc[:, selected_columns].copy()
    logger.info(
        "Selected %d/%d IF feature columns after scoring",
        len(selected_columns),
        len(feature_scores),
    )
    return FeatureSelectionResult(
        feature_frame=selected_frame,
        selected_columns=selected_columns,
        dropped_all_missing_columns=[str(column) for column in all_missing_cols],
        dropped_constant_columns=[str(column) for column in zero_var_cols],
        dropped_low_score_columns=dropped_low_score_columns,
        feature_scores=feature_scores,
    )


def _validate_isolation_forest_feature_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    if feature_frame.empty or feature_frame.shape[1] == 0:
        raise WorkbenchValidationError(
            "No usable feature columns were produced from the selected SQL-joined dataset and feature rules.",
            suggestion="Choose at least one numeric/date-like column or add feature rules that generate numeric signals.",
            details={
                "feature_column_count": int(feature_frame.shape[1]),
                "row_count": int(len(feature_frame.index)),
            },
        )

    non_missing_by_column = feature_frame.notna().sum(axis=0)
    usable_columns = non_missing_by_column[non_missing_by_column > 0].index.tolist()
    if not usable_columns:
        raise WorkbenchValidationError(
            "The engineered feature set only contains missing values.",
            suggestion="Adjust the selected feature rules or choose source columns that contain real numeric/date data.",
            details={
                "feature_column_count": int(feature_frame.shape[1]),
                "all_missing_columns": [str(column) for column in feature_frame.columns],
            },
        )

    non_missing_by_row = feature_frame[usable_columns].notna().sum(axis=1)
    if int((non_missing_by_row > 0).sum()) == 0:
        raise WorkbenchValidationError(
            "Every workbench row is missing all engineered feature values after preprocessing.",
            suggestion="Relax the source filters or use feature rules mapped to columns that actually contain values.",
            details={
                "usable_feature_columns": usable_columns,
                "row_count": int(len(feature_frame.index)),
            },
        )

    return feature_frame


def _score_feature_columns(feature_frame: pd.DataFrame) -> dict[str, float]:
    non_missing_ratio = feature_frame.notna().mean(axis=0)
    non_missing_counts = feature_frame.notna().sum(axis=0)
    unique_counts = feature_frame.nunique(dropna=True)
    variance = feature_frame.var(numeric_only=True).reindex(feature_frame.columns).fillna(0.0)

    positive_variance = variance[variance > 0]
    if positive_variance.empty:
        variance_rank = pd.Series(0.0, index=feature_frame.columns, dtype=float)
    else:
        variance_rank = variance.rank(pct=True).reindex(feature_frame.columns).fillna(0.0)

    uniqueness_ratio = pd.Series(0.0, index=feature_frame.columns, dtype=float)
    valid_unique_mask = non_missing_counts > 1
    uniqueness_ratio.loc[valid_unique_mask] = (
        (unique_counts.loc[valid_unique_mask] - 1).clip(lower=0)
        / (non_missing_counts.loc[valid_unique_mask] - 1).clip(lower=1)
    ).clip(lower=0.0, upper=1.0)

    scores = (
        (non_missing_ratio * 0.45)
        + (variance_rank * 0.35)
        + (uniqueness_ratio * 0.20)
    ).fillna(0.0)
    return {
        str(column): float(scores.loc[column])
        for column in feature_frame.columns
    }


def _select_feature_columns(feature_scores: dict[str, float]) -> tuple[list[str], list[str]]:
    if not feature_scores:
        return [], []

    ordered = sorted(feature_scores.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) <= settings.anomaly_feature_max_columns:
        selected = [column for column, score in ordered if score >= settings.anomaly_feature_min_score]
    else:
        selected = [
            column
            for column, score in ordered[:settings.anomaly_feature_max_columns]
            if score >= settings.anomaly_feature_min_score
        ]

    if not selected:
        selected = [ordered[0][0]]

    selected_set = set(selected)
    dropped = [column for column, _score in ordered if column not in selected_set]
    return selected, dropped
