from dataclasses import dataclass
import re

import pandas as pd
import numpy as np

from app.core.errors import WorkbenchValidationError
from app.services.workbench.constants import logger
from app.services.workbench.source_db import _is_date_like_column_name
from app.services.workbench.utils import _is_amount_like_column


@dataclass(frozen=True)
class FeatureSelectionResult:
    feature_frame: pd.DataFrame
    selected_columns: list[str]
    dropped_all_missing_columns: list[str]
    dropped_constant_columns: list[str]

def _is_frequency_encoded_column(column_name: str) -> bool:
    plain_name = str(column_name).strip().lower().split(".")[-1]
    return (
        plain_name.startswith("fk_")
        or plain_name.endswith("_no")
        or plain_name.endswith("_number")
    )


def _safe_encoded_feature_name(column_name: str, value: object) -> str:
    value_text = "missing" if pd.isna(value) else str(value)
    value_token = re.sub(r"[^a-zA-Z0-9]+", "_", value_text).strip("_").lower()
    if not value_token:
        value_token = "blank"
    if len(value_token) > 48:
        value_token = value_token[:48].rstrip("_") or "value"
    return f"{column_name}::{value_token}"


def _datetime_series_to_numeric(series: pd.Series) -> pd.Series:
    dt_series = pd.to_datetime(series, errors="coerce", utc=True)
    numeric = pd.Series(np.nan, index=series.index, dtype=float)
    valid = dt_series.notna()
    if valid.any():
        numeric.loc[valid] = (
            dt_series.loc[valid].astype("int64") / 1_000_000_000.0
        )
    return numeric


def _frequency_encode_series(series: pd.Series) -> pd.Series:
    encoded = pd.Series(np.nan, index=series.index, dtype=float)
    non_missing = series.notna()
    if non_missing.any():
        frequencies = series.loc[non_missing].astype("string").value_counts(normalize=True)
        encoded.loc[non_missing] = (
            series.loc[non_missing]
            .astype("string")
            .map(frequencies)
            .astype(float)
        )
    return encoded


def _one_hot_encode_series(series: pd.Series, column_name: str) -> pd.DataFrame:
    if series.notna().sum() == 0:
        return pd.DataFrame(index=series.index)

    text_series = series.astype("string")
    dummies = pd.get_dummies(text_series, dummy_na=False, dtype=float)
    dummies = dummies.rename(
        columns={
            value: _safe_encoded_feature_name(column_name, value)
            for value in dummies.columns
        }
    )
    return dummies


def _coerce_joined_feature_series(series: pd.Series, column_name: str) -> pd.Series | pd.DataFrame:
    if _is_frequency_encoded_column(column_name):
        return _frequency_encode_series(series)

    if pd.api.types.is_bool_dtype(series):
        return _one_hot_encode_series(series, column_name)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    if pd.api.types.is_datetime64_any_dtype(series):
        return _datetime_series_to_numeric(series)

    non_missing = int(series.notna().sum())
    if non_missing == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)

    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(numeric_series.notna().sum()) / float(non_missing)

    datetime_like_name = _is_date_like_column_name(column_name)
    if datetime_like_name:
        datetime_series = _datetime_series_to_numeric(series)
        datetime_ratio = float(datetime_series.notna().sum()) / float(non_missing)
    else:
        datetime_series = pd.Series(np.nan, index=series.index, dtype=float)
        datetime_ratio = 0.0

    if numeric_ratio >= 0.80 and numeric_ratio >= datetime_ratio:
        return numeric_series.astype(float)

    if datetime_like_name or datetime_ratio >= 0.80:
        return datetime_series

    return _one_hot_encode_series(series, column_name)


def _build_auto_feature_frame(
    joined: pd.DataFrame,
    *,
    excluded_columns: set[str] | None = None,
) -> pd.DataFrame:
    excluded = {str(column) for column in (excluded_columns or set())}
    features = pd.DataFrame(index=joined.index)

    for index, column in enumerate(joined.columns):
        column_name = str(column)
        if column_name in excluded:
            continue
        encoded = _coerce_joined_feature_series(joined.iloc[:, index], column_name)
        if isinstance(encoded, pd.DataFrame):
            features = pd.concat([features, encoded], axis=1)
        else:
            features = pd.concat([features, encoded.rename(column_name)], axis=1)

    return features

def _add_statistical_outlier_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        (index, column)
        for index, column in enumerate(out.columns)
        if _is_amount_like_column(str(column))
    ]
    for index, column in numeric_cols[:]:
        series = pd.to_numeric(out.iloc[:, index], errors="coerce")
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
        )

    logger.info(
        "Keeping %d IF feature columns after dropping missing/constant columns",
        len(feature_frame.columns),
    )
    return FeatureSelectionResult(
        feature_frame=feature_frame.copy(),
        selected_columns=[str(column) for column in feature_frame.columns],
        dropped_all_missing_columns=[str(column) for column in all_missing_cols],
        dropped_constant_columns=[str(column) for column in zero_var_cols],
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
