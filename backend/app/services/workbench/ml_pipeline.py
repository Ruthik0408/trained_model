import pandas as pd
import numpy as np

from app.services.workbench.constants import logger
from app.services.workbench.utils import _is_amount_like_column

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

