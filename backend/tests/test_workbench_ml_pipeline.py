import numpy as np
import pandas as pd
import pytest

from app.services.workbench import ml_pipeline
from app.services.workbench.ml_pipeline import _prepare_isolation_forest_feature_frame, _validate_isolation_forest_feature_frame


def test_validate_feature_frame_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="No usable feature columns"):
        _validate_isolation_forest_feature_frame(pd.DataFrame())


def test_validate_feature_frame_rejects_all_missing_columns() -> None:
    frame = pd.DataFrame(
        {
            "amount_delta": [np.nan, np.nan],
            "days_between": [np.nan, np.nan],
        }
    )

    with pytest.raises(ValueError, match="only contains missing values"):
        _validate_isolation_forest_feature_frame(frame)


def test_validate_feature_frame_accepts_partially_populated_data() -> None:
    frame = pd.DataFrame(
        {
            "amount_delta": [1.0, np.nan, 2.5],
            "days_between": [np.nan, 4.0, np.nan],
        }
    )

    validated = _validate_isolation_forest_feature_frame(frame)

    assert validated.equals(frame)


def test_prepare_feature_frame_drops_all_missing_and_constant_columns() -> None:
    frame = pd.DataFrame(
        {
            "all_missing": [np.nan, np.nan, np.nan],
            "constant": [5.0, 5.0, 5.0],
            "useful": [1.0, 2.0, 3.0],
        }
    )

    result = _prepare_isolation_forest_feature_frame(frame)

    assert result.selected_columns == ["useful"]
    assert "all_missing" in result.dropped_all_missing_columns
    assert "constant" in result.dropped_constant_columns
    assert list(result.feature_frame.columns) == ["useful"]


def test_prepare_feature_frame_scores_and_limits_columns(monkeypatch) -> None:
    monkeypatch.setattr(ml_pipeline.settings, "anomaly_feature_max_columns", 2)
    monkeypatch.setattr(ml_pipeline.settings, "anomaly_feature_min_score", 0.0)
    frame = pd.DataFrame(
        {
            "weak": [1.0, 1.0, 2.0, np.nan],
            "strong_a": [1.0, 4.0, 7.0, 10.0],
            "strong_b": [0.0, 1.0, 0.0, 1.0],
        }
    )

    result = _prepare_isolation_forest_feature_frame(frame)

    assert len(result.selected_columns) == 2
    assert "strong_a" in result.selected_columns
    assert set(result.dropped_low_score_columns) == {"weak"}
