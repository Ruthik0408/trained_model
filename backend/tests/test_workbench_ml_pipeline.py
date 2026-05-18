import numpy as np
import pandas as pd
import pytest

from app.services.workbench.ml_pipeline import (
    _build_auto_feature_frame,
    _prepare_isolation_forest_feature_frame,
    _validate_isolation_forest_feature_frame,
)


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


def test_prepare_feature_frame_keeps_all_usable_columns() -> None:
    frame = pd.DataFrame(
        {
            "weak": [1.0, 1.0, 2.0, np.nan],
            "strong_a": [1.0, 4.0, 7.0, 10.0],
            "strong_b": [0.0, 1.0, 0.0, 1.0],
        }
    )

    result = _prepare_isolation_forest_feature_frame(frame)

    assert result.selected_columns == ["weak", "strong_a", "strong_b"]


def test_build_auto_feature_frame_uses_non_identifier_columns() -> None:
    joined = pd.DataFrame(
        {
            "dak.id": [1, 2, 3],
            "dak.fk_dak": [101, 102, 103],
            "dak.fk_vendor": [10, 10, 12],
            "dak.bill_no": ["A", "A", "C"],
            "dak.amount": ["100.5", "200.0", None],
            "dak.bill_date": ["2025-01-01", "2025-01-02", None],
            "dak.status": ["open", "closed", "open"],
            "dak.approved": [True, False, True],
            "sql_rule_flag": [True, False, True],
        }
    )

    feature_frame = _build_auto_feature_frame(
        joined,
        excluded_columns={"sql_rule_flag"},
    )

    assert "dak.id" not in feature_frame.columns
    assert "dak.fk_dak" not in feature_frame.columns
    assert "sql_rule_flag" not in feature_frame.columns
    assert {"dak.fk_vendor", "dak.bill_no", "dak.amount", "dak.bill_date"} <= set(feature_frame.columns)
    assert {"dak.status::closed", "dak.status::open"} <= set(feature_frame.columns)
    assert {"dak.approved::false", "dak.approved::true"} <= set(feature_frame.columns)
    assert feature_frame["dak.fk_vendor"].tolist() == [2 / 3, 2 / 3, 1 / 3]
    assert feature_frame["dak.bill_no"].tolist() == [2 / 3, 2 / 3, 1 / 3]
    assert feature_frame["dak.amount"].tolist()[:2] == [100.5, 200.0]
    assert feature_frame["dak.bill_date"].notna().sum() == 2
