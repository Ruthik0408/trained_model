import pandas as pd
import numpy as np
from scipy import sparse

from app.services.workbench.anomaly_data_policy import (
    SavedModelPreprocessingPolicy,
    build_saved_model_preprocessing_policy,
)
from app.services.workbench.saved_model_inference import (
    _apply_saved_training_cleaning,
    _as_2d_numpy,
    score_with_saved_model,
)


class FakePreprocessor:
    def transform(self, feature_frame):
        return feature_frame


class FakeModel:
    offset_ = -0.58

    def score_samples(self, _transformed):
        return np.array([-0.62, -0.63, -0.64])


class FakePipeline:
    named_steps = {
        "preprocessor": FakePreprocessor(),
        "model": FakeModel(),
    }


def test_saved_training_cleaning_drops_void_duplicate_invoice_rows() -> None:
    raw_df = pd.DataFrame(
        {
            "invoice_no": ["A-1", "A-1", "B-1"],
            "invoice_date": ["2025-01-01", "2025-01-01", "2025-01-02"],
            "record_status": ["A", "V", "V"],
            "amount": [100, 100, 50],
        },
        index=[10, 11, 12],
    )
    artifact = {
        "cleaned_columns": ["invoice_no", "invoice_date", "record_status", "amount"],
        "invoice_row_filter_summary": {
            "tables_applied": [
                {
                    "invoice_column": "invoice_no",
                    "invoice_date_column": "invoice_date",
                    "record_status_column": "record_status",
                }
            ]
        },
    }

    cleaned = _apply_saved_training_cleaning(raw_df, artifact)

    assert cleaned.index.tolist() == [10, 12]
    assert cleaned["invoice_no"].tolist() == ["A-1", "B-1"]


def test_saved_model_preprocessing_policy_round_trips_artifact_fields() -> None:
    policy = build_saved_model_preprocessing_policy(
        cleaned_columns=["invoice_no", "invoice_date", "record_status", "amount"],
        date_sequence_checked_columns=["invoice_date", "approval_date"],
        sequence_filtered_columns=["amount", "gap_days_invoice_to_approval"],
        feature_input_columns=["amount", "gap_days_invoice_to_approval"],
        boolean_columns=["approved_flag"],
        invoice_row_filter_summary={
            "tables_applied": [
                {
                    "invoice_column": "invoice_no",
                    "invoice_date_column": "invoice_date",
                    "record_status_column": "record_status",
                }
            ]
        },
        date_sequence_summary={
            "checks": [
                {
                    "previous_column": "invoice_date",
                    "next_column": "approval_date",
                }
            ]
        },
        date_gap_features=[
            {
                "feature_name": "gap_days_invoice_to_approval",
                "previous_column": "invoice_date",
                "next_column": "approval_date",
            }
        ],
    )

    artifact = policy.to_artifact_fields()
    restored = SavedModelPreprocessingPolicy.from_artifact(artifact)

    assert artifact["training_row_drop_policies"] == [
        "duplicate_void_invoice_rows",
        "date_sequence_violation_rows",
        "night_timestamp_rows",
    ]
    assert "boolean_normalization" in artifact["shared_feature_transform_policies"]
    assert "date_sequence_violation_rows" in artifact["runtime_rule_flag_policies"]
    assert "night_timestamp_rows" in artifact["runtime_rule_flag_policies"]
    assert restored == policy


def test_saved_training_cleaning_drops_night_timestamp_rows() -> None:
    raw_df = pd.DataFrame(
        {
            "created_at": [
                "2025-01-01 00:00:00",
                "2025-01-01 20:59:00",
                "2025-01-01 21:00:00",
                "2025-01-02 05:59:00",
                "2025-01-02 06:00:00",
            ],
            "amount": [5, 10, 20, 30, 40],
        },
        index=[9, 10, 11, 12, 13],
    )
    artifact = {
        "cleaned_columns": ["created_at", "amount"],
        "date_sequence_checked_columns": ["created_at"],
        "night_timestamp_summary": {
            "columns": [
                {
                    "column_name": "created_at",
                    "rows_checked": 4,
                    "invalid_row_count": 2,
                    "status": "checked",
                }
            ]
        },
    }

    cleaned = _apply_saved_training_cleaning(raw_df, artifact)

    assert cleaned.index.tolist() == [9, 10, 13]


def test_saved_training_cleaning_builds_business_day_gap_features() -> None:
    raw_df = pd.DataFrame(
        {
            "invoice_date": ["2025-01-03"],  # Friday
            "approval_date": ["2025-01-06"],  # Monday
        },
        index=[10],
    )
    artifact = {
        "cleaned_columns": ["invoice_date", "approval_date"],
        "date_sequence_checked_columns": ["invoice_date", "approval_date"],
        "sequence_filtered_columns": ["gap_days_invoice_to_approval"],
        "feature_input_columns": ["gap_days_invoice_to_approval"],
        "date_gap_features": [
            {
                "feature_name": "gap_days_invoice_to_approval",
                "previous_column": "invoice_date",
                "next_column": "approval_date",
            }
        ],
    }

    cleaned = _apply_saved_training_cleaning(raw_df, artifact)

    assert cleaned["gap_days_invoice_to_approval"].astype("Float64").tolist() == [1.0]


def test_as_2d_numpy_converts_sparse_matrix_to_dense_2d_array() -> None:
    array = _as_2d_numpy(sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0]]))

    assert array.shape == (2, 2)
    assert array.tolist() == [[1.0, 0.0], [0.0, 2.0]]


def test_saved_model_flags_only_scores_above_threshold_margin() -> None:
    feature_frame = pd.DataFrame({"amount": [10.0, 20.0, 30.0]}, index=[101, 102, 103])

    _transformed, isolation_scores, ml_flag, ml_threshold = score_with_saved_model(
        feature_frame,
        {"pipeline": FakePipeline()},
    )

    assert ml_threshold == 0.58
    assert isolation_scores.tolist() == [0.62, 0.63, 0.64]
    assert ml_flag.to_dict() == {
        101: False,
        102: False,
        103: False,
    }


def test_saved_model_scoring_allows_empty_feature_frames() -> None:
    feature_frame = pd.DataFrame({"amount": []}, index=pd.Index([], dtype=int))

    transformed, isolation_scores, ml_flag, ml_threshold = score_with_saved_model(
        feature_frame,
        {"pipeline": FakePipeline()},
    )

    assert ml_threshold == 0.58
    assert transformed.shape == (0, 0)
    assert isolation_scores.tolist() == []
    assert ml_flag.empty
