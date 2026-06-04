import pandas as pd
import numpy as np
from scipy import sparse

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
        102: True,
        103: True,
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
