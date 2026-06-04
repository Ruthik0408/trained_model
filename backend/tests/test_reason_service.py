import sys
from pathlib import Path

import pandas as pd
from scipy import sparse

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.services.reason_service import build_feature_explanation_signals


def test_feature_explanation_signals_accept_sparse_transformed_matrix() -> None:
    feature_frame = pd.DataFrame({"amount": [100.0]}, index=[7])
    transformed = sparse.csr_matrix([[2.5]])

    signals = build_feature_explanation_signals(
        pipeline=None,
        feature_frame=feature_frame,
        transformed=transformed,
        anomaly_indices=[7],
        transformed_feature_labels=["amount"],
    )

    assert signals[7][0]["feature"] == "amount"
    assert signals[7][0]["scaled_value"] == 2.5
