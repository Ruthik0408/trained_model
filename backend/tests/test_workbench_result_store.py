import numpy as np
import pandas as pd

from app.schemas.workbench_schema import FeatureRuleInput, WorkbenchRunRequest
from app.services.workbench.result_store import DatasetBuildInputs, _build_dataset_frame


def test_build_dataset_frame_keeps_only_anomalies_and_preserves_reason_fields() -> None:
    joined = pd.DataFrame(
        {
            "bill_no": ["A-1", "A-2"],
            "claimed_amount": [100.0, 400.0],
        },
        index=[10, 11],
    )
    feature_frame = pd.DataFrame(
        {
            "amount_delta": [0.5, 2.5],
        },
        index=joined.index,
    )
    payload = WorkbenchRunRequest(
        selected_tables=["bills"],
        feature_rules=[
            FeatureRuleInput(
                name="Amount delta",
                feature_type="numeric",
                first_column="claimed_amount",
            )
        ],
    )

    dataset = _build_dataset_frame(
        DatasetBuildInputs(
            joined=joined,
            feature_frame=feature_frame,
            payload=payload,
            dataset_table="ML_Features",
            dataset_run_id=7,
            human_outlier_flag=pd.Series([False, True], index=joined.index),
            human_reasons=["Human-defined outlier rule matched"],
            human_reason_series=pd.Series([None, "Manual threshold"], index=joined.index),
            builtin_reason_series=None,
            isolation_scores=np.array([0.2, 0.8]),
            ml_flag=pd.Series([False, True], index=joined.index),
            ml_threshold=0.6,
            final_flag=pd.Series([False, True], index=joined.index),
            filtered_joined_override=joined.loc[[11]],
            explanation_signals_override={11: [{"feature": "amount_delta", "impact": 0.7}]},
        )
    )

    assert len(dataset) == 1
    row = dataset.iloc[0]
    assert row["feature_name"].startswith("bills")
    assert row["human_rule_name"] == "Manual threshold"
    assert bool(row["human_rule"]) is True
    assert bool(row["isolation_rule"]) is True
    assert row["ml_run_id"] == 7
    assert row["review_payload_json"]["bill_no"] == "A-2"
    assert row["feature_values_json"]["amount_delta"] == 2.5
    assert "__ml_explanation_signals" in row["feature_values_json"]
    assert "__ml_llm_if_reason" not in row["feature_values_json"]
