import pandas as pd

from app.services import dashboard_service


def test_dataset_row_to_prediction_keeps_llm_reason_fields_empty_until_requested() -> None:
    row = pd.Series(
        {
            "id": 101,
            "feature_name": "dak.bill",
            "human_rule_name": None,
            "human_rule": False,
            "isolation_rule": True,
            "ml_if_score": 0.49,
            "ml_threshold": 0.48,
            "feedback_score": None,
            "ml_run_id": 50,
            "review_payload_json": {
                "bill.amount_claimed": 14500.0,
                "dak.reference_date": "2025-04-04",
            },
            "feature_values_json": {
                "__ml_explanation_signals": [
                    {
                        "feature": "reference_date_to_list_date",
                        "value": 63.0,
                        "scaled_value": 4.35,
                        "direction": "high",
                    }
                ]
            },
        }
    )

    result = dashboard_service._dataset_row_to_prediction(row)

    assert result["reasons_json"]["ml_feature_signals"]
    assert result["reasons_json"]["llm_if_reason"] is None
    assert result["reasons_json"]["llm_if_reason_model"] is None
    assert result["reasons_json"]["llm_if_reason_fallback"] is False
