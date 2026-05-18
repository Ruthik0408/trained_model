from app.services.llm_reason_service import (
    _build_magnitude_signals,
    _translate_one_signal,
    build_deterministic_isolation_reason,
)
import pandas as pd


def test_translate_one_hot_text_signal_as_unusual_category_value() -> None:
    clause = _translate_one_signal(
        {
            "feature": "payment_detail::final adjustment on allindia ltc r o updesh kuma",
            "value": 1.0,
            "scaled_value": 4.2,
            "direction": "high",
        }
    )

    assert clause is not None
    assert "unusually high" not in clause
    assert "payment detail" in clause
    assert "unusual value" in clause


def test_deterministic_reason_prioritizes_date_gap_over_noisy_text_category() -> None:
    reason = build_deterministic_isolation_reason(
        [
            {
                "feature": "subject::gopi",
                "value": 1.0,
                "scaled_value": 6.0,
                "direction": "high",
            },
            {
                "feature": "invoice_date_to_bill_date",
                "value": 19.0,
                "scaled_value": 3.0,
                "direction": "high",
            },
        ],
        row_payload={},
    )

    assert reason is not None
    assert "gap from invoice date to bill date is 19 days" in reason.lower()


def test_translate_plain_datetime_signal_uses_date_not_epoch_number() -> None:
    clause = _translate_one_signal(
        {
            "feature": "aao_date",
            "value": 1746576000.0,
            "scaled_value": 3.7,
            "direction": "high",
        }
    )

    assert clause is not None
    assert "1,746,576,000" not in clause
    assert "unusually high" not in clause
    assert "aao date" in clause
    assert "later than usual" in clause
    assert "2025-05-07" in clause


def test_missing_indicator_signal_is_preserved_as_missing() -> None:
    feature_frame = pd.DataFrame(
        {
            "bill.fk_bill_type": [None],
        },
        index=[101],
    )
    transformed_frame = pd.DataFrame(
        {
            "bill.fk_bill_type__missing": [5.0],
        },
        index=[101],
    )

    signals = _build_magnitude_signals(feature_frame, transformed_frame, [101])

    assert signals[101][0]["feature"] == "bill.fk_bill_type__missing"
    clause = _translate_one_signal(signals[101][0])
    assert clause == "the fk bill type field is missing"
