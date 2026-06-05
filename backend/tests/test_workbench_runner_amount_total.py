from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.schemas.workbench_schema import WorkbenchRunRequest
import app.services.workbench.runner as runner
from app.services.workbench.runner import (
    RuleFlagState,
    WorkbenchExecutionState,
    _build_builtin_reason_lookup,
    _calculate_amount_total,
    _run_isolation_forest,
)
from app.services.workbench.result_store import (
    _feature_name_for_tables,
    _review_payload_cache_entries,
)


def test_calculate_amount_total_uses_first_duplicate_amount_column() -> None:
    payload = WorkbenchRunRequest(
        selected_tables=["bills"],
        amount_field="bills.amount",
    )
    filtered_joined = pd.DataFrame(
        [[100, 900], [250, 800]],
        columns=["bills.amount", "bills.amount"],
    )

    amount_total = _calculate_amount_total(payload, filtered_joined)

    assert amount_total == 350.0


def test_feature_name_for_tables_does_not_pad_nulls() -> None:
    assert _feature_name_for_tables(["dak"]) == "dak"
    assert _feature_name_for_tables(["dak", "bill"]) == "dak.bill"
    assert _feature_name_for_tables(["dak", "cheque_slip", "schedule3"]) == "dak.cheque_slip.schedule3"


def test_review_payload_cache_entries_store_table_dot_column_keys() -> None:
    rows = pd.DataFrame(
        [
            {
                "dak_amount": 100,
                "bill_invoice_no": "A-1",
                "bill.invoice_date": "2026-01-01",
                "__derived_feature": 9,
            }
        ]
    )

    payloads = _review_payload_cache_entries(
        rows,
        {"__derived_feature"},
        [101],
        ["dak", "bill"],
    )

    assert payloads == {
        "101": {
            "dak.amount": 100,
            "bill.invoice_no": "A-1",
            "bill.invoice_date": "2026-01-01",
        }
    }


def test_build_builtin_reason_lookup_raises_on_id_reason_mismatch() -> None:
    rule_flags = RuleFlagState(
        user_rule_flag=pd.Series([False, False]),
        default_rule_flag=pd.Series([True, True]),
        combined_rule_flag=pd.Series([True, True]),
        user_reason_series=None,
        default_reason_series=pd.Series(["First reason", "Second reason"], index=[10, 20]),
    )
    model_state = type(
        "ModelState",
        (),
        {"final_flag": pd.Series([True, True], index=[10, 20])},
    )()

    with pytest.raises(ValueError, match="Mismatch: 1 inserted IDs but 2 reason entries"):
        _build_builtin_reason_lookup(
            rule_flags,
            model_state,
            {"inserted_ids": [101]},
        )


def test_build_builtin_reason_lookup_maps_aligned_inserted_ids() -> None:
    rule_flags = RuleFlagState(
        user_rule_flag=pd.Series([False, False]),
        default_rule_flag=pd.Series([True, True]),
        combined_rule_flag=pd.Series([True, True]),
        user_reason_series=None,
        default_reason_series=pd.Series(["First reason", "Second reason"], index=[10, 20]),
    )
    model_state = type(
        "ModelState",
        (),
        {"final_flag": pd.Series([True, True], index=[10, 20])},
    )()

    lookup = _build_builtin_reason_lookup(
        rule_flags,
        model_state,
        {"inserted_ids": [101, 102]},
    )

    assert lookup == {"101": "First reason", "102": "Second reason"}


def test_run_workbench_does_not_create_run_when_dataset_build_fails(monkeypatch) -> None:
    payload = WorkbenchRunRequest(selected_tables=["dak"])
    execution = WorkbenchExecutionState(
        joined=pd.DataFrame({"value": [1]}),
        source_row_counts={},
        join_debug={},
        warnings=[],
        executed_sql="SELECT 1",
        user_reasons=[],
        applied_feature_rule_count=0,
        applied_user_rule_count=0,
        batch_id="test_batch",
        dataset_table="ML_Features",
        dataset_run_id=1,
    )
    rule_flags = RuleFlagState(
        user_rule_flag=pd.Series([False]),
        default_rule_flag=pd.Series([False]),
        combined_rule_flag=pd.Series([False]),
        user_reason_series=None,
        default_reason_series=None,
    )
    model_state = SimpleNamespace(
        feature_frame=pd.DataFrame({"feature": [1]}, index=[0]),
        isolation_scores=[0.1, 0.2],
        ml_flag=pd.Series([False], index=[0]),
        ml_threshold=0.5,
        final_flag=pd.Series([True], index=[0]),
        filtered_joined=pd.DataFrame({"value": [1]}, index=[0]),
        explanation_signals={},
    )
    created_run = False

    monkeypatch.setattr(runner, "get_run_execution_artifact", lambda _payload: {"cached": True})
    monkeypatch.setattr(runner, "get_isolation_forest_artifact", lambda _payload: {"cached": True})
    monkeypatch.setattr(runner, "_execution_state_from_artifact", lambda _payload, _artifact: execution)
    monkeypatch.setattr(runner, "_extract_rule_flags", lambda _joined: rule_flags)
    monkeypatch.setattr(runner, "_isolation_forest_state_from_artifact", lambda _artifact: model_state)
    monkeypatch.setattr(runner, "_calculate_amount_total", lambda _payload, _filtered: 0.0)
    monkeypatch.setattr(
        runner,
        "_build_dataset_frame",
        lambda _inputs: (_ for _ in ()).throw(ValueError("bad scoring length")),
    )

    def fake_create_run_record(*_args, **_kwargs):
        nonlocal created_run
        created_run = True
        return SimpleNamespace(run_id=1)

    monkeypatch.setattr(runner, "_create_run_record", fake_create_run_record)

    with pytest.raises(ValueError, match="bad scoring length"):
        runner.run_workbench(SimpleNamespace(), payload)

    assert created_run is False


def test_run_isolation_forest_scores_only_non_rule_rows(monkeypatch) -> None:
    payload = WorkbenchRunRequest(selected_tables=["dak"])
    execution = WorkbenchExecutionState(
        joined=pd.DataFrame({"value": [1, 2]}, index=[10, 20]),
        source_row_counts={},
        join_debug={},
        warnings=[],
        executed_sql="SELECT 1",
        user_reasons=[],
        applied_feature_rule_count=0,
        applied_user_rule_count=0,
        batch_id="test_batch",
        dataset_table="ML_Features",
        dataset_run_id=1,
    )
    rule_flags = RuleFlagState(
        user_rule_flag=pd.Series([False, False], index=[10, 20]),
        default_rule_flag=pd.Series([True, False], index=[10, 20]),
        combined_rule_flag=pd.Series([True, False], index=[10, 20]),
        user_reason_series=None,
        default_reason_series=pd.Series(["rule hit", None], index=[10, 20]),
    )
    full_feature_frame = pd.DataFrame({"feature": [100.0, 200.0]}, index=[10, 20])
    scored_inputs: list[pd.DataFrame] = []

    monkeypatch.setattr(runner, "load_saved_model_artifact", lambda _payload: {"pipeline": object(), "transformed_feature_names": []})
    monkeypatch.setattr(
        runner,
        "build_saved_model_feature_frame",
        lambda _joined, _artifact: (
            full_feature_frame,
            SimpleNamespace(
                feature_frame=full_feature_frame,
                selected_columns=["feature"],
                dropped_all_missing_columns=[],
                dropped_constant_columns=[],
            ),
        ),
    )

    def fake_score(feature_frame, _artifact):
        scored_inputs.append(feature_frame.copy())
        return np.asarray([[1.0]]), np.asarray([0.91]), pd.Series([True], index=feature_frame.index, dtype=bool), 0.5

    monkeypatch.setattr(runner, "score_with_saved_model", fake_score)
    monkeypatch.setattr(runner, "build_feature_explanation_signals", lambda *_args, **_kwargs: {20: [{"signal": "ml"}]})

    model_state = _run_isolation_forest(payload, execution, rule_flags)

    assert len(scored_inputs) == 1
    assert scored_inputs[0].index.tolist() == [20]
    assert model_state.ml_flag.to_dict() == {10: False, 20: True}
    assert model_state.final_flag.to_dict() == {10: True, 20: True}
    assert np.isnan(model_state.isolation_scores[0])
    assert float(model_state.isolation_scores[1]) == 0.91
    assert model_state.filtered_joined.index.tolist() == [10, 20]
