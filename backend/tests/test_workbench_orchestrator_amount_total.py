from types import SimpleNamespace

import pandas as pd
import pytest

from app.core.errors import WorkbenchValidationError
from app.schemas.workbench_schema import WorkbenchRunRequest
import app.services.workbench.orchestrator as orchestrator
from app.services.workbench.orchestrator import (
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
        staging_table=None,
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

    monkeypatch.setattr(orchestrator, "get_run_execution_artifact", lambda _payload: {"cached": True})
    monkeypatch.setattr(orchestrator, "get_isolation_forest_artifact", lambda _payload: {"cached": True})
    monkeypatch.setattr(orchestrator, "_execution_state_from_artifact", lambda _payload, _artifact: execution)
    monkeypatch.setattr(orchestrator, "_extract_rule_flags", lambda _joined: rule_flags)
    monkeypatch.setattr(orchestrator, "_isolation_forest_state_from_artifact", lambda _artifact: model_state)
    monkeypatch.setattr(orchestrator, "_calculate_amount_total", lambda _payload, _filtered: 0.0)
    monkeypatch.setattr(
        orchestrator,
        "_build_dataset_frame",
        lambda _inputs: (_ for _ in ()).throw(ValueError("bad scoring length")),
    )

    def fake_create_run_record(*_args, **_kwargs):
        nonlocal created_run
        created_run = True
        return SimpleNamespace(run_id=1)

    monkeypatch.setattr(orchestrator, "_create_run_record", fake_create_run_record)

    with pytest.raises(ValueError, match="bad scoring length"):
        orchestrator.run_workbench(SimpleNamespace(), payload)

    assert created_run is False


def test_run_isolation_forest_reports_staging_cache_inconsistency() -> None:
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
        staging_table=None,
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

    with pytest.raises(WorkbenchValidationError) as exc_info:
        _run_isolation_forest(payload, execution, SimpleNamespace(), rule_flags)

    assert exc_info.value.error_code == "WORKBENCH_STAGING_TABLE_UNAVAILABLE"
    assert "temporary scoring table context" in exc_info.value.suggestion
    assert "Run the workbench again" not in exc_info.value.suggestion
