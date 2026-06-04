import argparse
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.train_models import (
    build_count_sql,
    build_training_pipeline,
    build_select_sql,
    dataset_list_date_from,
    get_transformed_feature_names,
    limit_clause,
    list_date_where_clause,
    normalize_boolean_feature_columns,
    validate_date_window,
    validate_max_training_rows,
)
from app.services.workbench.model_cleaning import business_day_gap_series


def test_list_date_where_clause_supports_from_and_cutoff() -> None:
    schema_map = {
        "dak": pd.DataFrame({"column_name": ["id", "list_date"]}),
    }

    where_sql = list_date_where_clause(
        "dak",
        schema_map,
        list_date_from="2025-01-01",
        list_date_cutoff="2026-01-01",
    )

    assert where_sql == "WHERE dak.list_date >= '2025-01-01' AND dak.list_date < '2026-01-01'"


def test_validate_date_window_rejects_inverted_range() -> None:
    parser = argparse.ArgumentParser()

    with pytest.raises(SystemExit):
        validate_date_window("2026-01-01", "2025-01-01", parser)


def test_validate_max_training_rows_rejects_negative_values() -> None:
    parser = argparse.ArgumentParser()

    with pytest.raises(SystemExit):
        validate_max_training_rows(-1, parser)


def test_limit_clause_allows_zero_to_disable_limit() -> None:
    assert limit_clause(0) == ""
    assert limit_clause(250) == "LIMIT 250"


def test_build_select_sql_applies_max_training_rows() -> None:
    schema_map = {
        "dak": pd.DataFrame({"column_name": ["id", "list_date"]}),
    }

    sql = build_select_sql(
        "dak",
        [],
        schema_map,
        list_date_from="2025-08-01",
        list_date_cutoff="2026-01-01",
        max_training_rows=100_000,
    )

    assert "WHERE dak.list_date >= '2025-08-01' AND dak.list_date < '2026-01-01'" in sql
    assert "ORDER BY dak.\"list_date\", dak.\"id\"" in sql
    assert "LIMIT 100000" in sql


def test_build_select_sql_filters_cheque_schedule_rules_before_limit() -> None:
    schema_map = {
        "dak": pd.DataFrame({"column_name": ["id", "list_date"]}),
        "cheque_slip": pd.DataFrame(
            {"column_name": ["id", "fk_dak", "record_status", "approved", "fk_aao", "fk_ao", "fk_auditor"]}
        ),
        "schedule3": pd.DataFrame({"column_name": ["id", "fk_dak", "record_status"]}),
    }

    sql = build_select_sql(
        "dak",
        ["cheque_slip", "schedule3"],
        schema_map,
        dataset_name="dak.cheque_slip.schedule3",
        list_date_from=None,
        list_date_cutoff="2026-01-01",
        max_training_rows=100_000,
    )

    assert "WITH joined_base AS" in sql
    assert "WHERE NOT" in sql
    assert "cheque_slip_approved_by_dak.cheque_slip_v_approved_count" in sql
    assert 'base."cheque_slip_fk_aao" IS NULL' in sql
    assert 'base."cheque_slip_fk_ao" IS NULL' in sql
    assert 'base."cheque_slip_fk_auditor" IS NULL' in sql
    assert "ORDER BY \"dak_list_date\", \"dak_id\", \"cheque_slip_id\", \"schedule3_id\"" in sql
    assert sql.rfind("WHERE NOT") < sql.rfind("LIMIT 100000")


def test_build_select_sql_filters_cheque_slip_missing_approval_owners_for_all_cheque_datasets() -> None:
    schema_map = {
        "dak": pd.DataFrame({"column_name": ["id", "list_date"]}),
        "cheque_slip": pd.DataFrame(
            {"column_name": ["id", "fk_dak", "approved", "fk_aao", "fk_ao", "fk_auditor"]}
        ),
        "ecs": pd.DataFrame({"column_name": ["id", "fk_dak"]}),
    }

    sql = build_select_sql(
        "dak",
        ["cheque_slip", "ecs"],
        schema_map,
        dataset_name="dak.cheque_slip.ecs",
        list_date_from=None,
        list_date_cutoff="2026-01-01",
        max_training_rows=100_000,
    )

    assert "WITH joined_base AS" in sql
    assert 'base."cheque_slip_approved" = true' in sql
    assert 'base."cheque_slip_fk_aao" IS NULL' in sql
    assert 'base."cheque_slip_fk_ao" IS NULL' in sql
    assert 'base."cheque_slip_fk_auditor" IS NULL' in sql
    assert sql.rfind("WHERE NOT") < sql.rfind("LIMIT 100000")


def test_build_count_sql_filters_cheque_slip_missing_approval_owners() -> None:
    schema_map = {
        "dak": pd.DataFrame({"column_name": ["id", "list_date"]}),
        "cheque_slip": pd.DataFrame(
            {"column_name": ["id", "fk_dak", "approved", "fk_aao", "fk_ao", "fk_auditor"]}
        ),
        "ecs": pd.DataFrame({"column_name": ["id", "fk_dak"]}),
    }

    sql = build_count_sql(
        "dak",
        ["cheque_slip", "ecs"],
        schema_map,
        list_date_from=None,
        list_date_cutoff="2026-01-01",
    )

    assert 'WHERE NOT (' in sql or 'AND NOT (' in sql
    assert 'cheque_slip."approved" = true' in sql
    assert 'cheque_slip."fk_aao" IS NULL' in sql
    assert 'cheque_slip."fk_ao" IS NULL' in sql
    assert 'cheque_slip."fk_auditor" IS NULL' in sql


def test_dataset_list_date_from_uses_dataset_specific_defaults() -> None:
    args = argparse.Namespace(list_date_from=None)

    assert dataset_list_date_from("dak", args) == "2024-01-01"
    assert dataset_list_date_from("dak.echs_medical_bill", args) == "2025-01-01"
    assert dataset_list_date_from("dak.bill", args) is None


def test_dataset_list_date_from_allows_global_override() -> None:
    args = argparse.Namespace(list_date_from="2025-06-01")

    assert dataset_list_date_from("dak.bill", args) == "2025-06-01"


def test_boolean_features_are_converted_directly_instead_of_one_hot_encoded() -> None:
    feature_df = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0, 40.0],
            "schedule3_approved": [True, False, True, False],
            "record_status": ["V", "R", "V", "R"],
        }
    )
    pipeline = build_training_pipeline(
        ["amount"],
        ["schedule3_approved"],
        ["record_status"],
        contamination=0.25,
        random_state=42,
    )

    feature_df = normalize_boolean_feature_columns(feature_df, ["schedule3_approved"])
    pipeline.fit(feature_df)
    feature_names = get_transformed_feature_names(
        pipeline,
        ["amount"],
        ["schedule3_approved"],
        ["record_status"],
    )

    assert "schedule3_approved" in feature_names
    assert "schedule3_approved_f" not in feature_names
    assert "schedule3_approved_t" not in feature_names


def test_business_day_gap_series_excludes_weekends() -> None:
    previous = pd.Series(["2025-01-03", "2025-01-06"])  # Friday, Monday
    next_value = pd.Series(["2025-01-06", "2025-01-07"])  # Monday, Tuesday

    gap = business_day_gap_series(previous, next_value)

    assert gap.astype("Float64").tolist() == [1.0, 1.0]
