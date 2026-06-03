import sys
import types

redis_stub = types.ModuleType("redis")
redis_stub.Redis = object
redis_exceptions_stub = types.ModuleType("redis.exceptions")
redis_exceptions_stub.RedisError = Exception
sys.modules.setdefault("redis", redis_stub)
sys.modules.setdefault("redis.exceptions", redis_exceptions_stub)

from app.services.workbench.sql_runtime import (
    _build_date_sequence_anomaly_conditions,
    _build_dynamic_date_gap_rules,
    _qualified_builtin_scoring_columns,
)


def test_build_dynamic_date_gap_rules_keeps_prefixed_dates_in_same_family() -> None:
    available = {
        "dak.dak_list_date": "date",
        "dak.dak_auditor_date": "date",
        "dak.dak_aao_date": "date",
        "bill.bill_list_date": "date",
        "bill.bill_auditor_date": "date",
        "bill.bill_aao_date": "date",
    }

    rules = _build_dynamic_date_gap_rules(available)
    column_pairs = {(rule["first_column"], rule["second_column"]) for rule in rules}

    assert ("dak.dak_aao_date", "dak.dak_auditor_date") in column_pairs
    assert ("bill.bill_aao_date", "bill.bill_auditor_date") in column_pairs
    assert ("dak.dak_auditor_date", "dak.dak_list_date") not in column_pairs
    assert ("bill.bill_auditor_date", "bill.bill_list_date") not in column_pairs
    assert ("bill.bill_auditor_date", "dak.dak_list_date") not in column_pairs
    assert ("bill.bill_aao_date", "dak.dak_auditor_date") not in column_pairs


def test_qualified_builtin_scoring_columns_includes_prefixed_stage_dates() -> None:
    source_columns = {
        "dak": [
            {"column_name": "dak_list_date"},
            {"column_name": "dak_auditor_date"},
        ],
        "bill": [
            {"column_name": "bill_auditor_date"},
            {"column_name": "bill_aao_date"},
        ],
    }

    required = _qualified_builtin_scoring_columns(["dak", "bill"], source_columns)

    assert "dak.dak_auditor_date" in required
    assert "bill.bill_auditor_date" in required
    assert "bill.bill_aao_date" in required
    assert "dak.dak_list_date" not in required


def test_build_date_sequence_anomaly_conditions_avoids_cross_family_comparisons() -> None:
    source_columns = {
        "dak": [
            {"column_name": "dak_list_date"},
            {"column_name": "dak_auditor_date"},
            {"column_name": "dak_aao_date"},
        ],
        "bill": [
            {"column_name": "bill_list_date"},
            {"column_name": "bill_auditor_date"},
            {"column_name": "bill_aao_date"},
        ],
    }

    comparisons = _build_date_sequence_anomaly_conditions(["dak", "bill"], source_columns)
    combined_sql = "\n".join(condition for condition, _message in comparisons)

    assert 'base."dak.dak_auditor_date"' in combined_sql
    assert 'base."bill.bill_auditor_date"' in combined_sql
    assert 'base."dak.dak_list_date"' not in combined_sql
    assert 'base."bill.bill_list_date"' not in combined_sql
    assert 'base."dak.dak_list_date") > (CAST(base."bill.bill_auditor_date"' not in combined_sql
    assert 'base."bill.bill_list_date") > (CAST(base."dak.dak_auditor_date"' not in combined_sql
