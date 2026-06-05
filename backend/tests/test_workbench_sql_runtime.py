import sys
import types

redis_stub = types.ModuleType("redis")
redis_stub.Redis = object
redis_exceptions_stub = types.ModuleType("redis.exceptions")
redis_exceptions_stub.RedisError = Exception
sys.modules.setdefault("redis", redis_stub)
sys.modules.setdefault("redis.exceptions", redis_exceptions_stub)

from app.services.workbench.sql_runtime import (
    _build_join_sql,
    _build_date_sequence_anomaly_conditions,
    _build_night_timestamp_anomaly_conditions,
    _build_sql_anomaly_expressions,
    _qualified_builtin_scoring_columns,
)
from app.schemas.workbench_schema import WorkbenchRunRequest
from app.services.workbench.trained_datasets import apply_trained_dataset_defaults


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


def test_build_night_timestamp_anomaly_conditions_only_uses_timestamp_columns() -> None:
    source_columns = {
        "dak": [
            {"column_name": "list_date", "data_type": "date"},
            {"column_name": "created_at", "data_type": "timestamp without time zone"},
            {"column_name": "approved_time", "data_type": "time without time zone"},
        ],
    }

    conditions = _build_night_timestamp_anomaly_conditions(["dak"], source_columns)
    combined_sql = "\n".join(condition for condition, _message in conditions)
    combined_messages = "\n".join(message for _condition, message in conditions)

    assert len(conditions) == 2
    assert 'base."dak.created_at"' in combined_sql
    assert 'base."dak.approved_time"' in combined_sql
    assert 'base."dak.list_date"' not in combined_sql
    assert ">= 21" in combined_sql
    assert "< 6" in combined_sql
    assert "EXTRACT(HOUR" in combined_sql
    assert "<> 0" in combined_sql
    assert "dak.created_at" in combined_messages


def test_invoice_duplicate_rule_uses_preaggregated_ctes() -> None:
    source_columns = {
        "bill": [
            {"column_name": "invoice_no"},
            {"column_name": "invoice_date"},
            {"column_name": "record_status"},
            {"column_name": "fk_dak"},
        ],
        "gem_bill": [
            {"column_name": "invoice_number"},
            {"column_name": "invoice_date"},
            {"column_name": "record_status"},
        ],
    }

    conditions, ctes, outer_joins, evidence_expressions = _build_sql_anomaly_expressions(
        ["bill", "gem_bill"],
        source_columns,
    )
    combined_sql = "\n".join(
        [
            *(condition for condition, _reason in conditions),
            *ctes,
            *outer_joins,
            *evidence_expressions,
        ]
    )

    assert "duplicate_invoice_keys_bill AS" in combined_sql
    assert "duplicate_invoice_keys_gem_bill AS" in combined_sql
    assert "HAVING COUNT(*) >= 2" in combined_sql
    assert "LEFT JOIN duplicate_invoice_keys_bill" in combined_sql
    assert "LEFT JOIN duplicate_invoice_keys_gem_bill" in combined_sql
    assert "similar_fk_daks" in combined_sql
    assert "duplicate_invoice_fk_daks" in combined_sql
    assert "SELECT COUNT(*)" not in combined_sql
    assert 'NULLIF(BTRIM(CAST("invoice_no" AS text)), \'\') AS invoice_key' not in combined_sql
    assert 'NULLIF(BTRIM(CAST(base."bill.invoice_no" AS text)), \'\')' not in combined_sql
    assert '"invoice_no" AS invoice_key' in combined_sql
    assert 'duplicate_invoice_keys_bill.invoice_key = base."bill.invoice_no"' in combined_sql


def test_join_sql_parameterizes_date_filters_and_rule_reasons() -> None:
    payload = WorkbenchRunRequest(
        selected_tables=["dak"],
        from_date="2026-01-01",
        to_date="2026-01-31",
    )
    source_columns = {
        "dak": [
            {"column_name": "id", "data_type": "integer"},
            {"column_name": "list_date", "data_type": "date"},
            {"column_name": "record_status", "data_type": "character varying"},
            {"column_name": "dak_auditor_date", "data_type": "date"},
            {"column_name": "dak_aao_date", "data_type": "date"},
        ],
    }

    sql, _debug, _warnings, params = _build_join_sql(
        payload,
        source_columns,
        row_limit=None,
        projection_mode="full",
    )

    assert "DATE '2026-01-01'" not in sql
    assert "DATE '2026-01-31'" not in sql
    assert "CAST(:date_filter_from AS date)" in sql
    assert "CAST(:date_filter_to AS date)" in sql
    assert params["date_filter_from"] == "2026-01-01"
    assert params["date_filter_to"] == "2026-01-31"
    assert "THEN 'Date sequence violated" not in sql
    assert "THEN :sql_rule_reason_" in sql
    assert any(
        str(value).startswith("Date sequence violated")
        for key, value in params.items()
        if key.startswith("sql_rule_reason_")
    )


def test_cheque_slip_schedule3_uses_left_join_so_missing_schedule_rows_can_be_flagged() -> None:
    payload = apply_trained_dataset_defaults(
        WorkbenchRunRequest(
            selected_tables=["cheque_slip", "schedule3"],
            from_date="2026-01-01",
            to_date="2026-03-10",
        )
    )
    source_columns = {
        "dak": [
            {"column_name": "id", "data_type": "bigint"},
            {"column_name": "list_date", "data_type": "date"},
        ],
        "cheque_slip": [
            {"column_name": "id", "data_type": "bigint"},
            {"column_name": "fk_dak", "data_type": "bigint"},
            {"column_name": "record_status", "data_type": "character varying"},
            {"column_name": "approved", "data_type": "boolean"},
            {"column_name": "fk_aao", "data_type": "bigint"},
            {"column_name": "fk_ao", "data_type": "bigint"},
            {"column_name": "fk_auditor", "data_type": "bigint"},
            {"column_name": "fk_go", "data_type": "bigint"},
        ],
        "schedule3": [
            {"column_name": "id", "data_type": "bigint"},
            {"column_name": "fk_dak", "data_type": "bigint"},
            {"column_name": "record_status", "data_type": "character varying"},
        ],
    }

    sql, join_debug, _warnings, _params = _build_join_sql(
        payload,
        source_columns,
        row_limit=None,
        projection_mode="full",
    )

    assert [(join["left_table"], join["right_table"], join["join_sql"].split()[0]) for join in join_debug] == [
        ("dak", "cheque_slip", "INNER"),
        ("cheque_slip", "schedule3", "LEFT"),
    ]
    assert 'LEFT JOIN' in sql
    assert '"schedule3" AS "schedule3"' in sql
    assert "Approved V cheque_slip count does not match schedule3 P/V count for same fk_dak" in _params.values()
    assert 'base."cheque_slip.fk_aao" IS NULL' in sql
    assert 'base."cheque_slip.fk_ao" IS NULL' in sql
    assert 'base."cheque_slip.fk_auditor" IS NULL' in sql
    assert 'base."cheque_slip.fk_go" IS NULL' not in sql
