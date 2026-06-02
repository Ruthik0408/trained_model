from app.core.database import _missing_workbench_run_column_ddls


def test_missing_workbench_run_column_ddls_only_adds_missing_columns() -> None:
    existing_columns = {
        "run_id",
        "run_name",
        "source_tables_json",
        "join_config_json",
        "feature_rules_json",
        "amount_field",
        "total_rows",
    }

    statements = _missing_workbench_run_column_ddls(existing_columns)

    assert any('"user_rule_count" INTEGER NOT NULL DEFAULT 0' in stmt for stmt in statements)
    assert any('"ml_anomaly_count" INTEGER NOT NULL DEFAULT 0' in stmt for stmt in statements)
    assert any('"final_anomaly_count" INTEGER NOT NULL DEFAULT 0' in stmt for stmt in statements)
    assert any('"status" VARCHAR(30) NOT NULL DEFAULT \'COMPLETED\'' in stmt for stmt in statements)
    assert all('"run_name"' not in stmt for stmt in statements)


def test_missing_workbench_run_column_ddls_returns_empty_when_schema_is_current() -> None:
    existing_columns = {
        "run_name",
        "source_tables_json",
        "join_config_json",
        "feature_rules_json",
        "amount_field",
        "total_rows",
        "user_rule_count",
        "ml_anomaly_count",
        "final_anomaly_count",
        "selected_model",
        "metrics_json",
        "status",
    }

    assert _missing_workbench_run_column_ddls(existing_columns) == []
