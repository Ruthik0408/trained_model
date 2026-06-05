"""Print EXPLAIN ANALYZE plans for dashboard/review result-table queries.

Run from the backend directory:
    ../venv/bin/python scripts/profile_dashboard_queries.py --dataset-table ML_Features
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.dashboard_service import (
    _add_run_filter,
    _dashboard_amount_sql,
    _dataset_query_filter,
)
from app.services.workbench.constants import (
    FEATURE_VALUES_COLUMN,
    FEEDBACK_SCORE_COLUMN,
    FK_DAK_COLUMN,
    IF_SCORE_COLUMN,
    ISOLATION_RULE_COLUMN,
    ML_THRESHOLD_COLUMN,
    RUN_ID_COLUMN,
    SELECTED_TABLES_COLUMN,
    SERIAL_COLUMN,
    USER_RULE_COLUMN,
    USER_RULE_NAME_COLUMN,
)
from app.services.workbench.source_db import (
    _quote,
    _result_table_columns,
    _result_table_exists,
    _result_table_ref,
    _source_connect,
)


def _review_query(dataset_table: str, anomaly_filter: str, run_id: int | None, limit: int, offset: int) -> str:
    available_columns = _result_table_columns(dataset_table)
    selected_columns = [
        column_name
        for column_name in (
            SERIAL_COLUMN,
            SELECTED_TABLES_COLUMN,
            USER_RULE_NAME_COLUMN,
            USER_RULE_COLUMN,
            ISOLATION_RULE_COLUMN,
            IF_SCORE_COLUMN,
            ML_THRESHOLD_COLUMN,
            FEEDBACK_SCORE_COLUMN,
            RUN_ID_COLUMN,
            FK_DAK_COLUMN,
            FEATURE_VALUES_COLUMN,
        )
        if column_name in available_columns
    ]
    projection = ", ".join(_quote(column_name) for column_name in selected_columns) if selected_columns else "*"
    where_clause = _add_run_filter(
        _dataset_query_filter(anomaly_filter),
        run_id,
        dataset_table=dataset_table,
    )
    return (
        f"SELECT {projection} FROM {_result_table_ref(dataset_table)} "
        f"{where_clause} "
        f"ORDER BY {_quote(SERIAL_COLUMN)} ASC "
        f"LIMIT {int(limit)} OFFSET {int(offset)}"
    )


def _summary_query(dataset_table: str, anomaly_filter: str, run_id: int | None) -> str:
    where_clause = _add_run_filter(
        _dataset_query_filter(anomaly_filter),
        run_id,
        dataset_table=dataset_table,
    )
    return (
        "SELECT "
        "COUNT(*) AS total_rows, "
        f"COALESCE(SUM({_dashboard_amount_sql(dataset_table)}), 0.0) AS total_amount "
        f"FROM {_result_table_ref(dataset_table)} "
        f"{where_clause}"
    )


def _print_explain(label: str, sql: str) -> None:
    explain_sql = text(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql}")
    print(f"\n--- {label} ---")
    with _source_connect() as conn:
        for row in conn.execute(explain_sql):
            print(row[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-table", default="ML_Features")
    parser.add_argument("--anomaly-filter", default="all")
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    try:
        if not _result_table_exists(args.dataset_table):
            raise SystemExit(f"Result table does not exist: {args.dataset_table}")

        _print_explain(
            "review_rows",
            _review_query(
                args.dataset_table,
                args.anomaly_filter,
                args.run_id,
                max(1, args.limit),
                max(0, args.offset),
            ),
        )
        _print_explain(
            "summary",
            _summary_query(args.dataset_table, args.anomaly_filter, args.run_id),
        )
    except ConnectionError as exc:
        raise SystemExit(f"Could not connect to the source database: {exc}") from exc


if __name__ == "__main__":
    main()
