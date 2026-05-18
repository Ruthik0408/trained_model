from sqlalchemy import create_engine, text

from app.schemas.workbench_schema import OutlierRuleInput
from app.schemas.workbench_schema import WorkbenchRunRequest
from app.services.workbench.constants import TEMP_ROW_ID_COLUMN
from app.services.workbench.sql_runtime import _build_sql_outlier_predicate, _read_temp_scoring_frame


def test_read_temp_scoring_frame_keeps_full_joined_columns() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    payload = WorkbenchRunRequest(selected_tables=["dak", "bill"])

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE temp_scoring (
                    "__ml_row_number" INTEGER,
                    "dak.amount" REAL,
                    "bill.fk_ao" INTEGER,
                    "bill.approved" BOOLEAN,
                    "bill.passed" BOOLEAN,
                    "sql_rule_flag" BOOLEAN,
                    "invoice_date_to_bill_date" REAL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO temp_scoring (
                    "__ml_row_number",
                    "dak.amount",
                    "bill.fk_ao",
                    "bill.approved",
                    "bill.passed",
                    "sql_rule_flag",
                    "invoice_date_to_bill_date"
                ) VALUES
                    (1, 125.5, 10, 1, 1, 0, 2.0),
                    (2, 225.0, 20, 0, NULL, 1, 5.0)
                """
            )
        )

        frame = _read_temp_scoring_frame(conn, "temp_scoring", payload)

    assert frame.index.tolist() == [1, 2]
    assert TEMP_ROW_ID_COLUMN not in frame.columns
    assert "dak.amount" in frame.columns
    assert "bill.fk_ao" in frame.columns
    assert "bill.approved" in frame.columns
    assert "bill.passed" in frame.columns
    assert "sql_rule_flag" in frame.columns
    assert "invoice_date_to_bill_date" in frame.columns
    assert frame.loc[1, "dak.amount"] == 125.5
    assert frame.loc[2, "bill.fk_ao"] == 20


def test_build_sql_outlier_predicate_compares_datetime_columns_as_timestamps() -> None:
    params = {}
    source_columns = {
        "dak": [
            {"column_name": "created_at", "data_type": "timestamp without time zone"},
        ],
        "bill": [
            {"column_name": "created_at", "data_type": "timestamp without time zone"},
        ],
    }
    rule = OutlierRuleInput(
        name="dak after bill",
        first_column="dak.created_at",
        second_column="bill.created_at",
        operator=">",
    )

    predicate = _build_sql_outlier_predicate(
        ["dak", "bill"],
        source_columns,
        rule,
        params,
        alias="src",
    )

    assert "::double precision" not in predicate
    assert "CAST" not in predicate or "CAST(" in predicate
    assert "REPLACE" in predicate
    assert "::timestamp" in predicate
    assert ">" in predicate
    assert params == {}


def test_build_sql_outlier_predicate_compares_datetime_literal_as_timestamp() -> None:
    params = {}
    source_columns = {
        "dak": [
            {"column_name": "created_at", "data_type": "timestamp without time zone"},
        ],
    }
    rule = OutlierRuleInput(
        name="dak after cutoff",
        first_column="dak.created_at",
        operator=">=",
        value="2025-01-01 10:30:00",
    )

    predicate = _build_sql_outlier_predicate(
        ["dak"],
        source_columns,
        rule,
        params,
        alias="src",
    )

    assert "CAST(:rule_ts_1 AS timestamp)" in predicate
    assert params["rule_ts_1"] == "2025-01-01 10:30:00"
