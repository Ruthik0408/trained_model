from app.services.workbench.constants import (
    FEEDBACK_SCORE_COLUMN,
    ISOLATION_RULE_COLUMN,
    RUN_ID_COLUMN,
    SELECTED_TABLES_COLUMN,
    SERIAL_COLUMN,
    USER_RULE_COLUMN,
)
from app.services.workbench.result_store import _ensure_result_table_indexes


class RecordingConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement, *_args, **_kwargs):
        self.statements.append(str(statement))


def test_ensure_result_table_indexes_creates_dashboard_indexes() -> None:
    conn = RecordingConnection()

    _ensure_result_table_indexes(
        conn,
        "ML_Features",
        {
            SERIAL_COLUMN,
            RUN_ID_COLUMN,
            SELECTED_TABLES_COLUMN,
            FEEDBACK_SCORE_COLUMN,
            USER_RULE_COLUMN,
            ISOLATION_RULE_COLUMN,
        },
    )

    sql = "\n".join(conn.statements)

    assert '"id"' in sql
    assert f'"{RUN_ID_COLUMN}"' in sql
    assert '"selected_tables"' in sql
    assert '"feedback_score"' in sql
    assert f'LOWER(BTRIM("{USER_RULE_COLUMN}"::text))' in sql
    assert f'LOWER(BTRIM("{ISOLATION_RULE_COLUMN}"::text))' in sql
    assert f'"{RUN_ID_COLUMN}", "id" DESC' in sql


def test_ensure_result_table_indexes_skips_missing_optional_columns() -> None:
    conn = RecordingConnection()

    _ensure_result_table_indexes(conn, "ML_Features", {SERIAL_COLUMN})

    sql = "\n".join(conn.statements)

    assert '"id"' in sql
    assert f'"{RUN_ID_COLUMN}"' not in sql
    assert USER_RULE_COLUMN not in sql
