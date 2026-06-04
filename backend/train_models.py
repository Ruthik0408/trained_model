from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from business_rules import (
    cheque_slip_approval_owner_columns,
    cheque_slip_approval_owner_training_condition_sql,
    cheque_slip_schedule3_count_mismatch_rule,
    cheque_slip_schedule3_not_approved_rule,
    cheque_slip_schedule3_shared_sql_fragments,
)
from model_cleaning import (
    build_date_sequence_summary,
    build_invoice_row_filter_summary,
    normalize_boolean_feature_columns,
    apply_date_sequence_summary,
    apply_invoice_row_filter_summary,
)


def load_env_file(env_path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file."""
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ[key.strip()] = value


def find_env_file() -> Path:
    """Look for the project .env from common notebook/module locations."""
    module_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        module_dir / ".env",
        module_dir.parent / ".env",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find .env in the current directory or its parent.")


ENV_PATH = find_env_file()
load_env_file(ENV_PATH)

DB_CONFIG = {
    "host": os.environ.get("TULIP_SOURCE_DB_HOST"),
    "port": os.environ.get("TULIP_SOURCE_DB_PORT", "5432"),
    "dbname": os.environ.get("TULIP_SOURCE_DB_NAME"),
    "user": os.environ.get("TULIP_SOURCE_DB_USER"),
    "password": os.environ.get("TULIP_SOURCE_DB_PASSWORD"),
    "schema": os.environ.get("TULIP_SOURCE_DB_SCHEMA", "public"),
}

missing = [key for key, value in DB_CONFIG.items() if key != "schema" and not value]
if missing:
    raise ValueError(f"Missing required database settings: {missing}")


def query_postgres(sql: str) -> pd.DataFrame:
    """Run a SQL query through psql and return the result as a dataframe."""
    cleaned_sql = sql.strip().rstrip(";")
    psql_sql = f"COPY ({cleaned_sql}) TO STDOUT WITH CSV HEADER"

    env = os.environ.copy()
    env["PGPASSWORD"] = DB_CONFIG["password"]
    env["PGOPTIONS"] = f"{env.get('PGOPTIONS', '')} -c search_path={DB_CONFIG['schema']}".strip()

    command = [
        "psql",
        "-h",
        DB_CONFIG["host"],
        "-p",
        str(DB_CONFIG["port"]),
        "-U",
        DB_CONFIG["user"],
        "-d",
        DB_CONFIG["dbname"],
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        psql_sql,
    ]

    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", newline="", suffix=".csv") as csv_file:
        try:
            subprocess.run(
                command,
                stdout=csv_file,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or str(exc)).strip()
            raise RuntimeError(f"psql query failed: {detail}") from exc

        csv_file.flush()
        df = pd.read_csv(csv_file.name, low_memory=False)
    if "SET" in df.columns:
        df = df.drop(columns=["SET"])

    return df


def get_connection_check_df() -> pd.DataFrame:
    return query_postgres(
        """
        SELECT
            current_database() AS database_name,
            current_schema() AS schema_name,
            current_user AS database_user,
            now() AS connected_at
        """
    )



BASE_TABLE = "dak"
DEFAULT_LIST_DATE_CUTOFF = "2026-01-01"
DEFAULT_MAX_TRAINING_ROWS = 300000
DEFAULT_DATASET_LIST_DATE_FROM: dict[str, str] = {
    "dak": "2024-01-01",
    "dak.echs_medical_bill": "2025-01-01",
    "dak.cheque_slip.schedule3": "2025-01-01",
    "dak.cheque_slip.ecs": "2025-01-01",
}

NUMERIC_DATA_TYPES = {
    "smallint",
    "integer",
    "bigint",
    "decimal",
    "numeric",
    "real",
    "double precision",
}
BOOLEAN_DATA_TYPES = {"boolean"}
DATE_DATA_TYPES = {
    "date",
    "timestamp without time zone",
    "timestamp with time zone",
    "time without time zone",
    "time with time zone",
}
TEXT_DATA_TYPES = {"text"}

DATE_SEQUENCE_STAGE_ALIASES = [
    ("invoice_date", ["invoice_date"]),
    ("bill_date", ["bill_date"]),
    ("reference_date", ["reference_date"]),
    ("auditor_stage", ["auditor_date", "aud_date", "auditor_disposal_date"]),
    ("aao_stage", ["aao_date", "aao_disposal_date"]),
    ("ao_stage", ["ao_date", "ao_disposal_date"]),
    ("go_date", ["go_date"]),
    ("dp_sheet_date", ["dp_sheet_date"]),
    ("cmp_date", ["cmp_date", "cmp_batch_date", "cmp_file_gen_date"]),
    ("disposal_date", ["disposal_date"]),
]
SAME_TABLE_DATE_SEQUENCE_STAGES = {
    "auditor_stage",
    "aao_stage",
    "ao_stage",
    "go_date",

}

GLOBAL_SEQUENCE_STAGE_ORDER = [
    "invoice_date",
    "bill_date",
    "reference_date",
    "dp_sheet_date",
    "cmp_date",
    "disposal_date",
]

TABLE_SEQUENCE_STAGE_ORDER = [
    "invoice_date",
    "bill_date",
    "reference_date",
    "auditor_stage",
    "aao_stage",
    "ao_stage",
    "go_date",
    "dp_sheet_date",
    "cmp_date",
    "disposal_date",
]


@dataclass(frozen=True)
class JoinSpec:
    right_table: str
    left_table: str
    left_column: str
    right_column: str
    join_type: str = "JOIN"


JOIN_SPECS: dict[str, JoinSpec] = {
    "bill": JoinSpec(
        right_table="bill",
        left_table="dak",
        left_column="id",
        right_column="fk_dak",
    ),
    "gem_bill": JoinSpec(
        right_table="gem_bill",
        left_table="dak",
        left_column="id",
        right_column="fk_dak",
    ),
    "civ_medical_bill": JoinSpec(
        right_table="civ_medical_bill",
        left_table="dak",
        left_column="id",
        right_column="fk_dak",
    ),
    "civ_paybill": JoinSpec(
        right_table="civ_paybill",
        left_table="dak",
        left_column="id",
        right_column="fk_dak",
    ),
    "civ_tada_ltc_bill": JoinSpec(
        right_table="civ_tada_ltc_bill",
        left_table="dak",
        left_column="id",
        right_column="fk_dak",
    ),
    "echs_medical_bill": JoinSpec(
        right_table="echs_medical_bill",
        left_table="dak",
        left_column="id",
        right_column="fk_dak",
    ),
    "cheque_slip": JoinSpec(
        right_table="cheque_slip",
        left_table="dak",
        left_column="id",
        right_column="fk_dak",
    ),
    "schedule3": JoinSpec(
        right_table="schedule3",
        left_table="cheque_slip",
        left_column="fk_dak",
        right_column="fk_dak",
        join_type="LEFT JOIN",
    ),
    "ecs": JoinSpec(
        right_table="ecs",
        left_table="cheque_slip",
        left_column="fk_dak",
        right_column="fk_dak",
    ),
}

DATASET_TABLES: dict[str, list[str]] = {
    "dak": [],
    "dak.bill": ["bill"],
    "dak.gem_bill": ["gem_bill"],
    "dak_info": [],
    "dak.civ_medical_bill": ["civ_medical_bill"],
    "dak.civ_paybill": ["civ_paybill"],
    "dak.civ_tada_ltc_bill": ["civ_tada_ltc_bill"],
    "dak.echs_medical_bill": ["echs_medical_bill"],
    "dak.cheque_slip.schedule3": ["cheque_slip", "schedule3"],
    "dak.cheque_slip.ecs": ["cheque_slip", "ecs"],
}

DATASET_BASE_TABLES: dict[str, str] = {
    "dak_info": "dak_info",
}


def dataset_base_table(dataset_name: str) -> str:
    return DATASET_BASE_TABLES.get(dataset_name, BASE_TABLE)


def dataset_list_date_from(dataset_name: str, args: argparse.Namespace) -> str | None:
    if args.list_date_from:
        return args.list_date_from
    return DEFAULT_DATASET_LIST_DATE_FROM.get(dataset_name)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train anomaly-detection pipelines for dak and dak join datasets "
            "using dataset-specific cleaning, date-sequence validation, and "
            "date-gap feature engineering."
        )
    )
    parser.add_argument(
        "--list-date-from",
        default=None,
        help=(
            "Optional global lower bound: only rows with list_date on or after this date are included. "
            "Format: YYYY-MM-DD. If omitted, dak uses 2024-01-01, dak.echs_medical_bill uses "
            "2025-01-01, and remaining datasets have no lower bound."
        ),
    )
    parser.add_argument(
        "--list-date-cutoff",
        default=DEFAULT_LIST_DATE_CUTOFF,
        help="Only rows with dak.list_date earlier than this date are included. Format: YYYY-MM-DD.",
    )
    parser.add_argument(
        "--max-training-rows",
        type=int,
        default=DEFAULT_MAX_TRAINING_ROWS,
        help=(
            "Maximum joined rows to load into pandas for each dataset. "
            "Use 0 to train on every matching row."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="backend/artifacts",
        help="Directory where trained models and JSON reports will be saved.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASET_TABLES),
        help=(
            "Datasets to train. Example: --datasets dak dak_info dak.bill dak.gem_bill "
            "dak.civ_medical_bill dak.civ_paybill dak.civ_tada_ltc_bill dak.echs_medical_bill "
            "dak.cheque_slip.schedule3 dak.cheque_slip.ecs"
        ),
    )
    parser.add_argument(
        "--null-threshold",
        type=float,
        default=80.0,
        help="Drop columns whose null percentage is greater than or equal to this value.",
    )
    parser.add_argument(
        "--varchar-length-threshold",
        type=int,
        default=10,
        help="Drop character varying columns longer than this threshold.",
    )
    parser.add_argument(
        "--categorical-cardinality-threshold",
        type=int,
        default=10,
        help="One-hot encode remaining categorical columns up to this distinct-value count.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="IsolationForest contamination value.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for model training.",
    )
    args = parser.parse_args()
    validate_date_window(args.list_date_from, args.list_date_cutoff, parser)
    validate_max_training_rows(args.max_training_rows, parser)
    return args


def parse_iso_date(value: str, field_name: str, parser: argparse.ArgumentParser) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        parser.error(f"{field_name} must be in YYYY-MM-DD format.")


def validate_date_window(
    list_date_from: str | None,
    list_date_cutoff: str,
    parser: argparse.ArgumentParser,
) -> None:
    from_date = parse_iso_date(list_date_from, "--list-date-from", parser) if list_date_from else None
    cutoff_date = parse_iso_date(list_date_cutoff, "--list-date-cutoff", parser)
    if from_date and from_date >= cutoff_date:
        parser.error("--list-date-from must be earlier than --list-date-cutoff.")


def validate_max_training_rows(
    max_training_rows: int,
    parser: argparse.ArgumentParser,
) -> None:
    if max_training_rows < 0:
        parser.error("--max-training-rows must be 0 or greater.")


def limit_clause(max_training_rows: int) -> str:
    if max_training_rows <= 0:
        return ""
    return f"LIMIT {max_training_rows}"


def training_order_clause(base_table: str, schema_map: dict[str, pd.DataFrame]) -> str:
    column_names = set(schema_map[base_table]["column_name"].tolist())
    order_columns = []
    for column_name in ("list_date", "id"):
        if column_name in column_names:
            order_columns.append(f'{base_table}."{column_name}"')
    if not order_columns:
        return ""
    return "ORDER BY " + ", ".join(order_columns)


def joined_base_order_clause(base_table: str, join_tables: list[str], schema_map: dict[str, pd.DataFrame]) -> str:
    order_columns = []
    for table_name, column_name in (
        (base_table, "list_date"),
        (base_table, "id"),
        ("cheque_slip", "id"),
        ("schedule3", "id"),
    ):
        if table_name != base_table and table_name not in join_tables:
            continue
        if column_name in set(schema_map[table_name]["column_name"].tolist()):
            order_columns.append(f'"{table_name}_{column_name}"')
    if not order_columns:
        return ""
    return "ORDER BY " + ", ".join(order_columns)


def get_table_schema(table_name: str) -> pd.DataFrame:
    return query_postgres(
        f"""
        SELECT
            column_name,
            data_type,
            udt_name,
            character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = '{DB_CONFIG["schema"]}'
          AND table_name = '{table_name}'
        ORDER BY ordinal_position
        """
    )


def prefixed_schema(table_name: str) -> pd.DataFrame:
    schema_df = get_table_schema(table_name).copy()
    schema_df["source_table"] = table_name
    schema_df["prefixed_column_name"] = schema_df["column_name"].map(
        lambda column_name: f"{table_name}_{column_name}"
    )
    return schema_df


def raw_joined_columns_for_tables(
    table_names: list[str],
    schema_map: dict[str, pd.DataFrame],
) -> list[str]:
    return [
        f"{table_name}.{column_name}"
        for table_name in table_names
        for column_name in schema_map[table_name]["column_name"].tolist()
    ]


def list_date_where_clause(
    base_table: str,
    schema_map: dict[str, pd.DataFrame],
    *,
    list_date_from: str | None,
    list_date_cutoff: str,
) -> str:
    base_columns = set(schema_map[base_table]["column_name"].tolist())
    if "list_date" not in base_columns:
        return ""
    filters = []
    if list_date_from:
        filters.append(f"{base_table}.list_date >= '{list_date_from}'")
    filters.append(f"{base_table}.list_date < '{list_date_cutoff}'")
    return f"WHERE {' AND '.join(filters)}"


def build_select_sql(
    base_table: str,
    join_tables: list[str],
    schema_map: dict[str, pd.DataFrame],
    *,
    dataset_name: str | None = None,
    list_date_from: str | None,
    list_date_cutoff: str,
    max_training_rows: int,
) -> str:
    select_clauses: list[str] = []

    for table_name in [base_table, *join_tables]:
        for column_name in schema_map[table_name]["column_name"].tolist():
            select_clauses.append(
                f'{table_name}."{column_name}" AS "{table_name}_{column_name}"'
            )

    join_clauses: list[str] = []
    for table_name in join_tables:
        join_spec = JOIN_SPECS[table_name]
        join_clauses.append(
            (
                f"{join_spec.join_type} {join_spec.right_table} "
                f'ON {join_spec.left_table}."{join_spec.left_column}" = '
                f'{join_spec.right_table}."{join_spec.right_column}"'
            )
        )

    join_sql = "\n".join(join_clauses)
    where_sql = list_date_where_clause(
        base_table,
        schema_map,
        list_date_from=list_date_from,
        list_date_cutoff=list_date_cutoff,
    )
    limit_sql = limit_clause(max_training_rows)
    order_sql = training_order_clause(base_table, schema_map)

    base_sql = f"""
        SELECT
            {", ".join(select_clauses)}
        FROM {base_table}
        {join_sql}
        {where_sql}
    """

    cheque_slip_columns = (
        set(schema_map["cheque_slip"]["column_name"].tolist())
        if "cheque_slip" in schema_map
        else set()
    )
    cheque_slip_owner_columns = cheque_slip_approval_owner_columns(cheque_slip_columns)
    has_cheque_slip_owner_rule = (
        "cheque_slip" in join_tables
        and "approved" in cheque_slip_columns
        and bool(cheque_slip_owner_columns)
    )
    cheque_slip_owner_rule_sql = (
        cheque_slip_approval_owner_training_condition_sql(
            cheque_slip_owner_columns,
            approved_expr='base."cheque_slip_approved"',
            column_expr_template='base."cheque_slip_{column_name}"',
        )
        if has_cheque_slip_owner_rule
        else ""
    )

    if dataset_name == "dak.cheque_slip.schedule3":
        filtered_order_sql = joined_base_order_clause(base_table, join_tables, schema_map)
        schedule3_fragments = cheque_slip_schedule3_shared_sql_fragments(
            schedule3_table_ref="schedule3",
            cheque_slip_table_ref="cheque_slip",
            fk_dak_join_expr='base."cheque_slip_fk_dak"',
        )
        rule_clauses = [
            cheque_slip_schedule3_not_approved_rule(
                record_status_expr='base."cheque_slip_record_status"',
                approved_expr='base."cheque_slip_approved"',
                fk_dak_expr='base."cheque_slip_fk_dak"',
            ).condition_sql,
            cheque_slip_schedule3_count_mismatch_rule(
                record_status_expr='base."cheque_slip_record_status"',
                approved_expr='base."cheque_slip_approved"',
                fk_dak_expr='base."cheque_slip_fk_dak"',
            ).condition_sql,
        ]
        if cheque_slip_owner_rule_sql:
            rule_clauses.append(cheque_slip_owner_rule_sql)
        return f"""
            WITH joined_base AS (
                {base_sql}
            ),
            {",".join(schedule3_fragments.ctes)}
            SELECT base.*
            FROM joined_base base
            {"".join(schedule3_fragments.outer_joins)}
            WHERE NOT (
                {" OR ".join(rule_clauses)}
            )
            {filtered_order_sql}
            {limit_sql}
        """

    if has_cheque_slip_owner_rule:
        filtered_order_sql = joined_base_order_clause(base_table, join_tables, schema_map)
        return f"""
            WITH joined_base AS (
                {base_sql}
            )
            SELECT base.*
            FROM joined_base base
            WHERE NOT (
                {cheque_slip_owner_rule_sql}
            )
            {filtered_order_sql}
            {limit_sql}
        """

    return f"""
        {base_sql}
        {order_sql}
        {limit_sql}
    """


def build_count_sql(
    base_table: str,
    join_tables: list[str],
    schema_map: dict[str, pd.DataFrame],
    *,
    list_date_from: str | None,
    list_date_cutoff: str,
) -> str:
    join_clauses: list[str] = []
    for table_name in join_tables:
        join_spec = JOIN_SPECS[table_name]
        join_clauses.append(
            (
                f"{join_spec.join_type} {join_spec.right_table} "
                f'ON {join_spec.left_table}."{join_spec.left_column}" = '
                f'{join_spec.right_table}."{join_spec.right_column}"'
            )
        )

    join_sql = "\n".join(join_clauses)
    where_sql = list_date_where_clause(
        base_table,
        schema_map,
        list_date_from=list_date_from,
        list_date_cutoff=list_date_cutoff,
    )

    cheque_slip_columns = (
        set(schema_map["cheque_slip"]["column_name"].tolist())
        if "cheque_slip" in schema_map
        else set()
    )
    cheque_slip_owner_columns = cheque_slip_approval_owner_columns(cheque_slip_columns)
    if (
        "cheque_slip" in join_tables
        and "approved" in cheque_slip_columns
        and cheque_slip_owner_columns
    ):
        filter_sql = "\n            AND ".join(
            f'cheque_slip."{column_name}" IS NULL'
            for column_name in cheque_slip_owner_columns
        )
        return f"""
            SELECT COUNT(*) AS row_count
            FROM {base_table}
            {join_sql}
            {where_sql}
            {"AND" if where_sql else "WHERE"} NOT (
                cheque_slip."approved" = true
                AND {filter_sql}
            )
        """

    return f"""
        SELECT COUNT(*) AS row_count
        FROM {base_table}
        {join_sql}
        {where_sql}
    """


def merged_schema_for_tables(
    base_table: str,
    join_tables: list[str],
    schema_map: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    return pd.concat(
        [schema_map[table_name] for table_name in [base_table, *join_tables]],
        ignore_index=True,
    )


def append_column_operation(
    operation_map: dict[str, list[str]],
    column_name: str,
    operation: str,
) -> None:
    operation_map.setdefault(column_name, [])
    if operation not in operation_map[column_name]:
        operation_map[column_name].append(operation)


def first_matching_column(
    available_columns: list[str],
    table_name: str,
    aliases: list[str],
) -> str | None:
    for alias in aliases:
        candidate = f"{table_name}_{alias}"
        if candidate in available_columns:
            return candidate
    return None


def drop_voided_invoice_rows(
    df: pd.DataFrame,
    schema_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    summary = build_invoice_row_filter_summary(df, schema_df)
    return apply_invoice_row_filter_summary(df, summary), summary


def filter_columns(
    df: pd.DataFrame,
    schema_df: pd.DataFrame,
    *,
    null_threshold: float = 80,
    varchar_length_threshold: int = 10,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    null_percentage = df.isnull().mean().mul(100)
    high_null_columns = null_percentage[null_percentage >= null_threshold].index.tolist()
    working_df = df.drop(columns=high_null_columns)

    id_like_columns: list[str] = [
        column
        for column in working_df.columns
        if (
            column.endswith("_id")
            or "_fk_" in column
            or "_no_" in column
            or column.endswith("_no")
            or "_number_" in column
            or column.endswith("_number")
            or "_code_" in column
            or column.endswith("_code")
        )
    ]

    text_like_columns = schema_df.loc[
        schema_df["data_type"].isin(TEXT_DATA_TYPES),
        "prefixed_column_name",
    ].tolist()
    text_like_columns = [column for column in text_like_columns if column in working_df.columns]

    long_varchar_columns = schema_df.loc[
        (schema_df["data_type"] == "character varying")
        & (schema_df["character_maximum_length"].fillna(0) > varchar_length_threshold),
        "prefixed_column_name",
    ].tolist()
    long_varchar_columns = [
        column for column in long_varchar_columns if column in working_df.columns
    ]

    columns_to_drop = sorted(set(id_like_columns + text_like_columns + long_varchar_columns))
    filtered_df = working_df.drop(columns=columns_to_drop, errors="ignore")

    dropped_summary = {
        "high_null_columns": sorted(set(high_null_columns)),
        "id_like_columns": sorted(set(id_like_columns)),
        "text_like_columns": sorted(set(text_like_columns)),
        "long_varchar_columns": sorted(set(long_varchar_columns)),
        "final_dropped_columns": sorted(set(high_null_columns + columns_to_drop)),
    }
    return filtered_df, dropped_summary


def coerce_date_columns(
    df: pd.DataFrame,
    schema_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    converted_df = df.copy()
    converted_columns: list[str] = []

    date_columns = schema_df.loc[
        schema_df["data_type"].isin(DATE_DATA_TYPES),
        "prefixed_column_name",
    ].tolist()

    for column_name in date_columns:
        if column_name not in converted_df.columns:
            continue
        converted_df[column_name] = pd.to_datetime(
            converted_df[column_name],
            errors="coerce",
        )
        converted_columns.append(column_name)

    return converted_df, converted_columns


def resolve_date_stage_plan(
    available_columns: list[str],
    tables_in_dataset: list[str],
) -> dict[str, object]:
    alias_map = {stage_name: aliases for stage_name, aliases in DATE_SEQUENCE_STAGE_ALIASES}

    global_stage_columns: dict[str, str | None] = {}
    for stage_name in GLOBAL_SEQUENCE_STAGE_ORDER:
        global_stage_columns[stage_name] = None
        aliases = alias_map[stage_name]
        for table_name in tables_in_dataset:
            candidate = first_matching_column(available_columns, table_name, aliases)
            if candidate:
                global_stage_columns[stage_name] = candidate
                break

    same_table_stage_columns: dict[str, dict[str, str | None]] = {}
    for table_name in tables_in_dataset:
        same_table_stage_columns[table_name] = {}
        for stage_name in SAME_TABLE_DATE_SEQUENCE_STAGES:
            aliases = alias_map[stage_name]
            same_table_stage_columns[table_name][stage_name] = first_matching_column(
                available_columns,
                table_name,
                aliases,
            )

    table_paths: dict[str, list[dict[str, str | None]]] = {}
    for table_name in tables_in_dataset:
        table_paths[table_name] = []
        for stage_name in TABLE_SEQUENCE_STAGE_ORDER:
            if stage_name in SAME_TABLE_DATE_SEQUENCE_STAGES:
                column_name = same_table_stage_columns[table_name][stage_name]
            else:
                column_name = global_stage_columns[stage_name]
            table_paths[table_name].append(
                {
                    "stage_name": stage_name,
                    "column_name": column_name,
                }
            )

    return {
        "global_stage_columns": global_stage_columns,
        "same_table_stage_columns": same_table_stage_columns,
        "table_paths": table_paths,
    }


def drop_invalid_date_sequence_rows(
    df: pd.DataFrame,
    date_stage_plan: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    summary = build_date_sequence_summary(df, date_stage_plan)
    return apply_date_sequence_summary(df, summary), summary


def engineer_date_gap_features(
    df: pd.DataFrame,
    schema_df: pd.DataFrame,
    date_stage_plan: dict[str, object],
) -> tuple[pd.DataFrame, list[str], list[dict[str, str]]]:
    working_df = df.copy()
    original_date_columns = [
        column_name
        for column_name in schema_df.loc[
            schema_df["data_type"].isin(DATE_DATA_TYPES),
            "prefixed_column_name",
        ].tolist()
        if column_name in working_df.columns
    ]
    created_gap_features: list[dict[str, str]] = []
    created_feature_names: set[str] = set()

    table_paths: dict[str, list[dict[str, str | None]]] = date_stage_plan["table_paths"]
    for table_name, path_entries in table_paths.items():
        for previous_stage, next_stage in zip(path_entries, path_entries[1:]):
            previous_column = previous_stage["column_name"]
            next_column = next_stage["column_name"]
            if not previous_column or not next_column:
                continue

            feature_name = f"gap_days_{previous_column}_to_{next_column}"
            if feature_name in created_feature_names:
                continue

            gap_days = (
                working_df[next_column] - working_df[previous_column]
            ).dt.total_seconds() / 86_400.0
            working_df[feature_name] = gap_days.where(
                working_df[previous_column].notna() & working_df[next_column].notna(),
                pd.NA,
            )
            created_feature_names.add(feature_name)
            created_gap_features.append(
                {
                    "table_name": table_name,
                    "feature_name": feature_name,
                    "previous_stage": previous_stage["stage_name"],
                    "next_stage": next_stage["stage_name"],
                    "previous_column": previous_column,
                    "next_column": next_column,
                }
            )

    working_df = working_df.drop(columns=original_date_columns, errors="ignore")
    return working_df, original_date_columns, created_gap_features


def select_feature_columns(
    df: pd.DataFrame,
    schema_df: pd.DataFrame,
    *,
    categorical_cardinality_threshold: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    numeric_columns = schema_df.loc[
        schema_df["data_type"].isin(NUMERIC_DATA_TYPES),
        "prefixed_column_name",
    ].tolist()
    numeric_columns = [column for column in numeric_columns if column in df.columns]

    engineered_gap_columns = [
        column for column in df.columns if column.startswith("gap_days_")
    ]
    numeric_columns = sorted(set(numeric_columns + engineered_gap_columns))

    boolean_columns = schema_df.loc[
        schema_df["data_type"].isin(BOOLEAN_DATA_TYPES),
        "prefixed_column_name",
    ].tolist()
    boolean_columns = [column for column in boolean_columns if column in df.columns]

    categorical_candidates = schema_df.loc[
        ~schema_df["data_type"].isin(NUMERIC_DATA_TYPES | BOOLEAN_DATA_TYPES | DATE_DATA_TYPES | TEXT_DATA_TYPES),
        "prefixed_column_name",
    ].tolist()
    categorical_candidates = [column for column in categorical_candidates if column in df.columns]

    categorical_columns: list[str] = []
    for column_name in categorical_candidates:
        distinct_count = int(df[column_name].dropna().astype("string").nunique())
        if distinct_count <= categorical_cardinality_threshold:
            categorical_columns.append(column_name)

    supported_columns = set(numeric_columns) | set(boolean_columns) | set(categorical_columns)
    unsupported_columns = sorted(column for column in df.columns if column not in supported_columns)

    return numeric_columns, boolean_columns, categorical_columns, unsupported_columns


def build_training_pipeline(
    numeric_columns: list[str],
    boolean_columns: list[str],
    categorical_columns: list[str],
    *,
    contamination: float,
    random_state: int,
) -> Pipeline:
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if boolean_columns:
        boolean_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        transformers.append(("boolean", boolean_pipeline, boolean_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                IsolationForest(
                    contamination=contamination,
                    random_state=random_state,
                ),
            ),
        ]
    )


def get_transformed_feature_names(
    fitted_pipeline: Pipeline,
    numeric_columns: list[str],
    boolean_columns: list[str],
    categorical_columns: list[str],
) -> list[str]:
    feature_names: list[str] = []
    preprocessor: ColumnTransformer = fitted_pipeline.named_steps["preprocessor"]

    if numeric_columns:
        feature_names.extend(numeric_columns)

    if boolean_columns:
        feature_names.extend(boolean_columns)

    if categorical_columns:
        categorical_transformer = preprocessor.named_transformers_["categorical"]
        categorical_one_hot: OneHotEncoder = categorical_transformer.named_steps["one_hot"]
        feature_names.extend(
            categorical_one_hot.get_feature_names_out(categorical_columns).tolist()
        )

    return feature_names


def build_column_operation_map(
    raw_columns: list[str],
    invoice_row_filter_summary: dict[str, object],
    dropped_summary: dict[str, list[str]],
    converted_date_columns: list[str],
    date_sequence_summary: dict[str, object],
    original_date_columns_dropped: list[str],
    created_gap_features: list[dict[str, str]],
    numeric_columns: list[str],
    boolean_columns: list[str],
    categorical_columns: list[str],
    unsupported_columns: list[str],
) -> dict[str, list[str]]:
    operation_map = {column_name: ["selected_from_sql"] for column_name in raw_columns}

    for table_summary in invoice_row_filter_summary["tables_applied"]:
        append_column_operation(
            operation_map,
            table_summary["invoice_column"],
            "used_for_void_invoice_row_filter",
        )
        append_column_operation(
            operation_map,
            table_summary["invoice_date_column"],
            "used_for_void_invoice_row_filter",
        )
        append_column_operation(
            operation_map,
            table_summary["record_status_column"],
            "used_for_void_invoice_row_filter",
        )

    for column_name in dropped_summary["high_null_columns"]:
        append_column_operation(operation_map, column_name, "dropped_high_null")
    for column_name in dropped_summary["text_like_columns"]:
        append_column_operation(operation_map, column_name, "dropped_text_like")
    for column_name in dropped_summary["long_varchar_columns"]:
        append_column_operation(operation_map, column_name, "dropped_long_varchar")
    for column_name in dropped_summary["id_like_columns"]:
        append_column_operation(operation_map, column_name, "identified_as_id_like")

    for column_name in converted_date_columns:
        append_column_operation(operation_map, column_name, "converted_to_datetime_for_checks")

    for check_summary in date_sequence_summary["checks"]:
        previous_column = check_summary["previous_column"]
        next_column = check_summary["next_column"]
        if previous_column:
            append_column_operation(operation_map, previous_column, "used_in_date_sequence_check")
        if next_column:
            append_column_operation(operation_map, next_column, "used_in_date_sequence_check")

    for column_name in original_date_columns_dropped:
        append_column_operation(operation_map, column_name, "dropped_raw_date_before_training")

    for gap_feature in created_gap_features:
        feature_name = gap_feature["feature_name"]
        operation_map[feature_name] = [
            "engineered_date_gap_feature",
            f"from:{gap_feature['previous_column']}->{gap_feature['next_column']}",
        ]
        append_column_operation(
            operation_map,
            gap_feature["previous_column"],
            "used_for_date_gap_feature",
        )
        append_column_operation(
            operation_map,
            gap_feature["next_column"],
            "used_for_date_gap_feature",
        )

    for column_name in numeric_columns:
        append_column_operation(operation_map, column_name, "numeric_standardized")
    for column_name in boolean_columns:
        append_column_operation(operation_map, column_name, "boolean_converted_to_numeric")
    for column_name in categorical_columns:
        append_column_operation(operation_map, column_name, "categorical_one_hot_encoded")
    for column_name in unsupported_columns:
        append_column_operation(operation_map, column_name, "dropped_before_training")

    return dict(sorted(operation_map.items()))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def train_dataset(
    dataset_name: str,
    base_table: str,
    join_tables: list[str],
    schema_map: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    models_dir: Path,
    reports_dir: Path,
) -> dict:
    effective_list_date_from = dataset_list_date_from(dataset_name, args)
    tables_in_dataset = [base_table, *join_tables]
    schema_df = merged_schema_for_tables(base_table, join_tables, schema_map)
    raw_joined_columns = raw_joined_columns_for_tables(tables_in_dataset, schema_map)
    count_sql = build_count_sql(
        base_table,
        join_tables,
        schema_map,
        list_date_from=effective_list_date_from,
        list_date_cutoff=args.list_date_cutoff,
    )
    source_row_count = int(query_postgres(count_sql).iloc[0]["row_count"])

    sql = build_select_sql(
        base_table,
        join_tables,
        schema_map,
        dataset_name=dataset_name,
        list_date_from=effective_list_date_from,
        list_date_cutoff=args.list_date_cutoff,
        max_training_rows=args.max_training_rows,
    )
    raw_df = query_postgres(sql)
    row_filtered_df, invoice_row_filter_summary = drop_voided_invoice_rows(raw_df, schema_df)

    cleaned_df, dropped_summary = filter_columns(
        row_filtered_df,
        schema_df,
        null_threshold=args.null_threshold,
        varchar_length_threshold=args.varchar_length_threshold,
    )
    dated_df, converted_date_columns = coerce_date_columns(cleaned_df, schema_df)
    date_stage_plan = resolve_date_stage_plan(dated_df.columns.tolist(), tables_in_dataset)
    sequence_filtered_df, date_sequence_summary = drop_invalid_date_sequence_rows(
        dated_df,
        date_stage_plan,
    )
    feature_ready_df, original_date_columns_dropped, created_gap_features = (
        engineer_date_gap_features(
            sequence_filtered_df,
            schema_df,
            date_stage_plan,
        )
    )

    numeric_columns, boolean_columns, categorical_columns, unsupported_columns = (
        select_feature_columns(
            feature_ready_df,
            schema_df,
            categorical_cardinality_threshold=args.categorical_cardinality_threshold,
        )
    )
    feature_ready_df = normalize_boolean_feature_columns(feature_ready_df, boolean_columns)
    feature_df = feature_ready_df.drop(columns=unsupported_columns, errors="ignore")

    if not numeric_columns and not boolean_columns and not categorical_columns:
        raise ValueError(
            f"No supported numeric/boolean/categorical/date-gap columns were available for dataset {dataset_name}."
        )

    training_pipeline = build_training_pipeline(
        numeric_columns,
        boolean_columns,
        categorical_columns,
        contamination=args.contamination,
        random_state=args.random_state,
    )
    training_pipeline.fit(feature_df)
    training_predictions = training_pipeline.predict(feature_df)
    anomaly_count = int((training_predictions == -1).sum())
    inlier_count = int((training_predictions == 1).sum())

    transformed_feature_names = get_transformed_feature_names(
        training_pipeline,
        numeric_columns,
        boolean_columns,
        categorical_columns,
    )

    column_operation_map = build_column_operation_map(
        raw_columns=raw_df.columns.tolist(),
        invoice_row_filter_summary=invoice_row_filter_summary,
        dropped_summary=dropped_summary,
        converted_date_columns=converted_date_columns,
        date_sequence_summary=date_sequence_summary,
        original_date_columns_dropped=original_date_columns_dropped,
        created_gap_features=created_gap_features,
        numeric_columns=numeric_columns,
        boolean_columns=boolean_columns,
        categorical_columns=categorical_columns,
        unsupported_columns=unsupported_columns,
    )

    model_path = models_dir / f"{dataset_name}_pipeline.joblib"
    joblib.dump(
        {
            "dataset_name": dataset_name,
            "base_table": base_table,
            "join_tables": join_tables,
            "list_date_from": effective_list_date_from,
            "list_date_cutoff": args.list_date_cutoff,
            "max_training_rows": args.max_training_rows,
            "sql": sql,
            "pipeline": training_pipeline,
            "raw_columns": raw_joined_columns,
            "model_raw_columns": raw_df.columns.tolist(),
            "raw_joined_columns": raw_joined_columns,
            "cleaned_columns": cleaned_df.columns.tolist(),
            "date_sequence_checked_columns": dated_df.columns.tolist(),
            "sequence_filtered_columns": sequence_filtered_df.columns.tolist(),
            "feature_input_columns": feature_df.columns.tolist(),
            "transformed_feature_names": transformed_feature_names,
            "numeric_columns": numeric_columns,
            "boolean_columns": boolean_columns,
            "categorical_columns": categorical_columns,
            "training_row_count": int(len(feature_df.index)),
            "source_row_count": source_row_count,
            "post_invoice_filter_row_count": int(len(row_filtered_df.index)),
            "post_date_sequence_filter_row_count": int(len(sequence_filtered_df.index)),
            "anomaly_count": anomaly_count,
            "inlier_count": inlier_count,
            "invoice_row_filter_summary": invoice_row_filter_summary,
            "date_stage_plan": date_stage_plan,
            "date_sequence_summary": date_sequence_summary,
            "date_gap_features": created_gap_features,
            "column_operation_map": column_operation_map,
        },
        model_path,
    )

    report = {
        "dataset_name": dataset_name,
        "base_table": base_table,
        "join_tables": join_tables,
        "list_date_from": effective_list_date_from,
        "list_date_cutoff": args.list_date_cutoff,
        "max_training_rows": args.max_training_rows,
        "join_sql": sql.strip(),
        "source_row_count": source_row_count,
        "queried_row_count": int(len(raw_df.index)),
        "post_invoice_filter_row_count": int(len(row_filtered_df.index)),
        "post_date_sequence_filter_row_count": int(len(sequence_filtered_df.index)),
        "training_row_count": int(len(feature_df.index)),
        "anomaly_count": anomaly_count,
        "inlier_count": inlier_count,
        "raw_column_count": int(len(raw_df.columns)),
        "cleaned_column_count": int(len(cleaned_df.columns)),
        "feature_input_column_count": int(len(feature_df.columns)),
        "transformed_feature_count": int(len(transformed_feature_names)),
        "cleaned_columns": cleaned_df.columns.tolist(),
        "feature_input_columns": feature_df.columns.tolist(),
        "transformed_feature_names": transformed_feature_names,
        "invoice_row_filter_summary": invoice_row_filter_summary,
        "dropped_columns_summary": dropped_summary,
        "converted_date_columns": converted_date_columns,
        "date_stage_plan": date_stage_plan,
        "date_sequence_summary": date_sequence_summary,
        "original_date_columns_dropped_before_training": original_date_columns_dropped,
        "date_gap_features": created_gap_features,
        "numeric_columns_standardized": numeric_columns,
        "boolean_columns_converted_to_numeric": boolean_columns,
        "categorical_columns_one_hot_encoded": categorical_columns,
        "unsupported_columns_dropped_before_training": unsupported_columns,
        "column_operation_map": column_operation_map,
        "model_path": str(model_path),
    }

    report_path = reports_dir / f"{dataset_name}_columns.json"
    save_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()

    invalid_datasets = sorted(set(args.datasets) - set(DATASET_TABLES))
    if invalid_datasets:
        raise ValueError(
            f"Unsupported datasets: {invalid_datasets}. Available datasets: {sorted(DATASET_TABLES)}"
        )

    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tables_needed = sorted(
        {
            *(dataset_base_table(dataset_name) for dataset_name in args.datasets),
            *(table_name for dataset_name in args.datasets for table_name in DATASET_TABLES[dataset_name]),
        }
    )
    schema_map = {table_name: prefixed_schema(table_name) for table_name in tables_needed}

    manifest: dict[str, list[dict]] = {"trained_datasets": []}
    for dataset_name in args.datasets:
        base_table = dataset_base_table(dataset_name)
        join_tables = DATASET_TABLES[dataset_name]
        print(f"Training {dataset_name} ...", flush=True)
        report = train_dataset(
            dataset_name,
            base_table,
            join_tables,
            schema_map,
            args,
            models_dir,
            reports_dir,
        )
        manifest["trained_datasets"].append(
            {
                "dataset_name": report["dataset_name"],
                "base_table": report["base_table"],
                "join_tables": report["join_tables"],
                "list_date_from": report["list_date_from"],
                "list_date_cutoff": report["list_date_cutoff"],
                "max_training_rows": report["max_training_rows"],
                "source_row_count": report["source_row_count"],
                "queried_row_count": report["queried_row_count"],
                "post_invoice_filter_row_count": report["post_invoice_filter_row_count"],
                "post_date_sequence_filter_row_count": report["post_date_sequence_filter_row_count"],
                "training_row_count": report["training_row_count"],
                "anomaly_count": report["anomaly_count"],
                "inlier_count": report["inlier_count"],
                "raw_column_count": report["raw_column_count"],
                "cleaned_column_count": report["cleaned_column_count"],
                "feature_input_column_count": report["feature_input_column_count"],
                "transformed_feature_count": report["transformed_feature_count"],
                "high_null_column_count": len(report["dropped_columns_summary"]["high_null_columns"]),
                "id_like_column_count": len(report["dropped_columns_summary"]["id_like_columns"]),
                "text_like_column_count": len(report["dropped_columns_summary"]["text_like_columns"]),
                "long_varchar_column_count": len(report["dropped_columns_summary"]["long_varchar_columns"]),
                "dropped_column_count": len(report["dropped_columns_summary"]["final_dropped_columns"]),
                "converted_date_column_count": len(report["converted_date_columns"]),
                "date_gap_feature_count": len(report["date_gap_features"]),
                "unsupported_column_count": len(report["unsupported_columns_dropped_before_training"]),
            }
        )
        print(
            f"Trained {report['dataset_name']}: "
            f"{report['training_row_count']} training rows "
            f"(from {report['source_row_count']} source rows), "
            f"list_date_from={report['list_date_from'] or 'none'}, "
            f"list_date_cutoff={report['list_date_cutoff']}, "
            f"{report['transformed_feature_count']} transformed features, "
            f"{report['anomaly_count']} anomalies"
        )

    save_json(reports_dir / "training_manifest.json", manifest)


if __name__ == "__main__":
    main()
