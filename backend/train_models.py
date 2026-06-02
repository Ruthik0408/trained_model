from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from notebook_common import DB_CONFIG, query_postgres


BASE_TABLE = "dak"
DEFAULT_LIST_DATE_CUTOFF = "2026-01-01"

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
    "disposal_date",
}

GLOBAL_SEQUENCE_STAGE_ORDER = [
    "invoice_date",
    "bill_date",
    "reference_date",
    "dp_sheet_date",
    "cmp_date",
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
    "dak_bill": ["bill"],
    "dak_gem_bill": ["gem_bill"],
    "dak_civ_medical_bill": ["civ_medical_bill"],
    "dak_civ_paybill": ["civ_paybill"],
    "dak_civ_tada_ltc_bill": ["civ_tada_ltc_bill"],
    "dak_echs_medical_bill": ["echs_medical_bill"],
    "dak_cheque_slip_schedule3": ["cheque_slip", "schedule3"],
    "dak_cheque_slip_ecs": ["cheque_slip", "ecs"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train anomaly-detection pipelines for dak and dak join datasets "
            "using dataset-specific cleaning, date-sequence validation, and "
            "date-gap feature engineering."
        )
    )
    parser.add_argument(
        "--list-date-cutoff",
        default=DEFAULT_LIST_DATE_CUTOFF,
        help="Only rows with dak.list_date earlier than this date are included.",
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
            "Datasets to train. Example: --datasets dak dak_bill dak_gem_bill "
            "dak_civ_medical_bill dak_civ_paybill dak_civ_tada_ltc_bill dak_echs_medical_bill "
            "dak_cheque_slip_schedule3 dak_cheque_slip_ecs"
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
    return parser.parse_args()


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


def build_select_sql(
    base_table: str,
    join_tables: list[str],
    schema_map: dict[str, pd.DataFrame],
    *,
    list_date_cutoff: str,
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

    return f"""
        SELECT
            {", ".join(select_clauses)}
        FROM {base_table}
        {join_sql}
        WHERE {base_table}.list_date < '{list_date_cutoff}'
    """


def build_count_sql(
    base_table: str,
    join_tables: list[str],
    *,
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

    return f"""
        SELECT COUNT(*) AS row_count
        FROM {base_table}
        {join_sql}
        WHERE {base_table}.list_date < '{list_date_cutoff}'
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
    filtered_df = df.copy()
    dropped_row_indexes: set[int] = set()
    table_summaries: list[dict[str, object]] = []

    for table_name in schema_df["source_table"].drop_duplicates().tolist():
        invoice_column = None
        for alias in ("invoice_number", "invoice_no"):
            candidate = f"{table_name}_{alias}"
            if candidate in filtered_df.columns:
                invoice_column = candidate
                break

        invoice_date_column = f"{table_name}_invoice_date"
        record_status_column = f"{table_name}_record_status"
        required_columns = [invoice_column, invoice_date_column, record_status_column]
        if not invoice_column or not all(column in filtered_df.columns for column in required_columns):
            continue

        group_keys = [invoice_column, invoice_date_column]
        non_null_invoice_mask = filtered_df[group_keys].notna().all(axis=1)
        duplicate_group_mask = (
            filtered_df.loc[non_null_invoice_mask, group_keys]
            .duplicated(keep=False)
            .reindex(filtered_df.index, fill_value=False)
        )
        void_status_mask = (
            filtered_df[record_status_column]
            .astype("string")
            .str.strip()
            .str.lower()
            .eq("v")
            .fillna(False)
        )

        rows_to_drop_mask = non_null_invoice_mask & duplicate_group_mask & void_status_mask
        dropped_count = int(rows_to_drop_mask.sum())
        if dropped_count == 0:
            continue

        dropped_row_indexes.update(filtered_df.index[rows_to_drop_mask].tolist())
        table_summaries.append(
            {
                "table_name": table_name,
                "invoice_column": invoice_column,
                "invoice_date_column": invoice_date_column,
                "record_status_column": record_status_column,
                "dropped_row_count": dropped_count,
            }
        )

    if dropped_row_indexes:
        filtered_df = filtered_df.drop(index=sorted(dropped_row_indexes))

    return filtered_df, {
        "dropped_row_count": len(dropped_row_indexes),
        "tables_applied": table_summaries,
    }


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
    filtered_df = df.copy()
    dropped_row_indexes: set[int] = set()
    check_summaries: list[dict[str, object]] = []

    table_paths: dict[str, list[dict[str, str | None]]] = date_stage_plan["table_paths"]
    for table_name, path_entries in table_paths.items():
        for previous_stage, next_stage in zip(path_entries, path_entries[1:]):
            previous_column = previous_stage["column_name"]
            next_column = next_stage["column_name"]
            if not previous_column or not next_column:
                check_summaries.append(
                    {
                        "table_name": table_name,
                        "previous_stage": previous_stage["stage_name"],
                        "next_stage": next_stage["stage_name"],
                        "previous_column": previous_column,
                        "next_column": next_column,
                        "rows_checked": 0,
                        "invalid_row_count": 0,
                        "status": "skipped_missing_column",
                    }
                )
                continue

            comparable_mask = filtered_df[previous_column].notna() & filtered_df[next_column].notna()
            invalid_mask = comparable_mask & (filtered_df[next_column] < filtered_df[previous_column])
            invalid_count = int(invalid_mask.sum())
            rows_checked = int(comparable_mask.sum())

            if invalid_count:
                dropped_row_indexes.update(filtered_df.index[invalid_mask].tolist())

            check_summaries.append(
                {
                    "table_name": table_name,
                    "previous_stage": previous_stage["stage_name"],
                    "next_stage": next_stage["stage_name"],
                    "previous_column": previous_column,
                    "next_column": next_column,
                    "rows_checked": rows_checked,
                    "invalid_row_count": invalid_count,
                    "status": "checked",
                }
            )

    if dropped_row_indexes:
        filtered_df = filtered_df.drop(index=sorted(dropped_row_indexes))

    return filtered_df, {
        "dropped_row_count": len(dropped_row_indexes),
        "checks": check_summaries,
    }


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
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("boolean", boolean_pipeline, boolean_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
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
        boolean_transformer = preprocessor.named_transformers_["boolean"]
        boolean_one_hot: OneHotEncoder = boolean_transformer.named_steps["one_hot"]
        feature_names.extend(boolean_one_hot.get_feature_names_out(boolean_columns).tolist())

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
        append_column_operation(operation_map, column_name, "boolean_one_hot_encoded")
    for column_name in categorical_columns:
        append_column_operation(operation_map, column_name, "categorical_one_hot_encoded")
    for column_name in unsupported_columns:
        append_column_operation(operation_map, column_name, "dropped_before_training")

    return dict(sorted(operation_map.items()))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def train_dataset(
    dataset_name: str,
    join_tables: list[str],
    schema_map: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    models_dir: Path,
    reports_dir: Path,
) -> dict:
    tables_in_dataset = [BASE_TABLE, *join_tables]
    schema_df = merged_schema_for_tables(BASE_TABLE, join_tables, schema_map)
    count_sql = build_count_sql(
        BASE_TABLE,
        join_tables,
        list_date_cutoff=args.list_date_cutoff,
    )
    source_row_count = int(query_postgres(count_sql).iloc[0]["row_count"])

    sql = build_select_sql(
        BASE_TABLE,
        join_tables,
        schema_map,
        list_date_cutoff=args.list_date_cutoff,
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
            "join_tables": join_tables,
            "list_date_cutoff": args.list_date_cutoff,
            "sql": sql,
            "pipeline": training_pipeline,
            "raw_columns": raw_df.columns.tolist(),
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
        "base_table": BASE_TABLE,
        "join_tables": join_tables,
        "list_date_cutoff": args.list_date_cutoff,
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
        "raw_columns": raw_df.columns.tolist(),
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
        "boolean_columns_one_hot_encoded": boolean_columns,
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
            BASE_TABLE,
            *(table_name for dataset_name in args.datasets for table_name in DATASET_TABLES[dataset_name]),
        }
    )
    schema_map = {table_name: prefixed_schema(table_name) for table_name in tables_needed}

    manifest: dict[str, list[dict]] = {"trained_datasets": []}
    for dataset_name in args.datasets:
        join_tables = DATASET_TABLES[dataset_name]
        report = train_dataset(
            dataset_name,
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
            f"{report['transformed_feature_count']} transformed features, "
            f"{report['anomaly_count']} anomalies"
        )

    save_json(reports_dir / "training_manifest.json", manifest)


if __name__ == "__main__":
    main()
