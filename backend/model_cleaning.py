"""Shared row-cleaning helpers used by training and saved-model runtime preprocessing."""

from __future__ import annotations

from typing import Any

import pandas as pd


def normalized_bool_series(series: pd.Series) -> pd.Series:
    """Normalize common truthy/falsey strings into a nullable boolean series."""
    normalized = series.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[normalized.isin(["true", "t", "1", "yes", "y"])] = True
    result.loc[normalized.isin(["false", "f", "0", "no", "n"])] = False
    return result


def normalize_boolean_feature_columns(
    df: pd.DataFrame,
    boolean_columns: list[str],
) -> pd.DataFrame:
    """Convert boolean-like feature columns into the Float64 representation expected by the model."""
    if not boolean_columns:
        return df
    normalized_df = df.copy()
    for column_name in boolean_columns:
        if column_name not in normalized_df.columns:
            continue
        normalized_df[column_name] = normalized_bool_series(normalized_df[column_name]).astype("Float64")
    return normalized_df


def build_invoice_row_filter_summary(
    df: pd.DataFrame,
    schema_df: pd.DataFrame,
) -> dict[str, object]:
    """Describe which duplicate void-invoice rows should be removed for each source table."""
    table_summaries: list[dict[str, object]] = []

    for table_name in schema_df["source_table"].drop_duplicates().tolist():
        invoice_column = next(
            (
                candidate
                for alias in ("invoice_number", "invoice_no")
                for candidate in [f"{table_name}_{alias}"]
                if candidate in df.columns
            ),
            None,
        )
        invoice_date_column = f"{table_name}_invoice_date"
        record_status_column = f"{table_name}_record_status"
        required_columns = [invoice_column, invoice_date_column, record_status_column]
        if not invoice_column or not all(column in df.columns for column in required_columns):
            continue

        rows_to_drop = _invoice_row_indexes(
            df,
            invoice_column=invoice_column,
            invoice_date_column=invoice_date_column,
            record_status_column=record_status_column,
        )
        if not rows_to_drop:
            continue

        table_summaries.append(
            {
                "table_name": table_name,
                "invoice_column": invoice_column,
                "invoice_date_column": invoice_date_column,
                "record_status_column": record_status_column,
                "dropped_row_count": len(rows_to_drop),
            }
        )

    return {
        "dropped_row_count": int(sum(int(item["dropped_row_count"]) for item in table_summaries)),
        "tables_applied": table_summaries,
    }


def apply_invoice_row_filter_summary(
    df: pd.DataFrame,
    summary: dict[str, Any] | None,
) -> pd.DataFrame:
    """Drop duplicate void-invoice rows using a previously generated filter summary."""
    rows_to_drop: set[Any] = set()

    for table_summary in (summary or {}).get("tables_applied") or []:
        invoice_column = table_summary.get("invoice_column")
        invoice_date_column = table_summary.get("invoice_date_column")
        record_status_column = table_summary.get("record_status_column")
        required_columns = [invoice_column, invoice_date_column, record_status_column]
        if not all(column in df.columns for column in required_columns):
            continue

        rows_to_drop.update(
            _invoice_row_indexes(
                df,
                invoice_column=str(invoice_column),
                invoice_date_column=str(invoice_date_column),
                record_status_column=str(record_status_column),
            )
        )

    if rows_to_drop:
        return df.drop(index=sorted(rows_to_drop), errors="ignore")
    return df


def build_date_sequence_summary(
    df: pd.DataFrame,
    date_stage_plan: dict[str, object],
) -> dict[str, object]:
    """Describe which date-order checks are applied and how many rows each check invalidates."""
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

            invalid_indexes = _date_sequence_invalid_indexes(
                df,
                previous_column=str(previous_column),
                next_column=str(next_column),
            )
            comparable_mask = df[str(previous_column)].notna() & df[str(next_column)].notna()

            check_summaries.append(
                {
                    "table_name": table_name,
                    "previous_stage": previous_stage["stage_name"],
                    "next_stage": next_stage["stage_name"],
                    "previous_column": previous_column,
                    "next_column": next_column,
                    "rows_checked": int(comparable_mask.sum()),
                    "invalid_row_count": len(invalid_indexes),
                    "status": "checked",
                }
            )

    return {
        "dropped_row_count": int(sum(int(item["invalid_row_count"]) for item in check_summaries)),
        "checks": check_summaries,
    }


def apply_date_sequence_summary(
    df: pd.DataFrame,
    summary: dict[str, Any] | None,
) -> pd.DataFrame:
    """Drop rows that violate any previously generated date-sequence check summary."""
    rows_to_drop: set[Any] = set()

    for check in (summary or {}).get("checks") or []:
        previous_column = check.get("previous_column")
        next_column = check.get("next_column")
        if not previous_column or not next_column:
            continue
        if previous_column not in df.columns or next_column not in df.columns:
            continue
        rows_to_drop.update(
            _date_sequence_invalid_indexes(
                df,
                previous_column=str(previous_column),
                next_column=str(next_column),
            )
        )

    if rows_to_drop:
        return df.drop(index=sorted(rows_to_drop), errors="ignore")
    return df


def _invoice_row_indexes(
    df: pd.DataFrame,
    *,
    invoice_column: str,
    invoice_date_column: str,
    record_status_column: str,
) -> list[Any]:
    group_keys = [invoice_column, invoice_date_column]
    non_null_invoice_mask = df[group_keys].notna().all(axis=1)
    duplicate_group_mask = (
        df.loc[non_null_invoice_mask, group_keys]
        .duplicated(keep=False)
        .reindex(df.index, fill_value=False)
    )
    void_status_mask = (
        df[record_status_column]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("v")
        .fillna(False)
    )
    return df.index[non_null_invoice_mask & duplicate_group_mask & void_status_mask].tolist()


def _date_sequence_invalid_indexes(
    df: pd.DataFrame,
    *,
    previous_column: str,
    next_column: str,
) -> list[Any]:
    comparable = df[previous_column].notna() & df[next_column].notna()
    invalid = comparable & (df[next_column] < df[previous_column])
    return df.index[invalid].tolist()
