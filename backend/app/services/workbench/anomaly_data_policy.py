"""Shared preprocessing policy definitions for training and runtime anomaly flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.services.workbench.business_rules import RULE_REGISTRY_INDEX
from app.services.workbench.model_cleaning import (
    apply_date_sequence_summary,
    apply_invoice_row_filter_summary,
    business_day_gap_series,
    normalize_boolean_feature_columns,
)


TRAINING_ROW_DROP_POLICIES = [
    "duplicate_void_invoice_rows",
    "date_sequence_violation_rows",
]

RUNTIME_RULE_FLAG_POLICIES = [
    "duplicate_void_invoice_rows",
    "date_sequence_violation_rows",
    *[
        str(rule["rule_name"])
        for rule in RULE_REGISTRY_INDEX
    ],
]

SHARED_FEATURE_TRANSFORM_POLICIES = [
    "boolean_normalization",
    "date_gap_feature_engineering",
    "feature_input_column_ordering",
]


@dataclass(frozen=True)
class SavedModelPreprocessingPolicy:
    """Serialized training-time preprocessing contract reused during runtime scoring."""

    cleaned_columns: list[str]
    date_sequence_checked_columns: list[str]
    sequence_filtered_columns: list[str]
    feature_input_columns: list[str]
    boolean_columns: list[str]
    invoice_row_filter_summary: dict[str, Any]
    date_sequence_summary: dict[str, Any]
    date_gap_features: list[dict[str, Any]]
    training_row_drop_policies: list[str]
    runtime_rule_flag_policies: list[str]
    shared_feature_transform_policies: list[str]

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "SavedModelPreprocessingPolicy":
        """Load the preprocessing contract from a saved model artifact."""
        return cls(
            cleaned_columns=[str(column) for column in artifact.get("cleaned_columns") or []],
            date_sequence_checked_columns=[
                str(column)
                for column in artifact.get("date_sequence_checked_columns") or []
            ],
            sequence_filtered_columns=[
                str(column)
                for column in artifact.get("sequence_filtered_columns") or []
            ],
            feature_input_columns=[
                str(column)
                for column in artifact.get("feature_input_columns") or []
            ],
            boolean_columns=[str(column) for column in artifact.get("boolean_columns") or []],
            invoice_row_filter_summary=dict(artifact.get("invoice_row_filter_summary") or {}),
            date_sequence_summary=dict(artifact.get("date_sequence_summary") or {}),
            date_gap_features=[
                dict(item)
                for item in artifact.get("date_gap_features") or []
            ],
            training_row_drop_policies=[
                str(item)
                for item in artifact.get("training_row_drop_policies")
                or TRAINING_ROW_DROP_POLICIES
            ],
            runtime_rule_flag_policies=[
                str(item)
                for item in artifact.get("runtime_rule_flag_policies")
                or RUNTIME_RULE_FLAG_POLICIES
            ],
            shared_feature_transform_policies=[
                str(item)
                for item in artifact.get("shared_feature_transform_policies")
                or SHARED_FEATURE_TRANSFORM_POLICIES
            ],
        )

    def to_artifact_fields(self) -> dict[str, Any]:
        """Serialize the policy back into plain artifact fields."""
        return {
            "cleaned_columns": list(self.cleaned_columns),
            "date_sequence_checked_columns": list(self.date_sequence_checked_columns),
            "sequence_filtered_columns": list(self.sequence_filtered_columns),
            "feature_input_columns": list(self.feature_input_columns),
            "boolean_columns": list(self.boolean_columns),
            "invoice_row_filter_summary": dict(self.invoice_row_filter_summary),
            "date_sequence_summary": dict(self.date_sequence_summary),
            "date_gap_features": [dict(item) for item in self.date_gap_features],
            "training_row_drop_policies": list(self.training_row_drop_policies),
            "runtime_rule_flag_policies": list(self.runtime_rule_flag_policies),
            "shared_feature_transform_policies": list(self.shared_feature_transform_policies),
        }


def build_saved_model_preprocessing_policy(
    *,
    cleaned_columns: list[str],
    date_sequence_checked_columns: list[str],
    sequence_filtered_columns: list[str],
    feature_input_columns: list[str],
    boolean_columns: list[str],
    invoice_row_filter_summary: dict[str, Any],
    date_sequence_summary: dict[str, Any],
    date_gap_features: list[dict[str, Any]],
) -> SavedModelPreprocessingPolicy:
    """Create the shared preprocessing contract stored with each trained model."""
    return SavedModelPreprocessingPolicy(
        cleaned_columns=[str(column) for column in cleaned_columns],
        date_sequence_checked_columns=[str(column) for column in date_sequence_checked_columns],
        sequence_filtered_columns=[str(column) for column in sequence_filtered_columns],
        feature_input_columns=[str(column) for column in feature_input_columns],
        boolean_columns=[str(column) for column in boolean_columns],
        invoice_row_filter_summary=dict(invoice_row_filter_summary),
        date_sequence_summary=dict(date_sequence_summary),
        date_gap_features=[dict(item) for item in date_gap_features],
        training_row_drop_policies=list(TRAINING_ROW_DROP_POLICIES),
        runtime_rule_flag_policies=list(RUNTIME_RULE_FLAG_POLICIES),
        shared_feature_transform_policies=list(SHARED_FEATURE_TRANSFORM_POLICIES),
    )


def append_column_operation(
    operation_map: dict[str, list[str]],
    column_name: str,
    operation: str,
) -> None:
    """Record one processing operation for a column without duplicating labels."""
    operation_map.setdefault(column_name, [])
    if operation not in operation_map[column_name]:
        operation_map[column_name].append(operation)


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
    """Build a column lineage report for training-time selection and transformations."""
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


def apply_saved_model_preprocessing_policy(
    raw_df: pd.DataFrame,
    policy: SavedModelPreprocessingPolicy,
) -> pd.DataFrame:
    """Apply the training-time preprocessing contract to runtime scoring data."""
    working = raw_df.copy()
    if policy.cleaned_columns:
        working = working.reindex(columns=policy.cleaned_columns)

    for column_name in policy.date_sequence_checked_columns:
        if column_name in working.columns and _looks_like_date_column(column_name):
            working[column_name] = pd.to_datetime(working[column_name], errors="coerce")

    working = apply_invoice_row_filter_summary(working, policy.invoice_row_filter_summary)
    working = apply_date_sequence_summary(working, policy.date_sequence_summary)
    working = _add_saved_date_gap_features(working, policy)
    working = normalize_boolean_feature_columns(working, policy.boolean_columns)
    return working.where(pd.notna(working), np.nan)


def _add_saved_date_gap_features(
    df: pd.DataFrame,
    policy: SavedModelPreprocessingPolicy,
) -> pd.DataFrame:
    working = df.copy()
    original_date_columns = set()

    for gap in policy.date_gap_features:
        feature_name = str(gap.get("feature_name") or "")
        previous_column = str(gap.get("previous_column") or "")
        next_column = str(gap.get("next_column") or "")
        if not feature_name or previous_column not in working.columns or next_column not in working.columns:
            continue
        working[feature_name] = business_day_gap_series(
            working[previous_column],
            working[next_column],
        )
        original_date_columns.update([previous_column, next_column])

    configured_dropped_dates = {
        str(column)
        for column in policy.sequence_filtered_columns
        if str(column) not in policy.feature_input_columns
        and _looks_like_date_column(str(column))
    }
    return working.drop(columns=sorted(original_date_columns | configured_dropped_dates), errors="ignore")


def _looks_like_date_column(column_name: str) -> bool:
    """Heuristic used when re-coercing saved date columns during runtime scoring."""
    normalized = str(column_name).strip().lower()
    return "date" in normalized or normalized.endswith("_at") or normalized.endswith("_time")
