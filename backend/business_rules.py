"""Shared deterministic business-rule definitions used by training and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CHEQUE_SLIP_APPROVAL_OWNER_COLUMNS = (
    "fk_aao",
    "fk_ao",
    "fk_auditor",
)
CHEQUE_SLIP_APPROVAL_OWNER_RULE_NAME = "approved_cheque_requires_owner"
CHEQUE_SLIP_APPROVAL_OWNER_RULE_REASON = (
    "Approved cheque slip but all approval columns are null"
)
CHEQUE_SLIP_SCHEDULE3_NOT_APPROVED_RULE_NAME = (
    "cheque_slip_v_not_approved_but_schedule3_exists"
)
CHEQUE_SLIP_SCHEDULE3_NOT_APPROVED_RULE_REASON = (
    "Cheque slip record_status V and approved false but schedule3 exists for same fk_dak"
)
CHEQUE_SLIP_SCHEDULE3_COUNT_MISMATCH_RULE_NAME = (
    "cheque_slip_approved_count_not_matching_schedule3"
)
CHEQUE_SLIP_SCHEDULE3_COUNT_MISMATCH_RULE_REASON = (
    "Approved V cheque_slip count does not match schedule3 P/V count for same fk_dak"
)


@dataclass(frozen=True)
class RuleSqlDefinition:
    rule_name: str
    condition_sql: str
    reason: str


@dataclass(frozen=True)
class SharedSqlFragments:
    ctes: list[str]
    outer_joins: list[str]


@dataclass(frozen=True)
class RuleSqlBundle:
    rule: RuleSqlDefinition
    ctes: list[str]
    outer_joins: list[str]
    evidence_expressions: list[str]


CMP_SCROLL_PAYMENT_REFERENCE_RULE_NAME = "cmp_scroll_payment_reference_missing_in_ecs"
CMP_SCROLL_PAYMENT_REFERENCE_RULE_REASON = "CMP scroll has payment_reference_no but not found in ECS"
CHEQUE_SLIP_ECS_MODE_RULE_NAME = "cheque_slip_ecs_mode_conflict"
CHEQUE_SLIP_ECS_MODE_RULE_REASON = "Cheque slip ECS mode=1 but ECS record exists"
RULE_REGISTRY_INDEX = [
    {
        "rule_name": CMP_SCROLL_PAYMENT_REFERENCE_RULE_NAME,
        "scope": "runtime",
        "applies_to": ["cmp_scroll", "ecs"],
        "description": CMP_SCROLL_PAYMENT_REFERENCE_RULE_REASON,
    },
    {
        "rule_name": CHEQUE_SLIP_ECS_MODE_RULE_NAME,
        "scope": "runtime",
        "applies_to": ["cheque_slip", "ecs"],
        "description": CHEQUE_SLIP_ECS_MODE_RULE_REASON,
    },
    {
        "rule_name": CHEQUE_SLIP_SCHEDULE3_NOT_APPROVED_RULE_NAME,
        "scope": "training_and_runtime",
        "applies_to": ["cheque_slip", "schedule3"],
        "description": CHEQUE_SLIP_SCHEDULE3_NOT_APPROVED_RULE_REASON,
    },
    {
        "rule_name": CHEQUE_SLIP_SCHEDULE3_COUNT_MISMATCH_RULE_NAME,
        "scope": "training_and_runtime",
        "applies_to": ["cheque_slip", "schedule3"],
        "description": CHEQUE_SLIP_SCHEDULE3_COUNT_MISMATCH_RULE_REASON,
    },
    {
        "rule_name": CHEQUE_SLIP_APPROVAL_OWNER_RULE_NAME,
        "scope": "training_and_runtime",
        "applies_to": ["cheque_slip"],
        "description": CHEQUE_SLIP_APPROVAL_OWNER_RULE_REASON,
    },
    {
        "rule_name": "duplicate_void_invoice_<table>",
        "scope": "runtime",
        "applies_to": ["any table with invoice number, invoice date, and record_status"],
        "description": "Flags duplicate invoice/invoice_date rows with record_status V.",
    },
    {
        "rule_name": "date_sequence_<left_stage>_after_<right_stage>",
        "scope": "runtime",
        "applies_to": ["any joined dataset with matching date-stage columns"],
        "description": "Flags rows where a later processing stage date appears before an earlier stage date.",
    },
]


def cheque_slip_schedule3_shared_sql_fragments(
    *,
    schedule3_table_ref: str,
    cheque_slip_table_ref: str,
    fk_dak_join_expr: str,
) -> SharedSqlFragments:
    """Build the shared CTEs and joins used by cheque-slip/schedule3 business rules."""
    return SharedSqlFragments(
        ctes=[
            f"""
            schedule3_by_dak AS (
                SELECT
                    fk_dak,
                    COUNT(*) AS schedule3_total_count,
                    COUNT(*) FILTER (WHERE UPPER(BTRIM(CAST(record_status AS text))) IN ('P', 'V')) AS schedule3_pv_count
                FROM {schedule3_table_ref}
                WHERE fk_dak IS NOT NULL
                GROUP BY fk_dak
            )
            """,
            f"""
            cheque_slip_approved_by_dak AS (
                SELECT
                    fk_dak,
                    COUNT(*) FILTER (
                        WHERE UPPER(BTRIM(CAST(record_status AS text))) = 'V'
                          AND approved = true
                    ) AS cheque_slip_v_approved_count
                FROM {cheque_slip_table_ref}
                WHERE fk_dak IS NOT NULL
                GROUP BY fk_dak
            )
            """,
        ],
        outer_joins=[
            f"""
            LEFT JOIN schedule3_by_dak
                ON schedule3_by_dak.fk_dak = {fk_dak_join_expr}
            """,
            f"""
            LEFT JOIN cheque_slip_approved_by_dak
                ON cheque_slip_approved_by_dak.fk_dak = {fk_dak_join_expr}
            """,
        ],
    )


def cheque_slip_schedule3_not_approved_rule(
    *,
    record_status_expr: str,
    approved_expr: str,
    fk_dak_expr: str,
) -> RuleSqlDefinition:
    """Build the shared rule for V cheque slips that are not approved but have schedule3 rows."""
    return RuleSqlDefinition(
        rule_name=CHEQUE_SLIP_SCHEDULE3_NOT_APPROVED_RULE_NAME,
        condition_sql=(
            "\n            (\n"
            f"                UPPER(BTRIM(CAST({record_status_expr} AS text))) = 'V'\n"
            f"                AND {approved_expr} = false\n"
            f"                AND {fk_dak_expr} IS NOT NULL\n"
            "                AND COALESCE(schedule3_by_dak.schedule3_total_count, 0) > 0\n"
            "            )\n"
        ),
        reason=CHEQUE_SLIP_SCHEDULE3_NOT_APPROVED_RULE_REASON,
    )


def cheque_slip_schedule3_count_mismatch_rule(
    *,
    record_status_expr: str,
    approved_expr: str,
    fk_dak_expr: str,
) -> RuleSqlDefinition:
    """Build the shared rule for approved V cheque-slip counts that do not match schedule3."""
    return RuleSqlDefinition(
        rule_name=CHEQUE_SLIP_SCHEDULE3_COUNT_MISMATCH_RULE_NAME,
        condition_sql=(
            "\n            (\n"
            f"                UPPER(BTRIM(CAST({record_status_expr} AS text))) = 'V'\n"
            f"                AND {approved_expr} = true\n"
            f"                AND {fk_dak_expr} IS NOT NULL\n"
            "                AND COALESCE(cheque_slip_approved_by_dak.cheque_slip_v_approved_count, 0)\n"
            "                    <> COALESCE(schedule3_by_dak.schedule3_pv_count, 0)\n"
            "            )\n"
        ),
        reason=CHEQUE_SLIP_SCHEDULE3_COUNT_MISMATCH_RULE_REASON,
    )


def cheque_slip_approval_owner_columns(
    available_columns: Iterable[str],
) -> list[str]:
    """Return the approval-owner columns present in the available cheque-slip schema."""
    available = {str(column) for column in available_columns}
    return [
        column_name
        for column_name in CHEQUE_SLIP_APPROVAL_OWNER_COLUMNS
        if column_name in available
    ]


def cheque_slip_approval_owner_reason(owner_columns: Iterable[str]) -> str:
    """Build the user-facing reason text for the cheque-slip approval-owner rule."""
    columns = [str(column) for column in owner_columns]
    return f"{CHEQUE_SLIP_APPROVAL_OWNER_RULE_REASON}: {', '.join(columns)}"


def cheque_slip_approval_owner_training_condition_sql(
    owner_columns: Iterable[str],
    *,
    approved_expr: str,
    column_expr_template: str,
) -> str:
    """Build the training SQL condition that excludes invalid approved cheque-slip rows."""
    columns = [str(column) for column in owner_columns]
    if not columns:
        return ""

    all_owner_null = "\n                    AND ".join(
        column_expr_template.format(column_name=column_name) + " IS NULL"
        for column_name in columns
    )
    return (
        "(\n"
        f"                    {approved_expr} = true\n"
        f"                    AND {all_owner_null}\n"
        "                )"
    )


def cheque_slip_approval_owner_runtime_rule(
    owner_columns: Iterable[str],
) -> RuleSqlDefinition | None:
    """Build the runtime SQL rule definition for invalid approved cheque-slip rows."""
    columns = [str(column) for column in owner_columns]
    if not columns:
        return None

    all_owner_null = "\n                AND ".join(
        f'base."cheque_slip.{column_name}" IS NULL'
        for column_name in columns
    )
    return RuleSqlDefinition(
        rule_name=CHEQUE_SLIP_APPROVAL_OWNER_RULE_NAME,
        condition_sql=(
            "\n                (\n"
            '                    base."cheque_slip.approved" = true\n'
            f"                    AND {all_owner_null}\n"
            "                )\n"
        ),
        reason=cheque_slip_approval_owner_reason(columns),
    )


def cmp_scroll_payment_reference_rule(
    *,
    payment_reference_expr: str,
    cda_name_expr: str,
    ecs_table_ref: str,
) -> RuleSqlDefinition:
    """Build the runtime rule for CMP scroll references missing from ECS."""
    return RuleSqlDefinition(
        rule_name=CMP_SCROLL_PAYMENT_REFERENCE_RULE_NAME,
        condition_sql=(
            "\n            (\n"
            f"                {payment_reference_expr} IS NOT NULL\n"
            f"                AND {cda_name_expr} = 'CDA- Main Office Jabalpur'\n"
            "                AND NOT EXISTS (\n"
            "                    SELECT 1\n"
            f"                    FROM {ecs_table_ref} e\n"
            f"                    WHERE e.payment_reference_no = {payment_reference_expr}\n"
            "                )\n"
            "            )\n"
        ),
        reason=CMP_SCROLL_PAYMENT_REFERENCE_RULE_REASON,
    )


def cheque_slip_ecs_mode_rule(
    *,
    ecs_mode_expr: str,
    fk_dak_expr: str,
    ecs_table_ref: str,
) -> RuleSqlDefinition:
    """Build the runtime rule for cheque-slip ECS mode conflicts."""
    return RuleSqlDefinition(
        rule_name=CHEQUE_SLIP_ECS_MODE_RULE_NAME,
        condition_sql=(
            "\n            (\n"
            f"                {ecs_mode_expr} = 1\n"
            f"                AND {fk_dak_expr} IS NOT NULL\n"
            "                AND EXISTS (\n"
            "                    SELECT 1\n"
            f"                    FROM {ecs_table_ref} e\n"
            f"                    WHERE e.fk_dak = {fk_dak_expr}\n"
            "                )\n"
            "            )\n"
        ),
        reason=CHEQUE_SLIP_ECS_MODE_RULE_REASON,
    )


def duplicate_void_invoice_rule_bundle(
    *,
    table_name: str,
    invoice_column: str,
    source_table_ref: str,
    duplicate_cte_name: str,
    base_invoice_key_expr: str,
    base_invoice_date_expr: str,
    base_status_expr: str,
    cte_invoice_expr: str,
    cte_invoice_date_expr: str,
    cte_status_expr: str,
    cte_fk_dak_expr: str | None,
) -> RuleSqlBundle:
    """Build the runtime rule bundle for duplicate voided invoice anomalies."""
    cte_similar_fk_dak_select = (
        f"string_agg(DISTINCT CAST({cte_fk_dak_expr} AS text), ', ' ORDER BY CAST({cte_fk_dak_expr} AS text))"
        if cte_fk_dak_expr
        else "NULL::text"
    )
    condition_sql = (
        "\n            (\n"
        f"                {base_invoice_key_expr} IS NOT NULL\n"
        f"                AND {base_invoice_date_expr} IS NOT NULL\n"
        f"                AND UPPER(BTRIM(CAST({base_status_expr} AS text))) = 'V'\n"
        f"                AND COALESCE({duplicate_cte_name}.duplicate_count, 0) >= 2\n"
        "            )\n"
    )
    return RuleSqlBundle(
        rule=RuleSqlDefinition(
            rule_name=f"duplicate_void_invoice_{table_name}",
            condition_sql=condition_sql,
            reason=f"{table_name} has duplicate {invoice_column} and invoice_date with record_status V",
        ),
        ctes=[
            f"""
            {duplicate_cte_name} AS (
                SELECT
                    {cte_invoice_expr} AS invoice_key,
                    {cte_invoice_date_expr} AS invoice_date_key,
                    COUNT(*) AS duplicate_count,
                    {cte_similar_fk_dak_select} AS similar_fk_daks
                FROM {source_table_ref}
                WHERE {cte_invoice_expr} IS NOT NULL
                  AND {cte_invoice_date_expr} IS NOT NULL
                  AND UPPER(BTRIM(CAST({cte_status_expr} AS text))) = 'V'
                GROUP BY
                    {cte_invoice_expr},
                    {cte_invoice_date_expr}
                HAVING COUNT(*) >= 2
            )
            """
        ],
        outer_joins=[
            f"""
            LEFT JOIN {duplicate_cte_name}
                ON {duplicate_cte_name}.invoice_key = {base_invoice_key_expr}
               AND {duplicate_cte_name}.invoice_date_key = {base_invoice_date_expr}
            """
        ],
        evidence_expressions=[
            f"""
            CASE WHEN ({condition_sql}) THEN
                jsonb_build_object(
                    'kind', 'duplicate_invoice_fk_daks',
                    'table', '{table_name}',
                    'invoice_column', '{invoice_column}',
                    'invoice_number', CAST({base_invoice_key_expr} AS text),
                    'invoice_date', CAST({base_invoice_date_expr} AS text),
                    'record_status', 'V',
                    'duplicate_count', {duplicate_cte_name}.duplicate_count,
                    'similar_fk_daks', {duplicate_cte_name}.similar_fk_daks
                )::text
            ELSE NULL END
            """
        ],
    )


def date_sequence_rule(
    *,
    predicate_sql: str,
    left_label: str,
    right_label: str,
) -> RuleSqlDefinition:
    """Build the runtime rule for a date-sequence violation."""
    return RuleSqlDefinition(
        rule_name=f"date_sequence_{left_label}_after_{right_label}",
        condition_sql=predicate_sql,
        reason=f"Date sequence violated across processing stages: {left_label} after {right_label}",
    )
