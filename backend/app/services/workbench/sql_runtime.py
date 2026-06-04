"""SQL planning and execution helpers for the workbench join and scoring pipeline."""
from datetime import date, datetime
import time
from typing import Any
from uuid import uuid4

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.exc import SQLAlchemyError
from business_rules import (
    cmp_scroll_payment_reference_rule,
    date_sequence_rule,
    duplicate_void_invoice_rule_bundle,
    cheque_slip_approval_owner_columns,
    cheque_slip_approval_owner_runtime_rule,
    cheque_slip_ecs_mode_rule,
    cheque_slip_schedule3_count_mismatch_rule,
    cheque_slip_schedule3_not_approved_rule,
    cheque_slip_schedule3_shared_sql_fragments,
)

from app.core.cache import TTLCache
from app.core.config import settings
from app.schemas.workbench_schema import FeatureRuleInput, UserRuleInput, WorkbenchRunRequest
from app.services.workbench.constants import (
    DATE_FILTER_COLUMN_PRIORITY,
    ENABLE_EXPENSIVE_JOIN_DEBUG,
    PREVIEW_ROW_LIMIT,
    RESULT_TABLE,
    SQL_RULE_EVIDENCE_COLUMN,
    SQL_RULE_FLAG_COLUMN,
    SQL_RULE_REASONS_COLUMN,
    TEMP_ROW_ID_COLUMN,
    USER_RULE_FLAG_COLUMN,
    USER_RULE_REASONS_COLUMN,
    logger,
)
from app.services.workbench.source_db import (
    _approx_table_row_count,
    _index_name,
    _previous_dataset_row_count,
    _quote,
    _source_begin,
    _source_column_meta,
    _source_columns_map,
    _source_connect,
    _source_table_ref,
    _validate_identifier,
    _validate_selected_tables,
)
from app.services.workbench.utils import (
    _is_date_like_column_name,
    _safe_json,
    _safe_rule_name,
)


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
_JOIN_SQL_CACHE_TTL_SECONDS = 60.0
_join_sql_cache = TTLCache(ttl_seconds=_JOIN_SQL_CACHE_TTL_SECONDS, namespace="join_sql")


def _dataset_table_name(_selected_tables: list[str]) -> str:
    """Return the persisted result table name for the current workbench run."""
    return RESULT_TABLE


def _join_sql_cache_key(
    payload: WorkbenchRunRequest,
    *,
    row_limit: int | None,
    projection_mode: str,
) -> tuple[Any, ...]:
    """Build the stable cache key for a generated join SQL statement."""
    return (
        tuple(str(table_name) for table_name in payload.selected_tables),
        tuple(
            (
                str(join.left_table),
                str(join.left_column),
                str(join.right_table),
                str(join.right_column),
                str(join.join_type),
            )
            for join in payload.joins
        ),
        _safe_date_literal(getattr(payload, "from_date", None)),
        _safe_date_literal(getattr(payload, "to_date", None)),
        int(row_limit) if row_limit is not None else None,
        str(projection_mode),
    )


def _get_cached_join_sql(
    payload: WorkbenchRunRequest,
    *,
    row_limit: int | None,
    projection_mode: str,
) -> tuple[str, list[dict[str, Any]], list[str], dict[str, Any]] | None:
    """Return cached join SQL and metadata when the payload shape already exists."""
    cache_key = _join_sql_cache_key(
        payload,
        row_limit=row_limit,
        projection_mode=projection_mode,
    )
    cached = _join_sql_cache.get(repr(cache_key))
    if cached is None:
        return None

    sql = str(cached.get("sql") or "")
    join_debug = [dict(item) for item in cached.get("join_debug") or []]
    warnings = list(cached.get("warnings") or [])
    params = dict(cached.get("params") or {})
    logger.info(
        "Reusing cached workbench join SQL for tables=%s row_limit=%s",
        payload.selected_tables,
        row_limit,
    )
    return sql, join_debug, warnings, params


def _store_cached_join_sql(
    payload: WorkbenchRunRequest,
    *,
    row_limit: int | None,
    projection_mode: str,
    sql: str,
    join_debug: list[dict[str, Any]],
    warnings: list[str],
    params: dict[str, Any],
) -> None:
    """Persist generated join SQL and debug metadata into the shared TTL cache."""
    cache_key = _join_sql_cache_key(
        payload,
        row_limit=row_limit,
        projection_mode=projection_mode,
    )
    _join_sql_cache.set(
        repr(cache_key),
        {
            "sql": sql,
            "join_debug": [dict(item) for item in join_debug],
            "warnings": list(warnings),
            "params": dict(params),
        },
    )

def _builtin_stage_column_matches(available: dict[str, str]) -> dict[str, list[str]]:
    matches_by_alias: dict[str, list[str]] = {}
    for qualified_name, data_type in available.items():
        _, plain_column = qualified_name.split(".", 1)
        normalized_plain = plain_column.strip().lower()
        if not _is_date_like_column_name(normalized_plain, data_type):
            continue
        for _stage_name, aliases in DATE_SEQUENCE_STAGE_ALIASES:
            for alias in aliases:
                matched_alias = _matching_stage_alias(normalized_plain, alias)
                if matched_alias is None:
                    continue
                matches_by_alias.setdefault(matched_alias, []).append(qualified_name)
    return matches_by_alias

def _matching_stage_alias(column_name: str, alias: str) -> str | None:
    normalized_column = str(column_name).strip().lower()
    normalized_alias = str(alias).strip().lower()
    if normalized_column == normalized_alias:
        return normalized_alias
    suffix = f"_{normalized_alias}"
    if normalized_column.endswith(suffix):
        return normalized_alias
    return None

def _date_sequence_scope(
    qualified_name: str,
    matched_aliases: list[str],
) -> tuple[str, str]:
    table_name, plain_column = qualified_name.split(".", 1)
    normalized_plain = plain_column.strip().lower()
    for alias in matched_aliases:
        matched_alias = _matching_stage_alias(normalized_plain, alias)
        if matched_alias is None:
            continue
        if normalized_plain == matched_alias:
            return table_name, ""
        return table_name, normalized_plain[: -(len(matched_alias) + 1)]
    return table_name, ""

def _same_date_sequence_scope(
    left_column: str,
    left_aliases: list[str],
    right_column: str,
    right_aliases: list[str],
) -> bool:
    left_table, left_scope = _date_sequence_scope(left_column, left_aliases)
    right_table, right_scope = _date_sequence_scope(right_column, right_aliases)

    if left_scope or right_scope:
        return bool(left_scope) and left_scope == right_scope

    return left_table == right_table

def _type_family(data_type: str | None) -> str:
    value = str(data_type or "").strip().lower()
    if value in {"smallint", "integer", "bigint", "decimal", "numeric", "real", "double precision"}:
        return "numeric"
    if value in {"date", "timestamp without time zone", "timestamp with time zone", "time without time zone", "time with time zone"}:
        return "datetime"
    if value in {"character varying", "character", "text", "uuid"}:
        return "text"
    if value in {"boolean"}:
        return "boolean"
    return value

def _sql_join_keyword(join_type: str) -> str:
    normalized = (join_type or "inner").strip().lower()
    mapping = {
        "inner": "INNER JOIN",
        "left": "LEFT JOIN",
        "left outer": "LEFT JOIN",
        "right": "RIGHT JOIN",
        "right outer": "RIGHT JOIN",
        "full": "FULL OUTER JOIN",
        "full outer": "FULL OUTER JOIN",
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported join type '{join_type}'. Supported types: inner, left, right, full."
        )
    return mapping[normalized]

def _sql_table_column_samples(conn, table_name: str, column_name: str, size: int = 10) -> list[Any]:
    _validate_identifier(table_name, "table_name")
    _validate_identifier(column_name, "column_name")
    schema = _quote(settings.source_db_schema)
    table_q = _quote(table_name)
    column_q = _quote(column_name)
    safe_size = max(1, min(int(size), 100))
    sql = text(
        f"SELECT {column_q} FROM {schema}.{table_q} WHERE {column_q} IS NOT NULL LIMIT :sample_limit"
    )
    return [_safe_json(row[0]) for row in conn.execute(sql, {"sample_limit": safe_size}).fetchall()]

def _sql_common_key_count(conn, left_table: str, left_column: str, right_table: str, right_column: str) -> int:
    _validate_identifier(left_table, "left_table")
    _validate_identifier(left_column, "left_column")
    _validate_identifier(right_table, "right_table")
    _validate_identifier(right_column, "right_column")
    schema = _quote(settings.source_db_schema)
    left_table_q = _quote(left_table)
    right_table_q = _quote(right_table)
    left_key = _quote(left_column)
    right_key = _quote(right_column)
    sql = text(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT {left_key} AS join_key
            FROM {schema}.{left_table_q}
            WHERE {left_key} IS NOT NULL
        ) l
        INNER JOIN (
            SELECT DISTINCT {right_key} AS join_key
            FROM {schema}.{right_table_q}
            WHERE {right_key} IS NOT NULL
        ) r
        ON l.join_key = r.join_key
        """
    )
    return int(conn.execute(sql).scalar() or 0)

def _sql_common_key_sample(
    conn,
    left_table: str,
    left_column: str,
    right_table: str,
    right_column: str,
    size: int = 20,
) -> list[Any]:
    schema = _quote(settings.source_db_schema)
    left_table_q = _quote(left_table)
    right_table_q = _quote(right_table)
    left_key = _quote(left_column)
    right_key = _quote(right_column)
    safe_size = max(1, min(int(size), 100))
    sql = text(
        f"""
        SELECT l.join_key
        FROM (
            SELECT DISTINCT {left_key} AS join_key
            FROM {schema}.{left_table_q}
            WHERE {left_key} IS NOT NULL
        ) l
        INNER JOIN (
            SELECT DISTINCT {right_key} AS join_key
            FROM {schema}.{right_table_q}
            WHERE {right_key} IS NOT NULL
        ) r
        ON l.join_key = r.join_key
        ORDER BY l.join_key
        LIMIT :sample_limit
        """
    )
    return [_safe_json(row[0]) for row in conn.execute(sql, {"sample_limit": safe_size}).fetchall()]

def _payload_locator_column(table_name: str) -> str:
    return f"__ml_locator__{table_name}"


def _iter_projected_source_columns(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    projected_columns: set[str] | None,
):
    for table_name in selected_tables:
        for column in source_columns[table_name]:
            column_name = str(column["column_name"])
            qualified_name = f"{table_name}.{column_name}"
            if projected_columns is not None and qualified_name not in projected_columns:
                continue
            yield table_name, column_name, column


def _build_join_select_list(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    *,
    projected_columns: set[str] | None = None,
    include_locators: bool = False,
    cast_datetimes_as_text: bool = True,
) -> str:
    parts: list[str] = []
    if include_locators:
        for table_name in selected_tables:
            locator_name = _payload_locator_column(table_name)
            parts.append(
                f"CAST({_quote(table_name)}.ctid AS text) AS {_quote(locator_name)}"
            )

    for table_name, column_name, column in _iter_projected_source_columns(
        selected_tables,
        source_columns,
        projected_columns,
    ):
            source_expr = f"{_quote(table_name)}.{_quote(column_name)}"
            if cast_datetimes_as_text and _type_family(column.get("data_type")) == "datetime":
                source_expr = f"CAST({source_expr} AS text)"
            parts.append(f"{source_expr} AS {_quote(f'{table_name}.{column_name}')}")
    if not parts:
        raise ValueError("No source columns were found for the selected tables.")
    return ",\n    ".join(parts)

def _joined_column_meta(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    column_name: str,
) -> tuple[str, dict[str, Any]]:
    if "." in str(column_name):
        table_name, plain_column = str(column_name).split(".", 1)
        if table_name not in source_columns:
            raise ValueError(f"Table not found for column '{column_name}'.")
        meta = _source_column_meta(source_columns, table_name, plain_column)
        return f"{table_name}.{meta['column_name']}", meta

    matches: list[tuple[str, dict[str, Any]]] = []
    for table_name in selected_tables:
        for item in source_columns[table_name]:
            if item["column_name"] == column_name:
                matches.append((f"{table_name}.{item['column_name']}", item))
    if not matches:
        raise ValueError(f"Column not found: {column_name}")
    if len(matches) > 1:
        raise ValueError(f"Column is ambiguous: {column_name}. Matches: {[name for name, _ in matches[:10]]}")
    return matches[0]

def _joined_column_expr(joined_column_name: str, alias: str = "src") -> str:
    return f"{_quote(alias)}.{_quote(joined_column_name)}"

def _sql_safe_numeric(expr: str) -> str:
    trimmed = f"btrim(CAST({expr} AS text))"
    numeric_regex = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$"
    return (
        "CASE "
        f"WHEN {expr} IS NULL THEN NULL "
        f"WHEN lower({trimmed}) IN ('', 'nan', 'none', 'null', '<na>', 'nat') THEN NULL "
        f"WHEN {trimmed} ~ '{numeric_regex}' THEN ({trimmed})::double precision "
        "ELSE NULL "
        "END"
    )

def _sql_numeric_or_text_comparison(first_expr: str, second_expr: str, operator: str) -> str:
    comparator = "=" if operator == "=" else "<>"
    left_num = _sql_safe_numeric(first_expr)
    right_num = _sql_safe_numeric(second_expr)
    text_comparison = f"CAST({first_expr} AS text) {comparator} CAST({second_expr} AS text)"
    numeric_comparison = f"{left_num} {comparator} {right_num}"
    return (
        "("
        f"CASE WHEN {left_num} IS NOT NULL AND {right_num} IS NOT NULL "
        f"THEN {numeric_comparison} "
        f"ELSE {text_comparison} "
        "END"
        ")"
    )

def _sql_safe_timestamp(expr: str, data_type: str | None = None) -> str:
    text_expr = f"btrim(CAST({expr} AS text))"
    if _type_family(data_type) == "datetime":
        return (
            "CASE "
            f"WHEN {expr} IS NULL THEN NULL::timestamp "
            f"WHEN lower({text_expr}) IN ('', 'nan', 'none', 'null', '<na>', 'nat') THEN NULL::timestamp "
            f"ELSE REPLACE({text_expr}, 'T', ' ')::timestamp "
            "END"
        )
    return (
        "CASE "
        f"WHEN {expr} IS NULL THEN NULL::timestamp "
        f"WHEN lower({text_expr}) IN ('', 'nan', 'none', 'null', '<na>', 'nat') THEN NULL::timestamp "
        f"WHEN {text_expr} ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}(?:[ tT]\\d{{2}}:\\d{{2}}(?::\\d{{2}}(?:\\.\\d+)?)?(?:[+-]\\d{{2}}(?::?\\d{{2}})?|Z)?)?$' THEN REPLACE({text_expr}, 'T', ' ')::timestamp "
        f"WHEN {text_expr} ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}' THEN left({text_expr}, 10)::date::timestamp "
        "ELSE NULL::timestamp "
        "END"
    )

def _sql_boolean_as_double(predicate: str) -> str:
    return f"(CASE WHEN {predicate} THEN 1.0 ELSE 0.0 END)"

def _next_param_name(params: dict[str, Any], prefix: str) -> str:
    return f"{prefix}_{len(params) + 1}"

def _build_sql_feature_expression(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    rule: FeatureRuleInput,
    *,
    alias: str,
) -> str:
    first_name, first_meta = _joined_column_meta(selected_tables, source_columns, rule.first_column)
    first_expr = _joined_column_expr(first_name, alias)
    second_meta: dict[str, Any] | None = None
    second_expr: str | None = None
    if rule.second_column:
        second_name, second_meta = _joined_column_meta(selected_tables, source_columns, rule.second_column)
        second_expr = _joined_column_expr(second_name, alias)

    feature_type = str(rule.feature_type or "").strip().lower()

    if feature_type == "numeric":
        return _sql_safe_numeric(first_expr)
    if feature_type == "difference" and second_expr is not None:
        return f"({_sql_safe_numeric(first_expr)} - {_sql_safe_numeric(second_expr)})"
    if feature_type == "ratio" and second_expr is not None:
        denominator = _sql_safe_numeric(second_expr)
        return f"({_sql_safe_numeric(first_expr)} / NULLIF({denominator}, 0))"
    if feature_type == "sum" and second_expr is not None:
        return f"(COALESCE({_sql_safe_numeric(first_expr)}, 0) + COALESCE({_sql_safe_numeric(second_expr)}, 0))"
    if feature_type == "missingflag":
        return _sql_boolean_as_double(f"{first_expr} IS NULL")
    if feature_type == "daysbetween" and second_expr is not None and second_meta is not None:
        first_ts = _sql_safe_timestamp(first_expr, first_meta.get("data_type"))
        second_ts = _sql_safe_timestamp(second_expr, second_meta.get("data_type"))
        return f"(EXTRACT(EPOCH FROM ({first_ts} - {second_ts})) / 86400.0)"
    if feature_type == "isweekend":
        first_ts = _sql_safe_timestamp(first_expr, first_meta.get("data_type"))
        return _sql_boolean_as_double(f"EXTRACT(DOW FROM {first_ts}) IN (0, 6)")
    if feature_type == "isbusinesshour":
        first_ts = _sql_safe_timestamp(first_expr, first_meta.get("data_type"))
        return _sql_boolean_as_double(f"EXTRACT(HOUR FROM {first_ts}) BETWEEN 8 AND 21")
    raise ValueError(f"Unsupported feature type: {rule.feature_type}")


def _build_sql_feature_selects(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    rules: list[FeatureRuleInput],
    *,
    alias: str = "src",
) -> tuple[list[str], list[str], int]:
    warnings: list[str] = []
    selects: list[str] = []
    applied_count = 0
    used_aliases: set[str] = set()

    for index, rule in enumerate(rules, start=1):
        rule_name = _safe_rule_name(rule.name, f"Feature {index}")
        column_alias = rule_name
        while column_alias in used_aliases:
            column_alias = f"{column_alias}_{len(used_aliases) + 1}"
        try:
            expr = _build_sql_feature_expression(selected_tables, source_columns, rule, alias=alias)
            selects.append(f"{expr} AS {_quote(column_alias)}")
            used_aliases.add(column_alias)
            applied_count += 1
        except Exception as exc:
            warnings.append(f"Skipped feature rule '{rule_name}': {exc}")
    return selects, warnings, applied_count


def _feature_rule_aliases(rules: list[FeatureRuleInput]) -> list[str]:
    aliases: list[str] = []
    used_aliases: set[str] = set()
    for index, rule in enumerate(rules, start=1):
        column_alias = _safe_rule_name(rule.name, f"Feature {index}")
        while column_alias in used_aliases:
            column_alias = f"{column_alias}_{len(used_aliases) + 1}"
        used_aliases.add(column_alias)
        aliases.append(column_alias)
    return aliases


def _qualified_builtin_scoring_columns(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
) -> set[str]:
    required: set[str] = set()
    selected_set = set(selected_tables)

    def add_if_present(table_name: str, column_name: str) -> None:
        if table_name not in selected_set:
            return
        table_column_names = {
            str(item["column_name"])
            for item in source_columns.get(table_name, [])
        }
        if column_name in table_column_names:
            required.add(f"{table_name}.{column_name}")

    for table_name in selected_tables:
        table_column_names = {
            str(item["column_name"])
            for item in source_columns.get(table_name, [])
        }
        invoice_column = next(
            (column_name for column_name in ("invoice_number", "invoice_no") if column_name in table_column_names),
            None,
        )
        if invoice_column and "invoice_date" in table_column_names and "record_status" in table_column_names:
            add_if_present(table_name, invoice_column)
            add_if_present(table_name, "invoice_date")
            add_if_present(table_name, "record_status")

    add_if_present("cmp_scroll", "payment_reference_no")
    add_if_present("cmp_scroll", "cda_name")

    add_if_present("cheque_slip", "fk_ecs_payment_mode")
    add_if_present("cheque_slip", "fk_dak")
    add_if_present("cheque_slip", "record_status")
    add_if_present("cheque_slip", "approved")
    add_if_present("cheque_slip", "fk_aao")
    add_if_present("cheque_slip", "fk_ao")
    add_if_present("cheque_slip", "fk_go")
    add_if_present("cheque_slip", "fk_auditor")

    add_if_present("cmp_scroll", "payment_reference_no")

    for _stage_name, aliases in DATE_SEQUENCE_STAGE_ALIASES:
        for table_name in selected_tables:
            table_column_names = {
                str(item["column_name"])
                for item in source_columns.get(table_name, [])
            }
            for column_name in table_column_names:
                normalized_column = column_name.strip().lower()
                if any(_matching_stage_alias(normalized_column, alias) for alias in aliases):
                    required.add(f"{table_name}.{column_name}")

    return required


def _qualified_rule_source_columns(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
) -> set[str]:
    required: set[str] = set()
    for rule in payload.feature_rules:
        for column_name in (rule.first_column, rule.second_column):
            if not column_name:
                continue
            try:
                joined_name, _meta = _joined_column_meta(
                    payload.selected_tables,
                    source_columns,
                    column_name,
                )
            except Exception:
                continue
            required.add(joined_name)

    for rule in payload.user_rules:
        for column_name in (rule.first_column, rule.second_column):
            if not column_name:
                continue
            try:
                joined_name, _meta = _joined_column_meta(
                    payload.selected_tables,
                    source_columns,
                    column_name,
                )
            except Exception:
                continue
            required.add(joined_name)
    return required


def _qualified_auto_feature_columns(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
) -> set[str]:
    columns: set[str] = set()
    for table_name in payload.selected_tables:
        for item in source_columns.get(table_name, []):
            columns.add(f"{table_name}.{item['column_name']}")
    return columns


def _scoring_projection_columns(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
) -> set[str]:
    return (
        _qualified_auto_feature_columns(payload, source_columns)
        | _qualified_rule_source_columns(payload, source_columns)
        | _qualified_builtin_scoring_columns(payload.selected_tables, source_columns)
    )


def _is_datetime_meta(meta: dict[str, Any] | None) -> bool:
    return _type_family((meta or {}).get("data_type")) == "datetime"


def _parse_rule_timestamp_literal(value: Any) -> str:
    text_value = str(value or "").strip()
    if not text_value:
        raise ValueError("Rule value is missing.")

    normalized = text_value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).isoformat(sep=" ")
    except ValueError:
        pass

    try:
        return date.fromisoformat(normalized[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"Rule value '{value}' is not a valid date or timestamp.") from exc


def _sql_datetime_comparison(
    first_expr: str,
    first_meta: dict[str, Any] | None,
    operator: str,
    params: dict[str, Any],
    *,
    second_expr: str | None = None,
    second_meta: dict[str, Any] | None = None,
    rule_value: Any = None,
) -> str:
    left_ts = _sql_safe_timestamp(first_expr, (first_meta or {}).get("data_type"))

    if second_expr is not None:
        right_ts = _sql_safe_timestamp(second_expr, (second_meta or {}).get("data_type"))
        return f"({left_ts} {operator} {right_ts})"

    param_name = _next_param_name(params, "rule_ts")
    params[param_name] = _parse_rule_timestamp_literal(rule_value)
    return f"({left_ts} {operator} CAST(:{param_name} AS timestamp))"

def _build_sql_user_rule_predicate(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    rule: UserRuleInput,
    params: dict[str, Any],
    *,
    alias: str,
) -> str:
    first_name, first_meta = _joined_column_meta(selected_tables, source_columns, rule.first_column)
    first_expr = _joined_column_expr(first_name, alias)
    second_meta: dict[str, Any] | None = None
    second_expr: str | None = None
    if rule.second_column:
        second_name, second_meta = _joined_column_meta(selected_tables, source_columns, rule.second_column)
        second_expr = _joined_column_expr(second_name, alias)

    operator = str(rule.operator or "").strip().lower()
    if operator in {">", ">=", "<", "<="}:
        if _is_datetime_meta(first_meta) or _is_datetime_meta(second_meta):
            return _sql_datetime_comparison(
                first_expr,
                first_meta,
                operator,
                params,
                second_expr=second_expr,
                second_meta=second_meta,
                rule_value=rule.value,
            )
        left_num = _sql_safe_numeric(first_expr)
        if second_expr is not None:
            right_num = _sql_safe_numeric(second_expr)
            return f"({left_num} {operator} {right_num})"
        numeric_value = pd.to_numeric(pd.Series([rule.value]), errors="coerce").iloc[0]
        if pd.isna(numeric_value):
            raise ValueError(f"Rule value '{rule.value}' is not numeric.")
        param_name = _next_param_name(params, "rule_num")
        params[param_name] = float(numeric_value)
        return f"({left_num} {operator} :{param_name})"
    if operator == "null":
        return f"({first_expr} IS NULL)"
    if operator == "not null":
        return f"({first_expr} IS NOT NULL)"
    if operator in {"=", "!="}:
        if _is_datetime_meta(first_meta) or _is_datetime_meta(second_meta):
            comparator = "=" if operator == "=" else "<>"
            return _sql_datetime_comparison(
                first_expr,
                first_meta,
                comparator,
                params,
                second_expr=second_expr,
                second_meta=second_meta,
                rule_value=rule.value,
            )
        if second_expr is not None:
            return _sql_numeric_or_text_comparison(first_expr, second_expr, operator)
        param_name = _next_param_name(params, "rule_value")
        params[param_name] = None if rule.value is None else str(rule.value)
        comparator = "=" if operator == "=" else "<>"
        return f"(CAST({first_expr} AS text) {comparator} :{param_name})"
    raise ValueError(f"Unsupported user rule operator: {rule.operator}")

def _build_sql_user_rule_flag(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    rules: list[UserRuleInput],
    *,
    alias: str = "src",
) -> tuple[str, str, dict[str, Any], list[str], list[str], int]:
    if not rules:
        return "FALSE", "NULL::text", {}, [], [], 0

    predicates: list[str] = []
    reason_parts: list[str] = []
    warnings: list[str] = []
    labels: list[str] = []
    params: dict[str, Any] = {}
    applied_count = 0

    for index, rule in enumerate(rules, start=1):
        rule_name = _safe_rule_name(rule.name, f"User rule {index}")
        try:
            predicate = _build_sql_user_rule_predicate(selected_tables, source_columns, rule, params, alias=alias)
            label = f"RULE::{rule_name}"
            label_param = _next_param_name(params, "rule_label")
            params[label_param] = label
            predicates.append(predicate)
            reason_parts.append(f"CASE WHEN {predicate} THEN :{label_param} ELSE NULL END")
            labels.append(label)
            applied_count += 1
        except Exception as exc:
            warnings.append(f"Skipped user rule '{rule_name}': {exc}")

    if not predicates:
        return "FALSE", "NULL::text", params, [], warnings, applied_count
    flag_expr = "(" + " OR ".join(predicates) + ")"
    reason_expr = "array_to_string(array_remove(ARRAY[" + ", ".join(reason_parts) + "]::text[], NULL), ', ')"
    return flag_expr, reason_expr, params, labels, warnings, applied_count

def _build_sql_workbench_query(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
    joined_sql: str,
) -> tuple[str, dict[str, Any], list[str], int, list[str], list[str], int]:
    feature_selects, feature_warnings, applied_feature_rule_count = _build_sql_feature_selects(
        payload.selected_tables,
        source_columns,
        payload.feature_rules,
        alias="src",
    )
    (
        user_rule_expr,
        user_rule_reason_expr,
        params,
        user_rule_labels,
        user_rule_warnings,
        applied_user_rule_count,
    ) = _build_sql_user_rule_flag(
        payload.selected_tables,
        source_columns,
        payload.user_rules,
        alias="src",
    )

    select_parts = ["src.*"]
    select_parts.extend(feature_selects)
    select_parts.append(f"{user_rule_expr} AS {_quote('__ml_sql_rule_flag')}")
    select_parts.append(f"{user_rule_reason_expr} AS {_quote('__ml_sql_rule_reasons')}")
    sql = "WITH src AS (\n" + joined_sql + "\n)\nSELECT\n    " + ",\n    ".join(select_parts) + "\nFROM src"
    return (
        sql,
        params,
        user_rule_labels,
        applied_feature_rule_count,
        feature_warnings,
        user_rule_warnings,
        applied_user_rule_count,
    )

def _build_source_table_ref(table_name: str) -> str:
    schema = _quote(settings.source_db_schema)
    quoted_table = _quote(table_name)
    return f"{schema}.{quoted_table} AS {quoted_table}"

def _source_table_only_ref(table_name: str) -> str:
    _validate_identifier(table_name, "table_name")
    return f"{_quote(settings.source_db_schema)}.{_quote(table_name)}"

def _safe_date_literal(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        parsed = date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError):
        return None
    if parsed.year < 1 or parsed.year > 9999:
        return None
    return parsed.isoformat()

def _safe_sql_date_expr(expr: str) -> str:
    text_expr = f"btrim(CAST({expr} AS text))"
    return (
        "CASE "
        f"WHEN {expr} IS NULL THEN NULL::date "
        f"WHEN {text_expr} ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}' THEN left({text_expr}, 10)::date "
        "ELSE NULL::date "
        "END"
    )

def _date_range_filter_sql(
    table_name: str,
    column: dict[str, Any],
    *,
    from_date: str | None,
    to_date: str | None,
) -> str:
    column_name = str(column["column_name"])
    raw_expr = f"{_quote(table_name)}.{_quote(column_name)}"

    if _type_family(column.get("data_type")) == "datetime":
        parts = [f"{raw_expr} IS NOT NULL"]
        if from_date:
            parts.append(f"{raw_expr} >= CAST(:date_filter_from AS date)")
        if to_date:
            parts.append(f"{raw_expr} < (CAST(:date_filter_to AS date) + INTERVAL '1 day')")
    else:
        safe_date = _safe_sql_date_expr(raw_expr)
        parts = [f"{safe_date} IS NOT NULL"]
        if from_date:
            parts.append(f"{safe_date} >= CAST(:date_filter_from AS date)")
        if to_date:
            parts.append(f"{safe_date} <= CAST(:date_filter_to AS date)")

    return "(" + " AND ".join(parts) + ")"

def _join_sql_params(payload: WorkbenchRunRequest) -> dict[str, Any]:
    params: dict[str, Any] = {}
    from_date = _safe_date_literal(getattr(payload, "from_date", None))
    to_date = _safe_date_literal(getattr(payload, "to_date", None))
    if from_date:
        params["date_filter_from"] = from_date
    if to_date:
        params["date_filter_to"] = to_date
    return params

def _preferred_date_filter_target(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
) -> tuple[str, dict[str, Any]] | None:
    preferred_columns = DATE_FILTER_COLUMN_PRIORITY
    selected_lookup = {
        table_name: {
            str(column.get("column_name", "")).strip().lower(): column
            for column in source_columns.get(table_name, [])
        }
        for table_name in payload.selected_tables
    }

    for preferred_name in preferred_columns:
        if "dak" in payload.selected_tables:
            dak_column = selected_lookup.get("dak", {}).get(preferred_name)
            if dak_column is not None:
                return "dak", dak_column

        for table_name in payload.selected_tables:
            selected_column = selected_lookup.get(table_name, {}).get(preferred_name)
            if selected_column is not None:
                return table_name, selected_column

    return None

def _list_date_filter_sql(payload: WorkbenchRunRequest, source_columns: dict[str, list[dict]]) -> str | None:
    from_date = _safe_date_literal(getattr(payload, "from_date", None))
    to_date = _safe_date_literal(getattr(payload, "to_date", None))
    if not from_date and not to_date:
        return None
    if from_date and to_date and from_date > to_date:
        return None

    target = _preferred_date_filter_target(payload, source_columns)
    if target is None:
        return None
    table_name, column = target
    return _date_range_filter_sql(
        table_name,
        column,
        from_date=from_date,
        to_date=to_date,
    )

def _validate_join_payload_tables(payload: WorkbenchRunRequest, source_columns: dict[str, list[dict]]) -> None:
    _validate_selected_tables(payload.selected_tables, source_columns)

    if len(payload.selected_tables) > 1 and len(payload.joins) == 0:
        raise ValueError(
            "When selecting 2 or more tables, at least 1 join configuration is required to connect them."
        )

    selected = set(payload.selected_tables)
    for index, join in enumerate(payload.joins, start=1):
        if join.left_table not in selected:
            raise ValueError(
                f"Join step {index}: left table '{join.left_table}' is not present in selected_tables."
            )
        if join.right_table not in selected:
            raise ValueError(
                f"Join step {index}: right table '{join.right_table}' is not present in selected_tables."
            )
        if join.left_table == join.right_table:
            raise ValueError(
                f"Join step {index}: self-joins are not supported by the current SQL join builder: "
                f"'{join.left_table}' -> '{join.right_table}'."
            )

def _ensure_source_column_index(conn, table_name: str, column_name: str) -> None:
    index_name = _index_name(settings.source_db_schema, table_name, column_name, "join")
    conn.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS {_quote(index_name)} "
            f"ON {_source_table_ref(table_name)} ({_quote(column_name)})"
        )
    )

def _ensure_join_indexes(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
) -> list[str]:
    if not settings.auto_create_source_indexes:
        return []
    if not payload.joins:
        return []

    warnings: list[str] = []
    indexed_columns: set[tuple[str, str]] = set()
    try:
        with _source_begin() as conn:
            for join in payload.joins:
                left_meta = _source_column_meta(source_columns, join.left_table, join.left_column)
                right_meta = _source_column_meta(source_columns, join.right_table, join.right_column)
                for table_name, column_name in (
                    (join.left_table, left_meta["column_name"]),
                    (join.right_table, right_meta["column_name"]),
                ):
                    key = (table_name, column_name)
                    if key in indexed_columns:
                        continue
                    _ensure_source_column_index(conn, table_name, column_name)
                    indexed_columns.add(key)
    except (SQLAlchemyError, ValueError) as exc:
        warnings.append(
            "Could not automatically create indexes for selected join columns. "
            f"The join will still run, but it may be slower: {exc}"
        )

    if indexed_columns:
        warnings.append(
            "Ensured PostgreSQL indexes exist for selected join columns: "
            + ", ".join(f"{table}.{column}" for table, column in sorted(indexed_columns))
            + "."
        )
    return warnings

def _ensure_date_filter_indexes(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
) -> list[str]:
    if not settings.auto_create_source_indexes:
        return []
    if not _safe_date_literal(getattr(payload, "from_date", None)) and not _safe_date_literal(getattr(payload, "to_date", None)):
        return []

    warnings: list[str] = []
    indexed_columns: set[tuple[str, str]] = set()
    try:
        with _source_begin() as conn:
            target = _preferred_date_filter_target(payload, source_columns)
            if target is None:
                warnings.append("No supported date filter column was found for the selected tables.")
            else:
                table_name, date_column = target
                column_name = str(date_column["column_name"])
                key = (table_name, column_name)
                if key not in indexed_columns:
                    _ensure_source_column_index(conn, table_name, column_name)
                    indexed_columns.add(key)
    except (SQLAlchemyError, ValueError) as exc:
        warnings.append(
            "Could not automatically create indexes for selected date filters. "
            f"The run will still continue, but date filtering may be slower: {exc}"
        )

    if indexed_columns:
        warnings.append(
            "Ensured PostgreSQL indexes exist for selected date filters: "
            + ", ".join(f"{table}.{column}" for table, column in sorted(indexed_columns))
            + "."
        )
    return warnings

def _build_date_sequence_anomaly_conditions(
    joined_tables: list[str],
    source_columns: dict[str, list[dict]],
) -> list[tuple[str, str]]:
    stage_exprs: list[tuple[str, list[tuple[str, str]]]] = []

    for stage_name, aliases in DATE_SEQUENCE_STAGE_ALIASES:
        exprs: list[tuple[str, str]] = []

        for table_name in joined_tables:
            table_columns = [
                str(column.get("column_name"))
                for column in source_columns.get(table_name, [])
            ]

            for column_name in table_columns:
                normalized_column = column_name.strip().lower()
                if any(_matching_stage_alias(normalized_column, alias) for alias in aliases):
                    exprs.append(
                        (
                            f"{table_name}.{column_name}",
                            _safe_sql_date_expr(f'base."{table_name}.{column_name}"'),
                        )
                    )

        if not exprs:
            continue

        stage_exprs.append((stage_name, exprs))

    if len(stage_exprs) < 2:
        return []

    comparisons: list[tuple[str, str]] = []

    for index in range(len(stage_exprs) - 1):
        left_label, left_exprs = stage_exprs[index]
        right_label, right_exprs = stage_exprs[index + 1]
        left_aliases = next(aliases for stage_name, aliases in DATE_SEQUENCE_STAGE_ALIASES if stage_name == left_label)
        right_aliases = next(aliases for stage_name, aliases in DATE_SEQUENCE_STAGE_ALIASES if stage_name == right_label)
        predicate_parts: list[str] = []

        for left_column_name, left_expr in left_exprs:
            for right_column_name, right_expr in right_exprs:
                if not _same_date_sequence_scope(left_column_name, left_aliases, right_column_name, right_aliases):
                    continue
                predicate_parts.append(
                    f"""
                    (
                        ({left_expr}) IS NOT NULL
                        AND ({right_expr}) IS NOT NULL
                        AND ({left_expr}) > ({right_expr})
                    )
                    """
                )

        if not predicate_parts:
            continue

        comparisons.append(
            date_sequence_rule(
                predicate_sql="(" + " OR ".join(predicate_parts) + ")",
                left_label=left_label,
                right_label=right_label,
            )
        )
    return [(rule.condition_sql, rule.reason) for rule in comparisons]

def _build_sql_anomaly_expressions(
    joined_tables: list[str],
    source_columns: dict[str, list[dict]],
) -> tuple[list[tuple[str, str]], list[str], list[str], list[str]]:
    conditions: list[tuple[str, str]] = []
    ctes: list[str] = []
    outer_joins: list[str] = []
    evidence_expressions: list[str] = []

    joined_set = set(joined_tables)

    def has_table(table: str) -> bool:
        return table in joined_set

    def table_cols(table: str) -> set[str]:
        return {
            str(column.get("column_name"))
            for column in source_columns.get(table, [])
        }

    def text_key(expr: str) -> str:
        return f"NULLIF(BTRIM(CAST({expr} AS text)), '')"

    # CMP scroll payment_reference_no must exist in ECS
    if has_table("cmp_scroll") and has_table("ecs"):
        rule = cmp_scroll_payment_reference_rule(
            payment_reference_expr='base."cmp_scroll.payment_reference_no"',
            cda_name_expr='base."cmp_scroll.cda_name"',
            ecs_table_ref=_source_table_only_ref("ecs"),
        )
        conditions.append((rule.condition_sql, rule.reason))

    # cheque_slip ECS mode = 1 but ECS record exists
    if has_table("cheque_slip") and has_table("ecs"):
        rule = cheque_slip_ecs_mode_rule(
            ecs_mode_expr='base."cheque_slip.fk_ecs_payment_mode"',
            fk_dak_expr='base."cheque_slip.fk_dak"',
            ecs_table_ref=_source_table_only_ref("ecs"),
        )
        conditions.append((rule.condition_sql, rule.reason))

    # Cheque slip + schedule3 rules
    if has_table("cheque_slip") and has_table("schedule3"):
        schedule3_fragments = cheque_slip_schedule3_shared_sql_fragments(
            schedule3_table_ref=_source_table_only_ref("schedule3"),
            cheque_slip_table_ref=_source_table_only_ref("cheque_slip"),
            fk_dak_join_expr='base."cheque_slip.fk_dak"',
        )
        ctes.extend(schedule3_fragments.ctes)
        outer_joins.extend(schedule3_fragments.outer_joins)

        # cheque_slip V + approved false should not have schedule3 for same fk_dak
        not_approved_rule = cheque_slip_schedule3_not_approved_rule(
            record_status_expr='base."cheque_slip.record_status"',
            approved_expr='base."cheque_slip.approved"',
            fk_dak_expr='base."cheque_slip.fk_dak"',
        )
        conditions.append((
            not_approved_rule.condition_sql,
            not_approved_rule.reason,
        ))

        # approved V cheque_slip count should match schedule3 P/V count for same fk_dak
        count_mismatch_rule = cheque_slip_schedule3_count_mismatch_rule(
            record_status_expr='base."cheque_slip.record_status"',
            approved_expr='base."cheque_slip.approved"',
            fk_dak_expr='base."cheque_slip.fk_dak"',
        )
        conditions.append((
            count_mismatch_rule.condition_sql,
            count_mismatch_rule.reason,
        ))

    # approved cheque slip must have at least one approval officer/user
    if has_table("cheque_slip"):
        cheque_cols = table_cols("cheque_slip")
        officer_columns = cheque_slip_approval_owner_columns(cheque_cols)
        rule_definition = cheque_slip_approval_owner_runtime_rule(officer_columns)
        if rule_definition is not None:
            conditions.append((
                rule_definition.condition_sql,
                rule_definition.reason,
            ))

    # Rows removed before training as duplicate voided invoices should be shown as rule anomalies at test time.
    for table_name in joined_tables:
        column_names = table_cols(table_name)
        invoice_column = next(
            (column_name for column_name in ("invoice_number", "invoice_no") if column_name in column_names),
            None,
        )
        if not invoice_column or "invoice_date" not in column_names or "record_status" not in column_names:
            continue

        base_invoice_expr = f'base."{table_name}.{invoice_column}"'
        base_invoice_key = base_invoice_expr
        base_invoice_date_expr = _safe_sql_date_expr(f'base."{table_name}.invoice_date"')
        base_status_expr = f'base."{table_name}.record_status"'
        duplicate_cte_name = f"duplicate_invoice_keys_{_safe_rule_name(table_name, table_name)}"
        cte_invoice_expr = _quote(invoice_column)
        cte_invoice_key = cte_invoice_expr
        cte_invoice_date_expr = _safe_sql_date_expr(_quote("invoice_date"))
        cte_status_expr = _quote("record_status")
        cte_fk_dak_expr = _quote("fk_dak") if "fk_dak" in column_names else None
        bundle = duplicate_void_invoice_rule_bundle(
            table_name=table_name,
            invoice_column=invoice_column,
            source_table_ref=_source_table_only_ref(table_name),
            duplicate_cte_name=duplicate_cte_name,
            base_invoice_key_expr=base_invoice_key,
            base_invoice_date_expr=base_invoice_date_expr,
            base_status_expr=base_status_expr,
            cte_invoice_expr=cte_invoice_key,
            cte_invoice_date_expr=cte_invoice_date_expr,
            cte_status_expr=cte_status_expr,
            cte_fk_dak_expr=cte_fk_dak_expr,
        )
        ctes.extend(bundle.ctes)
        outer_joins.extend(bundle.outer_joins)
        conditions.append((bundle.rule.condition_sql, bundle.rule.reason))
        evidence_expressions.extend(bundle.evidence_expressions)

    # Date sequence rule
    date_sequence_conditions = _build_date_sequence_anomaly_conditions(
        joined_tables,
        source_columns,
    )

    conditions.extend(date_sequence_conditions)

    return conditions, ctes, outer_joins, evidence_expressions

def _build_join_sql(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
    *,
    row_limit: int | None,
    projection_mode: str,
) -> tuple[str, list[dict[str, Any]], list[str], dict[str, Any]]:
    _validate_join_payload_tables(payload, source_columns)

    warnings: list[str] = []
    params = _join_sql_params(payload)
    selected_tables = payload.selected_tables
    first_table = selected_tables[0]
    joined_aliases: set[str] = {first_table}
    used_tables: set[str] = {first_table}
    scoring_projection = projection_mode == "scoring"
    projected_columns = (
        _scoring_projection_columns(payload, source_columns)
        if scoring_projection
        else None
    )
    select_clause = _build_join_select_list(
        selected_tables,
        source_columns,
        projected_columns=projected_columns,
        include_locators=scoring_projection,
        cast_datetimes_as_text=not scoring_projection,
    )
    first_source_ref = _build_source_table_ref(first_table)
    from_clause = f"FROM {first_source_ref}"

    join_clauses: list[str] = []
    join_debug: list[dict[str, Any]] = []

    for index, join in enumerate(payload.joins, start=1):
        left_table = join.left_table
        right_table = join.right_table

        if left_table not in joined_aliases:
            raise ValueError(
                f"Join step {index} is invalid. Left table '{left_table}' is not yet part of the SQL join chain. "
                f"Start from '{first_table}' and chain joins in order."
            )
        if right_table in joined_aliases:
            raise ValueError(
                f"Join step {index} is invalid. Right table '{right_table}' is already part of the SQL join chain. "
                "Duplicate/repeated table joins are not supported by the current builder."
            )
        if right_table not in source_columns:
            raise ValueError(f"Join step {index}: right table '{right_table}' is not present in source schema.")

        left_meta = _source_column_meta(source_columns, left_table, join.left_column)
        right_meta = _source_column_meta(source_columns, right_table, join.right_column)
        left_column = left_meta["column_name"]
        right_column = right_meta["column_name"]
    
        join_condition=(
            f'{_quote(left_table)}.{_quote(left_column)} = ' 
            f'{_quote(right_table)}.{_quote(right_column)}'
        )
        join_mode = "raw_equal"

        right_source_ref = _build_source_table_ref(right_table)
        join_keyword = _sql_join_keyword(join.join_type)
        join_clause = f"{join_keyword} {right_source_ref} ON {join_condition}"
        join_clauses.append(join_clause)

        joined_aliases.add(right_table)
        used_tables.add(right_table)
        join_debug.append(
            {
                "step": index,
                "left_table": left_table,
                "right_table": right_table,
                "left_key": f"{left_table}.{left_column}",
                "right_key": f"{right_table}.{right_column}",
                "left_data_type": left_meta.get("data_type"),
                "right_data_type": right_meta.get("data_type"),
                "join_condition_mode": join_mode,
                "join_sql": join_clause,
            }
        )

    unjoined_tables = [table_name for table_name in selected_tables if table_name not in used_tables]
    if unjoined_tables:
        raise ValueError(
            "Some selected tables are not connected by the join configuration: "
            f"{unjoined_tables}. Add join steps for them or remove them from selected_tables."
        )

    base_sql = f"SELECT\n    {select_clause}\n{from_clause}"
    if join_clauses:
        base_sql += "\n" + "\n".join(join_clauses)

    filters = []
    list_date_filter = _list_date_filter_sql(payload, source_columns)
    if list_date_filter:
        filters.append(list_date_filter)

    sql = base_sql
    if filters:
        sql += "\nWHERE " + "\n  AND ".join(f"({item})" for item in filters)

    safe_limit = max(1, int(row_limit)) if row_limit and row_limit > 0 else None
    if safe_limit:
        warnings.append(
            "The final joined result is capped with LIMIT "
            f"{safe_limit}. This is a final-result limit, not a per-table source limit."
        )

    logger.info(
        "Workbench SQL join query built in projection_mode=%s with projected_source_columns=%s",
        projection_mode,
        "ALL" if projected_columns is None else len(projected_columns),
    )

    anomaly_conditions, anomaly_ctes, anomaly_outer_joins, anomaly_evidence_expressions = (
        _build_sql_anomaly_expressions(list(used_tables), source_columns)
    )
    if anomaly_conditions:
        anomaly_sql = " OR ".join([f"({condition})" for condition, _reason in anomaly_conditions])
        anomaly_reason_parts: list[str] = []
        for index, (condition, reason) in enumerate(anomaly_conditions, start=1):
            reason_param = f"sql_rule_reason_{index}"
            params[reason_param] = str(reason)
            anomaly_reason_parts.append(
                f"CASE WHEN ({condition}) THEN :{reason_param} ELSE NULL END"
            )
        anomaly_reason_sql = (
            "array_to_string(array_remove(ARRAY["
            + ", ".join(anomaly_reason_parts)
            + "]::text[], NULL), ', ')"
        )
        anomaly_evidence_sql = (
            "array_to_string(array_remove(ARRAY["
            + ", ".join(anomaly_evidence_expressions)
            + "]::text[], NULL), E'\\n')"
            if anomaly_evidence_expressions
            else "NULL::text"
        )
        cte_sql = [f"joined_base AS (\n{sql}\n)"]
        cte_sql.extend(anomaly_ctes)
        cte_sql_text = ",\n".join(cte_sql)
        outer_join_sql = "\n".join(anomaly_outer_joins)
        sql = f"""WITH {cte_sql_text}
SELECT base.*,
       CASE WHEN {anomaly_sql} THEN TRUE ELSE FALSE END AS sql_rule_flag,
       {anomaly_reason_sql} AS sql_rule_reasons,
       {anomaly_evidence_sql} AS {SQL_RULE_EVIDENCE_COLUMN}
FROM joined_base base"""
        if outer_join_sql:
            sql += "\n" + outer_join_sql

    if safe_limit:
        sql += f"\nLIMIT {safe_limit}"

    return sql, join_debug, warnings, params


def _get_or_build_join_sql(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
    *,
    row_limit: int | None,
    projection_mode: str,
) -> tuple[str, list[dict[str, Any]], list[str], dict[str, Any]]:
    """Reuse cached join SQL when possible, otherwise build and cache a fresh version."""
    cached = _get_cached_join_sql(
        payload,
        row_limit=row_limit,
        projection_mode=projection_mode,
    )
    if cached is not None:
        return cached

    sql, join_debug, warnings, params = _build_join_sql(
        payload,
        source_columns,
        row_limit=row_limit,
        projection_mode=projection_mode,
    )
    _store_cached_join_sql(
        payload,
        row_limit=row_limit,
        projection_mode=projection_mode,
        sql=sql,
        join_debug=join_debug,
        warnings=warnings,
        params=params,
    )
    return sql, join_debug, warnings, params

def _enrich_sql_join_debug(
    conn,
    join_debug: list[dict[str, Any]],
    source_row_counts: dict[str, int],
) -> list[dict[str, Any]]:
    """Attach row-count and key-sample diagnostics to the join debug payload."""
    if not ENABLE_EXPENSIVE_JOIN_DEBUG:
        return [
            {
                **item,
                "left_table_rows_estimate": source_row_counts.get(item["left_table"]),
                "right_table_rows_estimate": source_row_counts.get(item["right_table"]),
                "debug_mode": "light",
            }
            for item in join_debug
        ]

    enriched = []
    for item in join_debug:
        left_table = item["left_table"]
        right_table = item["right_table"]
        # Safely extract column name from "table.column" format
        left_parts = item["left_key"].split(".", 1)
        right_parts = item["right_key"].split(".", 1)
        left_col = left_parts[1] if len(left_parts) > 1 else left_parts[0]
        right_col = right_parts[1] if len(right_parts) > 1 else right_parts[0]

        try:
            common_count = _sql_common_key_count(conn, left_table, left_col, right_table, right_col)
            common_sample = _sql_common_key_sample(conn, left_table, left_col, right_table, right_col)
        except Exception as exc:
            common_count = -1
            common_sample = [f"debug_error: {exc}"]

        try:
            left_raw_sample = _sql_table_column_samples(conn, left_table, left_col)
        except Exception as exc:
            left_raw_sample = [f"debug_error: {exc}"]

        try:
            right_raw_sample = _sql_table_column_samples(conn, right_table, right_col)
        except Exception as exc:
            right_raw_sample = [f"debug_error: {exc}"]

        enriched.append(
            {
                **item,
                "left_table_rows_estimate": source_row_counts.get(left_table),
                "right_table_rows_estimate": source_row_counts.get(right_table),
                "left_raw_sample": left_raw_sample,
                "right_raw_sample": right_raw_sample,
                "common_raw_key_count_table_level": common_count,
                "common_raw_key_sample_table_level": common_sample,
                "debug_mode": "full",
            }
        )
    return enriched

def _execute_sql_joined_frame(
    payload: WorkbenchRunRequest,
    *,
    for_preview: bool = False,
) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, Any]], list[str], str]:
    """Execute the raw joined SQL and return a DataFrame for preview or downstream scoring."""
    row_limit = PREVIEW_ROW_LIMIT if for_preview else None
    source_columns = _source_columns_map(payload.selected_tables)
    sql, join_debug, warnings, params = _get_or_build_join_sql(
        payload,
        source_columns,
        row_limit=row_limit,
        projection_mode="full",
    )
    warnings.extend(_ensure_join_indexes(payload, source_columns))
    warnings.extend(_ensure_date_filter_indexes(payload, source_columns))

    source_row_counts: dict[str, int] = {}
    with _source_connect() as conn:
        for table_name in payload.selected_tables:
            source_row_counts[table_name] = _approx_table_row_count(conn, table_name)

        joined = pd.read_sql_query(text(sql), conn, params=params)
        join_debug = _enrich_sql_join_debug(conn, join_debug, source_row_counts)

    if joined.empty:
        previous_count = _previous_dataset_row_count(_dataset_table_name(payload.selected_tables))
        if previous_count > 0:
            warnings.append(
                "No new rows were available after skipping previously joined rows. "
                f"Previously saved rows: {previous_count}."
            )
        else:
            raise ValueError("The selected SQL join returned no rows. Review the chosen join keys and join type.")

    return joined, source_row_counts, join_debug, warnings, sql

def _execute_sql_workbench_frame(
    payload: WorkbenchRunRequest,
    conn,
) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, Any]], list[str], str, list[str], int, list[str], int, str]:
    """Build scoring SQL, materialize the temp table, and return the scoring-ready frame."""
    source_columns = _source_columns_map(payload.selected_tables)
    joined_sql, join_debug, warnings, joined_params = _get_or_build_join_sql(
        payload,
        source_columns,
        row_limit=None,
        projection_mode="scoring",
    )
    warnings.extend(_ensure_join_indexes(payload, source_columns))
    warnings.extend(_ensure_date_filter_indexes(payload, source_columns))
    (
        workbench_sql,
        params,
        user_rule_labels,
        applied_feature_rule_count,
        feature_warnings,
        user_rule_warnings,
        applied_user_rule_count,
    ) = _build_sql_workbench_query(payload, source_columns, joined_sql)
    params = {**joined_params, **params}

    source_row_counts: dict[str, int] = {}
    staging_table: str | None = None
    for table_name in payload.selected_tables:
        source_row_counts[table_name] = _approx_table_row_count(conn, table_name)

    _log_workbench_query_plan(conn, workbench_sql, params)
    staging_table, staged_row_count = _materialize_workbench_temp_table(conn, workbench_sql, params)
    joined = _read_temp_scoring_frame(conn, staging_table, payload)
    join_debug = _enrich_sql_join_debug(conn, join_debug, source_row_counts)

    if staged_row_count == 0:
        previous_count = _previous_dataset_row_count(_dataset_table_name(payload.selected_tables))
        if previous_count > 0:
            warnings.append(
                "No new rows were available after skipping previously joined rows. "
                f"Previously saved rows: {previous_count}."
            )
        else:
            raise ValueError("The selected SQL join returned no rows. Review the chosen join keys and join type.")

    return (
        joined,
        source_row_counts,
        join_debug,
        warnings,
        joined_sql,
        user_rule_labels,
        applied_feature_rule_count,
        feature_warnings + user_rule_warnings,
        applied_user_rule_count,
        staging_table,
    )

def _workbench_temp_table_name() -> str:
    """Generate a unique transaction-scoped temp-table name for one workbench run."""
    return f"tmp_ml_join_{uuid4().hex[:12]}"

def _workbench_temp_table_ref(temp_table: str) -> str:
    """Quote the generated temp-table name before embedding it in SQL."""
    return _quote(temp_table)

def _materialize_workbench_temp_table(conn, workbench_sql: str, params: dict[str, Any]) -> tuple[str, int]:
    """Create the transaction-scoped PostgreSQL temp table used for model scoring."""
    temp_table = _workbench_temp_table_name()
    temp_ref = _workbench_temp_table_ref(temp_table)
    started_at = time.monotonic()
    logger.info("Materializing workbench join into PostgreSQL temp table %s", temp_table)
    conn.execute(
        text(
            f"""
            -- This staging table is intentionally transaction-scoped.
            -- Callers must read it before the surrounding source transaction commits.
            CREATE TEMP TABLE {temp_ref}
            ON COMMIT DROP
            AS
            SELECT row_number() OVER () AS {_quote(TEMP_ROW_ID_COLUMN)}, src.*
            FROM (
            {workbench_sql}
            ) src
            """
        ),
        params,
    )
    conn.execute(text(f"CREATE INDEX ON {temp_ref} ({_quote(TEMP_ROW_ID_COLUMN)})"))
    conn.execute(text(f"ANALYZE {temp_ref}"))
    row_count = int(conn.execute(text(f"SELECT COUNT(*) FROM {temp_ref}")).scalar() or 0)
    logger.info(
        "Materialized %s rows into PostgreSQL temp table %s in %.2fs",
        row_count,
        temp_table,
        time.monotonic() - started_at,
    )
    return temp_table, row_count

def _temp_table_columns(conn, temp_table: str) -> list[str]:
    """Read the visible column list from the materialized temp table."""
    result = conn.execute(text(f"SELECT * FROM {_workbench_temp_table_ref(temp_table)} LIMIT 0"))
    return [str(column) for column in result.keys()]


def _read_temp_scoring_frame(
    conn,
    temp_table: str,
    payload: WorkbenchRunRequest,
) -> pd.DataFrame:
    """Load the scoring frame from the temp table after dropping sparse feature columns."""
    del payload

    available_columns = _temp_table_columns(conn, temp_table)
    required_columns = [
        TEMP_ROW_ID_COLUMN,
        USER_RULE_FLAG_COLUMN,
        USER_RULE_REASONS_COLUMN,
        SQL_RULE_FLAG_COLUMN,
        SQL_RULE_REASONS_COLUMN,
        SQL_RULE_EVIDENCE_COLUMN,
    ]
    candidate_columns = [
        column
        for column in available_columns
        if column not in required_columns
    ]
    _total_rows, present_ratios = _temp_table_column_presence_ratios(
        conn,
        temp_table,
        candidate_columns,
    )
    min_present_ratio = max(0.0, min(float(settings.anomaly_feature_min_present_ratio), 1.0))
    kept_candidate_columns = [
        column
        for column in candidate_columns
        if present_ratios.get(column, 0.0) >= min_present_ratio
    ]
    dropped_sparse_columns = [
        column
        for column in candidate_columns
        if column not in kept_candidate_columns
    ]

    if dropped_sparse_columns:
        logger.info(
            "Skipped %d sparse scoring columns from temp table %s with present ratio below %.2f: %s",
            len(dropped_sparse_columns),
            temp_table,
            min_present_ratio,
            [
                {
                    "column": column,
                    "present_ratio": round(float(present_ratios.get(column, 0.0)), 3),
                }
                for column in dropped_sparse_columns[:20]
            ],
        )

    selected_columns = [
        column
        for column in required_columns
        if column in available_columns and column != TEMP_ROW_ID_COLUMN
    ]
    selected_columns.extend(kept_candidate_columns)
    selected_columns.insert(0, TEMP_ROW_ID_COLUMN)
    select_sql = ", ".join(_quote(column) for column in selected_columns)
    df = pd.read_sql_query(
        text(
            f"""
            SELECT {select_sql}
            FROM {_workbench_temp_table_ref(temp_table)}
            ORDER BY {_quote(TEMP_ROW_ID_COLUMN)}
            """
        ),
        conn,
    )
    if TEMP_ROW_ID_COLUMN in df.columns:
        df = df.set_index(TEMP_ROW_ID_COLUMN, drop=True)
    logger.info("Loaded %s scoring columns from temp table %s", len(df.columns), temp_table)
    return df


def _temp_table_column_presence_ratios(
    conn,
    temp_table: str,
    candidate_columns: list[str],
) -> tuple[int, dict[str, float]]:
    """Compute non-null presence ratios for candidate scoring columns in the temp table."""
    if not candidate_columns:
        return 0, {}

    select_parts = ["COUNT(*) AS total_rows"]
    alias_by_column: dict[str, str] = {}

    for index, column_name in enumerate(candidate_columns):
        alias = f"present_{index}"
        alias_by_column[column_name] = alias
        select_parts.append(f"COUNT({_quote(column_name)}) AS {_quote(alias)}")

    row = conn.execute(
        text(
            f"""
            SELECT {", ".join(select_parts)}
            FROM {_workbench_temp_table_ref(temp_table)}
            """
        )
    ).mappings().first()

    total_rows = int((row or {}).get("total_rows") or 0)
    ratios: dict[str, float] = {}

    for column_name in candidate_columns:
        present_count = int((row or {}).get(alias_by_column[column_name]) or 0)
        if total_rows <= 0:
            ratios[column_name] = 0.0
            continue
        ratios[column_name] = present_count / total_rows

    return total_rows, ratios

def _log_workbench_query_plan(conn, workbench_sql: str, params: dict[str, Any]) -> None:
    """Log the PostgreSQL query plan for the generated workbench SQL when enabled."""
    explain_mode = "ANALYZE, BUFFERS, FORMAT TEXT" if settings.workbench_explain_analyze else "FORMAT TEXT"
    started_at = time.monotonic()
    try:
        rows = conn.execute(text(f"EXPLAIN ({explain_mode})\n{workbench_sql}"), params).fetchall()
    except Exception as exc:
        logger.warning("Unable to collect workbench query plan: %s", exc)
        return

    plan_text = "\n".join(str(row[0]) for row in rows)
    logger.info(
        "Workbench query plan collected in %.2fs using EXPLAIN (%s):\n%s",
        time.monotonic() - started_at,
        explain_mode,
        plan_text[:12000],
    )



def _read_temp_anomaly_payload_frame(
    conn,
    temp_table: str,
    row_ids: list[int],
    payload: WorkbenchRunRequest,
) -> pd.DataFrame:
    """Rehydrate final anomaly rows from source tables using temp-table locator columns."""
    if not row_ids:
        return pd.DataFrame()
    started_at = time.monotonic()
    available_columns = _temp_table_columns(conn, temp_table)
    locator_columns = [
        _payload_locator_column(table_name)
        for table_name in payload.selected_tables
        if _payload_locator_column(table_name) in available_columns
    ]
    select_columns = [_quote(TEMP_ROW_ID_COLUMN), *[_quote(column) for column in locator_columns]]
    frames: list[pd.DataFrame] = []
    chunk_size = 5000
    for start in range(0, len(row_ids), chunk_size):
        chunk = row_ids[start:start + chunk_size]
        query = text(
            f"""
            SELECT {", ".join(select_columns)}
            FROM {_workbench_temp_table_ref(temp_table)}
            WHERE {_quote(TEMP_ROW_ID_COLUMN)} IN :row_ids
            ORDER BY {_quote(TEMP_ROW_ID_COLUMN)}
            """
        ).bindparams(bindparam("row_ids", expanding=True))
        frames.append(
            pd.read_sql_query(
                query,
                conn,
                params={"row_ids": [int(row_id) for row_id in chunk]},
            )
        )
    locator_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if TEMP_ROW_ID_COLUMN in locator_frame.columns:
        locator_frame = locator_frame.set_index(TEMP_ROW_ID_COLUMN, drop=True)

    row_payloads: dict[int, dict[str, Any]] = {
        int(row_id): {}
        for row_id in locator_frame.index.tolist()
    }

    source_columns = _source_columns_map(payload.selected_tables)
    for table_name in payload.selected_tables:
        locator_column = _payload_locator_column(table_name)
        if locator_column not in locator_frame.columns:
            continue
        table_locators = [
            str(value)
            for value in locator_frame[locator_column].dropna().astype(str).tolist()
            if str(value).strip()
        ]
        if not table_locators:
            continue
        locator_to_payload = _table_payload_by_locator(
            conn,
            table_name,
            source_columns.get(table_name, []),
            table_locators,
        )
        for row_id, locator in locator_frame[locator_column].items():
            if pd.isna(locator):
                continue
            payload_values = locator_to_payload.get(str(locator))
            if payload_values:
                row_payloads[int(row_id)].update(payload_values)

    df = pd.DataFrame.from_dict(row_payloads, orient="index")
    if not df.empty:
        df.index.name = TEMP_ROW_ID_COLUMN
    logger.info(
        "Loaded %s anomaly payload rows from source tables via locators from temp table %s in %.2fs",
        len(locator_frame),
        temp_table,
        time.monotonic() - started_at,
    )
    return df


def _table_payload_by_locator(
    conn,
    table_name: str,
    columns: list[dict[str, Any]],
    locators: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch full source-row payloads keyed by the saved PostgreSQL row locator."""
    unique_locators = sorted({str(locator).strip() for locator in locators if str(locator).strip()})
    if not unique_locators:
        return {}

    select_parts = ['CAST(ctid AS text) AS "__ml_locator__"']
    for column in columns:
        column_name = str(column["column_name"])
        source_expr = f"{_quote(column_name)}"
        if _type_family(column.get("data_type")) == "datetime":
            source_expr = f"CAST({source_expr} AS text)"
        select_parts.append(
            f'{source_expr} AS {_quote(f"{table_name}.{column_name}")}'
        )

    query = text(
        f"""
        SELECT {", ".join(select_parts)}
        FROM {_source_table_only_ref(table_name)}
        WHERE CAST(ctid AS text) IN :locators
        """
    ).bindparams(bindparam("locators", expanding=True))

    frame = pd.read_sql_query(query, conn, params={"locators": unique_locators})
    if frame.empty:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        locator = str(row.get("__ml_locator__") or "").strip()
        if not locator:
            continue
        payload = {
            str(column): _safe_json(value)
            for column, value in row.items()
            if str(column) != "__ml_locator__"
        }
        result[locator] = payload
    return result

def _resolve_column(df: pd.DataFrame, column_name: str) -> str:
    """Resolve an exact or table-qualified DataFrame column reference safely."""
    exact_matches = [column for column in df.columns if column == column_name]
    if len(exact_matches) == 1:
        return column_name
    if len(exact_matches) > 1:
        raise ValueError(
            f"Column is ambiguous: {column_name}. Duplicate exact matches found."
        )
    matches = [column for column in df.columns if column.endswith(f".{column_name}")]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Column not found: {column_name}")
    raise ValueError(f"Column is ambiguous: {column_name}. Matches: {matches[:10]}")
