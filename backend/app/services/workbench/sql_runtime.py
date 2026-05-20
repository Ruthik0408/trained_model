from datetime import date, datetime, timezone
import time
from typing import Any
from uuid import uuid4

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.schemas.workbench_schema import BuiltinRuleRequest, FeatureRuleInput, OutlierRuleInput, WorkbenchRunRequest
from app.services.workbench.constants import (
    DATE_FILTER_COLUMN_PRIORITY,
    DATE_SEQUENCE_STAGE_ALIASES,
    ENABLE_EXPENSIVE_JOIN_DEBUG,
    MIN_FEATURE_COLUMN_PRESENT_RATIO,
    PREVIEW_ROW_LIMIT,
    RESULT_TABLE,
    SAME_TABLE_DATE_SEQUENCE_STAGES,
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
    _is_date_like_column_name,
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
from app.services.workbench.utils import _is_identifier_like_column, _safe_json, _safe_rule_name


_JOIN_SQL_CACHE_TTL_SECONDS = 60.0
_join_sql_cache: dict[tuple[Any, ...], tuple[float, str, list[dict[str, Any]], list[str]]] = {}


def _dataset_table_name(_selected_tables: list[str]) -> str:
    return RESULT_TABLE


def _join_sql_cache_key(
    payload: BuiltinRuleRequest | WorkbenchRunRequest,
    *,
    row_limit: int | None,
) -> tuple[Any, ...]:
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
    )


def _get_cached_join_sql(
    payload: BuiltinRuleRequest | WorkbenchRunRequest,
    *,
    row_limit: int | None,
) -> tuple[str, list[dict[str, Any]], list[str]] | None:
    cache_key = _join_sql_cache_key(payload, row_limit=row_limit)
    cached = _join_sql_cache.get(cache_key)
    if cached is None:
        return None

    cached_at, sql, join_debug, warnings = cached
    now = datetime.now(tz=timezone.utc).timestamp()
    if now - cached_at >= _JOIN_SQL_CACHE_TTL_SECONDS:
        _join_sql_cache.pop(cache_key, None)
        return None

    logger.info(
        "Reusing cached workbench join SQL for tables=%s row_limit=%s",
        payload.selected_tables,
        row_limit,
    )
    return sql, [dict(item) for item in join_debug], list(warnings)


def _store_cached_join_sql(
    payload: BuiltinRuleRequest | WorkbenchRunRequest,
    *,
    row_limit: int | None,
    sql: str,
    join_debug: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    cache_key = _join_sql_cache_key(payload, row_limit=row_limit)
    _join_sql_cache[cache_key] = (
        datetime.now(tz=timezone.utc).timestamp(),
        sql,
        [dict(item) for item in join_debug],
        list(warnings),
    )

def _feature_column_presence_ratios(
    table_frames: dict[str, list[dict]],
) -> dict[str, float]:
    ratios: dict[str, float] = {}
    with _source_connect() as conn:
        for table_name, columns in table_frames.items():
            date_columns = [
                str(column["column_name"])
                for column in columns
                if _is_date_like_column_name(str(column["column_name"]), column.get("data_type"))
            ]
            if not date_columns:
                continue
            select_parts = ["COUNT(*) AS total_rows"]
            alias_by_column: dict[str, str] = {}
            for index, column_name in enumerate(date_columns):
                alias = f"present_{index}"
                alias_by_column[column_name] = alias
                select_parts.append(
                    f"COUNT({_quote(column_name)}) AS {_quote(alias)}"
                )
            sql = text(f"SELECT {', '.join(select_parts)} FROM {_source_table_ref(table_name)}")
            row = conn.execute(sql).mappings().first()
            total_rows = int((row or {}).get("total_rows") or 0)
            for column_name in date_columns:
                qualified_name = f"{table_name}.{column_name}"
                if total_rows == 0:
                    ratios[qualified_name] = 0.0
                    continue
                present_count = int((row or {}).get(alias_by_column[column_name]) or 0)
                ratios[qualified_name] = present_count / total_rows
    return ratios

def _joined_feature_column_presence_ratios(
    payload: BuiltinRuleRequest | WorkbenchRunRequest,
    table_frames: dict[str, list[dict]],
) -> dict[str, float]:
    sql, _join_debug, _warnings = _get_or_build_join_sql(payload, table_frames, row_limit=None)
    date_columns = [
        f"{table_name}.{column['column_name']}"
        for table_name, columns in table_frames.items()
        for column in columns
        if _is_date_like_column_name(str(column["column_name"]), column.get("data_type"))
    ]
    if not date_columns:
        return {}

    select_parts = ["COUNT(*) AS total_rows"]
    alias_by_column: dict[str, str] = {}
    for index, column_name in enumerate(date_columns):
        alias = f"present_{index}"
        alias_by_column[column_name] = alias
        select_parts.append(f"COUNT(joined_source.{_quote(column_name)}) AS {_quote(alias)}")

    query = text(
        f"""
        SELECT {', '.join(select_parts)}
        FROM ({sql}) AS joined_source
        """
    )

    ratios: dict[str, float] = {}
    with _source_connect() as conn:
        row = conn.execute(query).mappings().first()
        total_rows = int((row or {}).get("total_rows") or 0)
        for column_name in date_columns:
            if total_rows == 0:
                ratios[column_name] = 0.0
                continue
            present_count = int((row or {}).get(alias_by_column[column_name]) or 0)
            ratios[column_name] = present_count / total_rows
    return ratios

def _has_enough_values_for_builtin_features(present_ratio: float) -> bool:
    return present_ratio >= MIN_FEATURE_COLUMN_PRESENT_RATIO

def _builtin_stage_column_matches(available: dict[str, str]) -> dict[str, list[str]]:
    matches_by_alias: dict[str, list[str]] = {}
    for qualified_name, data_type in available.items():
        _, plain_column = qualified_name.split(".", 1)
        normalized_plain = plain_column.strip().lower()
        if not _is_date_like_column_name(normalized_plain, data_type):
            continue
        matches_by_alias.setdefault(normalized_plain, []).append(qualified_name)
    return matches_by_alias

def _same_table_date_sequence_pair(left_stage_name: str, right_stage_name: str) -> bool:
    return (
        left_stage_name in SAME_TABLE_DATE_SEQUENCE_STAGES
        and right_stage_name in SAME_TABLE_DATE_SEQUENCE_STAGES
    )

def _build_dynamic_date_gap_rules(available: dict[str, str]) -> list[dict]:
    matches_by_alias = _builtin_stage_column_matches(available)
    stage_columns: list[tuple[str, list[str]]] = []
    for stage_name, aliases in DATE_SEQUENCE_STAGE_ALIASES:
        matched_columns: list[str] = []
        for alias in aliases:
            matched_columns.extend(matches_by_alias.get(alias, []))
        if matched_columns:
            stage_columns.append((stage_name, matched_columns))

    rules: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index in range(len(stage_columns) - 1):
        left_stage_name, left_columns = stage_columns[index]
        right_stage_name, right_columns = stage_columns[index + 1]
        for left_column in left_columns:
            for right_column in right_columns:
                if left_column == right_column:
                    continue
                if _same_table_date_sequence_pair(left_stage_name, right_stage_name):
                    left_table = left_column.split(".", 1)[0]
                    right_table = right_column.split(".", 1)[0]
                    if left_table != right_table:
                        continue
                pair_key = (right_column, left_column)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                rules.append(
                    {
                        "name": f"{left_stage_name}_to_{right_stage_name}",
                        "feature_type": "daysbetween",
                        "first_column": right_column,
                        "second_column": left_column,
                    }
                )
    return rules

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
        "outer": "FULL OUTER JOIN",
        "full": "FULL OUTER JOIN",
        "full outer": "FULL OUTER JOIN",
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported join type '{join_type}'. Supported types: inner, left, right, outer/full."
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

def _build_join_select_list(selected_tables: list[str], source_columns: dict[str, list[dict]]) -> str:
    parts: list[str] = []
    for table_name in selected_tables:
        for column in source_columns[table_name]:
            column_name = column["column_name"]
            source_expr = f"{_quote(table_name)}.{_quote(column_name)}"
            if _type_family(column.get("data_type")) == "datetime":
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
        column_alias = f"{rule_name}"
        while column_alias in used_aliases:
            column_alias = f"{column_alias}_{len(used_aliases) + 1}"
        try:
            feature_columns = [rule.first_column, rule.second_column]
            if any(_is_identifier_like_column(column) for column in feature_columns if column):
                warnings.append(f"Skipped feature rule '{rule_name}': identifier columns are not used as ML features.")
                continue
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

def _build_sql_outlier_predicate(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    rule: OutlierRuleInput,
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
    raise ValueError(f"Unsupported outlier operator: {rule.operator}")

def _build_sql_outlier_flag(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
    rules: list[OutlierRuleInput],
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
        rule_name = _safe_rule_name(rule.name, f"Outlier rule {index}")
        try:
            predicate = _build_sql_outlier_predicate(selected_tables, source_columns, rule, params, alias=alias)
            label = f"OUTLIER::{rule_name}"
            label_param = _next_param_name(params, "rule_label")
            params[label_param] = label
            predicates.append(predicate)
            reason_parts.append(f"CASE WHEN {predicate} THEN :{label_param} ELSE NULL END")
            labels.append(label)
            applied_count += 1
        except Exception as exc:
            warnings.append(f"Skipped outlier rule '{rule_name}': {exc}")

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
        outlier_expr,
        outlier_reason_expr,
        params,
        outlier_labels,
        outlier_warnings,
        applied_outlier_rule_count,
    ) = _build_sql_outlier_flag(
        payload.selected_tables,
        source_columns,
        payload.outlier_rules,
        alias="src",
    )

    select_parts = ["src.*"]
    select_parts.extend(feature_selects)
    select_parts.append(f"{outlier_expr} AS {_quote('__ml_sql_rule_flag')}")
    select_parts.append(f"{outlier_reason_expr} AS {_quote('__ml_sql_rule_reasons')}")
    sql = "WITH src AS (\n" + joined_sql + "\n)\nSELECT\n    " + ",\n    ".join(select_parts) + "\nFROM src"
    return (
        sql,
        params,
        outlier_labels,
        applied_feature_rule_count,
        feature_warnings,
        outlier_warnings,
        applied_outlier_rule_count,
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
            parts.append(f"{raw_expr} >= DATE '{from_date}'")
        if to_date:
            parts.append(f"{raw_expr} < (DATE '{to_date}' + INTERVAL '1 day')")
    else:
        safe_date = _safe_sql_date_expr(raw_expr)
        parts = [f"{safe_date} IS NOT NULL"]
        if from_date:
            parts.append(f"{safe_date} >= DATE '{from_date}'")
        if to_date:
            parts.append(f"{safe_date} <= DATE '{to_date}'")

    return "(" + " AND ".join(parts) + ")"

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
            table_columns = {
                str(column.get("column_name"))
                for column in source_columns.get(table_name, [])
            }

            for alias in aliases:
                if alias in table_columns:
                    exprs.append(
                        (
                            table_name,
                            _safe_sql_date_expr(f'base."{table_name}.{alias}"'),
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
        predicate_parts: list[str] = []

        for left_table_name, left_expr in left_exprs:
            for right_table_name, right_expr in right_exprs:
                if _same_table_date_sequence_pair(left_label, right_label) and left_table_name != right_table_name:
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
            (
                "(" + " OR ".join(predicate_parts) + ")",
                f"Date sequence violated across processing stages: {left_label} after {right_label}",
            )
        )
    return comparisons

def _build_sql_anomaly_expressions(
    joined_tables: list[str],
    source_columns: dict[str, list[dict]],
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    conditions: list[tuple[str, str]] = []
    ctes: list[str] = []
    outer_joins: list[str] = []

    joined_set = set(joined_tables)

    def has_table(table: str) -> bool:
        return table in joined_set

    def table_cols(table: str) -> set[str]:
        return {
            str(column.get("column_name"))
            for column in source_columns.get(table, [])
        }

    # Duplicate invoice check: bill / gem_bill
    for table_name in ("bill", "gem_bill"):
        if not has_table(table_name):
            continue

        cols = table_cols(table_name)
        invoice_column = next(
            (col for col in ("invoice_no", "invoice_number") if col in cols),
            None,
        )

        if not invoice_column:
            continue

        required_cols = {"invoice_date", "record_status"}
        if not required_cols.issubset(cols):
            continue

        conditions.append((
            f"""
            (
                base."{table_name}.record_status" = 'V'
                AND base."{table_name}.{invoice_column}" IS NOT NULL
                AND base."{table_name}.invoice_date" IS NOT NULL
                AND (
                    SELECT COUNT(*)
                    FROM {_source_table_only_ref(table_name)} b2
                    WHERE b2.record_status = 'V'
                      AND b2.{invoice_column} = base."{table_name}.{invoice_column}"
                      AND {_safe_sql_date_expr(f'b2.invoice_date')}
                          = {_safe_sql_date_expr(f'base."{table_name}.invoice_date"')}
                ) > 1
            )
            """,
            f"{table_name} has duplicate valid invoice number + invoice_date",
        ))

    # CMP scroll payment_reference_no must exist in ECS
    if has_table("cmp_scroll") and has_table("ecs"):
        conditions.append((
            f"""
            (
                base."cmp_scroll.payment_reference_no" IS NOT NULL
                AND base."cmp_scroll.cda_name" = 'CDA- Main Office Jabalpur'
                AND NOT EXISTS (
                    SELECT 1
                    FROM {_source_table_only_ref("ecs")} e
                    WHERE e.payment_reference_no = base."cmp_scroll.payment_reference_no"
                )
            )
            """,
            "CMP scroll has payment_reference_no but not found in ECS",
        ))

    # cheque_slip ECS mode = 1 but ECS record exists
    if has_table("cheque_slip") and has_table("ecs"):
        conditions.append((
            f"""
            (
                base."cheque_slip.fk_ecs_payment_mode" = 1
                AND base."cheque_slip.fk_dak" IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM {_source_table_only_ref("ecs")} e
                    WHERE e.fk_dak = base."cheque_slip.fk_dak"
                )
            )
            """,
            "Cheque slip ECS mode=1 but ECS record exists",
        ))

    # Cheque slip + schedule3 rules
    if has_table("cheque_slip") and has_table("schedule3"):
        ctes.append(
            f"""
            schedule3_by_dak AS (
                SELECT
                    fk_dak,
                    COUNT(*) AS schedule3_total_count,
                    COUNT(*) FILTER (WHERE record_status IN ('P', 'V')) AS schedule3_pv_count
                FROM {_source_table_only_ref("schedule3")}
                WHERE fk_dak IS NOT NULL
                GROUP BY fk_dak
            )
            """
        )
        ctes.append(
            f"""
            cheque_slip_approved_by_dak AS (
                SELECT
                    fk_dak,
                    COUNT(*) FILTER (WHERE record_status = 'V' AND approved = true) AS cheque_slip_v_approved_count
                FROM {_source_table_only_ref("cheque_slip")}
                WHERE fk_dak IS NOT NULL
                GROUP BY fk_dak
            )
            """
        )
        outer_joins.append(
            """
            LEFT JOIN schedule3_by_dak
                ON schedule3_by_dak.fk_dak = base."cheque_slip.fk_dak"
            """
        )
        outer_joins.append(
            """
            LEFT JOIN cheque_slip_approved_by_dak
                ON cheque_slip_approved_by_dak.fk_dak = base."cheque_slip.fk_dak"
            """
        )

        # cheque_slip V + approved false should not have schedule3 for same fk_dak
        conditions.append((
            f"""
            (
                base."cheque_slip.record_status" = 'V'
                AND base."cheque_slip.approved" = false
                AND base."cheque_slip.fk_dak" IS NOT NULL
                AND COALESCE(schedule3_by_dak.schedule3_total_count, 0) > 0
            )
            """,
            "Cheque slip record_status V and approved false but schedule3 exists for same fk_dak",
        ))

        # approved V cheque_slip count should match schedule3 P/V count for same fk_dak
        conditions.append((
            f"""
            (
                base."cheque_slip.record_status" = 'V'
                AND base."cheque_slip.approved" = true
                AND base."cheque_slip.fk_dak" IS NOT NULL
                AND COALESCE(cheque_slip_approved_by_dak.cheque_slip_v_approved_count, 0)
                    <> COALESCE(schedule3_by_dak.schedule3_pv_count, 0)
            )
            """,
            "Approved V cheque_slip count does not match schedule3 P/V count for same fk_dak",
        ))

    # approved cheque slip must have at least one approval officer/user
    if has_table("cheque_slip"):
        cheque_cols = table_cols("cheque_slip")

        officer_columns = [
            col for col in ("fk_aao", "fk_ao", "fk_go", "fk_auditor")
            if col in cheque_cols
        ]

        if officer_columns:
            all_officers_null = "\n                AND ".join(
                f'base."cheque_slip.{col}" IS NULL'
                for col in officer_columns
            )

            conditions.append((
                f"""
                (
                    base."cheque_slip.approved" = true
                    AND {all_officers_null}
                )
                """,
                f"Approved cheque slip but all approval columns are null: {', '.join(officer_columns)}",
            ))

    # Date sequence rule
    date_sequence_conditions = _build_date_sequence_anomaly_conditions(
        joined_tables,
        source_columns,
    )

    conditions.extend(date_sequence_conditions)

    return conditions, ctes, outer_joins

def _build_join_sql(
    payload: WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
    *,
    row_limit: int | None,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    _validate_join_payload_tables(payload, source_columns)

    warnings: list[str] = []
    selected_tables = payload.selected_tables
    first_table = selected_tables[0]
    joined_aliases: set[str] = {first_table}
    used_tables: set[str] = {first_table}
    select_clause = _build_join_select_list(selected_tables, source_columns)
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

    if row_limit and row_limit > 0:
        safe_limit = max(1, int(row_limit))
        sql += f"\nLIMIT {safe_limit}"
        warnings.append(
            "The final joined result is capped with LIMIT "
            f"{safe_limit}. This is a final-result limit, not a per-table source limit."
        )

    logger.info("Workbench SQL join query built: %s", sql)

    anomaly_conditions, anomaly_ctes, anomaly_outer_joins = _build_sql_anomaly_expressions(list(used_tables), source_columns)
    if anomaly_conditions:
        anomaly_sql = " OR ".join([f"({condition})" for condition, _reason in anomaly_conditions])
        anomaly_reason_parts: list[str] = []
        for condition, reason in anomaly_conditions:
            escaped_reason = str(reason).replace("'", "''")
            anomaly_reason_parts.append(
                f"CASE WHEN ({condition}) THEN '{escaped_reason}' ELSE NULL END"
            )
        anomaly_reason_sql = (
            "array_to_string(array_remove(ARRAY["
            + ", ".join(anomaly_reason_parts)
            + "]::text[], NULL), ', ')"
        )
        cte_sql = [f"joined_base AS (\n{sql}\n)"]
        cte_sql.extend(anomaly_ctes)
        cte_sql_text = ",\n".join(cte_sql)
        outer_join_sql = "\n".join(anomaly_outer_joins)
        sql = f"""WITH {cte_sql_text}
SELECT base.*,
       CASE WHEN {anomaly_sql} THEN TRUE ELSE FALSE END AS sql_rule_flag,
       {anomaly_reason_sql} AS sql_rule_reasons
FROM joined_base base"""
        if outer_join_sql:
            sql += "\n" + outer_join_sql

    return sql, join_debug, warnings


def _get_or_build_join_sql(
    payload: BuiltinRuleRequest | WorkbenchRunRequest,
    source_columns: dict[str, list[dict]],
    *,
    row_limit: int | None,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    cached = _get_cached_join_sql(payload, row_limit=row_limit)
    if cached is not None:
        return cached

    sql, join_debug, warnings = _build_join_sql(payload, source_columns, row_limit=row_limit)
    _store_cached_join_sql(
        payload,
        row_limit=row_limit,
        sql=sql,
        join_debug=join_debug,
        warnings=warnings,
    )
    return sql, join_debug, warnings

def _enrich_sql_join_debug(
    conn,
    join_debug: list[dict[str, Any]],
    source_row_counts: dict[str, int],
) -> list[dict[str, Any]]:
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
    row_limit = PREVIEW_ROW_LIMIT if for_preview else None
    source_columns = _source_columns_map(payload.selected_tables)
    sql, join_debug, warnings = _get_or_build_join_sql(payload, source_columns, row_limit=row_limit)
    warnings.extend(_ensure_join_indexes(payload, source_columns))
    warnings.extend(_ensure_date_filter_indexes(payload, source_columns))

    source_row_counts: dict[str, int] = {}
    with _source_connect() as conn:
        for table_name in payload.selected_tables:
            source_row_counts[table_name] = _approx_table_row_count(conn, table_name)

        joined = pd.read_sql_query(text(sql), conn)
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
    source_columns = _source_columns_map(payload.selected_tables)
    joined_sql, join_debug, warnings = _get_or_build_join_sql(payload, source_columns, row_limit=None)
    warnings.extend(_ensure_join_indexes(payload, source_columns))
    warnings.extend(_ensure_date_filter_indexes(payload, source_columns))
    (
        workbench_sql,
        params,
        outlier_labels,
        applied_feature_rule_count,
        feature_warnings,
        outlier_warnings,
        applied_outlier_rule_count,
    ) = _build_sql_workbench_query(payload, source_columns, joined_sql)

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
        outlier_labels,
        applied_feature_rule_count,
        feature_warnings + outlier_warnings,
        applied_outlier_rule_count,
        staging_table,
    )

def _workbench_temp_table_name() -> str:
    return f"tmp_ml_join_{uuid4().hex[:12]}"

def _workbench_temp_table_ref(temp_table: str) -> str:
    return _quote(temp_table)

def _materialize_workbench_temp_table(conn, workbench_sql: str, params: dict[str, Any]) -> tuple[str, int]:
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

def _drop_workbench_temp_table(temp_table: str | None) -> None:
    # Temp tables now live only for the current transaction and are removed by
    # PostgreSQL via ON COMMIT DROP, so there is no cross-session cleanup to do.
    return

def _temp_table_columns(conn, temp_table: str) -> list[str]:
    result = conn.execute(text(f"SELECT * FROM {_workbench_temp_table_ref(temp_table)} LIMIT 0"))
    return [str(column) for column in result.keys()]


def _temp_table_column_presence_ratios(
    conn,
    temp_table: str,
    candidate_columns: list[str],
) -> tuple[int, dict[str, float]]:
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

def _read_temp_scoring_frame(
    conn,
    temp_table: str,
    payload: WorkbenchRunRequest,
) -> pd.DataFrame:
    available_columns = _temp_table_columns(conn, temp_table)
    source_columns = _source_columns_map(payload.selected_tables)
    raw_date_columns = {
        f"{table_name}.{column['column_name']}"
        for table_name, columns in source_columns.items()
        for column in columns
        if _is_date_like_column_name(str(column["column_name"]), column.get("data_type"))
    }
    required_columns = [
        TEMP_ROW_ID_COLUMN,
        USER_RULE_FLAG_COLUMN,
        USER_RULE_REASONS_COLUMN,
        SQL_RULE_FLAG_COLUMN,
        SQL_RULE_REASONS_COLUMN,
    ]
    candidate_columns = [
        column
        for column in available_columns
        if (
            column not in required_columns
            and column not in raw_date_columns
            and not _is_identifier_like_column(column)
        )
    ]
    dropped_date_columns = [
        column
        for column in available_columns
        if column in raw_date_columns
    ]
    if dropped_date_columns:
        logger.info(
            "Skipped %d raw date/time scoring columns from temp table %s: %s",
            len(dropped_date_columns),
            temp_table,
            dropped_date_columns[:20],
        )
    dropped_identifier_columns = [
        column
        for column in available_columns
        if column not in required_columns and _is_identifier_like_column(column)
    ]
    if dropped_identifier_columns:
        logger.info(
            "Skipped %d identifier-like scoring columns from temp table %s: %s",
            len(dropped_identifier_columns),
            temp_table,
            dropped_identifier_columns[:20],
        )
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

def _read_temp_anomaly_payload_frame(
    conn,
    temp_table: str,
    row_ids: list[int],
    payload: WorkbenchRunRequest,
) -> pd.DataFrame:
    if not row_ids:
        return pd.DataFrame()
    started_at = time.monotonic()
    available_columns = _temp_table_columns(conn, temp_table)
    excluded = {
        TEMP_ROW_ID_COLUMN,
        USER_RULE_FLAG_COLUMN,
        USER_RULE_REASONS_COLUMN,
        SQL_RULE_FLAG_COLUMN,
        SQL_RULE_REASONS_COLUMN,
        *_feature_rule_aliases(payload.feature_rules),
    }
    payload_columns = [column for column in available_columns if column not in excluded]
    select_columns = [_quote(TEMP_ROW_ID_COLUMN), *[_quote(column) for column in payload_columns]]
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
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if TEMP_ROW_ID_COLUMN in df.columns:
        df = df.set_index(TEMP_ROW_ID_COLUMN, drop=True)
    logger.info(
        "Loaded %s anomaly payload rows from temp table %s in %.2fs",
        len(df),
        temp_table,
        time.monotonic() - started_at,
    )
    return df

def _resolve_column(df: pd.DataFrame, column_name: str) -> str:
    if column_name in df.columns:
        return column_name
    matches = [column for column in df.columns if column.endswith(f".{column_name}")]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Column not found: {column_name}")
    raise ValueError(f"Column is ambiguous: {column_name}. Matches: {matches[:10]}")
