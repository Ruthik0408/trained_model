from datetime import datetime, timezone
import hashlib

from app.schemas.workbench_schema import BuiltinRuleRequest, WorkbenchRunRequest
from app.services.workbench.constants import (
    BUILTIN_FEATURE_RULES_CACHE_TTL,
    SINGLE_FEATURE_TYPES,
    _builtin_feature_rules_cache,
    logger,
)
from app.services.workbench.source_db import _source_columns_map
from app.services.workbench.sql_runtime import (
    _build_dynamic_date_gap_rules,
    _feature_column_presence_ratios,
    _has_enough_values_for_builtin_features,
    _is_date_like_column_name,
    _joined_feature_column_presence_ratios,
    _safe_date_literal,
)

def builtin_feature_rules(
    payload: BuiltinRuleRequest | WorkbenchRunRequest | list[str],
) -> list[dict]:
    if isinstance(payload, list):
        selected_tables = payload
        request_payload = None
    else:
        selected_tables = payload.selected_tables
        request_payload = payload

    join_signature: tuple[tuple[str, str, str, str, str], ...] = ()
    from_date: str | None = None
    to_date: str | None = None

    if request_payload is not None:
        join_signature = tuple(
            sorted(
                (
                    str(join.left_table),
                    str(join.left_column),
                    str(join.right_table),
                    str(join.right_column),
                    str(join.join_type),
                )
                for join in request_payload.joins
            )
        )
        from_date = _safe_date_literal(getattr(request_payload, "from_date", None))
        to_date = _safe_date_literal(getattr(request_payload, "to_date", None))

    join_hash = hashlib.md5(repr(join_signature).encode("utf-8")).hexdigest()

    cache_key = (
        tuple(sorted(selected_tables)),
        join_hash,
        from_date,
        to_date,
    )

    now = datetime.now(tz=timezone.utc).timestamp()
    cached = _builtin_feature_rules_cache.get(cache_key)

    if cached is not None and now - cached[0] < BUILTIN_FEATURE_RULES_CACHE_TTL:
        return cached[1]

    table_frames = _source_columns_map(selected_tables)

    use_joined_presence = bool(
        request_payload
        and (
            request_payload.joins
            or from_date
            or to_date
        )
    )

    if use_joined_presence and request_payload is not None:
        try:
            present_ratios = _joined_feature_column_presence_ratios(
                request_payload,
                table_frames,
            )
        except Exception as exc:
            logger.warning(
                "Falling back to table-level built-in feature detection for tables=%s because "
                "join/date-aware feature profiling failed: %s",
                selected_tables,
                exc,
            )
            present_ratios = _feature_column_presence_ratios(table_frames)
    else:
        present_ratios = _feature_column_presence_ratios(table_frames)

    available = {
        f"{table_name}.{column['column_name']}": column.get("data_type", "")
        for table_name, columns in table_frames.items()
        for column in columns
        if _has_enough_values_for_builtin_features(
            present_ratios.get(f"{table_name}.{column['column_name']}", 0.0)
        )
    }

    rules: list[dict] = []
    rules.extend(_build_dynamic_date_gap_rules(available))

    for column_name, data_type in available.items():
        if not _is_date_like_column_name(column_name, data_type):
            continue

        for feature_type in SINGLE_FEATURE_TYPES:
            pretty = feature_type.title()
            rules.append(
                {
                    "name": f"{pretty}-{column_name}",
                    "feature_type": feature_type,
                    "first_column": column_name,
                    "second_column": "",
                    "operator": "",
                }
            )

    deduped_rules: list[dict] = []
    seen_rule_keys: set[tuple[str, str, str, str]] = set()

    for rule in rules:
        rule_key = (
            str(rule.get("feature_type") or ""),
            str(rule.get("first_column") or ""),
            str(rule.get("second_column") or ""),
            str(rule.get("operator") or ""),
        )

        if rule_key in seen_rule_keys:
            continue

        seen_rule_keys.add(rule_key)
        deduped_rules.append(rule)

    _builtin_feature_rules_cache[cache_key] = (now, deduped_rules)
    return deduped_rules