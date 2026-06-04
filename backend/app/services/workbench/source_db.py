"""Database access helpers for source tables and persisted result tables."""
from collections import Counter
import re
from contextlib import contextmanager
from functools import lru_cache
from hashlib import blake2s
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from app.core.cache import TABLE_METADATA_CACHE
from app.core.config import settings
from app.services.workbench.constants import RESULT_SCHEMA, RUN_ID_COLUMN, logger
from app.services.workbench.utils import _slug_token


_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def list_source_tables() -> list[dict]:
    """Return source schema tables grouped with their visible columns and types."""
    cached_result = TABLE_METADATA_CACHE.get("source_tables")
    if cached_result is not None:
        logger.debug("Returning cached source tables list")
        return cached_result
    query = text(
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = :schema
        ORDER BY table_name, ordinal_position
        """
    )
    grouped: dict[str, list[dict]] = {}

    with _source_connect() as conn:
        for row in conn.execute(query, {"schema": settings.source_db_schema}):
            grouped.setdefault(row.table_name, []).append(
                {
                    "table_name": row.table_name,
                    "column_name": row.column_name,
                    "data_type": row.data_type,
                }
            )

    result = [
        {"table_name": table_name, "columns": columns}
        for table_name, columns in grouped.items()
    ]

    TABLE_METADATA_CACHE.set("source_tables", result)
    return result


def source_connection_status() -> dict:
    """Probe the source PostgreSQL database and return a UI-friendly health payload."""
    try:
        with _source_connect() as conn:
            conn.execute(text("SELECT 1"))
            table_count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = :schema"
                ),
                {"schema": settings.source_db_schema},
            ).scalar_one()

        return {
            "connected": True,
            "host": settings.source_db_host,
            "database": settings.source_db_name,
            "table_count": int(table_count),
        }
    except Exception as exc:
        return {
            "connected": False,
            "host": settings.source_db_host,
            "database": settings.source_db_name,
            "table_count": 0,
            "error": _friendly_source_db_error(exc),
        }


@lru_cache(maxsize=1)
def _source_engine():
    """Create and cache the SQLAlchemy engine used for source/result PostgreSQL access."""
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=settings.source_db_user,
        password=settings.source_db_password,
        host=settings.source_db_host,
        port=settings.source_db_port,
        database=settings.source_db_name,
    )

    return create_engine(
        url,
        future=True,
        pool_size=settings.source_db_pool_size,
        max_overflow=settings.source_db_max_overflow,
        pool_pre_ping=True,
        pool_recycle=settings.source_db_pool_recycle_seconds,
        pool_timeout=settings.source_db_pool_timeout_seconds,
    )


def _dispose_source_engine() -> None:
    """Dispose the cached source engine after connectivity errors or pool issues."""
    try:
        engine = _source_engine()
        engine.dispose()
    except Exception:
        logger.exception("Failed to dispose source engine")
    finally:
        _source_engine.cache_clear()


def _friendly_source_db_error(exc: Exception) -> str:
    """Translate raw database exceptions into clearer operator-facing error text."""
    message = str(exc).strip()
    lowered = message.lower()

    if "too many clients already" in lowered:
        return (
            "PostgreSQL is currently handling too many database connections at the same time. "
            "Please wait a moment and try again, or reduce the number of simultaneous jobs/users."
        )

    if "timeout" in lowered and "queuepool" in lowered:
        return (
            "Timed out while waiting for a PostgreSQL connection from the SQLAlchemy pool. "
            "The pool is saturated or queries are holding connections too long."
        )

    if not message:
        return (
            "Could not connect to the source PostgreSQL database. "
            f"Check TULIP_SOURCE_DB_HOST={settings.source_db_host}, "
            f"TULIP_SOURCE_DB_PORT={settings.source_db_port}, "
            f"TULIP_SOURCE_DB_NAME={settings.source_db_name}, "
            "and the source DB credentials."
        )
    
    return message


@contextmanager
def _source_connect():
    """Yield a plain source DB connection and normalize connection failures."""
    try:
        with _source_engine().connect() as conn:
            yield conn
    except OperationalError as exc:
        _dispose_source_engine()
        raise ConnectionError(_friendly_source_db_error(exc)) from exc


@contextmanager
def _source_begin():
    """Yield a transactional source DB connection for writes or temp-table work."""
    try:
        with _source_engine().begin() as conn:
            yield conn
    except OperationalError as exc:
        _dispose_source_engine()
        raise ConnectionError(_friendly_source_db_error(exc)) from exc


def _quote(name: str) -> str:
    """Quote a SQL identifier using PostgreSQL double-quote escaping."""
    return '"' + str(name).replace('"', '""') + '"'


def _index_name(*parts: str) -> str:
    """Build a deterministic PostgreSQL index name that respects name-length limits."""
    base = "idx_" + "_".join(_slug_token(part) for part in parts)

    if len(base) <= 55:
        return base

    digest = blake2s(base.encode("utf-8")).hexdigest()[:8]
    return f"{base[:46].rstrip('_')}_{digest}"


def _storage_column_name(column_name: Any, used_names: set[str]) -> str:
    """Normalize one DataFrame column name into a collision-safe SQL column name."""
    raw_name = str(column_name).strip()

    safe_base = re.sub(r"[^A-Za-z0-9_]+", "_", raw_name)
    safe_base = re.sub(r"_+", "_", safe_base).strip("_")

    if not safe_base:
        safe_base = "column"

    if safe_base[0].isdigit():
        safe_base = f"col_{safe_base}"

    if len(safe_base) <= 55 and safe_base not in used_names:
        used_names.add(safe_base)
        return safe_base

    digest = blake2s(raw_name.encode("utf-8")).hexdigest()[:8]
    base = safe_base[:46].rstrip("_")
    candidate = f"{base}_{digest}"

    suffix = 1
    while candidate in used_names:
        suffix_text = f"_{suffix}"
        base = safe_base[: 46 - len(suffix_text)].rstrip("_")
        candidate = f"{base}_{digest}{suffix_text}"
        suffix += 1

    used_names.add(candidate)
    return candidate


def _normalize_storage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DataFrame columns into result-table-safe storage identifiers when needed."""
    used_names: set[str] = set()
    renamed = [_storage_column_name(column, used_names) for column in df.columns]

    if renamed == [str(column) for column in df.columns]:
        return df

    normalized = df.copy()
    normalized.columns = renamed
    return normalized

def _source_table_ref(table_name: str) -> str:
    """Return a fully qualified source table reference after identifier validation."""
    _validate_identifier(table_name, "source table")
    return f"{_quote(settings.source_db_schema)}.{_quote(table_name)}"


def _source_columns_map(
    selected_tables: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Build a quick lookup of source column metadata keyed by table name."""
    tables = list_source_tables()
    out: dict[str, list[dict]] = {}

    for item in tables:
        table_name = item["table_name"]

        if selected_tables and table_name not in selected_tables:
            continue

        out[table_name] = item["columns"]

    return out


def _approx_table_row_count(conn, table_name: str) -> int:
    """Read PostgreSQL planner statistics for an approximate source table row count."""
    sql = text(
        """
        SELECT COALESCE(c.reltuples, 0)::bigint
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema_name
          AND c.relname = :table_name
        """
    )

    value = conn.execute(
        sql,
        {
            "schema_name": settings.source_db_schema,
            "table_name": table_name,
        },
    ).scalar()

    return int(value or 0)


def _validate_selected_tables(
    selected_tables: list[str],
    source_columns: dict[str, list[dict]],
) -> None:
    """Reject empty, missing, or duplicated source table selections."""
    if not selected_tables:
        raise ValueError("At least one source table must be selected.")

    missing = [
        table_name
        for table_name in selected_tables
        if table_name not in source_columns
    ]

    if missing:
        raise ValueError(
            f"Selected source tables not found in schema"
            f"'{settings.source_db_schema}': {missing}"
        )

    counts = Counter(selected_tables)
    duplicates = [table for table, count in counts.items() if count > 1]

    if duplicates:
        raise ValueError(
            "Duplicate source tables are not supported in the current SQL join builder. "
            f"Remove duplicates from selected_tables: {sorted(set(duplicates))}"
        )


def _resolve_source_column_name(
    source_columns: dict[str, list[dict]],
    table_name: str,
    column_name: str,
) -> str:
    """Resolve a table-qualified or shorthand column reference into the actual source column."""
    if not column_name:
        raise ValueError(f"Join column missing for table: {table_name}")

    if table_name not in source_columns:
        raise ValueError(
            f"Table '{table_name}' is not available in source schema "
            f"'{settings.source_db_schema}'."
        )

    raw_column = str(column_name).strip()
    column_parts = [part for part in raw_column.split(".") if part]

    if len(column_parts) == 1:
        plain_column = column_parts[0]
    elif table_name in column_parts:
        table_position = column_parts.index(table_name)
        if table_position == len(column_parts) - 1:
            raise ValueError(
                f"Column '{column_name}' is missing the field name after table '{table_name}'."
            )
        plain_column = column_parts[table_position + 1]
    else:
        plain_column = column_parts[-1]

    available = {item["column_name"] for item in source_columns[table_name]}

    if plain_column not in available:
        raise ValueError(
            f"Column '{column_name}' was not found in source table '{table_name}'. "
            f"Available columns sample: {sorted(list(available))[:20]}"
        )

    return plain_column


def _source_column_meta(
    source_columns: dict[str, list[dict]],
    table_name: str,
    column_name: str,
) -> dict[str, Any]:
    """Return full metadata for a validated source column reference."""
    plain_column = _resolve_source_column_name(source_columns, table_name, column_name)

    for item in source_columns[table_name]:
        if item["column_name"] == plain_column:
            return item

    raise ValueError(f"Column metadata not found for {table_name}.{plain_column}")


def _validate_identifier(value: str, label: str) -> None:
    """Block unsafe SQL identifiers before they reach string-built SQL."""
    value = str(value)
    if not _SAFE_IDENTIFIER_RE.match(value):
        raise ValueError(f"Unsafe SQL identifier for {label}: {value!r}")


def _result_table_ref(table_name: str) -> str:
    """Return a fully qualified result-table reference after validation."""
    _validate_identifier(table_name, "result table")
    return f"{_quote(RESULT_SCHEMA)}.{_quote(table_name)}"


def _previous_dataset_row_count(dataset_table: str) -> int:
    """Return the current row count of the persisted anomaly dataset table."""
    _validate_identifier(dataset_table, "dataset_table")

    if not _result_table_exists(dataset_table):
        return 0

    with _source_connect() as conn:
        total_rows = conn.execute(
            text(f"SELECT COUNT(*) FROM {_result_table_ref(dataset_table)}")
        ).scalar()

    return int(total_rows or 0)


def _next_dataset_run_id(dataset_table: str) -> int:
    """Compute the next dataset-local run id stored in the result table."""
    _validate_identifier(dataset_table, "dataset_table")

    if not _result_table_exists(dataset_table):
        return 1

    if RUN_ID_COLUMN not in _result_table_columns(dataset_table):
        return 1

    sql = text(
        f"""
        SELECT COALESCE(MAX({_quote(RUN_ID_COLUMN)}), 0)
        FROM {_result_table_ref(dataset_table)}
        """
    )

    with _source_connect() as conn:
        current_max = conn.execute(sql).scalar()

    return int(current_max or 0) + 1


def _result_table_exists(dataset_table: str) -> bool:
    """Check whether the configured result table exists in PostgreSQL."""
    if not dataset_table:
        return False

    _validate_identifier(dataset_table, "dataset_table")
    cache_key = f"result_table_exists:{dataset_table}"
    cached = TABLE_METADATA_CACHE.get(cache_key)
    if cached is not None:
        return bool(cached)

    sql = text(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = :schema_name
          AND table_name = :table_name
        LIMIT 1
        """
    )

    try:
        with _source_connect() as conn:
            exists = (
                conn.execute(
                    sql,
                    {
                        "schema_name": RESULT_SCHEMA,
                        "table_name": dataset_table,
                    },
                ).scalar()
                == 1
            )
            TABLE_METADATA_CACHE.set(cache_key, bool(exists))
            return bool(exists)
    except (SQLAlchemyError, ValueError) as exc:
        logger.warning(
            "Unable to verify result table %s.%s: %s",
            RESULT_SCHEMA,
            dataset_table,
            exc,
        )
        return False


def _clear_result_table_exists_cache(dataset_table: str | None = None) -> None:
    """Invalidate one or all cached result-table existence lookups."""
    if dataset_table:
        TABLE_METADATA_CACHE.invalidate(f"result_table_exists:{dataset_table}")
        return
    TABLE_METADATA_CACHE.invalidate_prefix("result_table_exists:")


def _result_table_columns(dataset_table: str) -> frozenset[str]:
    """Return the visible columns of the persisted result table."""
    if not dataset_table:
        return frozenset()

    _validate_identifier(dataset_table, "dataset_table")
    cache_key = f"result_table_columns:{dataset_table}"
    cached = TABLE_METADATA_CACHE.get(cache_key)
    if cached is not None:
        return frozenset(str(column) for column in cached)

    sql = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema_name
          AND table_name = :table_name
        ORDER BY ordinal_position
        """
    )

    with _source_connect() as conn:
        columns = frozenset(
            str(row.column_name)
            for row in conn.execute(
                sql,
                {
                    "schema_name": RESULT_SCHEMA,
                    "table_name": dataset_table,
                },
            )
        )
    TABLE_METADATA_CACHE.set(cache_key, sorted(columns))
    return columns


def _clear_result_table_columns_cache(dataset_table: str | None = None) -> None:
    """Invalidate one or all cached result-table column lookups."""
    if dataset_table:
        TABLE_METADATA_CACHE.invalidate(f"result_table_columns:{dataset_table}")
        return
    TABLE_METADATA_CACHE.invalidate_prefix("result_table_columns:")
