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
    """
    Get list of tables from source database.
    
    Results are cached for 5 minutes to avoid repeated information_schema queries.
    """
    # Try to get from cache first
    cached_result = TABLE_METADATA_CACHE.get("source_tables")
    if cached_result is not None:
        logger.debug("Returning cached source tables list")
        return cached_result
    
    schema = settings.source_db_schema
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
        for row in conn.execute(query, {"schema": schema}):
            grouped.setdefault(row.table_name, []).append(
                {
                    "table_name": row.table_name,
                    "column_name": row.column_name,
                    "data_type": row.data_type,
                }
            )
    
    result = [{"table_name": table, "columns": columns} for table, columns in grouped.items()]
    TABLE_METADATA_CACHE.set("source_tables", result)
    return result

def source_connection_status() -> dict:
    resolved_database = settings.source_db_name
    try:
        tables = list_source_tables()
        return {
            "connected": True,
            "host": settings.source_db_host,
            "database": resolved_database,
            "table_count": len(tables),
        }
    except Exception as exc:
        return {
            "connected": False,
            "host": settings.source_db_host,
            "database": resolved_database,
            "table_count": 0,
            "error": _friendly_source_db_error(exc),
        }

@lru_cache(maxsize=16)
def _source_engine():
    resolved_database = settings.source_db_name
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=settings.source_db_user,
        password=settings.source_db_password,
        host=settings.source_db_host,
        port=settings.source_db_port,
        database=resolved_database,
    )
    return create_engine(
        url,
        future=True,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )

def _dispose_source_engine() -> None:
    try:
        _source_engine().dispose()
        _source_engine.cache_clear()
    except Exception:
        logger.exception("Failed to dispose source engine")

def _friendly_source_db_error(exc: Exception) -> str:
    message = str(exc).strip()
    lowered = message.lower()
    missing_database_match = re.search(r'database "([^"]+)" does not exist', message, flags=re.IGNORECASE)
    if missing_database_match:
        missing_database = missing_database_match.group(1)
        return (
            f'PostgreSQL database "{missing_database}" does not exist on '
            f'{settings.source_db_host}:{settings.source_db_port}. '
            "Update TULIP_SOURCE_DB_NAME in your .env to the exact database name."
        )
    if "too many clients already" in lowered:
        return (
            "PostgreSQL has no free client connections right now. "
            "Reduce concurrent workbench/scheduler jobs and use a smaller SQLAlchemy pool."
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
            f"TULIP_SOURCE_DB_NAME={settings.source_db_name}, and the source DB credentials."
        )
    return message

@contextmanager
def _source_connect():
    engine = _source_engine()
    try:
        with engine.connect() as conn:
            yield conn
    except OperationalError as exc:
        _dispose_source_engine()
        raise ConnectionError(_friendly_source_db_error(exc)) from exc

@contextmanager
def _source_begin():
    engine = _source_engine()
    try:
        with engine.begin() as conn:
            yield conn
    except OperationalError as exc:
        _dispose_source_engine()
        raise ConnectionError(_friendly_source_db_error(exc)) from exc

def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'

def _index_name(*parts: str) -> str:
    base = "idx_" + "_".join(_slug_token(part) for part in parts)
    if len(base) <= 55:
        return base
    digest = blake2s(base.encode("utf-8")).hexdigest()[:8]
    return f"{base[:46].rstrip('_')}_{digest}"

def _storage_column_name(column_name: Any, used_names: set[str]) -> str:
    raw_name = str(column_name)
    if len(raw_name) <= 55 and raw_name not in used_names:
        used_names.add(raw_name)
        return raw_name

    digest = blake2s(raw_name.encode("utf-8")).hexdigest()[:8]
    candidate = f"{raw_name[:46].rstrip('_')}_{digest}"
    suffix = 1
    while candidate in used_names:
        suffix_text = f"_{suffix}"
        candidate = f"{raw_name[:46 - len(suffix_text)].rstrip('_')}_{digest}{suffix_text}"
        suffix += 1
    used_names.add(candidate)
    return candidate

def _normalize_storage_columns(df: pd.DataFrame) -> pd.DataFrame:
    used_names: set[str] = set()
    renamed = [_storage_column_name(column, used_names) for column in df.columns]
    if renamed == [str(column) for column in df.columns]:
        return df
    normalized = df.copy()
    normalized.columns = renamed
    return normalized

def _source_table_ref(table_name: str) -> str:
    return f"{_quote(settings.source_db_schema)}.{_quote(table_name)}"

def _source_columns_map(
    selected_tables: list[str] | None = None,
) -> dict[str, list[dict]]:
    tables = list_source_tables()
    out: dict[str, list[dict]] = {}
    for item in tables:
        table_name = item["table_name"]
        if selected_tables and table_name not in selected_tables:
            continue
        out[table_name] = item["columns"]
    return out

def _is_date_like_column_name(column_name: str, data_type: str | None = None) -> bool:
    lower_name = str(column_name).strip().lower()
    lower_type = str(data_type or "").strip().lower()
    return any(token in lower_type for token in ["date", "time"]) or any(
        token in lower_name for token in ["date", "time", "created_at", "updated_at", "timestamp"]
    )

def _approx_table_row_count(conn, table_name: str) -> int:
    sql = text(
        """
        SELECT COALESCE(c.reltuples, 0)::bigint
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema_name
          AND c.relname = :table_name
        """
    )
    return int(conn.execute(sql, {"schema_name": settings.source_db_schema, "table_name": table_name}).scalar() or 0)

def _validate_selected_tables(selected_tables: list[str], source_columns: dict[str, list[dict]]) -> None:
    if not selected_tables:
        raise ValueError("At least one source table must be selected.")

    missing = [table_name for table_name in selected_tables if table_name not in source_columns]
    if missing:
        raise ValueError(f"Selected source tables not found in schema '{settings.source_db_schema}': {missing}")

    counts = Counter(selected_tables)
    duplicates = [table for table, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(
            "Duplicate source tables are not supported in the current SQL join builder. "
            f"Remove duplicates from selected_tables: {sorted(set(duplicates))}"
        )

def _resolve_source_column_name(source_columns: dict[str, list[dict]], table_name: str, column_name: str) -> str:
    if not column_name:
        raise ValueError(f"Join column missing for table: {table_name}")
    if table_name not in source_columns:
        raise ValueError(f"Table '{table_name}' is not available in source schema '{settings.source_db_schema}'.")

    plain_column = column_name.split(".", 1)[1] if column_name.startswith(f"{table_name}.") else column_name
    available = {item["column_name"] for item in source_columns[table_name]}
    if plain_column not in available:
        raise ValueError(
            f"Column '{column_name}' was not found in source table '{table_name}'. "
            f"Available columns sample: {sorted(list(available))[:20]}"
        )
    return plain_column

def _source_column_meta(source_columns: dict[str, list[dict]], table_name: str, column_name: str) -> dict[str, Any]:
    plain_column = _resolve_source_column_name(source_columns, table_name, column_name)
    for item in source_columns[table_name]:
        if item["column_name"] == plain_column:
            return item
    raise ValueError(f"Column metadata not found for {table_name}.{plain_column}")

def _validate_identifier(value: str, label: str) -> None:
    if not _SAFE_IDENTIFIER_RE.match(value):
        raise ValueError(f"Unsafe SQL identifier for {label}: {value!r}")

def _result_table_ref(table_name: str) -> str:
    return f"{_quote(RESULT_SCHEMA)}.{_quote(table_name)}"

def _previous_dataset_row_count(dataset_table: str) -> int:
    _validate_identifier(dataset_table, "dataset_table")
    if not _result_table_exists(dataset_table, ):
        return 0
    with _source_connect() as conn:
        total_rows = conn.execute(text(f"SELECT COUNT(*) FROM {_result_table_ref(dataset_table)}")).scalar()
    return int(total_rows or 0)

def _next_dataset_run_id(dataset_table: str) -> int:
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

@lru_cache(maxsize=16)
def _result_table_exists(dataset_table: str) -> bool:
    if not dataset_table:
        return False
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
            return conn.execute(sql, {"schema_name": RESULT_SCHEMA, "table_name": dataset_table}).scalar() == 1
    except (SQLAlchemyError, ValueError) as exc:
        logger.warning("Unable to verify result table %s.%s: %s", RESULT_SCHEMA, dataset_table, exc)
        return False

def _clear_result_table_exists_cache(dataset_table: str | None = None) -> None:
    """Clear the cache for _result_table_exists. If dataset_table is provided, clear all caches."""
    try:
        _result_table_exists.cache_clear()
    except Exception:
        pass

@lru_cache(maxsize=16)
def _result_table_columns(dataset_table: str) -> frozenset[str]:
    if not dataset_table:
        return frozenset()
    sql = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema_name
          AND table_name = :table_name
        """
    )
    with _source_connect() as conn:
        return frozenset(
            str(row.column_name)
            for row in conn.execute(sql, {"schema_name": RESULT_SCHEMA, "table_name": dataset_table})
        )

def _clear_result_table_columns_cache(dataset_table: str | None = None) -> None:
    """Clear the cache for _result_table_columns. If dataset_table is provided, only that table's cache is cleared."""
    if dataset_table is None:
        _result_table_columns.cache_clear()
    else:
        try:
            _result_table_columns.cache_clear()
        except Exception:
            pass
