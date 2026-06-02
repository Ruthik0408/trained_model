import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import OperationalError
from app.core.config import settings

logger = logging.getLogger(__name__)

# PostgreSQL-only engine for the application database (WorkbenchRun metadata, etc.)
engine = create_engine(
    settings.db_url,
    future=True,
    echo=False,
    pool_pre_ping=True,       # Verify connections before using them
    pool_recycle=settings.app_db_pool_recycle_seconds,
    pool_timeout=settings.app_db_pool_timeout_seconds,
    pool_size=settings.app_db_pool_size,
    max_overflow=settings.app_db_max_overflow,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

WORKBENCH_RUN_TABLE = "anomaly_workbench_runs"
WORKBENCH_RUN_COLUMN_DEFS: dict[str, str] = {
    "run_name": "VARCHAR(200) NOT NULL DEFAULT 'Ad hoc workbench run'",
    "source_tables_json": "JSON",
    "join_config_json": "JSON",
    "feature_rules_json": "JSON",
    "amount_field": "VARCHAR(150)",
    "total_rows": "INTEGER NOT NULL DEFAULT 0",
    "user_rule_count": "INTEGER NOT NULL DEFAULT 0",
    "ml_anomaly_count": "INTEGER NOT NULL DEFAULT 0",
    "final_anomaly_count": "INTEGER NOT NULL DEFAULT 0",
    "selected_model": "VARCHAR(100)",
    "metrics_json": "JSON",
    "status": "VARCHAR(30) NOT NULL DEFAULT 'COMPLETED'",
}


def get_db():
    """Dependency to get database session with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
    except Exception as exc:
        logger.error(f"Database session error: {exc}")
        db.rollback()
        raise
    finally:
        db.close()


def check_app_db_connection() -> dict:
    """
    Verify the application PostgreSQL database is reachable.
    Returns a dict with 'connected' bool and optional 'error' string.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"connected": True}
    except OperationalError as exc:
        message = str(exc).strip() or "Could not connect to the application PostgreSQL database."
        return {"connected": False, "error": message}


def _missing_workbench_run_column_ddls(existing_columns: set[str]) -> list[str]:
    statements: list[str] = []

    for column_name, column_sql in WORKBENCH_RUN_COLUMN_DEFS.items():
        if column_name in existing_columns:
            continue

        statements.append(
            f'ALTER TABLE "{WORKBENCH_RUN_TABLE}" '
            f'ADD COLUMN IF NOT EXISTS "{column_name}" {column_sql}'
        )

    return statements


def ensure_workbench_run_schema() -> None:
    """
    Backfill columns added after the table was first created.

    `create_all()` only creates missing tables; it does not alter existing ones.
    This keeps older app databases compatible with the current ORM model.
    """
    inspector = inspect(engine)

    if not inspector.has_table(WORKBENCH_RUN_TABLE):
        return

    existing_columns = {
        str(column["name"])
        for column in inspector.get_columns(WORKBENCH_RUN_TABLE)
    }
    statements = _missing_workbench_run_column_ddls(existing_columns)

    if not statements:
        return

    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))

    logger.info(
        "App DB schema updated for %s; added columns: %s",
        WORKBENCH_RUN_TABLE,
        ", ".join(
            column_name
            for column_name in WORKBENCH_RUN_COLUMN_DEFS
            if column_name not in existing_columns
        ),
    )
