import logging
from sqlalchemy import create_engine, text
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
    pool_recycle=3600,        # Recycle connections every hour
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


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
