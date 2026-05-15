import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BASE_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.rstrip("\r\n")
    if not value:
        return value
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value.strip()


def load_local_env() -> None:
    for env_path in (PROJECT_DIR / ".env", BASE_DIR / ".env", WORKSPACE_DIR / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = _parse_env_value(value)
            if key and key not in os.environ:
                os.environ[key] = value


load_local_env()


def _build_app_db_url() -> str:
    """
    Build the PostgreSQL connection URL for the application database
    (stores WorkbenchRun metadata, feedback, etc.).

    Priority:
      1. TULIP_APP_DB_URL  – full DSN, used as-is
      2. Falls back to constructing a DSN from individual source-DB env vars,
         but targeting a separate database named by TULIP_APP_DB_NAME
         (defaults to 'tulip_anomaly').
    """
    explicit = os.getenv("TULIP_APP_DB_URL", "").strip()
    if explicit:
        if not explicit.lower().startswith(("postgresql://", "postgresql+psycopg2://")):
            raise ValueError(
                "TULIP_APP_DB_URL must be a PostgreSQL DSN. "
                "Set it to a postgresql+psycopg2:// URL or leave it unset to "
                "auto-derive it from the TULIP_SOURCE_DB_* variables."
            )
        return explicit

    host = str(os.getenv("TULIP_SOURCE_DB_HOST", "localhost")).strip()
    port = str(os.getenv("TULIP_SOURCE_DB_PORT", "5432")).strip()
    user = str(os.getenv("TULIP_SOURCE_DB_USER", "postgres")).strip()
    password = str(os.getenv("TULIP_SOURCE_DB_PASSWORD", "")).strip()
    # Use a dedicated app DB; override with TULIP_APP_DB_NAME if needed
    app_db_name = str(os.getenv("TULIP_APP_DB_NAME", "tulip_anomaly")).strip()

    if password:
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{app_db_name}"
    return f"postgresql+psycopg2://{user}@{host}:{port}/{app_db_name}"


class Settings(BaseModel):
    app_name: str = "Tulip 2.0 Anomaly System"
    db_url: str = _build_app_db_url()
    log_json: bool = str(os.getenv("TULIP_LOG_JSON", "true")).strip().lower() in {"1", "true", "yes", "on"}
    source_db_host: str = str(os.getenv("TULIP_SOURCE_DB_HOST", "localhost")).strip()
    source_db_port: int = int(str(os.getenv("TULIP_SOURCE_DB_PORT", "5432")).strip())
    source_db_name: str = str(os.getenv("TULIP_SOURCE_DB_NAME", "tulip 2"))
    source_db_user: str = str(os.getenv("TULIP_SOURCE_DB_USER", "postgres")).strip()
    source_db_password: str = str(os.getenv("TULIP_SOURCE_DB_PASSWORD", ""))
    source_db_schema: str = str(os.getenv("TULIP_SOURCE_DB_SCHEMA", "public")).strip()
    auto_create_source_indexes: bool = str(os.getenv("TULIP_AUTO_CREATE_SOURCE_INDEXES", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    app_db_pool_size: int = int(str(os.getenv("TULIP_APP_DB_POOL_SIZE", "5")).strip())
    app_db_max_overflow: int = int(str(os.getenv("TULIP_APP_DB_MAX_OVERFLOW", "10")).strip())
    app_db_pool_timeout_seconds: int = int(str(os.getenv("TULIP_APP_DB_POOL_TIMEOUT_SECONDS", "30")).strip())
    app_db_pool_recycle_seconds: int = int(str(os.getenv("TULIP_APP_DB_POOL_RECYCLE_SECONDS", "3600")).strip())
    source_db_pool_size: int = int(str(os.getenv("TULIP_SOURCE_DB_POOL_SIZE", "10")).strip())
    source_db_max_overflow: int = int(str(os.getenv("TULIP_SOURCE_DB_MAX_OVERFLOW", "20")).strip())
    source_db_pool_timeout_seconds: int = int(str(os.getenv("TULIP_SOURCE_DB_POOL_TIMEOUT_SECONDS", "30")).strip())
    source_db_pool_recycle_seconds: int = int(str(os.getenv("TULIP_SOURCE_DB_POOL_RECYCLE_SECONDS", "3600")).strip())
    active_model_path: str = str(MODEL_DIR / "active_model.joblib")
    random_state: int = 42
    ollama_base_url: str = str(os.getenv("TULIP_OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
    anomaly_reason_model: str = str(os.getenv("TULIP_ANOMALY_REASON_MODEL", "qwen3:4b")).strip()
    anomaly_reason_timeout_seconds: float = float(str(os.getenv("TULIP_ANOMALY_REASON_TIMEOUT_SECONDS", "60")).strip())
    anomaly_reason_timeout_min_seconds: float = float(str(os.getenv("TULIP_ANOMALY_REASON_TIMEOUT_MIN_SECONDS", "15")).strip())
    anomaly_reason_timeout_per_1k_chars_seconds: float = float(
        str(os.getenv("TULIP_ANOMALY_REASON_TIMEOUT_PER_1K_CHARS_SECONDS", "6")).strip()
    )
    anomaly_reason_timeout_load_penalty_seconds: float = float(
        str(os.getenv("TULIP_ANOMALY_REASON_TIMEOUT_LOAD_PENALTY_SECONDS", "5")).strip()
    )
    anomaly_reason_retry_attempts: int = int(str(os.getenv("TULIP_ANOMALY_REASON_RETRY_ATTEMPTS", "2")).strip())
    anomaly_reason_retry_backoff_seconds: float = float(str(os.getenv("TULIP_ANOMALY_REASON_RETRY_BACKOFF_SECONDS", "1.5")).strip())
    anomaly_reason_cache_ttl_seconds: float = float(str(os.getenv("TULIP_ANOMALY_REASON_CACHE_TTL_SECONDS", "900")).strip())
    anomaly_reason_circuit_fail_threshold: int = int(str(os.getenv("TULIP_ANOMALY_REASON_CIRCUIT_FAIL_THRESHOLD", "3")).strip())
    anomaly_reason_circuit_reset_seconds: float = float(str(os.getenv("TULIP_ANOMALY_REASON_CIRCUIT_RESET_SECONDS", "45")).strip())
    rate_limit_enabled: bool = str(os.getenv("TULIP_RATE_LIMIT_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
    rate_limit_requests: int = int(str(os.getenv("TULIP_RATE_LIMIT_REQUESTS", "120")).strip())
    rate_limit_window_seconds: int = int(str(os.getenv("TULIP_RATE_LIMIT_WINDOW_SECONDS", "60")).strip())
    anomaly_feature_max_columns: int = int(str(os.getenv("TULIP_ANOMALY_FEATURE_MAX_COLUMNS", "40")).strip())
    anomaly_feature_min_score: float = float(str(os.getenv("TULIP_ANOMALY_FEATURE_MIN_SCORE", "0.15")).strip())
    workbench_explain_analyze: bool = str(os.getenv("TULIP_WORKBENCH_EXPLAIN_ANALYZE", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


settings = Settings()
