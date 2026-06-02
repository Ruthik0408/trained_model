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

    host = str(os.getenv("TULIP_SOURCE_DB_HOST", "localhost"))
    port = str(os.getenv("TULIP_SOURCE_DB_PORT", "5432"))
    user = str(os.getenv("TULIP_SOURCE_DB_USER", "postgres"))
    password = str(os.getenv("TULIP_SOURCE_DB_PASSWORD", ""))
    # Use a dedicated app DB; override with TULIP_APP_DB_NAME if needed
    app_db_name = str(os.getenv("TULIP_APP_DB_NAME", "tulip_anomaly"))

    if password:
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{app_db_name}"
    return f"postgresql+psycopg2://{user}@{host}:{port}/{app_db_name}"


def _trained_model_dir() -> str:
    explicit = os.getenv("TULIP_TRAINED_MODEL_DIR", "").strip()
    if explicit:
        return explicit

    candidates = [
        BASE_DIR / "artifacts" / "models",
        PROJECT_DIR / "artifacts" / "models",
        WORKSPACE_DIR / "backend" / "artifacts" / "models",
        PROJECT_DIR.parent.parent / "backend" / "artifacts" / "models",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


class Settings(BaseModel):
    app_name: str = "Tulip 2.0 Anomaly System"
    db_url: str = _build_app_db_url()
    log_json: bool = str(os.getenv("TULIP_LOG_JSON", "true")).lower() in {"1", "true", "yes", "on"}
    source_db_host: str = str(os.getenv("TULIP_SOURCE_DB_HOST", "localhost"))
    source_db_port: int = int(str(os.getenv("TULIP_SOURCE_DB_PORT", "5432")))
    source_db_name: str = str(os.getenv("TULIP_SOURCE_DB_NAME", "tulip 2"))
    source_db_user: str = str(os.getenv("TULIP_SOURCE_DB_USER", "postgres"))
    source_db_password: str = str(os.getenv("TULIP_SOURCE_DB_PASSWORD", ""))
    source_db_schema: str = str(os.getenv("TULIP_SOURCE_DB_SCHEMA", "public"))
    auto_create_source_indexes: bool = str(os.getenv("TULIP_AUTO_CREATE_SOURCE_INDEXES", "false")) in {
        "1",
        "true",
        "yes",
        "on",
    }
    app_db_pool_size: int = int(str(os.getenv("TULIP_APP_DB_POOL_SIZE", "5")))
    app_db_max_overflow: int = int(str(os.getenv("TULIP_APP_DB_MAX_OVERFLOW", "10")))
    app_db_pool_timeout_seconds: int = int(str(os.getenv("TULIP_APP_DB_POOL_TIMEOUT_SECONDS", "30")))
    app_db_pool_recycle_seconds: int = int(str(os.getenv("TULIP_APP_DB_POOL_RECYCLE_SECONDS", "3600")))
    source_db_pool_size: int = int(str(os.getenv("TULIP_SOURCE_DB_POOL_SIZE", "10")))
    source_db_max_overflow: int = int(str(os.getenv("TULIP_SOURCE_DB_MAX_OVERFLOW", "20")))
    source_db_pool_timeout_seconds: int = int(str(os.getenv("TULIP_SOURCE_DB_POOL_TIMEOUT_SECONDS", "30")))
    source_db_pool_recycle_seconds: int = int(str(os.getenv("TULIP_SOURCE_DB_POOL_RECYCLE_SECONDS", "3600")))
    active_model_path: str = str(MODEL_DIR / "active_model.joblib")
    trained_model_dir: str = _trained_model_dir()
    random_state: int = 42
    ollama_base_url: str = str(os.getenv("TULIP_OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
    anomaly_reason_model: str = str(os.getenv("TULIP_ANOMALY_REASON_MODEL", "qwen3:4b")).strip()
    anomaly_reason_keep_alive: str = str(os.getenv("TULIP_ANOMALY_REASON_KEEP_ALIVE", "30m")).strip()
    anomaly_reason_num_predict: int = int(str(os.getenv("TULIP_ANOMALY_REASON_NUM_PREDICT", "120")))
    anomaly_reason_max_signals: int = int(str(os.getenv("TULIP_ANOMALY_REASON_MAX_SIGNALS", "5")))
    anomaly_reason_max_row_facts: int = int(str(os.getenv("TULIP_ANOMALY_REASON_MAX_ROW_FACTS", "12")))
    anomaly_reason_timeout_seconds: float = float(str(os.getenv("TULIP_ANOMALY_REASON_TIMEOUT_SECONDS", "60")))
    anomaly_reason_timeout_min_seconds: float = float(str(os.getenv("TULIP_ANOMALY_REASON_TIMEOUT_MIN_SECONDS", "15")))
    anomaly_reason_timeout_per_1k_chars_seconds: float = float(
        str(os.getenv("TULIP_ANOMALY_REASON_TIMEOUT_PER_1K_CHARS_SECONDS", "6"))
    )
    anomaly_reason_timeout_load_penalty_seconds: float = float(
        str(os.getenv("TULIP_ANOMALY_REASON_TIMEOUT_LOAD_PENALTY_SECONDS", "5"))
    )
    anomaly_reason_retry_attempts: int = int(str(os.getenv("TULIP_ANOMALY_REASON_RETRY_ATTEMPTS", "2")))
    anomaly_reason_retry_backoff_seconds: float = float(str(os.getenv("TULIP_ANOMALY_REASON_RETRY_BACKOFF_SECONDS", "1.5")))
    anomaly_reason_cache_ttl_seconds: float = float(str(os.getenv("TULIP_ANOMALY_REASON_CACHE_TTL_SECONDS", "900")))
    anomaly_reason_circuit_fail_threshold: int = int(str(os.getenv("TULIP_ANOMALY_REASON_CIRCUIT_FAIL_THRESHOLD", "3")))
    anomaly_reason_circuit_reset_seconds: float = float(str(os.getenv("TULIP_ANOMALY_REASON_CIRCUIT_RESET_SECONDS", "45")))
    rate_limit_enabled: bool = str(os.getenv("TULIP_RATE_LIMIT_ENABLED", "true")) in {"1", "true", "yes", "on"}
    rate_limit_requests: int = int(str(os.getenv("TULIP_RATE_LIMIT_REQUESTS", "120")))
    rate_limit_window_seconds: int = int(str(os.getenv("TULIP_RATE_LIMIT_WINDOW_SECONDS", "60")))
    valkey_host: str = str(os.getenv("VALKEY_HOST", "localhost"))
    valkey_port: int = int(str(os.getenv("VALKEY_PORT", "6379")))
    valkey_db: int = int(str(os.getenv("VALKEY_DB", "0")))
    valkey_password: str = str(os.getenv("VALKEY_PASSWORD", ""))
    valkey_enabled: bool = str(os.getenv("TULIP_VALKEY_ENABLED", "true")).lower() in {"1", "true", "yes", "on"}
    valkey_key_prefix: str = str(os.getenv("TULIP_VALKEY_KEY_PREFIX", "tulip"))
    valkey_socket_timeout_seconds: float = float(str(os.getenv("TULIP_VALKEY_SOCKET_TIMEOUT_SECONDS", "1.5")))
    anomaly_feature_min_score: float = float(str(os.getenv("TULIP_ANOMALY_FEATURE_MIN_SCORE", "0.15")))
    anomaly_feature_min_present_ratio: float = float(str(os.getenv("TULIP_ANOMALY_FEATURE_MIN_PRESENT_RATIO", "0.5")))
    workbench_explain_analyze: bool = str(os.getenv("TULIP_WORKBENCH_EXPLAIN_ANALYZE", "false")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


settings = Settings()
