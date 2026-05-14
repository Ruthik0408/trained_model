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
    active_model_path: str = str(MODEL_DIR / "active_model.joblib")
    random_state: int = 42
    ollama_base_url: str = str(os.getenv("TULIP_OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
    anomaly_reason_model: str = str(os.getenv("TULIP_ANOMALY_REASON_MODEL", "qwen3:4b")).strip()
    anomaly_reason_timeout_seconds: float = float(str(os.getenv("TULIP_ANOMALY_REASON_TIMEOUT_SECONDS", "60")).strip())
    workbench_explain_analyze: bool = str(os.getenv("TULIP_WORKBENCH_EXPLAIN_ANALYZE", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


settings = Settings()
