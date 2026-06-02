import os
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd


def load_env_file(env_path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file."""
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ[key.strip()] = value


def find_env_file() -> Path:
    """Look for the project .env from common notebook/module locations."""
    module_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        module_dir / ".env",
        module_dir.parent / ".env",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find .env in the current directory or its parent.")


ENV_PATH = find_env_file()
load_env_file(ENV_PATH)

DB_CONFIG = {
    "host": os.environ.get("TULIP_SOURCE_DB_HOST"),
    "port": os.environ.get("TULIP_SOURCE_DB_PORT", "5432"),
    "dbname": os.environ.get("TULIP_SOURCE_DB_NAME"),
    "user": os.environ.get("TULIP_SOURCE_DB_USER"),
    "password": os.environ.get("TULIP_SOURCE_DB_PASSWORD"),
    "schema": os.environ.get("TULIP_SOURCE_DB_SCHEMA", "public"),
}

missing = [key for key, value in DB_CONFIG.items() if key != "schema" and not value]
if missing:
    raise ValueError(f"Missing required database settings: {missing}")


def query_postgres(sql: str) -> pd.DataFrame:
    """Run a SQL query through psql and return the result as a dataframe."""
    cleaned_sql = sql.strip().rstrip(";")
    psql_sql = (
        f"SET search_path TO {DB_CONFIG['schema']}; "
        f"COPY ({cleaned_sql}) TO STDOUT WITH CSV HEADER"
    )

    env = os.environ.copy()
    env["PGPASSWORD"] = DB_CONFIG["password"]

    result = subprocess.run(
        [
            "psql",
            "-h",
            DB_CONFIG["host"],
            "-p",
            str(DB_CONFIG["port"]),
            "-U",
            DB_CONFIG["user"],
            "-d",
            DB_CONFIG["dbname"],
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            psql_sql,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    stdout = result.stdout
    lines = stdout.splitlines()

    while lines and (not lines[0].strip() or lines[0].strip() == "SET"):
        lines = lines[1:]

    df = pd.read_csv(StringIO("\n".join(lines)), low_memory=False)
    if "SET" in df.columns:
        df = df.drop(columns=["SET"])

    return df


def get_connection_check_df() -> pd.DataFrame:
    return query_postgres(
        """
        SELECT
            current_database() AS database_name,
            current_schema() AS schema_name,
            current_user AS database_user,
            now() AS connected_at
        """
    )
