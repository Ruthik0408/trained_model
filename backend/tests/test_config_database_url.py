from app.core.config import _build_app_db_url, mask_database_url


def test_build_app_db_url_escapes_password_at_sign(monkeypatch) -> None:
    monkeypatch.delenv("TULIP_APP_DB_URL", raising=False)
    monkeypatch.setenv("TULIP_SOURCE_DB_HOST", "db.example")
    monkeypatch.setenv("TULIP_SOURCE_DB_PORT", "5432")
    monkeypatch.setenv("TULIP_SOURCE_DB_USER", "app_user")
    monkeypatch.setenv("TULIP_SOURCE_DB_PASSWORD", "pa@ss")
    monkeypatch.setenv("TULIP_APP_DB_NAME", "app_db")

    db_url = _build_app_db_url()

    assert "pa%40ss" in db_url
    assert mask_database_url(db_url) == "postgresql+psycopg2://app_user:***@db.example:5432/app_db"
