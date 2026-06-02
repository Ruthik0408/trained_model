from sqlalchemy import Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class WorkbenchRun(Base):
    """
    Stores metadata for every anomaly workbench execution.

    Written to the *application* PostgreSQL database (TULIP_APP_DB_NAME /
    TULIP_APP_DB_URL), **not** the source data database.
    """

    __tablename__ = "anomaly_workbench_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_name: Mapped[str] = mapped_column(String(200), default="Ad hoc workbench run")
    source_tables_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    join_config_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    feature_rules_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    amount_field: Mapped[str | None] = mapped_column(String(150), nullable=True)
    total_rows: Mapped[int] = mapped_column(Integer, default=0)
    user_rule_count: Mapped[int] = mapped_column(Integer, default=0)
    ml_anomaly_count: Mapped[int] = mapped_column(Integer, default=0)
    final_anomaly_count: Mapped[int] = mapped_column(Integer, default=0)
    selected_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="COMPLETED")
