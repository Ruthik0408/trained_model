from typing import Annotated, Literal

from pydantic import BaseModel, Field

class JoinConfig(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    join_type: str = Field(default="inner", pattern=r"^(inner|left|right|outer)$")

class OutlierRuleInput(BaseModel):
    name: str
    first_column: str
    second_column: str | None = None
    operator: str = Field(
        pattern=r"^(=|!=|>|>=|<|<=|null|not null)$"
    )
    value: str | None = None
    second_value: str | None = None

class FeatureRuleInput(BaseModel):
    name: str
    feature_type: str = Field(
        pattern=(
            r"^(numeric|difference|ratio|sum|comparisonflag|missingflag|daysbetween|"
            r"isweekend|isbusinesshour)$"
        )
    )
    first_column: str
    second_column: str | None = None
    operator: str | None = Field(
        default=None,
        pattern=r"^(=|!=|>|>=|<|<=|null|not null)$",
    )

class WorkbenchRunRequest(BaseModel):
    run_name: str = "Ad hoc anomaly workbench run"
    source_database: str | None = None
    selected_tables: list[str] = Field(min_length=1, max_length=3)
    joins: list[JoinConfig] = Field(default_factory=list)
    amount_field: str | None = None
    outlier_rules: list[OutlierRuleInput] = Field(default_factory=list)
    feature_rules: list[FeatureRuleInput] = Field(default_factory=list)
    contamination: Literal["auto"] | Annotated[float, Field(gt=0.0, lt=1.0)] = "auto"
    from_date: str | None = None
    to_date: str | None = None


class BuiltinRuleRequest(BaseModel):
    source_database: str | None = None
    selected_tables: list[str] = Field(default_factory=list)
    joins: list[JoinConfig] = Field(default_factory=list)
    from_date: str | None = None
    to_date: str | None = None

class ColumnInfo(BaseModel):
    table_name: str
    column_name: str
    data_type: str
class TableInfo(BaseModel):
    table_name: str
    columns: list[ColumnInfo]
class WorkbenchRunResponse(BaseModel):
    run_id: int
    run_name: str
    total_rows: int
    human_outlier_count: int
    ml_anomaly_count: int
    final_anomaly_count: int
    amount_total: float
    selected_model: str
    metrics: dict
class ReviewTableRow(BaseModel):
    serial_no: int
    prediction_id: int
    anomaly: str
    reason: str | None = None
    amount: float
    total_amount: float
    review_status: str | None = None
    feedback: str | None = None
    bill_no: str | None = None
    vendor_name: str | None = None
    office_name: str | None = None
    detected_on: str | None = None
    anomaly_type: str | None = None
    risk_score: float | None = None
class ReviewTableResponse(BaseModel):
    rows: list[ReviewTableRow]
    total_amount: float
    total_rows: int
    dataset_table: str | None = None
    run_id: int | None = None
class LatestRunReport(BaseModel):
    run_id: int | None = None
    run_name: str | None = None
    selected_tables: list[str] = Field(default_factory=list)
    total_rows: int | None = None
    human_outlier_count: int | None = None
    ml_anomaly_count: int | None = None
    final_anomaly_count: int | None = None
    selected_model: str | None = None
    amount_field: str | None = None
    dataset_table: str | None = None
class FeedbackSummary(BaseModel):
    accept: int = 0
    reject: int = 0
    maybe: int = 0
    total: int = 0
class ReportResponse(BaseModel):
    run_id: int | None = None
    dataset_table: str | None = None
    run_name: str | None = None
    selected_tables: list[str] = Field(default_factory=list)
    total_rows: int = 0
    anomaly_count: int = 0
    reviewed_count: int = 0
    pending_count: int = 0
    accepted_count: int = 0
    amount: float = 0.0
    from_date: str | None = None
    to_date: str | None = None
    selected_model: str | None = None
class DatasetFeedbackRequest(BaseModel):
    dataset_table: str
    record_id: int
    feedback: str = Field(pattern=r"^(accept|reject|maybe|ACCEPT|REJECT|MAYBE)$")


class IsolationReasonRequest(BaseModel):
    prediction_id: int | None = None
    review_key: str | None = None
    if_score: float | None = None
    ml_threshold: float | None = None
    rule_anomaly: bool | None = None
    rule_count: int | None = None
    existing_reasons: list[str] = Field(default_factory=list)
    feature_signals: list[dict] = Field(default_factory=list)
    row_payload: dict = Field(default_factory=dict)


class IsolationReasonResponse(BaseModel):
    reason: str
    model: str
    fallback: bool = False
