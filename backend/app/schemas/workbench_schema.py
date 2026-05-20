from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

class JoinConfig(BaseModel):
    """Describes how two source tables should be joined for a workbench run."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "left_table": "bills",
                "left_column": "vendor_id",
                "right_table": "vendors",
                "right_column": "vendor_id",
                "join_type": "left",
            }
        }
    )

    left_table: str
    left_column: str
    right_table: str
    right_column: str
    join_type: str = Field(default="inner", pattern=r"^(inner|left|right|outer)$")

class OutlierRuleInput(BaseModel):
    """Represents a SQL-style rule that marks a row as human-defined anomalous."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Large amount threshold",
                "first_column": "bills.amount",
                "operator": ">",
                "value": "100000",
            }
        }
    )

    name: str
    first_column: str
    second_column: str | None = None
    operator: str = Field(
        pattern=r"^(=|!=|>|>=|<|<=|null|not null)$"
    )
    value: str | None = None
    second_value: str | None = None

class FeatureRuleInput(BaseModel):
    """Defines a derived feature used by the Isolation Forest pipeline."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "Amount difference",
                    "feature_type": "difference",
                    "first_column": "bills.claimed_amount",
                    "second_column": "bills.approved_amount",
                },
                {
                    "name": "Weekend posting",
                    "feature_type": "isweekend",
                    "first_column": "bills.bill_date",
                },
            ]
        }
    )

    name: str
    feature_type: str = Field(
        pattern=(
            r"^(numeric|difference|ratio|sum|missingflag|daysbetween|"
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
    """Payload for previewing or executing an anomaly workbench run.

    Edge cases:
    - `selected_tables` must contain between 1 and 3 tables.
    - `contamination` may be `"auto"` or a float strictly between 0 and 1.
    - feature rules that produce only missing values will now fail validation at runtime.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_name": "Vendor amount anomaly scan",
                "selected_tables": ["bills", "vendors"],
                "joins": [
                    {
                        "left_table": "bills",
                        "left_column": "vendor_id",
                        "right_table": "vendors",
                        "right_column": "vendor_id",
                        "join_type": "left",
                    }
                ],
                "amount_field": "bills.amount",
                "outlier_rules": [
                    {
                        "name": "Large amount threshold",
                        "first_column": "bills.amount",
                        "operator": ">",
                        "value": "100000",
                    }
                ],
                "feature_rules": [
                    {
                        "name": "Approval lag",
                        "feature_type": "daysbetween",
                        "first_column": "bills.bill_date",
                        "second_column": "bills.approval_date",
                    }
                ],
                "contamination": 0.05,
                "from_date": "2025-01-01",
                "to_date": "2025-12-31",
            }
        }
    )

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
    """Payload for suggesting built-in feature rules from selected tables."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "selected_tables": ["bills", "vendors"],
                "joins": [
                    {
                        "left_table": "bills",
                        "left_column": "vendor_id",
                        "right_table": "vendors",
                        "right_column": "vendor_id",
                        "join_type": "left",
                    }
                ],
                "from_date": "2025-01-01",
                "to_date": "2025-12-31",
            }
        }
    )

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
    """Summary returned after a successful workbench execution."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": 42,
                "run_name": "Vendor amount anomaly scan",
                "total_rows": 1280,
                "human_outlier_count": 12,
                "ml_anomaly_count": 64,
                "final_anomaly_count": 70,
                "amount_total": 845230.75,
                "selected_model": "IsolationForest",
                "metrics": {
                    "dataset_table": "ML_Features",
                    "feature_count": 7,
                    "warnings": [],
                },
            }
        }
    )

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
    """Review feedback attached to a persisted anomaly record."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dataset_table": "ML_Features",
                "record_id": 101,
                "feedback": "accept",
            }
        }
    )

    dataset_table: str
    record_id: int
    feedback: str = Field(pattern=r"^(accept|reject|maybe|ACCEPT|REJECT|MAYBE)$")


class IsolationReasonRequest(BaseModel):
    """Inputs required to generate a natural-language explanation for an IF anomaly."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_id": 101,
                "if_score": 0.83,
                "ml_threshold": 0.61,
                "rule_anomaly": True,
                "rule_count": 1,
                "existing_reasons": ["Amount exceeds expected range"],
                "feature_signals": [{"feature": "amount_delta", "impact": 0.42}],
                "row_payload": {"vendor_name": "ABC Supplies", "amount": 120000},
            }
        }
    )

    prediction_id: int | None = None
    dataset_table: str | None = None
    review_key: str | None = None
    if_score: float | None = None
    ml_threshold: float | None = None
    rule_anomaly: bool | None = None
    rule_count: int | None = None
    existing_reasons: list[str] = Field(default_factory=list)
    feature_signals: list[dict] = Field(default_factory=list)
    row_payload: dict = Field(default_factory=dict)
