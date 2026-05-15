import logging
from typing import Any

logger = logging.getLogger(__name__)

PREVIEW_ROW_LIMIT = 50
ENABLE_EXPENSIVE_JOIN_DEBUG = False
DATE_SEQUENCE_STAGE_ALIASES = [
    ("invoice_date"),
    ("bill_date"),
    ("reference_date"),
    ("list_date"),
    ("auditor_stage", ["auditor_date", "aud_date", "auditor_disposal_date"]),
    ("aao_stage", ["aao_date", "aao_disposal_date"]),
    ("ao_stage", ["ao_date", "ao_disposal_date"]),
    ("go_date"),
    ("dp_sheet_date"),
    ("cmp_date", ["cmp_date", "cmp_batch_date","cmp_file_gen_date"]),
    ("disposal_date"),
]
SAME_TABLE_DATE_SEQUENCE_STAGES = {
    "auditor_stage",
    "aao_stage",
    "ao_stage",
    "go_date",
    "disposal_date",
}
SINGLE_FEATURE_TYPES = [
    "isweekend",
    "isbusinesshour",
]
MIN_FEATURE_COLUMN_PRESENT_RATIO = 0.70
DATE_FILTER_COLUMN_PRIORITY = [
    "list_date",
    "created_at",
]
SYSTEM_COLUMN_PREFIX = "ml_"
RESULT_SCHEMA = "public"
RESULT_TABLE = "ML_Features"
ML_FEATURES_TABLE = RESULT_TABLE
SERIAL_COLUMN = "id"

BUILTIN_FEATURE_RULES_CACHE_TTL = 60.0
_builtin_feature_rules_cache: dict[tuple[Any, ...], tuple[float, list[dict]]] = {}
FEATURE_NAME_COLUMN = "feature_name"
HUMAN_RULE_NAME_COLUMN = "human_rule_name"
HUMAN_RULE_COLUMN = "human_rule"
ISOLATION_RULE_COLUMN = "isolation_rule"
IF_SCORE_COLUMN = "ml_if_score"
ML_THRESHOLD_COLUMN = "ml_threshold"
FEEDBACK_SCORE_COLUMN = "feedback_score"
RUN_ID_COLUMN = "ml_run_id"
REVIEW_PAYLOAD_COLUMN = "review_payload_json"
FEATURE_VALUES_COLUMN = "feature_values_json"
SYSTEM_COLUMNS = {
    SERIAL_COLUMN,
    FEATURE_NAME_COLUMN,
    HUMAN_RULE_NAME_COLUMN,
    HUMAN_RULE_COLUMN,
    ISOLATION_RULE_COLUMN,
    IF_SCORE_COLUMN,
    ML_THRESHOLD_COLUMN,
    FEEDBACK_SCORE_COLUMN,
    RUN_ID_COLUMN,
    REVIEW_PAYLOAD_COLUMN,
    FEATURE_VALUES_COLUMN,
}
TEMP_ROW_ID_COLUMN = "__ml_row_number"
SQL_RULE_FLAG_COLUMN = "sql_rule_flag"
SQL_RULE_REASONS_COLUMN = "sql_rule_reasons"
USER_RULE_FLAG_COLUMN = "__ml_sql_rule_flag"
USER_RULE_REASONS_COLUMN = "__ml_sql_rule_reasons"

FEEDBACK_TO_SCORE = {
    "accept": 1.0,
    "reject": 0.0,
    "maybe": 0.5,
}
SCORE_TO_FEEDBACK = {v: k for k, v in FEEDBACK_TO_SCORE.items()}
