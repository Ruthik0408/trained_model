"""Shared date-stage metadata for training and runtime anomaly checks."""

from __future__ import annotations


DATE_SEQUENCE_STAGE_ALIASES = [
    ("invoice_date", ["invoice_date"]),
    ("bill_date", ["bill_date"]),
    ("reference_date", ["reference_date"]),
    ("auditor_stage", ["auditor_date", "aud_date", "auditor_disposal_date"]),
    ("aao_stage", ["aao_date", "aao_disposal_date"]),
    ("ao_stage", ["ao_date", "ao_disposal_date"]),
    ("go_date", ["go_date"]),
    ("dp_sheet_date", ["dp_sheet_date"]),
    ("cmp_date", ["cmp_date", "cmp_batch_date", "cmp_file_gen_date"]),
    ("disposal_date", ["disposal_date"]),
]

DATE_SEQUENCE_STAGE_ALIAS_MAP = {
    stage_name: list(aliases)
    for stage_name, aliases in DATE_SEQUENCE_STAGE_ALIASES
}

SAME_TABLE_DATE_SEQUENCE_STAGES = {
    "auditor_stage",
    "aao_stage",
    "ao_stage",
    "go_date",
    "disposal_date",
}

GLOBAL_SEQUENCE_STAGE_ORDER = [
    "invoice_date",
    "bill_date",
    "reference_date",
    "dp_sheet_date",
    "cmp_date",
    "disposal_date",
]

TABLE_SEQUENCE_STAGE_ORDER = [
    "invoice_date",
    "bill_date",
    "reference_date",
    "auditor_stage",
    "aao_stage",
    "ao_stage",
    "go_date",
    "dp_sheet_date",
    "cmp_date",
    "disposal_date",
]


def get_date_stage_aliases(stage_name: str) -> list[str]:
    """Return the configured aliases for one logical date stage."""
    return DATE_SEQUENCE_STAGE_ALIAS_MAP[str(stage_name)]
