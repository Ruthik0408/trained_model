from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import WorkbenchValidationError
from app.schemas.workbench_schema import JoinConfig


BASE_TABLE = "dak"


@dataclass(frozen=True)
class TrainedJoinSpec:
    right_table: str
    left_table: str
    left_column: str
    right_column: str
    join_type: str = "inner"


TRAINED_JOIN_SPECS: dict[str, TrainedJoinSpec] = {
    "bill": TrainedJoinSpec("bill", "dak", "id", "fk_dak"),
    "gem_bill": TrainedJoinSpec("gem_bill", "dak", "id", "fk_dak"),
    "civ_medical_bill": TrainedJoinSpec("civ_medical_bill", "dak", "id", "fk_dak"),
    "civ_paybill": TrainedJoinSpec("civ_paybill", "dak", "id", "fk_dak"),
    "civ_tada_ltc_bill": TrainedJoinSpec("civ_tada_ltc_bill", "dak", "id", "fk_dak"),
    "echs_medical_bill": TrainedJoinSpec("echs_medical_bill", "dak", "id", "fk_dak"),
    "cheque_slip": TrainedJoinSpec("cheque_slip", "dak", "id", "fk_dak"),
    "schedule3": TrainedJoinSpec("schedule3", "cheque_slip", "fk_dak", "fk_dak"),
    "ecs": TrainedJoinSpec("ecs", "cheque_slip", "fk_dak", "fk_dak"),
}

TRAINED_DATASET_TABLES: dict[str, list[str]] = {
    "dak": [],
    "dak.bill": ["bill"],
    "dak.gem_bill": ["gem_bill"],
    "dak_info": [],
    "dak.civ_medical_bill": ["civ_medical_bill"],
    "dak.civ_paybill": ["civ_paybill"],
    "dak.civ_tada_ltc_bill": ["civ_tada_ltc_bill"],
    "dak.echs_medical_bill": ["echs_medical_bill"],
    "dak.cheque_slip.schedule3": ["cheque_slip", "schedule3"],
    "dak.cheque_slip.ecs": ["cheque_slip", "ecs"],
}

TRAINED_DATASET_BASE_TABLES: dict[str, str] = {
    "dak_info": "dak_info",
}


def trained_dataset_base_table(dataset_name: str) -> str:
    return TRAINED_DATASET_BASE_TABLES.get(dataset_name, BASE_TABLE)


def trained_selectable_tables() -> list[str]:
    seen: set[str] = set()
    table_names: list[str] = []
    for dataset_name, tables in TRAINED_DATASET_TABLES.items():
        for table_name in [trained_dataset_base_table(dataset_name), *tables]:
            if table_name in seen:
                continue
            seen.add(table_name)
            table_names.append(table_name)
    return table_names


def resolve_dataset_name(selected_tables: list[str]) -> str:
    normalized = {str(table).strip().lower() for table in selected_tables}
    trained_sets = {
        "dak": {"dak"},
        "dak_info": {"dak_info"},
        "dak.bill": {"bill"},
        "dak.gem_bill": {"gem_bill"},
        "dak.civ_medical_bill": {"civ_medical_bill"},
        "dak.civ_paybill": {"civ_paybill"},
        "dak.civ_tada_ltc_bill": {"civ_tada_ltc_bill"},
        "dak.echs_medical_bill": {"echs_medical_bill"},
        "dak.cheque_slip.schedule3": {"cheque_slip", "schedule3"},
        "dak.cheque_slip.ecs": {"cheque_slip", "ecs"},
    }
    without_base = normalized - {BASE_TABLE}

    for dataset_name, required_tables in trained_sets.items():
        base_table = trained_dataset_base_table(dataset_name)
        if normalized == required_tables:
            return dataset_name
        if base_table == BASE_TABLE and without_base == required_tables:
            return dataset_name

    raise WorkbenchValidationError(
        "No saved trained model is available for the selected tables.",
        suggestion=(
            "Select dak, dak_info, dak + bill, dak + gem_bill, dak + civ_medical_bill, "
            "dak + civ_paybill, dak + civ_tada_ltc_bill, or dak + echs_medical_bill "
            "to use the trained anomaly pipeline."
        ),
        details={"selected_tables": sorted(normalized)},
    )


def trained_dataset_tables(selected_tables: list[str]) -> list[str]:
    dataset_name = resolve_dataset_name(selected_tables)
    return [trained_dataset_base_table(dataset_name), *TRAINED_DATASET_TABLES[dataset_name]]


def trained_join_configs(selected_tables: list[str]) -> list[JoinConfig]:
    dataset_name = resolve_dataset_name(selected_tables)
    join_tables = TRAINED_DATASET_TABLES[dataset_name]
    configs: list[JoinConfig] = []
    for table_name in join_tables:
        spec = TRAINED_JOIN_SPECS[table_name]
        configs.append(
            JoinConfig(
                left_table=spec.left_table,
                left_column=spec.left_column,
                right_table=spec.right_table,
                right_column=spec.right_column,
                join_type=spec.join_type,
            )
        )
    return configs


def apply_trained_dataset_defaults(payload):
    return payload.model_copy(
        update={
            "selected_tables": trained_dataset_tables(payload.selected_tables),
            "joins": trained_join_configs(payload.selected_tables),
        }
    )
