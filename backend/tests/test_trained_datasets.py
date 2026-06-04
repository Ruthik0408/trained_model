from app.schemas.workbench_schema import WorkbenchRunRequest
from app.core.errors import WorkbenchValidationError
from app.services.workbench.trained_datasets import (
    apply_trained_dataset_defaults,
    default_amount_field_for_tables,
    trained_selectable_tables,
    trained_join_configs,
    trained_dataset_tables,
)


def test_trained_dataset_defaults_expand_single_bill_selection() -> None:
    payload = WorkbenchRunRequest(
        selected_tables=["bill"],
        from_date="2025-01-01",
        to_date="2025-12-31",
    )

    effective_payload = apply_trained_dataset_defaults(payload)

    assert effective_payload.selected_tables == ["dak", "bill"]
    assert len(effective_payload.joins) == 1
    assert effective_payload.joins[0].left_table == "dak"
    assert effective_payload.joins[0].left_column == "id"
    assert effective_payload.joins[0].right_table == "bill"
    assert effective_payload.joins[0].right_column == "fk_dak"
    assert effective_payload.amount_field == "dak.amount"


def test_trained_selectable_tables_only_include_trained_sources() -> None:
    tables = trained_selectable_tables()

    assert tables == [
        "dak",
        "bill",
        "gem_bill",
        "dak_info",
        "civ_medical_bill",
        "civ_paybill",
        "civ_tada_ltc_bill",
        "echs_medical_bill",
        "cheque_slip",
        "schedule3",
        "ecs",
    ]


def test_trained_dataset_defaults_ignore_user_join_payload(caplog) -> None:
    payload = WorkbenchRunRequest(
        selected_tables=["dak", "gem_bill"],
        joins=[
            {
                "left_table": "dak",
                "left_column": "wrong_column",
                "right_table": "gem_bill",
                "right_column": "wrong_column",
                "join_type": "left",
            }
        ],
    )

    effective_payload = apply_trained_dataset_defaults(payload)

    assert effective_payload.selected_tables == ["dak", "gem_bill"]
    assert len(effective_payload.joins) == 1
    assert effective_payload.joins[0].left_column == "id"
    assert effective_payload.joins[0].right_column == "fk_dak"
    assert effective_payload.joins[0].join_type == "inner"
    assert "Ignoring caller-provided joins" in caplog.text


def test_trained_dataset_defaults_expand_dak_info_selection() -> None:
    payload = WorkbenchRunRequest(selected_tables=["dak_info"])

    effective_payload = apply_trained_dataset_defaults(payload)

    assert effective_payload.selected_tables == ["dak_info"]
    assert effective_payload.joins == []
    assert effective_payload.amount_field == "dak_info.amount"


def test_trained_dataset_defaults_keep_explicit_amount_field() -> None:
    payload = WorkbenchRunRequest(
        selected_tables=["dak"],
        amount_field="dak.custom_amount",
    )

    effective_payload = apply_trained_dataset_defaults(payload)

    assert effective_payload.amount_field == "dak.custom_amount"


def test_default_amount_field_for_tables_prefers_dak_then_dak_info() -> None:
    assert default_amount_field_for_tables(["dak"]) == "dak.amount"
    assert default_amount_field_for_tables(["dak", "bill"]) == "dak.amount"
    assert default_amount_field_for_tables(["dak_info"]) == "dak_info.amount"
    assert default_amount_field_for_tables(["bill"]) is None


def test_trained_dataset_tables_and_joins_follow_training_chain() -> None:
    assert trained_dataset_tables(["schedule3", "cheque_slip"]) == [
        "dak",
        "cheque_slip",
        "schedule3",
    ]

    joins = trained_join_configs(["schedule3", "cheque_slip"])

    assert [(join.left_table, join.right_table) for join in joins] == [
        ("dak", "cheque_slip"),
        ("cheque_slip", "schedule3"),
    ]
    assert [join.join_type for join in joins] == ["inner", "left"]


def test_trained_dataset_defaults_reject_ambiguous_table_mix() -> None:
    payload = WorkbenchRunRequest(selected_tables=["bill", "civ_paybill"])

    try:
        apply_trained_dataset_defaults(payload)
    except WorkbenchValidationError as exc:
        assert exc.error_code == "VALIDATION_ERROR"
    else:
        raise AssertionError("Expected ambiguous selected tables to be rejected.")


def test_trained_dataset_defaults_reject_dak_info_join_mix() -> None:
    payload = WorkbenchRunRequest(selected_tables=["dak", "dak_info"])

    try:
        apply_trained_dataset_defaults(payload)
    except WorkbenchValidationError as exc:
        assert exc.error_code == "VALIDATION_ERROR"
    else:
        raise AssertionError("Expected dak plus standalone dak_info to be rejected.")
