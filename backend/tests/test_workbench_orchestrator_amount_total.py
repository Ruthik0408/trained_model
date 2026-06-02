import pandas as pd

from app.schemas.workbench_schema import WorkbenchRunRequest
from app.services.workbench.orchestrator import _calculate_amount_total


def test_calculate_amount_total_uses_first_duplicate_amount_column() -> None:
    payload = WorkbenchRunRequest(
        selected_tables=["bills"],
        amount_field="bills.amount",
    )
    filtered_joined = pd.DataFrame(
        [[100, 900], [250, 800]],
        columns=["bills.amount", "bills.amount"],
    )

    amount_total = _calculate_amount_total(payload, filtered_joined)

    assert amount_total == 350.0
