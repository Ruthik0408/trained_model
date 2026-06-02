import sys
import types

import pandas as pd

redis_stub = types.ModuleType("redis")
redis_stub.Redis = object
redis_exceptions_stub = types.ModuleType("redis.exceptions")
redis_exceptions_stub.RedisError = Exception
sys.modules.setdefault("redis", redis_stub)
sys.modules.setdefault("redis.exceptions", redis_exceptions_stub)

from app.services.workbench.utils import _select_series_column


def test_select_series_column_uses_first_duplicate_match() -> None:
    frame = pd.DataFrame(
        [[100, 900], [250, 800]],
        columns=["bills.amount", "bills.amount"],
    )

    selected = _select_series_column(frame, "bills.amount")

    assert isinstance(selected, pd.Series)
    assert selected.tolist() == [100, 250]
