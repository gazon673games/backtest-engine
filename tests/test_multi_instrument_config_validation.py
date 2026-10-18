from __future__ import annotations

import pytest
from pydantic import ValidationError

from backtest_engine.portfolio.multi.config import MultiInstrumentExecutionConfig


@pytest.mark.parametrize(
    ("field", "value"),
    [("engine", "invalid"), ("rebalance_policy", "invalid")],
)
def test_multi_execution_rejects_invalid_choices(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match=field.replace("_", " ")):
        MultiInstrumentExecutionConfig(**{field: value})


def test_multi_execution_normalizes_choices() -> None:
    config = MultiInstrumentExecutionConfig(engine="V3", rebalance_policy="ALL_OR_NONE")

    assert config.engine == "v3"
    assert config.rebalance_policy == "all_or_none"
