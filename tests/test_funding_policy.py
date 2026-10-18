"""Funding assumptions remain explicit and preserve existing zero-cost runs."""

import math
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.accounting.funding import funding_payment_for_row
from backtest_engine.accounting.funding_policy import FundingConfig
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.intraday.event.config import IntradayEventExecutionConfig
from backtest_engine.portfolio.multi.config import MultiInstrumentExecutionConfig
from backtest_engine.research.candidate_schedule_state import prices_from_frame
from backtest_engine.workbench.models import WorkbenchRunRequest


def payment(row, **overrides):
    arguments = dict(
        quantity=2.0,
        point_value=1.0,
        mark_price=200.0,
        rate_column="funding_rate",
        default_rate=0.0,
        event_column="funding_event",
        require_event_flag=True,
    )
    return funding_payment_for_row(row, **(arguments | overrides))


@pytest.mark.parametrize(
    "config_class", [IntradayEventExecutionConfig, MultiInstrumentExecutionConfig]
)
def test_existing_execution_settings_keep_legacy_zero_funding(config_class):
    config = config_class.model_validate({})
    assert config.funding_mode == "legacy"
    assert config.default_funding_rate == 0
    assert payment({}) == 0.0


def test_disabled_mode_ignores_funding_columns_and_does_not_need_an_archive():
    assert payment({}, mode="disabled") == 0
    assert payment({"funding_rate": 0.01, "funding_event": True}, mode="disabled") == 0
    assert payment({"funding_rate": math.nan}, mode="disabled") == 0


def test_legacy_unpriced_funding_keeps_its_declared_bar_price():
    assert payment({"funding_rate": 0.01, "funding_event": True}) == pytest.approx(4.0)
    assert payment({"funding_rate": -0.01, "funding_event": True}) == pytest.approx(-4.0)


def test_historical_mode_uses_event_price_instead_of_the_bar_close():
    row = {"funding_rate": 0.01, "funding_event": True, "funding_price_rate": 1.0}
    assert payment(row, mode="historical") == pytest.approx(2.0)
    assert payment(row, mode="historical", quantity=-2) == pytest.approx(-2.0)
    assert payment({"funding_event": False}, mode="historical") == 0


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"funding_event": True},
        {"funding_event": True, "funding_rate": 0.01},
        {
            "funding_event": True,
            "funding_rate": 0.01,
            "funding_price_rate": 1.0,
            "funding_price_estimated": True,
        },
        {"funding_event": True, "funding_rate": math.nan, "funding_price_rate": 1.0},
    ],
)
def test_historical_mode_rejects_missing_or_unobserved_inputs(row):
    with pytest.raises(DatasetValidationError):
        payment(row, mode="historical")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "bad"])
def test_nonfinite_funding_cannot_corrupt_cash(value):
    with pytest.raises(DatasetValidationError, match="finite"):
        payment({"funding_rate": value, "funding_event": True})


@pytest.mark.parametrize(
    "overrides",
    [
        {"default_funding_rate": 0.001},
        {"funding_rate_column": "custom"},
        {"require_funding_event_flag": False},
    ],
)
def test_historical_policy_cannot_silently_enable_legacy_fallbacks(overrides):
    with pytest.raises(ValueError):
        FundingConfig(funding_mode="historical", **overrides)


def test_workbench_request_preserves_an_explicit_funding_mode():
    request = WorkbenchRunRequest(
        dataset_path=Path("synthetic.parquet"),
        script_source="// synthetic",
        funding_mode="disabled",
    )
    assert request.model_dump(mode="json")["funding_mode"] == "disabled"


def test_candidate_screening_can_disable_existing_funding_events():
    frame = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1, tzinfo=UTC)],
            "available_at": [datetime(2024, 1, 1, 0, 1, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [10.0],
            "mark_price": [100.0],
            "mark_low": [99.0],
            "mark_high": [101.0],
            "funding_rate": [0.01],
            "funding_event": [True],
            "funding_price_rate": [1.0],
        }
    )
    original = frame.clone()
    assert prices_from_frame(frame, perpetual=True).funding_unit.tolist() == [1.0]
    assert prices_from_frame(
        frame, perpetual=True, funding_mode="disabled"
    ).funding_unit.tolist() == [0.0]
    assert frame.equals(original)
