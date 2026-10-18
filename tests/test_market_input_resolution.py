"""Explicit currency, calendar and rule inputs determine accounting behavior."""

import json
from datetime import UTC, datetime

import polars as pl
import pytest
from neutral_adapter_fixture import NeutralAdapter

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.request_defaults import resolve_build_request
from backtest_engine.cli.parsing import parse_multi_instrument_leg
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.bar_availability import finalized_as_of
from backtest_engine.data.instrument_registry import InstrumentSpec
from backtest_engine.data.market_identity import account_currency, annualization_days
from backtest_engine.intraday.event.crypto_rules import validate_crypto_fill


def test_only_the_selected_adapter_supplies_build_defaults(tmp_path):
    request = resolve_build_request(
        DatasetBuildRequest(
            adapter_id=NeutralAdapter.id,
            security="GOLDUSD",
            from_date="2024-01-02",
            till_date="2024-01-02",
            output_root=tmp_path / "datasets",
        ),
        NeutralAdapter(),
    )
    assert (request.engine, request.market, request.board) == ("cash", "test-market", "test-board")
    assert request.cache_dir == tmp_path / "cache" / NeutralAdapter.id
    assert request.source_delay_minutes == 2
    assert request.roll_rule is None


def test_missing_or_mixed_currencies_require_an_account_choice():
    assert account_currency(None, ["EUR", "EUR"]) == "EUR"
    assert account_currency("CHF", ["USD", "EUR"]) == "CHF"
    for currencies in ([None], ["EUR", "USD"]):
        with pytest.raises(DatasetValidationError, match="base_currency"):
            account_currency(None, currencies)


def test_cli_leg_preserves_unspecified_currency_for_dataset_resolution():
    leg = parse_multi_instrument_leg("gold=prices.parquet")
    assert leg.price_currency is None
    assert leg.pnl_currency is None


def test_annualization_does_not_guess_an_exchange_calendar(tmp_path):
    assert annualization_days(None, [None]) is None
    assert annualization_days(250, [None]) == 250
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, factor in zip(paths, (260, 365), strict=True):
        path.write_text(json.dumps({"trading_days_per_year": factor}))
    assert annualization_days(None, paths[:1]) == 260
    with pytest.raises(DatasetValidationError, match="different annualization"):
        annualization_days(None, paths)
    paths[0].write_text('{"trading_days_per_year": 0}')
    with pytest.raises(DatasetValidationError, match="positive integer"):
        annualization_days(None, paths[:1])


def test_non_crypto_instrument_quantity_rules_are_enforced():
    instrument = InstrumentSpec(
        symbol="GOLDUSD",
        asset_class="commodity",
        instrument_type="spot",
        price_currency="USD",
        pnl_currency="USD",
        quantity_step=0.5,
        minimum_notional=50,
    )
    with pytest.raises(DatasetValidationError, match="quantity_step"):
        validate_crypto_fill(instrument, 0, 0.7, 100, 1)
    with pytest.raises(DatasetValidationError, match="minimum_notional"):
        validate_crypto_fill(instrument, 0, 0.5, 10, 1)


@pytest.mark.parametrize("aware", [False, True])
def test_finalization_uses_explicit_timestamp_basis(aware):
    frame = pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 1, 0, minute, tzinfo=UTC if aware else None)
                for minute in (0, 5, 10)
            ]
        }
    )
    result, audit = finalized_as_of(
        frame,
        timeframe="5m",
        source_as_of=datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
        source_delay_minutes=0,
        timestamp_timezone="UTC",
    )
    assert result.height == 2
    assert audit["timestamp_timezone"] == "UTC"
