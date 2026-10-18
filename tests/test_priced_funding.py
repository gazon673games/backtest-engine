"""Funding prices survive normalization, cancellation of rates and aggregation."""

from __future__ import annotations

import polars as pl
import pytest
from binance_fixture import exchange_info, kline

from backtest_engine.accounting.funding import funding_payment_for_row
from backtest_engine.adapters.binance.funding import funding_price_rate
from backtest_engine.adapters.binance.normalize import normalize_klines
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.bar_aggregation import aggregate_intraday_bars


def payment(row, **overrides):
    arguments = dict(
        quantity=2,
        point_value=1,
        mark_price=40000,
        rate_column="funding_rate",
        default_rate=0,
        event_column="funding_event",
        require_event_flag=True,
    )
    return funding_payment_for_row(row, **(arguments | overrides))


def test_offsetting_rates_at_different_prices_still_produce_a_cash_payment():
    bar = kline()
    events = [
        {"fundingTime": bar[0] + 1, "fundingRate": "0.001", "markPrice": "100"},
        {"fundingTime": bar[0] + 2, "fundingRate": "-0.001", "markPrice": "120"},
    ]
    frame = normalize_klines(
        product="perpetual",
        symbol="BTCUSDT",
        timeframe="1m",
        metadata=exchange_info("perpetual")["symbols"][0],
        klines=[bar],
        funding=events,
    )
    row = frame.row(0, named=True)
    assert row["funding_rate"] == 0
    assert row["funding_event"] is True
    assert row["funding_price_rate"] == pytest.approx(-0.02)
    assert payment(row) == pytest.approx(-0.04)


@pytest.mark.parametrize("price", ["nan", "inf", "0", "-1", "invalid"])
def test_invalid_funding_event_price_is_rejected_with_symbol_and_timestamp(price):
    event = {"fundingTime": 123, "fundingRate": "0.001", "markPrice": price}
    with pytest.raises(DatasetValidationError, match="BTCUSDT at 123"):
        funding_price_rate([event], "BTCUSDT")


def test_priced_payment_still_requires_the_event_flag():
    row = {"funding_rate": 0, "funding_price_rate": 0.1}
    with pytest.raises(DatasetValidationError, match="funding_event"):
        payment(row)
    assert payment(row | {"funding_event": False}) == 0
    assert payment(row | {"funding_event": True}) == pytest.approx(0.2)


def test_legacy_missing_prices_and_custom_rate_columns_keep_declared_bar_valuation():
    row = {"funding_rate": 0.001, "funding_event": True, "funding_price_rate": None}
    assert payment(row, mark_price=100) == pytest.approx(0.2)
    row.update(funding_price_rate=99, custom_rate=0.005)
    assert payment(row, mark_price=100, rate_column="custom_rate") == pytest.approx(1)


@pytest.mark.parametrize("price", ["120", None], ids=["priced", "legacy-unpriced"])
def test_aggregation_preserves_price_rate_sums_and_missing_price_information(price):
    first = kline()
    second = kline()
    second[0] += 60000
    second[6] += 60000
    events = [
        {"fundingTime": first[0] + 1, "fundingRate": "0.001", "markPrice": "100"},
        {"fundingTime": second[0] + 1, "fundingRate": "-0.001", "markPrice": price},
    ]
    frame = normalize_klines(
        product="perpetual",
        symbol="BTCUSDT",
        timeframe="1m",
        metadata=exchange_info("perpetual")["symbols"][0],
        klines=[first, second],
        funding=events,
    )
    before = frame.clone()
    result = aggregate_intraday_bars(frame, "5m")
    assert frame.equals(before)
    assert result.height == 1
    assert result["funding_rate"][0] == 0
    assert result["funding_event"][0] is True
    assert result.schema["funding_price_rate"] == pl.Float64
    if price is None:
        assert result["funding_price_rate"][0] is None
    else:
        assert result["funding_price_rate"][0] == pytest.approx(-0.02)
        assert payment(result.row(0, named=True)) == pytest.approx(-0.04)
