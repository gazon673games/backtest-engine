from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from binance_fixture import FixtureTransport, kline, timestamp_ms

from backtest_engine.adapters.binance.adapter import BinanceAdapter
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.intraday.event.config import IntradayEventInstrumentSpec
from backtest_engine.intraday.event.crypto_rules import validate_crypto_fill
from backtest_engine.workbench.datasets import register_workbench_dataset
from backtest_engine.workbench.models import WorkbenchRunRequest
from backtest_engine.workbench.runner import run_workbench_backtest

STRATEGY = """
globalThis.STRATEGY_METADATA = {
  name: "crypto_fixture", timeframe: "5m", schedule: "every-bar",
  missing_decision_policy: "carry", history_bars: 10,
  data_columns: ["ts", "open", "high", "low", "close", "volume", "timeframe"],
};
function onDecision() { return { target: 1, contracts: 0.01, reason: "fixture" }; }
"""


class SeriesTransport(FixtureTransport):
    def get(self, product, path, params=None):
        if path.endswith("fundingRate"):
            return [
                {
                    "symbol": "BTCUSDT",
                    "fundingTime": timestamp_ms(2024, 1, 1, 0, 15, 30),
                    "fundingRate": "0.001",
                    "markPrice": "40300",
                }
            ]
        if path.endswith("klines") or path.endswith("markPriceKlines"):
            self.calls.append((product, path, params or {}))
            result = []
            for index in range(8):
                row = kline()
                row[0] += index * 300_000
                row[6] = row[0] + 299_999
                close = 40_000.0 + 100 * index
                row[1:5] = [str(close), str(close + 50), str(close - 50), str(close)]
                if params["startTime"] <= row[0] <= params["endTime"]:
                    result.append(row)
            return result
        return super().get(product, path, params)


def rule_archive(tmp_path: Path, product: str) -> Path:
    path = tmp_path / f"{product}-rules.json"
    snapshot = {
        "valid_from": "2024-01-01",
        "valid_to": "2024-01-31",
        "tick_size": 0.1,
        "quantity_step": 0.001,
        "minimum_quantity": 0.001,
        "minimum_notional": 5.0,
        "maker_fee_bps": 2.0,
        "taker_fee_bps": 5.0,
        "liquidation_fee_bps": 10.0,
    }
    if product == "perpetual":
        snapshot["margin_tiers"] = [
            {
                "notional_floor": 0.0,
                "notional_cap": None,
                "maintenance_margin_rate": 0.005,
                "max_leverage": 20.0,
            }
        ]
    path.write_text(
        json.dumps(
            {"venue": "binance", "product": product, "symbol": "BTCUSDT", "snapshots": [snapshot]}
        ),
        encoding="utf-8",
    )
    return path


def build_request(tmp_path, product="spot", **overrides):
    values = dict(
        adapter_id="binance",
        engine=product,
        market="crypto",
        security="BTCUSDT",
        from_date="2024-01-01",
        till_date="2024-01-01",
        timeframe="5m",
        output_root=tmp_path / "datasets",
        history_source="rest",
        rules_path=rule_archive(tmp_path, product),
        rules_enforcement="strict",
    )
    return DatasetBuildRequest(**(values | overrides))


def run_request(result, tmp_path, **overrides):
    values = dict(
        dataset_path=result.dataset_path,
        quality_approval_path=result.approval_path,
        quality_approval_required=True,
        instrument_registry_path=result.instrument_registry_path,
        script_source=STRATEGY,
        from_date="2024-01-01",
        initial_capital=1000.0,
        output_root=tmp_path / "runs",
        run_id="crypto",
    )
    return WorkbenchRunRequest(**(values | overrides))


@pytest.mark.parametrize("product", ["spot", "perpetual"])
def test_strict_binance_build_runs_through_catalog_and_workbench(tmp_path, product):
    request = build_request(tmp_path, product)
    result = BinanceAdapter(SeriesTransport()).build_dataset(request)
    result = register_workbench_dataset(tmp_path / "runs", request, result)
    assert result.approval_status == "approved"
    response = run_workbench_backtest(
        run_request(
            result,
            tmp_path,
            catalog_path=result.catalog_path,
            dataset_id=result.dataset_id,
            catalog_enforcement="strict",
        )
    )
    assert response.status == "completed"
    config = json.loads((response.output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["portfolio"]["base_currency"] == "USDT"
    assert config["portfolio"]["allow_fractional_contracts"] is True
    assert config["execution"]["trading_days_per_year"] == 365
    events = pl.read_parquet(response.output_dir / "events.parquet")
    assert events.get_column("target_contracts").max() == pytest.approx(0.01)
    assert events.get_column("trading_fee_cash").sum() > 0
    if product == "perpetual":
        assert events.get_column("funding_cost_cash").sum() > 0
    assert pl.read_parquet(response.output_dir / "portfolio_ledger.parquet").height > 0


def test_rebuilt_dataset_does_not_overwrite_queued_run_inputs(tmp_path):
    request = build_request(tmp_path)
    adapter = BinanceAdapter(SeriesTransport())
    first = adapter.build_dataset(request)
    contents = first.dataset_path.read_bytes()
    second = adapter.build_dataset(request)
    assert first.dataset_path != second.dataset_path
    assert first.dataset_path.read_bytes() == contents


def test_margin_rules_change_without_resetting_an_open_position(tmp_path):
    class MidnightTransport(SeriesTransport):
        def get(self, product, path, params=None):
            if path.endswith("fundingRate"):
                return []
            rows = super().get(product, path, params)
            if path.endswith("klines") or path.endswith("markPriceKlines"):
                for row in rows:
                    row[0] += 85_200_000  # 23:40 UTC, with bars across midnight.
                    row[6] += 85_200_000
            return rows

    request = build_request(tmp_path, "perpetual", till_date="2024-01-02")
    archive = json.loads(request.rules_path.read_text(encoding="utf-8"))
    first = archive["snapshots"][0]
    first["valid_to"] = "2024-01-01"
    second = json.loads(json.dumps(first))
    second.update(valid_from="2024-01-02", valid_to="2024-01-02", liquidation_fee_bps=25.0)
    second["margin_tiers"][0]["maintenance_margin_rate"] = 0.02
    archive["snapshots"].append(second)
    request.rules_path.write_text(json.dumps(archive), encoding="utf-8")
    result = BinanceAdapter(MidnightTransport()).build_dataset(request)
    assert result.approval_status == "approved"
    response = run_workbench_backtest(run_request(result, tmp_path))
    events = pl.read_parquet(response.output_dir / "events.parquet")
    rows = events.to_dicts()
    assert {row["fill_ts"].date() for row in rows} == {date(2024, 1, 1), date(2024, 1, 2)}
    for row in rows:
        rate = 0.005 if row["fill_ts"].date().day == 1 else 0.02
        assert row["maintenance_margin"] == pytest.approx(0.01 * row["exit_price"] * rate)
        assert row["target_contracts"] == pytest.approx(0.01)
    assert events.get_column("trading_fee_cash").sum() == pytest.approx(0.01 * 40100 * 0.0005)


def test_approved_dataset_cannot_be_replaced_after_build(tmp_path):
    result = BinanceAdapter(SeriesTransport()).build_dataset(build_request(tmp_path))
    pl.read_parquet(result.dataset_path).with_columns(
        (pl.col("volume") + 1.0).alias("volume")
    ).write_parquet(result.dataset_path)
    with pytest.raises(DatasetValidationError, match="dataset_sha256 mismatch"):
        run_workbench_backtest(run_request(result, tmp_path))


def test_missing_mark_prices_block_perpetual_quality(tmp_path):
    class MissingMarks(SeriesTransport):
        def get(self, product, path, params=None):
            if path.endswith("markPriceKlines"):
                return []
            return super().get(product, path, params)

    result = BinanceAdapter(MissingMarks()).build_dataset(build_request(tmp_path, "perpetual"))
    assert result.approval_status == "rejected"
    assert "missing mark prices" in " ".join(result.quality["violations"])


@pytest.mark.parametrize("bad_price", ["nan", "-1", "999999"])
def test_invalid_mark_prices_block_perpetual_quality(tmp_path, bad_price):
    class InvalidMarks(SeriesTransport):
        def get(self, product, path, params=None):
            rows = super().get(product, path, params)
            if path.endswith("markPriceKlines"):
                rows[0][4] = bad_price
            return rows

    result = BinanceAdapter(InvalidMarks()).build_dataset(build_request(tmp_path, "perpetual"))
    assert result.approval_status == "rejected"
    assert "invalid mark prices" in " ".join(result.quality["violations"])


def test_timestamp_gaps_do_not_earn_approved_status(tmp_path):
    class GapTransport(SeriesTransport):
        def get(self, product, path, params=None):
            rows = super().get(product, path, params)
            if path.endswith("klines"):
                return rows[:2] + rows[3:]
            return rows

    result = BinanceAdapter(GapTransport()).build_dataset(build_request(tmp_path))
    assert result.approval_status == "warning"
    assert result.quality["gap_count"] == 1
    with pytest.raises(DatasetValidationError, match="not approved"):
        run_workbench_backtest(run_request(result, tmp_path))


@pytest.mark.parametrize(
    ("delta", "message"), [(0.0105, "quantity_step"), (0.001, "minimum_notional")]
)
def test_crypto_filters_are_enforced_on_actual_fills(delta, message):
    spec = IntradayEventInstrumentSpec(
        symbol="BTCUSDT",
        asset_class="crypto",
        instrument_type="spot",
        quantity_step=0.001,
        minimum_quantity=0.001,
        minimum_notional=5.0,
    )
    with pytest.raises(DatasetValidationError, match=message):
        validate_crypto_fill(spec, 0.0, delta, 1000.0, 1.0)


def test_historical_maximum_leverage_cannot_be_exceeded():
    spec = IntradayEventInstrumentSpec(
        symbol="BTCUSDT",
        asset_class="crypto",
        instrument_type="perpetual",
        margin_tiers=[
            {
                "notional_floor": 0.0,
                "notional_cap": None,
                "maintenance_margin_rate": 0.005,
                "max_leverage": 5.0,
            }
        ],
    )
    with pytest.raises(DatasetValidationError, match="historical max_leverage"):
        validate_crypto_fill(spec, 0.0, 1.0, 1000.0, 10.0)
