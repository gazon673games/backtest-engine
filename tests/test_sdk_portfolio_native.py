from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest
from strategy_sdk import DataRequirements, StrategyMetadata, hold_decision

from backtest_engine.data.instrument_registry import InstrumentSpec
from backtest_engine.portfolio.multi.config import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentExecutionConfig,
    MultiInstrumentLegConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
)
from backtest_engine.portfolio.multi.engine import run_multi_instrument_backtest
from backtest_engine.portfolio.multi.sdk_history import SDKHistory
from backtest_engine.portfolio.multi.strategy_config import (
    MultiInstrumentStrategyConfig,
    SDKSleeveConfig,
)
from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.sdk_adapter import SDKReplayStrategy


def bars(symbol, start_price):
    start_price = float(start_price)
    times = [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(4)]
    closes = [start_price] + [start_price * 1.1] * 3
    return pl.DataFrame(
        {
            "ts": times,
            "bar_open_ts": times,
            "bar_close_ts": [t + timedelta(hours=1) - timedelta(milliseconds=1) for t in times],
            "available_at": [t + timedelta(hours=1) - timedelta(milliseconds=1) for t in times],
            "open": [start_price, start_price, *closes[2:]],
            "high": closes,
            "low": [start_price, start_price, *closes[2:]],
            "close": closes,
            "volume": [10000.0] * 4,
            "timeframe": ["60m"] * 4,
            "symbol": [symbol] * 4,
            "active_contract": [symbol] * 4,
            "continuous_symbol": [symbol] * 4,
            "instrument_type": ["perpetual"] * 4,
            "asset_class": ["crypto"] * 4,
            "price_currency": ["USDT"] * 4,
            "currency": ["USDT"] * 4,
            "funding_event": [False, True, False, False],
            "funding_rate": [0.0, 0.01, 0.0, 0.0],
            "mark_price": closes,
            "mark_low": [start_price, start_price, *closes[2:]],
            "mark_high": closes,
            "dataset_type": ["perpetual"] * 4,
            "base_currency": [symbol] * 4,
            "quote_currency": ["USDT"] * 4,
            "settlement_currency": ["USDT"] * 4,
            "source": ["fixture"] * 4,
            "exchange": ["fixture"] * 4,
            "engine": ["perpetual"] * 4,
            "market": ["perpetual"] * 4,
            "board": ["fixture"] * 4,
            "value": [100000.0] * 4,
            "is_final": [True] * 4,
        }
    )


def test_sdk_signal_history_excludes_incomplete_and_future_buckets():
    frame = bars("A", 100)
    history = SDKHistory(frame, SimpleNamespace(timeframe="M120"))
    assert history.advance(datetime(2025, 1, 1, 1, tzinfo=UTC)) is None
    assert history.advance(datetime(2025, 1, 1, 2, tzinfo=UTC)) == 0
    assert history.advance(datetime(2025, 1, 1, 2, 30, tzinfo=UTC)) is None
    gap = SDKHistory(frame.filter(pl.col("ts").dt.hour() != 1), SimpleNamespace(timeframe="M120"))
    assert gap.frame.height == 1
    assert gap.frame["ts"][0].hour == 2


def test_sdk_hold_keeps_native_inventory_without_requesting_a_noise_order(tmp_path):
    def hold(ctx, params):
        ctx.state["observed"] = str(ctx.position_quantity)
        return hold_decision(reason="keep_position")

    source = SimpleNamespace(
        __strategy_source__=b"hold fixture",
        STRATEGY_METADATA=StrategyMetadata("hold_fixture"),
        data_requirements=lambda params: DataRequirements("M60", 1),
        generate_signal=hold,
    )
    adapter = SDKReplayStrategy(source, tmp_path / "hold.py")
    actual = 0.005999999999999999
    context = IntradayStrategyContext(
        bars("A", 100),
        end_index=0,
        timeframe="60m",
        previous_position=1,
        position_quantity=actual,
        point_value=1,
        base_currency="USDT",
        equity_cash=700,
        instrument_spec=InstrumentSpec(symbol="A", quantity_step=0.001),
    )
    assert adapter(context, {})["contracts"] == actual
    assert adapter.state["observed"] == "0.006"


def test_native_sdk_portfolio_reconciles_fees_funding_and_confirmed_quantities(tmp_path):
    source = tmp_path / "fixture_strategy.py"
    source.write_text(
        """from strategy_sdk import (
    DataRequirements, StrategyMetadata, entry_quantity, long_decision, hold_decision,
)
STRATEGY_METADATA = StrategyMetadata("fixture")
def data_requirements(params):
    return DataRequirements("M60", 1)
def generate_signal(ctx, params):
    if ctx.position_quantity:
        return hold_decision(reason="fixture_hold")
    quantity = entry_quantity(
        ctx.account.equity, ctx.current_bar().close, ctx.instrument, rounding="down",
    )
    return long_decision(contracts=quantity, reason="fixture_entry")
""",
        encoding="utf-8",
    )
    legs, sleeves = [], []
    for asset, price in (("A", 100), ("B", 50)):
        path = tmp_path / f"{asset}.parquet"
        bars(asset, price).write_parquet(path)
        legs.append(
            MultiInstrumentLegConfig(
                asset_id=asset,
                dataset_path=path,
                quality_approval_required=False,
                asset_class="crypto",
                instrument_type="perpetual",
                pnl_currency="USDT",
                price_currency="USDT",
                settlement_currency="USDT",
                quantity_step=0.001,
                minimum_quantity=0.001,
                minimum_notional=1,
                taker_fee_bps=5,
                maintenance_margin_rate=0.005,
            )
        )
        sleeves.append(
            SDKSleeveConfig(
                asset_id=asset, strategy_file=source, initial_capital=300, max_exposure=2
            )
        )
    result = run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=legs,
            strategy=MultiInstrumentStrategyConfig(sdk_sleeves=sleeves, strict_runtime=True),
            portfolio=MultiInstrumentPortfolioConfig(
                initial_capital=700, base_currency="USDT", leverage=3
            ),
            execution=MultiInstrumentExecutionConfig(
                timing="signal_close_to_next_bar_open_to_close",
                from_date=datetime(2025, 1, 1).date(),
                end_of_data_position_policy="mark",
                trading_days_per_year=365,
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path, run_id="golden-sdk"),
        )
    )
    assert result.metrics["final_equity_cash"] == pytest.approx(753.1)
    rows = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    assert rows["funding_cost_cash"].sum() == pytest.approx(6.6)
    assert rows["trading_fee_cash"].sum() == pytest.approx(0.3)
    ledger = pl.read_parquet(result.output_dir / "portfolio_ledger.parquet")
    assert ledger["net_cash_flow"].sum() + 700 == pytest.approx(753.1)
    assert rows.filter(pl.col("filled_contract_delta") != 0).height == 2
    assert (result.output_dir / "sdk_sources.json").exists()
