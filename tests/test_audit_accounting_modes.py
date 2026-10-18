"""Native multi and paper regressions for the shared accounting contracts."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from test_shared_cross_margin import leg, write_perpetual

from backtest_engine.core.exceptions import DatasetValidationError, StrategyPluginError
from backtest_engine.data.instrument_registry import InstrumentSpec
from backtest_engine.paper.engine import PaperTradingEngine
from backtest_engine.paper.models import PaperTradingConfig
from backtest_engine.portfolio.multi.engine import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentExecutionConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
    run_multi_instrument_backtest,
)


@pytest.mark.parametrize("include_marks", [True, False])
def test_native_cross_margin_uses_funding_event_price_and_requires_marks(tmp_path, include_marks):
    legs = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        path = write_perpetual(tmp_path / f"{symbol}.parquet", symbol, [100.0] * 3)
        frame = pl.read_parquet(path).with_columns(
            pl.Series("funding_rate", [0.0, 0.0, 0.01]),
            pl.Series("funding_price_rate", [0.0, 0.0, 0.5]),
            pl.Series("funding_event", [False, False, True]),
        )
        if not include_marks:
            frame = frame.drop("mark_price", "mark_low", "mark_high")
        frame.write_parquet(path)
        legs.append(leg(path, symbol).model_copy(update={"taker_fee_bps": 0.0}))
    strategy = tmp_path / "strategy.py"
    strategy.write_text(
        "def generate_targets(ctx):\n    return {key: 1 for key in ctx.tradable_assets}\n",
        encoding="utf-8",
    )
    config = MultiInstrumentBacktestConfig(
        legs=legs,
        strategy=MultiInstrumentStrategyConfig(strategy_file=strategy),
        portfolio=MultiInstrumentPortfolioConfig(
            initial_capital=10_000,
            base_currency="USDT",
            fixed_contracts=1,
            margin_mode="cross",
            leverage=10,
            require_mark_price=True,
        ),
        execution=MultiInstrumentExecutionConfig(
            from_date=datetime(2024, 1, 2).date(), end_of_data_position_policy="mark"
        ),
        output=MultiInstrumentOutputConfig(root=tmp_path / "runs", run_id="cross"),
    )
    if not include_marks:
        with pytest.raises(DatasetValidationError, match="mark"):
            run_multi_instrument_backtest(config)
        return
    result = run_multi_instrument_backtest(config)
    rows = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    assert rows["funding_cost_cash"].sum() == pytest.approx(1.0)
    assert result.metrics["final_equity_cash"] == pytest.approx(9999.0)


def spot_spec():
    return InstrumentSpec(
        symbol="SOLUSDT",
        asset_class="crypto",
        instrument_type="spot",
        contract_type="spot",
        base_currency="SOL",
        quote_currency="USDT",
        settlement_currency="USDT",
        price_currency="USDT",
        pnl_currency="USDT",
    )


def test_paper_enforces_strict_metadata(tmp_path):
    path = tmp_path / "unannotated.py"
    path.write_text("def generate_signal(ctx): return 0\n", encoding="utf-8")
    with pytest.raises(StrategyPluginError, match="STRATEGY_METADATA"):
        PaperTradingEngine(PaperTradingConfig(strategy_file=path, instrument=spot_spec()))


def test_paper_context_has_capital_and_bounded_history(tmp_path):
    path = tmp_path / "capital.py"
    path.write_text(
        """from backtest_engine.strategies.runtime import StrategyMetadata
STRATEGY_METADATA = StrategyMetadata(name="Capital", history_bars=2)
def generate_signal(ctx):
    assert ctx.base_currency == "USDT"
    assert ctx.instrument_spec.symbol == "SOLUSDT"
    assert ctx.equity_cash >= 1000
    assert ctx.available_funds >= 0
    assert ctx.point_value == 1
    assert len(ctx.history()) <= 2
    if ctx.position_quantity == 0:
        return {"target": 1, "contracts": ctx.equity_cash * 0.05 / 100}
    return {"target": 1, "contracts": ctx.position_quantity}
""",
        encoding="utf-8",
    )
    engine = PaperTradingEngine(
        PaperTradingConfig(
            strategy_file=path,
            instrument=spot_spec(),
            initial_capital=1000,
            base_currency="USDT",
            max_history_bars=2,
            max_state_rows=2,
        )
    )
    at = datetime(2024, 1, 1)
    for index in range(5):
        engine.on_event(
            {
                "ts": at + timedelta(minutes=index),
                "available_at": at + timedelta(minutes=index + 1),
                "open": 100.0,
                "close": 100.0,
                "low": 100.0,
                "high": 100.0,
                "volume": 100.0,
            }
        )
    result = engine.finish()
    assert result.event_count == 5
    assert result.final_position == pytest.approx(0.5)
    assert result.final_equity_cash == pytest.approx(1000.0)
    assert len(engine.history) == len(result.state_rows) == 2


def test_paper_perpetual_uses_same_priced_funding_contract(tmp_path):
    path = tmp_path / "perpetual.py"
    path.write_text(
        """from backtest_engine.strategies.runtime import StrategyMetadata
STRATEGY_METADATA = StrategyMetadata(name="Perpetual")
def generate_signal(ctx): return 1
""",
        encoding="utf-8",
    )
    spec = spot_spec().model_copy(
        update={"instrument_type": "perpetual", "contract_type": "linear"}
    )
    config = PaperTradingConfig(
        strategy_file=path,
        instrument=spec,
        initial_capital=1000,
        leverage=10,
        require_margin_tiers=False,
    )
    engine = PaperTradingEngine(config)
    at = datetime(2024, 1, 1)
    for index in range(3):
        engine.on_event(
            {
                "ts": at + timedelta(minutes=index),
                "available_at": at + timedelta(minutes=index + 1),
                "open": 100.0,
                "close": 100.0,
                "low": 100.0,
                "high": 100.0,
                "volume": 100.0,
                "mark_price": 100.0,
                "mark_low": 100.0,
                "mark_high": 100.0,
                "funding_event": index == 2,
                "funding_rate": 0.01 if index == 2 else 0.0,
                "funding_price_rate": 0.5 if index == 2 else 0.0,
            }
        )
    result = engine.finish()
    assert result.final_position == pytest.approx(1.0)
    assert result.final_equity_cash == pytest.approx(999.5)
