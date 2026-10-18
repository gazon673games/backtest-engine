from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from backtest_engine.portfolio.multi.engine import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentExecutionConfig,
    MultiInstrumentLegConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
    run_multi_instrument_backtest,
)


def write_perpetual(path: Path, symbol: str, closes: list[float]) -> Path:
    timestamps = [
        datetime(2024, 1, 2, 10, 0),
        datetime(2024, 1, 2, 10, 5),
        datetime(2024, 1, 2, 10, 10),
    ]
    pl.DataFrame(
        {
            "source": ["fixture"] * 3,
            "exchange": ["BINANCE"] * 3,
            "engine": ["perpetual"] * 3,
            "market": ["crypto"] * 3,
            "board": ["USD_M_PERPETUAL"] * 3,
            "symbol": [symbol] * 3,
            "timeframe": ["5m"] * 3,
            "ts": timestamps,
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "mark_price": closes,
            "mark_high": [value + 1 for value in closes],
            "mark_low": [value - 1 for value in closes],
            "funding_rate": [0.0, 0.0001, 0.0],
            "volume": [1000.0] * 3,
            "value": [100000.0] * 3,
            "currency": ["USDT"] * 3,
            "dataset_type": ["perpetual"] * 3,
            "base_currency": [symbol.removesuffix("USDT")] * 3,
            "quote_currency": ["USDT"] * 3,
            "settlement_currency": ["USDT"] * 3,
        }
    ).write_parquet(path)
    return path


def leg(path: Path, symbol: str) -> MultiInstrumentLegConfig:
    return MultiInstrumentLegConfig(
        asset_id=symbol,
        dataset_path=path,
        quality_approval_required=False,
        allow_legacy_bar_timestamps=True,
        price_currency="USDT",
        pnl_currency="USDT",
        asset_class="crypto",
        instrument_type="perpetual",
        base_currency=symbol.removesuffix("USDT"),
        quote_currency="USDT",
        settlement_currency="USDT",
        initial_margin=10.0,
        taker_fee_bps=5.0,
        maintenance_margin_rate=0.01,
        liquidation_fee_bps=1.0,
    )


def test_multi_perpetual_run_uses_shared_cross_margin(tmp_path: Path) -> None:
    btc = write_perpetual(tmp_path / "btc.parquet", "BTCUSDT", [100.0, 105.0, 110.0])
    eth = write_perpetual(tmp_path / "eth.parquet", "ETHUSDT", [50.0, 48.0, 52.0])
    strategy = tmp_path / "strategy.py"
    strategy.write_text(
        "def generate_targets(ctx):\n"
        "    return {asset_id: 1.0 for asset_id in ctx.tradable_assets}\n",
        encoding="utf-8",
    )
    result = run_multi_instrument_backtest(
        MultiInstrumentBacktestConfig(
            legs=[leg(btc, "BTCUSDT"), leg(eth, "ETHUSDT")],
            strategy=MultiInstrumentStrategyConfig(strategy_file=strategy),
            portfolio=MultiInstrumentPortfolioConfig(
                initial_capital=10_000.0,
                base_currency="USDT",
                fixed_contracts=1.0,
                margin_mode="cross",
                leverage=10.0,
            ),
            execution=MultiInstrumentExecutionConfig(
                from_date=datetime(2024, 1, 2).date(),
                timing="signal_close_to_next_bar_close",
                end_of_data_position_policy="mark",
                require_funding_event_flag=False,
            ),
            output=MultiInstrumentOutputConfig(root=tmp_path / "runs", run_id="shared"),
        )
    )

    rows = pl.read_parquet(result.output_dir / "portfolio_legs.parquet")
    ledger = pl.read_parquet(result.output_dir / "portfolio_ledger.parquet")
    assert set(rows.get_column("asset_id")) == {"BTCUSDT", "ETHUSDT"}
    assert rows.get_column("target_contracts").to_list() == [1.0, 1.0]
    assert ledger.get_column("symbol").to_list() == ["CROSS_MARGIN"]
    assert result.metrics["final_equity_cash"] > 10_000.0
