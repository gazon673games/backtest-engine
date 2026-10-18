from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.core.exceptions import BacktestError
from backtest_engine.research.strategy_audit import StrategyAuditConfig, audit_strategy


def write_dataset(tmp_path: Path, count: int = 40) -> Path:
    path = tmp_path / "market.parquet"
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=index) for index in range(count)]
    closes = [100.0 + index for index in range(count)]
    pl.DataFrame(
        {
            "source": ["fixture"] * count,
            "exchange": ["TEST"] * count,
            "engine": ["spot"] * count,
            "market": ["crypto"] * count,
            "board": ["SPOT"] * count,
            "symbol": ["BTCUSDT"] * count,
            "timeframe": ["1m"] * count,
            "ts": timestamps,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * count,
            "value": closes,
            "currency": ["USDT"] * count,
            "dataset_type": ["spot"] * count,
            "base_currency": ["BTC"] * count,
            "quote_currency": ["USDT"] * count,
        }
    ).write_parquet(path)
    return path


def test_event_strategy_audit_passes_causal_strategy(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path)
    strategy = tmp_path / "clean.py"
    strategy.write_text(
        "def generate_signal(ctx):\n"
        "    mean = ctx.sma('close', 3)\n"
        "    return {'target': float(mean is not None and ctx.current_bar()['close'] > mean), "
        "'features': {'sma': mean}}\n",
        encoding="utf-8",
    )
    result = audit_strategy(
        StrategyAuditConfig(
            strategy_file=strategy,
            dataset_path=dataset,
            recursive_windows=[8, 16, 32],
            gate=True,
        )
    )
    assert result.status == "passed"
    assert result.lookahead_mismatches == 0
    assert result.recursive_mismatches == 0


def test_event_strategy_audit_detects_private_future_access(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path)
    strategy = tmp_path / "biased.py"
    strategy.write_text(
        "def generate_signal(ctx):\n"
        "    future = ctx._history.get_column('close')[-1]\n"
        "    return float(ctx.current_bar()['close'] < future)\n",
        encoding="utf-8",
    )
    with pytest.raises(BacktestError, match="Strategy bias audit failed"):
        audit_strategy(
            StrategyAuditConfig(
                strategy_file=strategy,
                dataset_path=dataset,
                recursive_windows=[8, 16, 32],
                output_path=tmp_path / "audit.json",
                gate=True,
            )
        )
    assert (tmp_path / "audit.json").is_file()


def test_vectorized_strategy_audit_detects_lookahead(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path)
    strategy = tmp_path / "vectorized.py"
    strategy.write_text(
        "def generate_signals(ctx):\n"
        "    values = ctx.series('close').to_list()\n"
        "    return [float(value < values[-1]) for value in values]\n",
        encoding="utf-8",
    )
    result = audit_strategy(
        StrategyAuditConfig(
            strategy_file=strategy,
            dataset_path=dataset,
            mode="vectorized",
            recursive_windows=[8, 16, 32],
        )
    )
    assert result.status == "failed"
    assert result.lookahead_mismatches > 0
