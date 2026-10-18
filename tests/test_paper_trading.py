from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from backtest_engine.data.instrument_registry import InstrumentSpec
from backtest_engine.paper.engine import run_paper_feed
from backtest_engine.paper.feed import BinanceClosedKlineFeed
from backtest_engine.paper.models import PaperTradingConfig


def test_paper_replay_uses_strategy_execution_and_spot_ledger(tmp_path: Path) -> None:
    strategy = tmp_path / "paper_strategy.py"
    strategy.write_text(
        """
def initialize(params):
    return {"calls": 0, "fills": 0}

def generate_signal(ctx):
    ctx.state["calls"] += 1
    return {"target": 1 if ctx.state["calls"] == 1 else 0}

def on_fill(state, event):
    state["fills"] += 1
""".lstrip(),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "paper-checkpoint.json"
    config = PaperTradingConfig(
        strategy_file=strategy,
        strict_runtime=False,
        instrument=InstrumentSpec(
            symbol="BTCUSDT",
            asset_class="crypto",
            instrument_type="spot",
            contract_type="spot",
            base_currency="BTC",
            quote_currency="USDT",
            settlement_currency="USDT",
            taker_fee_bps=10.0,
        ),
        initial_capital=1_000.0,
        checkpoint_path=checkpoint,
    )
    start = datetime(2025, 1, 1, tzinfo=UTC)
    feed = [
        {
            "ts": start + timedelta(minutes=index),
            "available_at": start + timedelta(minutes=index + 1),
            "timeframe": "1m",
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 10.0,
        }
        for index, price in enumerate((100.0, 101.0, 102.0))
    ]

    result = run_paper_feed(config, feed)

    assert result.event_count == 3
    assert result.fill_count == 2
    assert result.final_position == 0.0
    assert result.final_equity_cash == 1_000.797
    assert checkpoint.exists()
    assert '"final": true' in checkpoint.read_text(encoding="utf-8")


def test_binance_closed_kline_feed_deduplicates_final_candles() -> None:
    class FakeTransport:
        def get(self, product: str, path: str, params: dict[str, object]) -> list[list[object]]:
            assert product == "spot"
            assert path.endswith("/klines")
            return [
                [1_000, "100", "102", "99", "101", "4", 60_999],
                [61_000, "101", "103", "100", "102", "5", 120_999],
            ]

    feed = BinanceClosedKlineFeed(
        symbol="btcusdt",
        timeframe="1m",
        transport=FakeTransport(),
    )
    now = datetime.fromtimestamp(121, tz=UTC)

    first = feed.poll(now)

    assert first is not None
    assert first["symbol"] == "BTCUSDT"
    assert first["close"] == 102.0
    assert feed.poll(now) is None


@pytest.mark.parametrize("max_events", [0, 1])
def test_paper_limit_does_not_fetch_an_extra_event(tmp_path: Path, max_events: int) -> None:
    strategy = tmp_path / "flat.py"
    strategy.write_text("def generate_signal(ctx):\n    return 0.0\n", encoding="utf-8")
    config = PaperTradingConfig(
        strategy_file=strategy,
        strict_runtime=False,
        instrument=InstrumentSpec(
            symbol="BTCUSDT",
            instrument_type="spot",
            base_currency="BTC",
            quote_currency="USDT",
            settlement_currency="USDT",
        ),
    )
    fetched = []

    def feed():
        for index in range(2):
            fetched.append(index)
            yield {
                "ts": datetime(2025, 1, 1, 0, index, tzinfo=UTC),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
            }

    result = run_paper_feed(config, feed(), max_events=max_events)

    assert result.event_count == max_events
    assert fetched == list(range(max_events))
