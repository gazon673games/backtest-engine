from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.lifecycle import (
    finish_strategy,
    notify_strategy_fill,
    strategy_lifecycle,
)
from backtest_engine.strategies.plugins.loader import (
    generate_intraday_event_plugin_decision_from_callable,
    load_intraday_strategy_plugin,
)


def test_python_strategy_lifecycle_owns_state_for_one_run(tmp_path: Path) -> None:
    strategy_path = tmp_path / "stateful.py"
    strategy_path.write_text(
        """
def initialize(params):
    return {"calls": 0, "fills": 0, "ended": False, "seed": params.get("seed")}

def generate_signal(ctx, params):
    ctx.state["calls"] += 1
    return {"target": 1 if ctx.state["calls"] == 1 else 0}

def on_fill(state, event):
    state["fills"] += 1

def on_end(state, summary):
    state["ended"] = summary["event_count"] == 2
""".lstrip(),
        encoding="utf-8",
    )
    strategy = load_intraday_strategy_plugin(strategy_path)
    frame = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 1)],
            "timeframe": ["1m", "1m"],
            "close": [100.0, 101.0],
        }
    )

    first = generate_intraday_event_plugin_decision_from_callable(
        strategy, IntradayStrategyContext(frame, end_index=0), {"seed": "7"}
    )
    second = generate_intraday_event_plugin_decision_from_callable(
        strategy, IntradayStrategyContext(frame, end_index=1), {"seed": "7"}
    )
    notify_strategy_fill(strategy, {"quantity": 1.0})
    finish_strategy(strategy, {"event_count": 2})

    lifecycle = strategy_lifecycle(strategy)
    assert lifecycle is not None
    assert first.target == 1.0
    assert second.target == 0.0
    assert lifecycle.state == {
        "calls": 2,
        "fills": 1,
        "ended": True,
        "seed": "7",
    }
