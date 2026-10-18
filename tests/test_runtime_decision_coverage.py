from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import polars as pl

from backtest_engine.intraday.event.config import (
    IntradayEventBacktestConfig,
    IntradayEventDatasetConfig,
    IntradayEventExecutionConfig,
    IntradayEventOutputConfig,
    IntradayEventStrategyConfig,
)
from backtest_engine.intraday.event.quality import (
    enforce_runtime_decision_coverage,
    runtime_decision_coverage,
)


def timed_config(tmp_path: Path) -> IntradayEventBacktestConfig:
    return IntradayEventBacktestConfig(
        dataset=IntradayEventDatasetConfig(path=tmp_path / "bars.parquet"),
        strategy=IntradayEventStrategyConfig(strategy_file=tmp_path / "strategy.js"),
        execution=IntradayEventExecutionConfig(
            from_date=date(2024, 1, 2),
            schedule="times",
            decision_times=[time(14, 40)],
            decision_tolerance_minutes=0,
            timing="signal_close_to_next_bar_close",
        ),
        output=IntradayEventOutputConfig(root=tmp_path, run_id="fixture"),
    )


def frame_with_trailing_partial_day(*, historical_decision_missing: bool = False) -> pl.DataFrame:
    first_day = [
        datetime(2024, 1, 2, 14, 30 if historical_decision_missing else 40),
        datetime(2024, 1, 2, 14, 45),
        datetime(2024, 1, 2, 14, 50),
    ]
    return pl.DataFrame(
        {
            "ts": [
                *first_day,
                datetime(2024, 1, 3, 14, 40),
                datetime(2024, 1, 3, 14, 45),
                datetime(2024, 1, 3, 14, 50),
                datetime(2024, 1, 4, 14, 40),
                datetime(2024, 1, 4, 14, 45),
            ]
        }
    )


def pipeline_approval() -> dict[str, object]:
    return {
        "source": {
            "approval_status": "approved",
            "criteria": {"max_gap_ratio": 0.005},
            "gap_audit": {
                "status": "warning",
                "session_aware": True,
                "gap_ratio": 0.002,
            },
        }
    }


def test_trailing_unexecutable_decision_does_not_reduce_coverage(tmp_path: Path) -> None:
    config = timed_config(tmp_path)
    frame = frame_with_trailing_partial_day()
    approval = pipeline_approval()

    enforce_runtime_decision_coverage(config, frame, 5, approval)

    assert runtime_decision_coverage(config, frame, 5) == (1.0, 2, 2)
    assert approval["deferred_trailing_decision_count"] == 1


def test_historical_missing_decision_still_reduces_coverage(tmp_path: Path) -> None:
    coverage = runtime_decision_coverage(
        timed_config(tmp_path),
        frame_with_trailing_partial_day(historical_decision_missing=True),
        5,
    )

    assert coverage == (0.5, 2, 1)
