from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from backtest_engine.research.validation.config import ResearchValidationConfig
from backtest_engine.research.validation.rolling import rolling_validation


def test_rolling_selection_never_uses_test_or_locked_holdout(tmp_path: Path) -> None:
    runs = [
        ({"run_id": "stable"}, event_frame([0.01] * 8 + [0.02] * 2)),
        ({"run_id": "overfit"}, event_frame([0.03] * 4 + [-0.04] * 4 + [0.50] * 2)),
    ]
    config = ResearchValidationConfig(
        experiment_path=tmp_path,
        locked_holdout_start=date(2024, 1, 9),
        rolling_folds=2,
        rolling_min_train_dates=4,
        purge_dates=1,
    )

    folds, holdout, warnings = rolling_validation(runs, config)

    assert not warnings
    assert folds.height == 2
    assert folds.get_column("test_end").max() < "2024-01-09"
    assert holdout.row(0, named=True)["selected_run_id"] == "stable"
    assert holdout.row(0, named=True)["locked"] is True
    assert holdout.row(0, named=True)["holdout_start"] == "2024-01-09"


def test_rolling_can_use_fixed_training_window(tmp_path: Path) -> None:
    runs = [({"run_id": "one"}, event_frame([0.01] * 12))]
    config = ResearchValidationConfig(
        experiment_path=tmp_path,
        locked_holdout_start=date(2024, 1, 11),
        rolling_folds=2,
        rolling_min_train_dates=4,
        rolling_train_dates=3,
    )

    folds, _holdout, _warnings = rolling_validation(runs, config)

    for row in folds.iter_rows(named=True):
        train_start = date.fromisoformat(row["train_start"])
        train_end = date.fromisoformat(row["train_end"])
        train_days = (train_end - train_start).days
        assert train_days <= 2


def event_frame(returns: list[float]) -> pl.DataFrame:
    timestamps = [datetime(2024, 1, 1) + timedelta(days=index) for index in range(len(returns))]
    return pl.DataFrame(
        {
            "ts": timestamps,
            "next_ts": [value + timedelta(days=1) for value in timestamps],
            "net_return": returns,
            "net_pnl": returns,
            "cost_cash": [0.0] * len(returns),
            "order_status": ["filled"] * len(returns),
        }
    )
