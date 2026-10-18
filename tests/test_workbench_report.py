from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from backtest_engine.workbench.report import (
    equity_frame,
    extract_plot_series,
    frame_records,
    position_frame,
    table_records,
)


def test_report_tables_are_not_limited_like_chart_series() -> None:
    frame = pl.DataFrame({"decision": list(range(284))})

    assert len(frame_records(frame, 200)) < frame.height
    assert len(table_records(frame)) == frame.height
    assert table_records(frame)[-1]["decision"] == 283


def test_equity_sampling_preserves_full_series_drawdown_extreme() -> None:
    start = datetime(2026, 1, 1)
    equity = [100.0] * 1000
    equity[501] = 10.0
    frame = pl.DataFrame(
        {
            "ts": [start + timedelta(minutes=index) for index in range(1000)],
            "equity_cash": equity,
        }
    )

    records = frame_records(
        equity_frame(frame, initial_capital=100.0), 50, extrema_column="drawdown"
    )

    assert len(records) <= 50
    assert min(record["drawdown"] for record in records) == -0.9
    assert records[-1]["ts"] == (start + timedelta(minutes=999)).isoformat()


def test_position_sampling_keeps_target_transitions_and_last_row() -> None:
    frame = pl.DataFrame(
        {
            "ts": list(range(10)),
            "target_position": [0, 0, 1, 1, 1, -1, -1, -1, 0, 0],
            "target_contracts": [0, 0, 30, 30, 30, -30, -30, -30, 0, 0],
        }
    )

    records = frame_records(
        frame,
        50,
        transition_columns=("target_position", "target_contracts"),
    )

    assert [record["ts"] for record in records] == [0, 2, 5, 8, 9]
    assert [record["target_contracts"] for record in records] == [0, 30, -30, 0, 0]


def test_report_series_use_realization_and_fill_timestamps() -> None:
    frame = pl.DataFrame(
        {
            "ts": [datetime(2026, 7, 22, 13, 30)],
            "fill_ts": [datetime(2026, 7, 22, 13, 35)],
            "next_ts": [datetime(2026, 7, 22, 13, 40)],
            "target_position": [-1.0],
            "target_contracts": [-30.0],
            "equity_cash": [108_118.0],
        }
    )

    assert position_frame(frame)["ts"].item() == datetime(2026, 7, 22, 13, 35)
    assert equity_frame(frame, initial_capital=100_000)["ts"].item() == datetime(
        2026, 7, 22, 13, 40
    )


def test_plot_series_are_bounded_like_other_chart_series() -> None:
    start = datetime(2026, 1, 1)
    events = pl.DataFrame(
        {
            "ts": [start + timedelta(minutes=index) for index in range(100)],
            "strategy_features": ['{"plots":[{"name":"sma","type":"line","value":1}]}'] * 100,
        }
    )

    plots = extract_plot_series(events, max_points=10)

    assert len(plots["sma"]) <= 10
    assert plots["sma"][-1]["ts"] == (start + timedelta(minutes=99)).isoformat()
