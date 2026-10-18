"""Builds detailed performance and diagnostic tearsheets for event runs."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from pydantic import BaseModel

from backtest_engine.core.exceptions import ReportingError
from backtest_engine.reporting.tearsheet.analytics import (
    cost_breakdown,
    drawdown_periods,
    monthly_returns,
    trade_distribution,
)
from backtest_engine.reporting.tearsheet.io import read_optional_parquet, render_report, write_table


class IntradayEventTearsheetConfig(BaseModel):
    run_path: Path
    output_path: Path | None = None


class IntradayEventTearsheetResult(BaseModel):
    output_path: Path
    monthly_returns_path: Path
    drawdown_periods_path: Path
    cost_breakdown_path: Path
    trade_distribution_path: Path
    report_path: Path


def generate_intraday_event_tearsheet(
    config: IntradayEventTearsheetConfig,
) -> IntradayEventTearsheetResult:
    events = _read_events(config.run_path)
    trades = read_optional_parquet(config.run_path / "trades.parquet")
    output_path = config.output_path or config.run_path / "tearsheet"
    output_path.mkdir(parents=True, exist_ok=True)

    monthly = monthly_returns(events)
    drawdowns = drawdown_periods(events)
    costs = cost_breakdown(events)
    distribution = trade_distribution(trades)

    monthly_returns_path = output_path / "monthly_returns.csv"
    drawdown_periods_path = output_path / "drawdown_periods.csv"
    cost_breakdown_path = output_path / "cost_breakdown.csv"
    trade_distribution_path = output_path / "trade_distribution.csv"
    write_table(monthly_returns_path.with_suffix(""), monthly)
    write_table(drawdown_periods_path.with_suffix(""), drawdowns)
    write_table(cost_breakdown_path.with_suffix(""), costs)
    write_table(trade_distribution_path.with_suffix(""), distribution)

    report_path = output_path / "tearsheet.md"
    report_path.write_text(
        render_report(
            run_path=config.run_path,
            events=events,
            trades=trades,
            monthly_returns=monthly,
            drawdown_periods=drawdowns,
            cost_breakdown=costs,
            trade_distribution=distribution,
        ),
        encoding="utf-8",
    )
    return IntradayEventTearsheetResult(
        output_path=output_path,
        monthly_returns_path=monthly_returns_path,
        drawdown_periods_path=drawdown_periods_path,
        cost_breakdown_path=cost_breakdown_path,
        trade_distribution_path=trade_distribution_path,
        report_path=report_path,
    )


def _read_events(run_path: Path) -> pl.DataFrame:
    events_path = run_path / "events.parquet"
    if not events_path.exists():
        raise ReportingError(f"Missing events.parquet in run path: {run_path}")
    return pl.read_parquet(events_path)
