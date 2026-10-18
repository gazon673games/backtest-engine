from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from backtest_engine.cli import app
from backtest_engine.reporting.tearsheet.report import (
    IntradayEventTearsheetConfig,
    generate_intraday_event_tearsheet,
)

runner = CliRunner()


def test_intraday_event_tearsheet_writes_run_analytics(tmp_path: Path) -> None:
    run_path = write_run_fixture(tmp_path)

    result = generate_intraday_event_tearsheet(IntradayEventTearsheetConfig(run_path=run_path))
    monthly = pl.read_parquet(result.monthly_returns_path.with_suffix(".parquet"))
    costs = pl.read_parquet(result.cost_breakdown_path.with_suffix(".parquet"))
    distribution = pl.read_parquet(result.trade_distribution_path.with_suffix(".parquet"))

    assert result.report_path.exists()
    assert monthly.height == 2
    assert costs.row(0, named=True)["total_cost_cash"] == 3.0
    assert distribution.get_column("direction").to_list() == ["long", "short"]
    assert "Cost Breakdown" in result.report_path.read_text(encoding="utf-8")


def test_cli_intraday_event_tearsheet_smoke(tmp_path: Path) -> None:
    run_path = write_run_fixture(tmp_path)
    output_path = tmp_path / "tearsheet-out"

    result = runner.invoke(
        app,
        [
            "intraday-event-tearsheet",
            "--run-path",
            str(run_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert f"output_path={output_path}" in result.stdout
    assert (output_path / "tearsheet.md").exists()


def write_run_fixture(tmp_path: Path) -> Path:
    run_path = tmp_path / "run"
    run_path.mkdir()
    pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 31, 13, 30),
                datetime(2024, 2, 1, 13, 30),
                datetime(2024, 2, 2, 13, 30),
            ],
            "next_ts": [
                datetime(2024, 2, 1, 13, 30),
                datetime(2024, 2, 2, 13, 30),
                datetime(2024, 2, 5, 13, 30),
            ],
            "net_return": [0.02, -0.01, 0.03],
            "equity": [1.02, 1.0098, 1.040094],
            "equity_cash": [1020.0, 1009.8, 1040.094],
            "gross_pnl": [25.0, -8.0, 35.0],
            "net_pnl": [20.0, -10.2, 30.294],
            "cost_cash": [1.0, 1.0, 1.0],
            "commission_cost_cash": [0.3, 0.3, 0.3],
            "slippage_cost_cash": [0.2, 0.2, 0.2],
            "spread_cost_cash": [0.4, 0.4, 0.4],
            "roll_cost_cash": [0.1, 0.1, 0.1],
            "turnover_notional": [1000.0, 500.0, 1000.0],
        }
    ).write_parquet(run_path / "events.parquet")
    pl.DataFrame(
        {
            "direction": ["long", "short"],
            "net_return": [0.02, -0.01],
            "net_pnl": [20.0, -10.0],
        }
    ).write_parquet(run_path / "trades.parquet")
    return run_path
