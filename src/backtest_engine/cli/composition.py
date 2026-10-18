"""Composes all thin Typer command adapters into the public CLI."""

from __future__ import annotations

import typer

from backtest_engine.cli.daily_commands import register as register_daily_commands
from backtest_engine.cli.data_commands import register as register_data_commands
from backtest_engine.cli.event_batch_command import register as register_event_batch_command
from backtest_engine.cli.event_command import register as register_event_command
from backtest_engine.cli.experiment_commands import register as register_experiment_commands
from backtest_engine.cli.intraday_commands import register as register_intraday_commands
from backtest_engine.cli.js_commands import register as register_js_commands
from backtest_engine.cli.paper_commands import register as register_paper_commands
from backtest_engine.cli.portfolio_commands import register as register_portfolio_commands
from backtest_engine.cli.research_commands import register as register_research_commands
from backtest_engine.cli.robustness_commands import register as register_robustness_commands
from backtest_engine.cli.workbench_commands import register as register_workbench_commands

app = typer.Typer(help="Run daily and intraday backtests on prepared market datasets.")

register_daily_commands(app)
register_intraday_commands(app)
register_event_command(app)
register_event_batch_command(app)
register_experiment_commands(app)
register_research_commands(app)
register_robustness_commands(app)
register_portfolio_commands(app)
register_data_commands(app)
register_js_commands(app)
register_paper_commands(app)
register_workbench_commands(app)

__all__ = ["app"]
