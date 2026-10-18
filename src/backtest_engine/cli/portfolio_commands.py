from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from backtest_engine.cli.common import CLI_EXCEPTIONS, raise_cli_error
from backtest_engine.cli.parsing import (
    parse_date,
    parse_intraday_event_timing,
    parse_multi_instrument_leg,
    parse_params,
)
from backtest_engine.portfolio.combine import (
    PortfolioCombineConfig,
    combine_intraday_event_portfolio,
)
from backtest_engine.portfolio.multi.engine import (
    MultiInstrumentBacktestConfig,
    MultiInstrumentExecutionConfig,
    MultiInstrumentOutputConfig,
    MultiInstrumentPortfolioConfig,
    MultiInstrumentStrategyConfig,
    run_multi_instrument_backtest,
)


def register(app: typer.Typer) -> None:
    @app.command(name="portfolio-run-config")
    def portfolio_run_config(
        config_path: Annotated[Path, typer.Option(exists=True)],
    ) -> None:
        """Run a frozen native multi-instrument configuration, including SDK sleeves."""
        try:
            config = MultiInstrumentBacktestConfig.model_validate_json(
                config_path.read_text(encoding="utf-8")
            )
            result = run_multi_instrument_backtest(config)
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_dir={result.output_dir}")
        typer.echo(f"final_equity_cash={result.metrics['final_equity_cash']:.12g}")

    @app.command(name="portfolio-combine")
    def portfolio_combine(
        run_path: Annotated[list[Path], typer.Option("--run-path")],
        output_path: Annotated[Path, typer.Option()],
        initial_capital: Annotated[float | None, typer.Option()] = None,
    ) -> None:
        try:
            result = combine_intraday_event_portfolio(
                PortfolioCombineConfig(
                    run_paths=run_path,
                    output_path=output_path,
                    initial_capital=initial_capital,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_path={result.output_path}")
        typer.echo(f"run_count={result.run_count}")
        typer.echo(f"event_count={result.event_count}")
        typer.echo(f"final_equity_cash={result.final_equity_cash:.12g}")
        typer.echo(f"total_return={result.total_return:.12g}")

    @app.command(name="multi-instrument-run")
    def multi_instrument_run(
        leg: Annotated[list[str], typer.Option("--leg")],
        strategy_file: Annotated[Path, typer.Option()],
        param: Annotated[list[str] | None, typer.Option("--param")] = None,
        from_date: Annotated[str, typer.Option()] = "0001-01-01",
        initial_capital: Annotated[float, typer.Option(min=0.0)] = 1.0,
        sizing_mode: Annotated[str, typer.Option()] = "fixed_contracts",
        fixed_contracts: Annotated[float, typer.Option(min=0.0)] = 1.0,
        target_notional: Annotated[float | None, typer.Option()] = None,
        max_leverage: Annotated[float | None, typer.Option()] = None,
        max_position_notional: Annotated[float | None, typer.Option()] = None,
        max_margin_usage: Annotated[float | None, typer.Option()] = None,
        contract_rounding: Annotated[str, typer.Option()] = "floor",
        risk_policy: Annotated[str, typer.Option()] = "reject",
        execution_timing: Annotated[str, typer.Option()] = "signal_close_to_next_bar_close",
        commission_bps: Annotated[float, typer.Option(min=0.0)] = 0.0,
        slippage_bps: Annotated[float, typer.Option(min=0.0)] = 0.0,
        slippage_ticks: Annotated[float, typer.Option(min=0.0)] = 0.0,
        bid_ask_spread_bps: Annotated[float, typer.Option(min=0.0)] = 0.0,
        bid_ask_spread_ticks: Annotated[float, typer.Option(min=0.0)] = 0.0,
        roll_cost_bps: Annotated[float, typer.Option(min=0.0)] = 0.0,
        roll_slippage_ticks: Annotated[float, typer.Option(min=0.0)] = 0.0,
        roll_pnl_policy: Annotated[str, typer.Option()] = "reject",
        output_root: Annotated[Path, typer.Option()] = Path("data/backtests"),
        run_id: Annotated[str, typer.Option(help="Deterministic run identifier.")] = ...,
    ) -> None:
        try:
            result = run_multi_instrument_backtest(
                MultiInstrumentBacktestConfig(
                    legs=[parse_multi_instrument_leg(value) for value in leg],
                    strategy=MultiInstrumentStrategyConfig(
                        strategy_file=strategy_file,
                        strategy_params=parse_params(param or []),
                    ),
                    portfolio=MultiInstrumentPortfolioConfig(
                        initial_capital=initial_capital,
                        sizing_mode=sizing_mode,
                        fixed_contracts=fixed_contracts,
                        target_notional=target_notional,
                        contract_rounding=contract_rounding,
                        max_position_notional=max_position_notional,
                        max_leverage=max_leverage,
                        max_margin_usage=max_margin_usage,
                        risk_policy=risk_policy,
                    ),
                    execution=MultiInstrumentExecutionConfig(
                        timing=parse_intraday_event_timing(execution_timing),
                        from_date=parse_date(from_date),
                        commission_bps=commission_bps,
                        slippage_bps=slippage_bps,
                        slippage_ticks=slippage_ticks,
                        bid_ask_spread_bps=bid_ask_spread_bps,
                        bid_ask_spread_ticks=bid_ask_spread_ticks,
                        roll_cost_bps=roll_cost_bps,
                        roll_slippage_ticks=roll_slippage_ticks,
                        roll_pnl_policy=roll_pnl_policy,
                    ),
                    output=MultiInstrumentOutputConfig(root=output_root, run_id=run_id),
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_dir={result.output_dir}")
        typer.echo(f"asset_count={result.metrics['asset_count']}")
        typer.echo(f"event_count={result.metrics['event_count']}")
        typer.echo(f"final_equity_cash={result.metrics['final_equity_cash']:.12g}")
        typer.echo(f"total_return={result.metrics['total_return']:.12g}")
