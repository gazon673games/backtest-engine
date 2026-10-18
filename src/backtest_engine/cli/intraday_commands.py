from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from backtest_engine.cli.builders import build_intraday_batch_config
from backtest_engine.cli.common import CLI_EXCEPTIONS, raise_cli_error
from backtest_engine.cli.parsing import parse_date, parse_output_format, parse_params, parse_time
from backtest_engine.intraday.batch.runner import run_intraday_batch
from backtest_engine.intraday.engine import (
    IntradayBacktestConfig,
    IntradayDatasetConfig,
    IntradayExecutionConfig,
    IntradayOutputConfig,
    IntradayStrategyConfig,
    run_intraday_backtest,
)
from backtest_engine.reporting.experiment import ReportConfig, generate_experiment_report


def register(app: typer.Typer) -> None:
    @app.command(name="run-intraday-script")
    def run_intraday_script(
        dataset_path: Annotated[
            Path,
            typer.Option(help="Path to the minute nearest-contract Parquet dataset."),
        ],
        strategy_file: Annotated[Path, typer.Option(help="Trusted intraday strategy file.")],
        decision_time: Annotated[str, typer.Option(help="Decision time as HH:MM.")],
        quality_approval_path: Annotated[Path | None, typer.Option()] = None,
        require_quality_approval: Annotated[bool, typer.Option()] = True,
        allow_legacy_bar_timestamps: Annotated[bool, typer.Option()] = False,
        param: Annotated[list[str] | None, typer.Option("--param")] = None,
        from_date: Annotated[str, typer.Option(help="First trading date as YYYY-MM-DD.")] = (
            "0001-01-01"
        ),
        decision_tolerance_minutes: Annotated[int | None, typer.Option(min=0)] = None,
        minimum_bars_since_session_start: Annotated[int, typer.Option(min=0)] = 0,
        commission_bps: Annotated[float, typer.Option(min=0.0)] = 0.0,
        slippage_bps: Annotated[float, typer.Option(min=0.0)] = 0.0,
        output_root: Annotated[Path, typer.Option()] = Path("data/backtests"),
        run_id: Annotated[str, typer.Option(help="Deterministic run identifier.")] = ...,
        write_csv: Annotated[bool, typer.Option()] = False,
    ) -> None:
        try:
            result = run_intraday_backtest(
                IntradayBacktestConfig(
                    dataset=IntradayDatasetConfig(
                        path=dataset_path,
                        quality_approval_path=quality_approval_path,
                        quality_approval_required=require_quality_approval,
                        allow_legacy_bar_timestamps=allow_legacy_bar_timestamps,
                    ),
                    strategy=IntradayStrategyConfig(
                        strategy_file=strategy_file,
                        strategy_params=parse_params(param or []),
                    ),
                    execution=IntradayExecutionConfig(
                        from_date=parse_date(from_date),
                        decision_time=parse_time(decision_time),
                        decision_tolerance_minutes=decision_tolerance_minutes,
                        minimum_bars_since_session_start=minimum_bars_since_session_start,
                        commission_bps=commission_bps,
                        slippage_bps=slippage_bps,
                    ),
                    output=IntradayOutputConfig(
                        root=output_root, run_id=run_id, write_csv=write_csv
                    ),
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_dir={result.output_dir}")
        typer.echo(f"strategy_file={strategy_file}")
        typer.echo(f"decision_count={result.metrics['decision_count']}")
        typer.echo(f"skipped_decision_day_count={result.metrics['skipped_decision_day_count']}")
        typer.echo(f"total_return={result.metrics['total_return']:.12g}")
        typer.echo(f"max_drawdown={result.metrics['max_drawdown']:.12g}")

    @app.command(name="intraday-batch-run")
    def intraday_batch_run(
        dataset_path: Annotated[Path | None, typer.Option()] = None,
        strategy_file: Annotated[Path | None, typer.Option()] = None,
        quality_approval_path: Annotated[Path | None, typer.Option()] = None,
        require_quality_approval: Annotated[bool | None, typer.Option()] = None,
        allow_legacy_bar_timestamps: Annotated[bool | None, typer.Option()] = None,
        grid: Annotated[list[str] | None, typer.Option("--grid")] = None,
        from_date: Annotated[str | None, typer.Option()] = None,
        decision_time: Annotated[str | None, typer.Option()] = None,
        decision_tolerance_minutes: Annotated[int | None, typer.Option(min=0)] = None,
        minimum_bars_since_session_start: Annotated[int | None, typer.Option(min=0)] = None,
        commission_bps: Annotated[float | None, typer.Option(min=0.0)] = None,
        slippage_bps: Annotated[float | None, typer.Option(min=0.0)] = None,
        output_root: Annotated[Path | None, typer.Option()] = None,
        experiment_id: Annotated[str | None, typer.Option()] = None,
        fail_fast: Annotated[bool, typer.Option()] = False,
        config: Annotated[Path | None, typer.Option(help="TOML intraday batch config.")] = None,
    ) -> None:
        try:
            result = run_intraday_batch(
                build_intraday_batch_config(
                    config=config,
                    dataset_path=dataset_path,
                    strategy_file=strategy_file,
                    quality_approval_path=quality_approval_path,
                    require_quality_approval=require_quality_approval,
                    allow_legacy_bar_timestamps=allow_legacy_bar_timestamps,
                    grid_options=grid,
                    from_date=from_date,
                    decision_time=decision_time,
                    decision_tolerance_minutes=decision_tolerance_minutes,
                    minimum_bars_since_session_start=minimum_bars_since_session_start,
                    commission_bps=commission_bps,
                    slippage_bps=slippage_bps,
                    output_root=output_root,
                    experiment_id=experiment_id,
                    fail_fast=fail_fast,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_dir={result.output_dir}")
        typer.echo(f"successful_run_count={result.successful_run_count}")
        typer.echo(f"failed_run_count={result.failed_run_count}")

    @app.command()
    def report(
        experiment_path: Annotated[Path, typer.Option(help="Path to batch experiment.")],
        sort_by: Annotated[str, typer.Option()] = "sharpe",
        top: Annotated[int, typer.Option(min=1)] = 10,
        descending: Annotated[bool, typer.Option("--descending/--ascending")] = True,
        group_by_dataset: Annotated[
            bool,
            typer.Option("--group-by-dataset/--no-group-by-dataset"),
        ] = True,
        output_format: Annotated[str, typer.Option()] = "all",
        charts: Annotated[bool, typer.Option("--charts/--no-charts")] = True,
        chart_metric: Annotated[str | None, typer.Option()] = None,
    ) -> None:
        try:
            result = generate_experiment_report(
                ReportConfig(
                    experiment_path=experiment_path,
                    sort_by=sort_by,
                    top=top,
                    descending=descending,
                    group_by_dataset=group_by_dataset,
                    output_format=parse_output_format(output_format),
                    charts=charts,
                    chart_metric=chart_metric,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"experiment_path={result.experiment_path}")
        typer.echo(f"top_run_count={result.top_run_count}")
        typer.echo(f"failed_run_count={result.failed_run_count}")
        if result.report_path is not None:
            typer.echo(f"report_path={result.report_path}")
