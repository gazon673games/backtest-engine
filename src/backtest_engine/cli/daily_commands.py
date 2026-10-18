from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from backtest_engine.cli.builders import build_batch_config, build_run_config
from backtest_engine.cli.common import CLI_EXCEPTIONS, raise_cli_error
from backtest_engine.cli.parsing import parse_params
from backtest_engine.core.models import (
    BacktestConfig,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
    StrategyConfig,
)
from backtest_engine.daily.batch import run_batch
from backtest_engine.daily.engine import inspect_dataset, run_backtest


def register(app: typer.Typer) -> None:
    @app.command()
    def run(
        dataset_path: Annotated[
            Path | None,
            typer.Option(help="Path to the Parquet dataset."),
        ] = None,
        quality_manifest: Annotated[
            Path | None,
            typer.Option(help="Path to the quality gate manifest."),
        ] = None,
        strategy: Annotated[
            str | None,
            typer.Option(help="Strategy: buy_and_hold or moving_average_crossover."),
        ] = None,
        fast_window: Annotated[int | None, typer.Option()] = None,
        slow_window: Annotated[int | None, typer.Option()] = None,
        commission_bps: Annotated[float | None, typer.Option()] = None,
        slippage_bps: Annotated[float | None, typer.Option()] = None,
        output_root: Annotated[Path | None, typer.Option()] = None,
        run_id: Annotated[str | None, typer.Option()] = None,
        write_csv: Annotated[bool | None, typer.Option()] = None,
        config: Annotated[Path | None, typer.Option(help="TOML config file.")] = None,
    ) -> None:
        try:
            result = run_backtest(
                build_run_config(
                    config=config,
                    dataset_path=dataset_path,
                    quality_manifest=quality_manifest,
                    strategy=strategy,
                    fast_window=fast_window,
                    slow_window=slow_window,
                    commission_bps=commission_bps,
                    slippage_bps=slippage_bps,
                    output_root=output_root,
                    run_id=run_id,
                    write_csv=write_csv,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_dir={result.output_dir}")
        typer.echo(f"status={result.dataset_quality.status}")
        typer.echo(f"total_return={result.metrics.total_return:.12g}")
        typer.echo(f"max_drawdown={result.metrics.max_drawdown:.12g}")

    @app.command(name="batch-run")
    def batch_run(
        dataset_path: Annotated[
            list[Path] | None,
            typer.Option("--dataset-path", help="Path to a Parquet dataset. Repeatable."),
        ] = None,
        quality_manifest: Annotated[Path | None, typer.Option()] = None,
        strategy_file: Annotated[Path | None, typer.Option()] = None,
        grid: Annotated[list[str] | None, typer.Option("--grid")] = None,
        constraint: Annotated[list[str] | None, typer.Option("--constraint")] = None,
        commission_bps: Annotated[float | None, typer.Option()] = None,
        slippage_bps: Annotated[float | None, typer.Option()] = None,
        output_root: Annotated[Path | None, typer.Option()] = None,
        experiment_id: Annotated[str | None, typer.Option()] = None,
        fail_fast: Annotated[bool, typer.Option()] = False,
        config: Annotated[Path | None, typer.Option(help="TOML batch config file.")] = None,
    ) -> None:
        try:
            result = run_batch(
                build_batch_config(
                    config=config,
                    dataset_paths=dataset_path,
                    quality_manifest=quality_manifest,
                    strategy_file=strategy_file,
                    grid_options=grid,
                    constraints=constraint,
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

    @app.command(name="run-script")
    def run_script(
        dataset_path: Annotated[Path, typer.Option(help="Path to the Parquet dataset.")],
        quality_manifest: Annotated[Path, typer.Option(help="Path to the quality gate manifest.")],
        strategy_file: Annotated[Path, typer.Option(help="Path to a trusted Python strategy.")],
        param: Annotated[list[str] | None, typer.Option("--param")] = None,
        commission_bps: Annotated[float, typer.Option()] = 0.0,
        slippage_bps: Annotated[float, typer.Option()] = 0.0,
        output_root: Annotated[Path, typer.Option()] = Path("data/backtests"),
        run_id: Annotated[str, typer.Option(help="Deterministic run identifier.")] = ...,
        write_csv: Annotated[bool, typer.Option()] = False,
    ) -> None:
        try:
            result = run_backtest(
                BacktestConfig(
                    dataset=DatasetConfig(path=dataset_path, quality_manifest=quality_manifest),
                    strategy=StrategyConfig(
                        name="plugin",
                        strategy_file=strategy_file,
                        strategy_params=parse_params(param or []),
                    ),
                    execution=ExecutionConfig(
                        commission_bps=commission_bps,
                        slippage_bps=slippage_bps,
                    ),
                    output=OutputConfig(root=output_root, run_id=run_id, write_csv=write_csv),
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_dir={result.output_dir}")
        typer.echo(f"status={result.dataset_quality.status}")
        typer.echo(f"strategy_file={strategy_file}")
        typer.echo(f"total_return={result.metrics.total_return:.12g}")
        typer.echo(f"max_drawdown={result.metrics.max_drawdown:.12g}")

    @app.command(name="inspect-dataset")
    def inspect_dataset_command(
        dataset_path: Annotated[Path, typer.Option(help="Path to the Parquet dataset.")],
        quality_manifest: Annotated[Path, typer.Option(help="Path to the quality manifest.")],
    ) -> None:
        try:
            inspection = inspect_dataset(dataset_path, quality_manifest)
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        metadata = inspection.metadata
        quality = inspection.quality
        typer.echo(f"dataset_path={metadata.path}")
        typer.echo(f"continuous_symbol={metadata.continuous_symbol}")
        typer.echo(f"adjustment_mode={metadata.adjustment_mode}")
        typer.echo(f"row_count={metadata.row_count}")
        typer.echo(f"min_ts={metadata.min_ts}")
        typer.echo(f"max_ts={metadata.max_ts}")
        typer.echo(f"quality_gate_status={quality.status}")
        typer.echo(f"approved_for_backtest={str(quality.approved_for_backtest).lower()}")
