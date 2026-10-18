from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from backtest_engine.cli.common import CLI_EXCEPTIONS, raise_cli_error
from backtest_engine.cli.parsing import parse_date
from backtest_engine.reporting.tearsheet.report import (
    IntradayEventTearsheetConfig,
    generate_intraday_event_tearsheet,
)
from backtest_engine.research.benchmark import DatasetBenchmarkConfig, benchmark_dataset
from backtest_engine.research.external_validation import (
    ExternalValidationConfig,
    validate_against_statement,
)
from backtest_engine.research.pipeline import (
    IntradayEventResearchPipelineConfig,
    run_intraday_event_research_pipeline,
)
from backtest_engine.research.strategy_audit import StrategyAuditConfig, audit_strategy
from backtest_engine.research.validation.gate import (
    ResearchValidationConfig,
    generate_research_validation_report,
)


def register(app: typer.Typer) -> None:
    @app.command(name="strategy-audit")
    def strategy_audit_command(
        strategy_file: Annotated[Path, typer.Option()],
        dataset_path: Annotated[Path, typer.Option()],
        output_path: Annotated[Path, typer.Option()],
        mode: Annotated[str, typer.Option()] = "intraday_event",
        sample_count: Annotated[int, typer.Option(min=1)] = 32,
        gate: Annotated[bool, typer.Option("--gate/--report-only")] = False,
    ) -> None:
        try:
            result = audit_strategy(
                StrategyAuditConfig(
                    strategy_file=strategy_file,
                    dataset_path=dataset_path,
                    output_path=output_path,
                    mode=mode,
                    sample_count=sample_count,
                    gate=gate,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_path={result.output_path}")
        typer.echo(f"status={result.status}")
        typer.echo(f"lookahead_mismatches={result.lookahead_mismatches}")
        typer.echo(f"recursive_mismatches={result.recursive_mismatches}")

    @app.command(name="validate-external")
    def validate_external(
        backtest_events_path: Annotated[Path, typer.Option()],
        statement_path: Annotated[Path, typer.Option()],
        output_path: Annotated[Path, typer.Option()],
        tolerance: Annotated[float, typer.Option(min=0.0)] = 0.01,
    ) -> None:
        try:
            result = validate_against_statement(
                ExternalValidationConfig(
                    backtest_events_path=backtest_events_path,
                    statement_path=statement_path,
                    output_path=output_path,
                    tolerance=tolerance,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_path={result.output_path}")
        typer.echo(f"status={result.status}")
        typer.echo(f"break_count={result.break_count}")

    @app.command(name="benchmark-dataset")
    def benchmark_dataset_command(
        dataset_path: Annotated[Path, typer.Option()],
        output_path: Annotated[Path, typer.Option()],
        iterations: Annotated[int, typer.Option(min=1)] = 3,
    ) -> None:
        try:
            result = benchmark_dataset(
                DatasetBenchmarkConfig(
                    dataset_path=dataset_path,
                    output_path=output_path,
                    iterations=iterations,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_path={result.output_path}")
        typer.echo(f"row_count={result.row_count}")
        typer.echo(f"best_seconds={result.best_seconds}")
        typer.echo(f"rows_per_second={result.rows_per_second}")

    @app.command(name="intraday-event-research-report")
    def intraday_event_research_report(
        experiment_path: Annotated[Path, typer.Option()],
        output_path: Annotated[Path | None, typer.Option()] = None,
        split_date: Annotated[str | None, typer.Option()] = None,
        rolling_folds: Annotated[int, typer.Option(min=1)] = 3,
        rolling_min_train_dates: Annotated[int, typer.Option(min=2)] = 20,
        rolling_train_dates: Annotated[int | None, typer.Option(min=2)] = None,
        purge_dates: Annotated[int, typer.Option(min=0)] = 0,
        embargo_dates: Annotated[int, typer.Option(min=0)] = 0,
        locked_holdout_start: Annotated[str | None, typer.Option()] = None,
        locked_holdout_fraction: Annotated[float, typer.Option(min=0.01, max=0.49)] = 0.2,
        selection_metric: Annotated[str, typer.Option()] = "total_return",
        group_column: Annotated[list[str] | None, typer.Option("--group-column")] = None,
        bootstrap_iterations: Annotated[int, typer.Option(min=0)] = 200,
        block_bootstrap_iterations: Annotated[int, typer.Option(min=0)] = 100,
        block_bootstrap_length: Annotated[int, typer.Option(min=1)] = 5,
        cost_stress_bps: Annotated[list[float] | None, typer.Option("--cost-stress-bps")] = None,
        max_parameter_runs_without_warning: Annotated[int, typer.Option(min=1)] = 50,
        gate: Annotated[bool, typer.Option("--gate/--report-only")] = False,
        max_negative_oos_run_share: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
        max_warning_count: Annotated[int, typer.Option(min=0)] = 0,
    ) -> None:
        try:
            result = generate_research_validation_report(
                ResearchValidationConfig(
                    experiment_path=experiment_path,
                    output_path=output_path,
                    split_date=parse_date(split_date) if split_date is not None else None,
                    rolling_folds=rolling_folds,
                    rolling_min_train_dates=rolling_min_train_dates,
                    rolling_train_dates=rolling_train_dates,
                    purge_dates=purge_dates,
                    embargo_dates=embargo_dates,
                    locked_holdout_start=(
                        parse_date(locked_holdout_start)
                        if locked_holdout_start is not None
                        else None
                    ),
                    locked_holdout_fraction=locked_holdout_fraction,
                    selection_metric=selection_metric,
                    group_columns=group_column or [],
                    bootstrap_iterations=bootstrap_iterations,
                    block_bootstrap_iterations=block_bootstrap_iterations,
                    block_bootstrap_length=block_bootstrap_length,
                    cost_stress_bps=cost_stress_bps or [1.0, 2.0, 5.0],
                    max_parameter_runs_without_warning=max_parameter_runs_without_warning,
                    gate=gate,
                    max_negative_oos_run_share=max_negative_oos_run_share,
                    max_warning_count=max_warning_count,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_path={result.output_path}")
        typer.echo(f"run_count={result.run_count}")
        typer.echo(f"warning_count={result.warning_count}")
        typer.echo(f"acceptance_status={result.acceptance_status}")
        typer.echo(f"gate_path={result.gate_path}")
        typer.echo(f"walk_forward_path={result.walk_forward_path}")
        typer.echo(f"rolling_walk_forward_path={result.rolling_walk_forward_path}")
        typer.echo(f"locked_holdout_path={result.locked_holdout_path}")
        typer.echo(f"robustness_path={result.robustness_path}")
        typer.echo(f"block_bootstrap_path={result.block_bootstrap_path}")
        typer.echo(f"cost_stress_path={result.cost_stress_path}")
        typer.echo(f"regime_path={result.regime_path}")
        typer.echo(f"report_path={result.report_path}")
        typer.echo(f"manifest_path={result.manifest_path}")

    @app.command(name="intraday-event-tearsheet")
    def intraday_event_tearsheet(
        run_path: Annotated[Path, typer.Option()],
        output_path: Annotated[Path | None, typer.Option()] = None,
    ) -> None:
        try:
            result = generate_intraday_event_tearsheet(
                IntradayEventTearsheetConfig(run_path=run_path, output_path=output_path)
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_path={result.output_path}")
        typer.echo(f"monthly_returns_path={result.monthly_returns_path}")
        typer.echo(f"drawdown_periods_path={result.drawdown_periods_path}")
        typer.echo(f"cost_breakdown_path={result.cost_breakdown_path}")
        typer.echo(f"trade_distribution_path={result.trade_distribution_path}")
        typer.echo(f"report_path={result.report_path}")

    @app.command(name="intraday-event-pipeline")
    def intraday_event_pipeline(
        config: Annotated[Path, typer.Option(help="TOML intraday event batch config file.")],
        catalog_enforcement: Annotated[str, typer.Option()] = "strict",
        sort_by: Annotated[str, typer.Option()] = "sharpe",
        top: Annotated[int, typer.Option(min=1)] = 20,
        charts: Annotated[bool, typer.Option("--charts/--no-charts")] = True,
        split_date: Annotated[str | None, typer.Option()] = None,
        strategy_audit_policy: Annotated[str, typer.Option()] = "strict",
    ) -> None:
        try:
            result = run_intraday_event_research_pipeline(
                IntradayEventResearchPipelineConfig(
                    config_path=config,
                    catalog_enforcement=catalog_enforcement,
                    report_sort_by=sort_by,
                    report_top=top,
                    report_charts=charts,
                    split_date=parse_date(split_date) if split_date is not None else None,
                    strategy_audit_policy=strategy_audit_policy,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"experiment_path={result.experiment_path}")
        typer.echo(f"successful_run_count={result.batch.successful_run_count}")
        typer.echo(f"failed_run_count={result.batch.failed_run_count}")
        typer.echo(f"report_path={result.report.report_path}")
        typer.echo(f"research_validation_report_path={result.research_validation.report_path}")
        typer.echo(f"manifest_path={result.manifest_path}")
        for path in result.strategy_audit_paths:
            typer.echo(f"strategy_audit_path={path}")
