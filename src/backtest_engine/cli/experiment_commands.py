from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from backtest_engine.cli.common import CLI_EXCEPTIONS, raise_cli_error
from backtest_engine.cli.parsing import parse_output_format
from backtest_engine.intraday.event.reporting.report import (
    IntradayEventReportConfig,
    generate_intraday_event_report,
)
from backtest_engine.research.experiments.registry import (
    ExperimentCompareConfig,
    ExperimentRegistryConfig,
    build_experiment_registry,
    generate_experiment_compare_report,
)


def register(app: typer.Typer) -> None:
    @app.command(name="experiment-index")
    def experiment_index(
        root: Annotated[Path, typer.Option()] = Path("data/backtests"),
        output_path: Annotated[Path | None, typer.Option()] = None,
        output_format: Annotated[str, typer.Option()] = "all",
    ) -> None:
        try:
            result = build_experiment_registry(
                ExperimentRegistryConfig(
                    root=root,
                    output_path=output_path or (root / "experiment_registry"),
                    output_format=parse_output_format(output_format),
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"root={result.root}")
        typer.echo(f"output_path={result.output_path}")
        typer.echo(f"registry_path={result.registry_path}")
        typer.echo(f"run_count={result.run_count}")
        typer.echo(f"successful_run_count={result.successful_run_count}")
        typer.echo(f"failed_run_count={result.failed_run_count}")
        typer.echo(f"experiment_count={result.experiment_count}")
        typer.echo(f"warning_count={len(result.warnings)}")

    @app.command(name="experiment-compare")
    def experiment_compare(
        registry_path: Annotated[Path | None, typer.Option()] = None,
        root: Annotated[Path | None, typer.Option()] = None,
        output_path: Annotated[Path, typer.Option()] = Path("data/backtests/experiment_compare"),
        sort_by: Annotated[str, typer.Option()] = "sharpe",
        top: Annotated[int, typer.Option(min=1)] = 20,
        descending: Annotated[bool, typer.Option("--descending/--ascending")] = True,
        group_by: Annotated[list[str] | None, typer.Option("--group-by")] = None,
        output_format: Annotated[str, typer.Option()] = "all",
    ) -> None:
        try:
            result = generate_experiment_compare_report(
                ExperimentCompareConfig(
                    registry_path=registry_path,
                    root=root,
                    output_path=output_path,
                    sort_by=sort_by,
                    top=top,
                    descending=descending,
                    group_by=group_by or [],
                    output_format=parse_output_format(output_format),
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_path={result.output_path}")
        typer.echo(f"registry_path={result.registry_path}")
        typer.echo(f"top_run_count={result.top_run_count}")
        typer.echo(f"successful_run_count={result.successful_run_count}")
        typer.echo(f"failed_run_count={result.failed_run_count}")
        if result.report_path is not None:
            typer.echo(f"report_path={result.report_path}")
        typer.echo(f"warning_count={len(result.warnings)}")

    @app.command(name="intraday-event-report")
    def intraday_event_report(
        experiment_path: Annotated[Path, typer.Option()],
        sort_by: Annotated[str, typer.Option()] = "sharpe",
        top: Annotated[int, typer.Option(min=1)] = 20,
        descending: Annotated[bool, typer.Option("--descending/--ascending")] = True,
        charts: Annotated[bool, typer.Option("--charts/--no-charts")] = True,
    ) -> None:
        try:
            result = generate_intraday_event_report(
                IntradayEventReportConfig(
                    experiment_path=experiment_path,
                    sort_by=sort_by,
                    top=top,
                    descending=descending,
                    charts=charts,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"experiment_path={result.experiment_path}")
        typer.echo(f"top_run_count={result.top_run_count}")
        typer.echo(f"failed_run_count={result.failed_run_count}")
        typer.echo(f"report_path={result.report_path}")
        typer.echo(f"chart_count={len(result.chart_paths)}")
