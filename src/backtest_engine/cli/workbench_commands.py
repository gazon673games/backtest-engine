from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.registry import get_adapter
from backtest_engine.cli.common import CLI_EXCEPTIONS, raise_cli_error
from backtest_engine.cli.parsing import parse_params
from backtest_engine.workbench.models import WorkbenchRunRequest
from backtest_engine.workbench.raw_compaction import compact_workbench_raw
from backtest_engine.workbench.runner import run_workbench_backtest
from backtest_engine.workbench.server import serve_workbench
from backtest_engine.workbench.settings import (
    datasets_root,
    load_workbench_settings,
    runs_root,
)
from backtest_engine.workbench.storage_cleanup import cleanup_workbench_storage


def register(app: typer.Typer) -> None:
    @app.command(name="workbench-clean-storage")
    def workbench_clean_storage(
        data_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
        apply: Annotated[bool, typer.Option(help="Delete candidates; otherwise dry-run.")] = False,
    ) -> None:
        try:
            report = cleanup_workbench_storage(data_root, apply=apply)
            compaction = compact_workbench_raw(data_root, apply=apply)
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"mode={'apply' if report.applied else 'dry-run'}")
        typer.echo(f"data_root={report.root}")
        for category, (count, size) in sorted(report.category_totals().items()):
            typer.echo(f"{category}_files={count}")
            typer.echo(f"{category}_bytes={size}")
        typer.echo(f"candidate_files={len(report.candidates)}")
        typer.echo(f"candidate_bytes={report.total_bytes}")
        typer.echo(f"raw_compression_files={len(compaction.candidates)}")
        typer.echo(f"raw_compression_source_bytes={compaction.source_bytes}")
        if apply:
            typer.echo(f"raw_compression_target_bytes={compaction.target_bytes}")
            typer.echo(f"raw_compression_saved_bytes={compaction.saved_bytes}")

    @app.command(name="workbench-run")
    def workbench_run(
        dataset_path: Annotated[Path, typer.Option()],
        script_path: Annotated[Path, typer.Option()],
        param: Annotated[list[str] | None, typer.Option("--param")] = None,
        decision_time: Annotated[list[str] | None, typer.Option("--decision-time")] = None,
        from_date: Annotated[str, typer.Option()] = "0001-01-01",
        output_root: Annotated[Path | None, typer.Option()] = None,
        run_id: Annotated[str | None, typer.Option()] = None,
    ) -> None:
        try:
            settings = load_workbench_settings()
            response = run_workbench_backtest(
                WorkbenchRunRequest(
                    dataset_path=dataset_path,
                    script_path=script_path,
                    params=parse_params(param or []),
                    decision_times=decision_time or [],
                    schedule="times" if decision_time else "every-bar",
                    from_date=from_date,
                    output_root=output_root or runs_root(settings),
                    run_id=run_id,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"status={response.status}")
        typer.echo(f"run_id={response.run_id}")
        typer.echo(f"output_dir={response.output_dir}")
        typer.echo(f"total_return={response.report['metrics'].get('total_return')}")

    @app.command(name="workbench-serve")
    def workbench_serve(
        host: Annotated[str | None, typer.Option()] = None,
        port: Annotated[int | None, typer.Option(min=1, max=65535)] = None,
        output_root: Annotated[Path | None, typer.Option()] = None,
        settings_path: Annotated[Path | None, typer.Option()] = None,
    ) -> None:
        settings = load_workbench_settings(settings_path)
        resolved_host = host or settings.server.host
        resolved_port = port or settings.server.port
        typer.echo(f"workbench_url=http://{resolved_host}:{resolved_port}")
        serve_workbench(
            host=resolved_host,
            port=resolved_port,
            output_root=output_root or runs_root(settings),
        )

    @app.command(name="workbench-build-dataset")
    def workbench_build_dataset(
        security: Annotated[str, typer.Option()],
        from_date: Annotated[str, typer.Option()],
        till_date: Annotated[str, typer.Option()],
        adapter_id: Annotated[str, typer.Option()],
        instrument_mode: Annotated[str, typer.Option()] = "security",
        timeframe: Annotated[str, typer.Option()] = "5m",
        engine: Annotated[str | None, typer.Option()] = None,
        market: Annotated[str | None, typer.Option()] = None,
        board: Annotated[str | None, typer.Option()] = None,
        cache_policy: Annotated[str, typer.Option()] = "prefer",
        roll_rule: Annotated[str | None, typer.Option()] = None,
        output_root: Annotated[Path | None, typer.Option()] = None,
    ) -> None:
        try:
            settings = load_workbench_settings()
            result = get_adapter(adapter_id).build_dataset(
                DatasetBuildRequest(
                    adapter_id=adapter_id,
                    security=security,
                    instrument_mode=instrument_mode,
                    from_date=from_date,
                    till_date=till_date,
                    timeframe=timeframe,
                    engine=engine,
                    market=market,
                    board=board,
                    cache_policy=cache_policy,
                    roll_rule=roll_rule,
                    output_root=output_root or datasets_root(settings),
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"dataset_path={result.dataset_path}")
        typer.echo(f"approval_status={result.approval_status}")
        typer.echo(f"row_count={result.row_count}")
