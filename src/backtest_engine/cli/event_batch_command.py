from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from backtest_engine.cli.common import CLI_EXCEPTIONS, raise_cli_error
from backtest_engine.cli.event_batch_builder import build_intraday_event_batch_config
from backtest_engine.intraday.event.batch.runner import run_intraday_event_batch


def register(app: typer.Typer) -> None:
    @app.command(name="intraday-event-batch-run")
    def intraday_event_batch_run(
        dataset_path: Annotated[
            list[Path] | None,
            typer.Option("--dataset-path", help="Prepared minute dataset. Repeatable."),
        ] = None,
        catalog_path: Annotated[Path | None, typer.Option()] = None,
        dataset_id: Annotated[list[str] | None, typer.Option("--dataset-id")] = None,
        catalog_enforcement: Annotated[str | None, typer.Option()] = None,
        strategy_file: Annotated[Path | None, typer.Option()] = None,
        quality_approval_path: Annotated[Path | None, typer.Option()] = None,
        require_quality_approval: Annotated[
            bool | None,
            typer.Option("--require-quality-approval/--no-require-quality-approval"),
        ] = None,
        min_usable_decision_day_pct: Annotated[float | None, typer.Option()] = None,
        allow_legacy_bar_timestamps: Annotated[bool | None, typer.Option()] = None,
        grid: Annotated[list[str] | None, typer.Option("--grid")] = None,
        from_date: Annotated[str | None, typer.Option()] = None,
        schedule: Annotated[str | None, typer.Option()] = None,
        execution_timing: Annotated[str | None, typer.Option()] = None,
        allow_noncausal_same_close: Annotated[bool | None, typer.Option()] = None,
        end_of_data_position_policy: Annotated[str | None, typer.Option()] = None,
        decision_time: Annotated[list[str] | None, typer.Option("--decision-time")] = None,
        decision_tolerance_minutes: Annotated[int | None, typer.Option(min=0)] = None,
        missing_decision_policy: Annotated[str | None, typer.Option()] = None,
        commission_bps: Annotated[float | None, typer.Option(min=0.0)] = None,
        slippage_bps: Annotated[float | None, typer.Option(min=0.0)] = None,
        bid_ask_spread_bps: Annotated[float | None, typer.Option(min=0.0)] = None,
        bid_ask_spread_ticks: Annotated[float | None, typer.Option(min=0.0)] = None,
        volume_participation_limit: Annotated[float | None, typer.Option(min=0.0)] = None,
        volume_to_contracts_multiplier: Annotated[float | None, typer.Option(min=0.0)] = None,
        partial_fill_policy: Annotated[str | None, typer.Option()] = None,
        intrabar_ambiguity_policy: Annotated[str | None, typer.Option()] = None,
        output_root: Annotated[Path | None, typer.Option()] = None,
        experiment_id: Annotated[str | None, typer.Option()] = None,
        max_workers: Annotated[int | None, typer.Option(min=1, max=64)] = None,
        fail_fast: Annotated[bool, typer.Option()] = False,
        config: Annotated[Path | None, typer.Option(help="TOML config file.")] = None,
    ) -> None:
        try:
            result = run_intraday_event_batch(
                build_intraday_event_batch_config(
                    config=config,
                    dataset_paths=dataset_path,
                    catalog_path=catalog_path,
                    dataset_ids=dataset_id,
                    catalog_enforcement=catalog_enforcement,
                    strategy_file=strategy_file,
                    quality_approval_path=quality_approval_path,
                    require_quality_approval=require_quality_approval,
                    min_usable_decision_day_pct=min_usable_decision_day_pct,
                    allow_legacy_bar_timestamps=allow_legacy_bar_timestamps,
                    grid_options=grid,
                    from_date=from_date,
                    schedule=schedule,
                    execution_timing=execution_timing,
                    allow_noncausal_same_close=allow_noncausal_same_close,
                    end_of_data_position_policy=end_of_data_position_policy,
                    decision_time=decision_time,
                    decision_tolerance_minutes=decision_tolerance_minutes,
                    missing_decision_policy=missing_decision_policy,
                    commission_bps=commission_bps,
                    slippage_bps=slippage_bps,
                    bid_ask_spread_bps=bid_ask_spread_bps,
                    bid_ask_spread_ticks=bid_ask_spread_ticks,
                    volume_participation_limit=volume_participation_limit,
                    volume_to_contracts_multiplier=volume_to_contracts_multiplier,
                    partial_fill_policy=partial_fill_policy,
                    intrabar_ambiguity_policy=intrabar_ambiguity_policy,
                    output_root=output_root,
                    experiment_id=experiment_id,
                    max_workers=max_workers,
                    fail_fast=fail_fast,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_dir={result.output_dir}")
        typer.echo(f"successful_run_count={result.successful_run_count}")
        typer.echo(f"failed_run_count={result.failed_run_count}")
