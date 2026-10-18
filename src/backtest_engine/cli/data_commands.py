from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from backtest_engine.cli.common import CLI_EXCEPTIONS, raise_cli_error
from backtest_engine.data.lifecycle import DataLifecycleAuditConfig, audit_data_lifecycle
from backtest_engine.data.store import build_data_store_from_catalog
from backtest_engine.research.pack import ResearchPackConfig, generate_research_pack


def register(app: typer.Typer) -> None:
    @app.command(name="data-doctor")
    def data_doctor(
        catalog_path: Annotated[Path, typer.Option(help="Data catalog JSON/TOML path.")],
        output_path: Annotated[Path | None, typer.Option()] = None,
        strict: Annotated[bool, typer.Option("--strict/--audit-only")] = True,
    ) -> None:
        try:
            result = audit_data_lifecycle(
                DataLifecycleAuditConfig(
                    catalog_path=catalog_path,
                    output_path=output_path,
                    strict=strict,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_path={result.output_path}")
        typer.echo(f"entry_count={result.entry_count}")
        typer.echo(f"approved_count={result.approved_count}")
        typer.echo(f"rejected_count={result.rejected_count}")
        typer.echo(f"audit_path={result.audit_path}")
        typer.echo(f"report_path={result.report_path}")

    @app.command(name="data-store-index")
    def data_store_index(
        catalog_path: Annotated[Path, typer.Option(help="Data catalog JSON/TOML path.")],
        output_path: Annotated[Path, typer.Option(help="Output data store manifest path.")],
        adapter_name: Annotated[str | None, typer.Option()] = None,
        adapter_version: Annotated[str | None, typer.Option()] = None,
    ) -> None:
        try:
            adapter_snapshot = {
                key: value
                for key, value in {
                    "adapter_name": adapter_name,
                    "adapter_version": adapter_version,
                }.items()
                if value is not None
            }
            manifest = build_data_store_from_catalog(
                catalog_path=catalog_path,
                output_path=output_path,
                adapter_snapshot=adapter_snapshot,
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_path={output_path}")
        typer.echo(f"dataset_version_count={len(manifest.versions)}")
        typer.echo(f"schema_migration_count={len(manifest.schema_migrations)}")

    @app.command(name="moex-registry-update")
    def moex_registry_update(
        dataset_path: Annotated[list[Path], typer.Option("--dataset-path")],
        instrument_output_path: Annotated[Path, typer.Option()],
        calendar_output_path: Annotated[Path, typer.Option()],
        audit_output_path: Annotated[Path, typer.Option()],
        contract_multiplier: Annotated[float, typer.Option(min=0.0)] = 1.0,
        lot_size: Annotated[float, typer.Option(min=0.0)] = 1.0,
        tick_size: Annotated[float | None, typer.Option(min=0.0)] = None,
        tick_value: Annotated[float | None, typer.Option(min=0.0)] = None,
        initial_margin: Annotated[float | None, typer.Option(min=0.0)] = None,
    ) -> None:
        try:
            from backtest_engine.adapters.moex.registry import (
                MoexRegistryUpdateConfig,
                update_moex_registry_and_calendar,
            )

            result = update_moex_registry_and_calendar(
                MoexRegistryUpdateConfig(
                    dataset_paths=dataset_path,
                    instrument_output_path=instrument_output_path,
                    calendar_output_path=calendar_output_path,
                    audit_output_path=audit_output_path,
                    contract_multiplier=contract_multiplier,
                    lot_size=lot_size,
                    tick_size=tick_size,
                    tick_value=tick_value,
                    initial_margin=initial_margin,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"instrument_output_path={result.instrument_output_path}")
        typer.echo(f"calendar_output_path={result.calendar_output_path}")
        typer.echo(f"audit_output_path={result.audit_output_path}")
        typer.echo(f"instrument_count={result.instrument_count}")
        typer.echo(f"trading_day_count={result.trading_day_count}")
        typer.echo(f"holiday_count={result.holiday_count}")
        typer.echo(f"short_day_count={result.short_day_count}")
        typer.echo(f"warning_count={len(result.warnings)}")

    @app.command(name="moex-iss-snapshot")
    def moex_iss_snapshot(
        snapshot_output_path: Annotated[Path, typer.Option()],
        audit_output_path: Annotated[Path, typer.Option()],
        source_url: Annotated[str | None, typer.Option()] = None,
        raw_snapshot_path: Annotated[Path | None, typer.Option()] = None,
        registry_output_path: Annotated[Path | None, typer.Option()] = None,
        calendar_output_path: Annotated[Path | None, typer.Option()] = None,
    ) -> None:
        try:
            from backtest_engine.adapters.moex.iss.snapshot import (
                MoexIssSnapshotConfig,
                build_moex_iss_snapshot,
            )

            result = build_moex_iss_snapshot(
                MoexIssSnapshotConfig(
                    snapshot_output_path=snapshot_output_path,
                    audit_output_path=audit_output_path,
                    source_url=source_url,
                    raw_snapshot_path=raw_snapshot_path,
                    registry_output_path=registry_output_path,
                    calendar_output_path=calendar_output_path,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"snapshot_output_path={result.snapshot_output_path}")
        typer.echo(f"audit_output_path={result.audit_output_path}")
        typer.echo(f"registry_output_path={result.registry_output_path}")
        typer.echo(f"calendar_output_path={result.calendar_output_path}")
        typer.echo(f"status={result.status}")
        typer.echo(f"warning_count={result.warning_count}")

    @app.command(name="moex-data-pipeline")
    def moex_data_pipeline(
        output_root: Annotated[Path, typer.Option()],
        raw_snapshot_path: Annotated[Path | None, typer.Option()] = None,
        source_url: Annotated[str | None, typer.Option()] = None,
        dataset_id: Annotated[str, typer.Option()] = "moex:dataset",
        continuous_symbol: Annotated[str, typer.Option()] = "MOEX_NEAREST",
        base_timeframe: Annotated[str, typer.Option()] = "1m",
        aggregate_timeframe: Annotated[
            list[str] | None,
            typer.Option("--aggregate-timeframe"),
        ] = None,
        gap_policy: Annotated[str, typer.Option()] = "reject",
        gap_repair_policy: Annotated[str, typer.Option()] = "none",
    ) -> None:
        try:
            from backtest_engine.adapters.moex.pipeline.models import MoexDataPipelineConfig
            from backtest_engine.adapters.moex.pipeline.runner import run_moex_data_pipeline

            result = run_moex_data_pipeline(
                MoexDataPipelineConfig(
                    output_root=output_root,
                    raw_snapshot_path=raw_snapshot_path,
                    source_url=source_url,
                    dataset_id=dataset_id,
                    continuous_symbol=continuous_symbol,
                    base_timeframe=base_timeframe,
                    aggregate_timeframes=aggregate_timeframe or ["5m"],
                    gap_policy=gap_policy,
                    gap_repair_policy=gap_repair_policy,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_root={result.output_root}")
        typer.echo(f"approval_status={result.approval_status}")
        typer.echo(f"raw_snapshot_path={result.raw_snapshot_path}")
        typer.echo(f"normalized_path={result.normalized_path}")
        typer.echo(f"nearest_path={result.nearest_path}")
        typer.echo(f"continuous_path={result.continuous_path}")
        typer.echo(f"approval_path={result.approval_path}")
        typer.echo(f"gap_repair_path={result.gap_repair_path}")
        typer.echo(f"manifest_path={result.manifest_path}")

    @app.command(name="research-pack")
    def research_pack(
        path: Annotated[Path, typer.Option(help="Run or experiment directory.")],
        output_path: Annotated[Path | None, typer.Option()] = None,
        top: Annotated[int, typer.Option(min=1)] = 10,
        tearsheet: Annotated[bool, typer.Option("--tearsheet/--no-tearsheet")] = True,
    ) -> None:
        try:
            result = generate_research_pack(
                ResearchPackConfig(
                    path=path,
                    output_path=output_path,
                    top=top,
                    generate_tearsheet=tearsheet,
                )
            )
        except CLI_EXCEPTIONS as exc:
            raise_cli_error(exc)
        typer.echo(f"output_path={result.output_path}")
        typer.echo(f"pack_type={result.pack_type}")
        typer.echo(f"markdown_path={result.markdown_path}")
        typer.echo(f"html_path={result.html_path}")
        typer.echo(f"manifest_path={result.manifest_path}")
