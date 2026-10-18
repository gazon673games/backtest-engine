"""Builds resolved event-run configs for batch cases."""

from __future__ import annotations

from pathlib import Path

from backtest_engine.data.catalog import DataCatalogEntry, resolve_data_catalog_entry
from backtest_engine.intraday.event.batch.config import IntradayEventBatchConfig
from backtest_engine.intraday.event.config import (
    IntradayEventBacktestConfig,
    IntradayEventDatasetConfig,
    IntradayEventExecutionConfig,
    IntradayEventInstrumentConfig,
    IntradayEventOutputConfig,
    IntradayEventSessionConfig,
    IntradayEventStrategyConfig,
)


def build_case_config(
    config: IntradayEventBatchConfig,
    runs_dir: Path,
    dataset_path: Path,
    catalog_entry: DataCatalogEntry | None,
    run_id: str,
    execution: IntradayEventExecutionConfig,
    plugin_params: dict[str, object],
) -> IntradayEventBacktestConfig:
    return IntradayEventBacktestConfig(
        dataset=IntradayEventDatasetConfig(
            path=dataset_path,
            catalog_path=config.catalog_path,
            dataset_id=catalog_entry.dataset_id if catalog_entry is not None else None,
            catalog_enforcement=config.catalog_enforcement,
            quality_approval_path=quality_approval_path(config, catalog_entry),
            quality_approval_required=config.quality_approval_required,
            min_usable_decision_day_pct=config.min_usable_decision_day_pct,
            allow_legacy_bar_timestamps=config.allow_legacy_bar_timestamps,
        ),
        strategy=IntradayEventStrategyConfig(
            strategy_file=config.strategy_file,
            strategy_params={key: str(value) for key, value in plugin_params.items()},
        ),
        instrument=instrument_for_catalog_entry(config, catalog_entry),
        portfolio=config.portfolio,
        session=session_for_catalog_entry(config, catalog_entry),
        execution=execution,
        output=IntradayEventOutputConfig(root=runs_dir, run_id=run_id),
    )


def resolved_dataset_sources(
    config: IntradayEventBatchConfig,
) -> list[tuple[Path, DataCatalogEntry | None]]:
    sources = [(path, None) for path in config.dataset_paths]
    if config.catalog_path is None:
        return sources
    for dataset_id in config.dataset_ids:
        entry = resolve_data_catalog_entry(config.catalog_path, dataset_id)
        sources.append((entry.dataset_path, entry))
    return sources


def quality_approval_path(
    config: IntradayEventBatchConfig,
    catalog_entry: DataCatalogEntry | None,
) -> Path | None:
    if config.quality_approval_path is not None:
        return config.quality_approval_path
    return None if catalog_entry is None else catalog_entry.quality_approval_path


def instrument_for_catalog_entry(
    config: IntradayEventBatchConfig,
    catalog_entry: DataCatalogEntry | None,
) -> IntradayEventInstrumentConfig:
    if (
        catalog_entry is None
        or catalog_entry.instrument_registry_path is None
        or config.instrument.registry_path is not None
        or config.instrument.spec_path is not None
        or config.instrument.specs
    ):
        return config.instrument
    return config.instrument.model_copy(
        update={"registry_path": catalog_entry.instrument_registry_path}
    )


def session_for_catalog_entry(
    config: IntradayEventBatchConfig,
    catalog_entry: DataCatalogEntry | None,
) -> IntradayEventSessionConfig:
    if (
        catalog_entry is None
        or catalog_entry.calendar_path is None
        or config.session.calendar_path is not None
    ):
        return config.session
    return config.session.model_copy(update={"calendar_path": catalog_entry.calendar_path})
