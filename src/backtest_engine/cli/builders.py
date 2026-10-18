from __future__ import annotations

from pathlib import Path

from backtest_engine.cli.parsing import parse_date, parse_time
from backtest_engine.core.config import load_config_file, merge_cli_overrides
from backtest_engine.core.models import (
    BacktestConfig,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
    StrategyConfig,
)
from backtest_engine.daily.batch import BatchConfig, load_batch_config, parse_grid_options
from backtest_engine.intraday.batch.runner import (
    IntradayBatchConfig,
    load_intraday_batch_config,
    parse_intraday_grid_options,
)


def build_run_config(
    *,
    config: Path | None,
    dataset_path: Path | None,
    quality_manifest: Path | None,
    strategy: str | None,
    fast_window: int | None,
    slow_window: int | None,
    commission_bps: float | None,
    slippage_bps: float | None,
    output_root: Path | None,
    run_id: str | None,
    write_csv: bool | None = None,
) -> BacktestConfig:
    if config is not None:
        return merge_cli_overrides(
            load_config_file(config),
            dataset_path=dataset_path,
            quality_manifest=quality_manifest,
            strategy_name=strategy,
            fast_window=fast_window,
            slow_window=slow_window,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            output_root=output_root,
            run_id=run_id,
            write_csv=write_csv,
        )
    require_options(
        (dataset_path, "--dataset-path"),
        (quality_manifest, "--quality-manifest"),
        (run_id, "--run-id"),
    )
    return BacktestConfig(
        dataset=DatasetConfig(path=dataset_path, quality_manifest=quality_manifest),
        strategy=StrategyConfig(
            name=strategy or "buy_and_hold",  # type: ignore[arg-type]
            fast_window=fast_window,
            slow_window=slow_window,
        ),
        execution=ExecutionConfig(
            commission_bps=commission_bps or 0.0,
            slippage_bps=slippage_bps or 0.0,
        ),
        output=OutputConfig(
            root=output_root or Path("data/backtests"),
            run_id=run_id,
            write_csv=bool(write_csv),
        ),
    )


def build_batch_config(
    *,
    config: Path | None,
    dataset_paths: list[Path] | None,
    quality_manifest: Path | None,
    strategy_file: Path | None,
    grid_options: list[str] | None,
    constraints: list[str] | None,
    commission_bps: float | None,
    slippage_bps: float | None,
    output_root: Path | None,
    experiment_id: str | None,
    fail_fast: bool,
) -> BatchConfig:
    raw_config = load_batch_config(config).model_dump(mode="python") if config else {}
    optional_updates(
        raw_config,
        dataset_paths=dataset_paths,
        quality_manifest=quality_manifest,
        strategy_file=strategy_file,
        constraints=constraints,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        output_root=output_root,
        experiment_id=experiment_id,
    )
    if grid_options is not None:
        raw_config["grid"] = parse_grid_options(grid_options)
    raw_config["continue_on_error"] = not fail_fast and raw_config.get("continue_on_error", True)
    require_config_keys(
        raw_config,
        ("--dataset-path", "dataset_paths"),
        ("--quality-manifest", "quality_manifest"),
        ("--strategy-file", "strategy_file"),
        ("--grid", "grid"),
        ("--experiment-id", "experiment_id"),
    )
    raw_config.setdefault("constraints", [])
    raw_config.setdefault("commission_bps", 0.0)
    raw_config.setdefault("slippage_bps", 0.0)
    raw_config.setdefault("output_root", Path("data/backtests"))
    return BatchConfig(**raw_config)


def build_intraday_batch_config(
    *,
    config: Path | None,
    dataset_path: Path | None,
    strategy_file: Path | None,
    quality_approval_path: Path | None,
    require_quality_approval: bool | None,
    allow_legacy_bar_timestamps: bool | None,
    grid_options: list[str] | None,
    from_date: str | None,
    decision_time: str | None,
    decision_tolerance_minutes: int | None,
    minimum_bars_since_session_start: int | None,
    commission_bps: float | None,
    slippage_bps: float | None,
    output_root: Path | None,
    experiment_id: str | None,
    fail_fast: bool,
) -> IntradayBatchConfig:
    raw_config = load_intraday_batch_config(config).model_dump(mode="python") if config else {}
    optional_updates(
        raw_config,
        dataset_path=dataset_path,
        strategy_file=strategy_file,
        quality_approval_path=quality_approval_path,
        quality_approval_required=require_quality_approval,
        allow_legacy_bar_timestamps=allow_legacy_bar_timestamps,
        decision_tolerance_minutes=decision_tolerance_minutes,
        minimum_bars_since_session_start=minimum_bars_since_session_start,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        output_root=output_root,
        experiment_id=experiment_id,
    )
    if grid_options is not None:
        raw_config["grid"] = parse_intraday_grid_options(grid_options)
    if from_date is not None:
        raw_config["from_date"] = parse_date(from_date)
    if decision_time is not None:
        raw_config["decision_time"] = parse_time(decision_time)
    raw_config["continue_on_error"] = not fail_fast and raw_config.get("continue_on_error", True)
    require_config_keys(
        raw_config,
        ("--dataset-path", "dataset_path"),
        ("--strategy-file", "strategy_file"),
        ("--grid", "grid"),
        ("--experiment-id", "experiment_id"),
    )
    raw_config.setdefault("from_date", parse_date("0001-01-01"))
    raw_config.setdefault("decision_time", None)
    raw_config.setdefault("decision_tolerance_minutes", None)
    raw_config.setdefault("minimum_bars_since_session_start", 0)
    raw_config.setdefault("commission_bps", 0.0)
    raw_config.setdefault("slippage_bps", 0.0)
    raw_config.setdefault("output_root", Path("data/backtests"))
    return IntradayBatchConfig(**raw_config)


def optional_updates(raw_config: dict, **values: object) -> None:
    for key, value in values.items():
        if value is not None:
            raw_config[key] = value


def require_options(*options: tuple[object, str]) -> None:
    missing = [label for value, label in options if value is None]
    if missing:
        raise ValueError(f"Missing required option(s): {', '.join(missing)}")


def require_config_keys(raw_config: dict, *options: tuple[str, str]) -> None:
    missing = [option_name for option_name, key in options if key not in raw_config]
    if missing:
        raise ValueError(f"Missing required option(s): {', '.join(missing)}")
