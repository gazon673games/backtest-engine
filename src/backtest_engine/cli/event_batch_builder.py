from __future__ import annotations

from pathlib import Path

from backtest_engine.cli.builders import require_config_keys
from backtest_engine.cli.parsing import (
    parse_date,
    parse_intraday_event_missing_decision_policy,
    parse_intraday_event_schedule,
    parse_intraday_event_timing,
    parse_time,
)
from backtest_engine.intraday.event.batch.runner import (
    IntradayEventBatchConfig,
    load_intraday_event_batch_config,
    parse_intraday_event_grid_options,
)


def build_intraday_event_batch_config(
    *,
    config: Path | None,
    dataset_paths: list[Path] | None,
    catalog_path: Path | None,
    dataset_ids: list[str] | None,
    catalog_enforcement: str | None,
    strategy_file: Path | None,
    quality_approval_path: Path | None,
    require_quality_approval: bool | None,
    min_usable_decision_day_pct: float | None,
    allow_legacy_bar_timestamps: bool | None,
    grid_options: list[str] | None,
    from_date: str | None,
    schedule: str | None,
    execution_timing: str | None,
    allow_noncausal_same_close: bool | None,
    end_of_data_position_policy: str | None,
    decision_time: list[str] | None,
    decision_tolerance_minutes: int | None,
    missing_decision_policy: str | None,
    commission_bps: float | None,
    slippage_bps: float | None,
    bid_ask_spread_bps: float | None,
    bid_ask_spread_ticks: float | None,
    volume_participation_limit: float | None,
    volume_to_contracts_multiplier: float | None,
    partial_fill_policy: str | None,
    intrabar_ambiguity_policy: str | None,
    output_root: Path | None,
    experiment_id: str | None,
    max_workers: int | None,
    fail_fast: bool,
) -> IntradayEventBatchConfig:
    raw_config = (
        load_intraday_event_batch_config(config).model_dump(mode="python") if config else {}
    )
    update_raw_event_batch_config(
        raw_config,
        dataset_paths=dataset_paths,
        catalog_path=catalog_path,
        dataset_ids=dataset_ids,
        catalog_enforcement=catalog_enforcement,
        strategy_file=strategy_file,
        quality_approval_path=quality_approval_path,
        min_usable_decision_day_pct=min_usable_decision_day_pct,
        allow_legacy_bar_timestamps=allow_legacy_bar_timestamps,
        allow_noncausal_same_close=allow_noncausal_same_close,
        end_of_data_position_policy=end_of_data_position_policy,
        decision_tolerance_minutes=decision_tolerance_minutes,
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
    )
    apply_parsed_event_batch_options(
        raw_config,
        grid_options,
        from_date,
        schedule,
        execution_timing,
        decision_time,
        missing_decision_policy,
    )
    if require_quality_approval is not None:
        raw_config["quality_approval_required"] = require_quality_approval
    raw_config["continue_on_error"] = not fail_fast and raw_config.get("continue_on_error", True)
    validate_event_batch_required_options(raw_config)
    apply_event_batch_defaults(raw_config)
    return IntradayEventBatchConfig(**raw_config)


def update_raw_event_batch_config(raw_config: dict[str, object], **values: object) -> None:
    for key, value in values.items():
        if value is not None:
            raw_config[key] = value


def apply_parsed_event_batch_options(
    raw_config: dict[str, object],
    grid_options: list[str] | None,
    from_date: str | None,
    schedule: str | None,
    execution_timing: str | None,
    decision_time: list[str] | None,
    missing_decision_policy: str | None,
) -> None:
    if grid_options is not None:
        raw_config["grid"] = parse_intraday_event_grid_options(grid_options)
    if from_date is not None:
        raw_config["from_date"] = parse_date(from_date)
    if schedule is not None:
        raw_config["schedule"] = parse_intraday_event_schedule(schedule)
    if execution_timing is not None:
        raw_config["timing"] = parse_intraday_event_timing(execution_timing)
    if decision_time is not None:
        raw_config["decision_times"] = [parse_time(value) for value in decision_time]
    if missing_decision_policy is not None:
        raw_config["missing_decision_policy"] = parse_intraday_event_missing_decision_policy(
            missing_decision_policy
        )


def validate_event_batch_required_options(raw_config: dict[str, object]) -> None:
    require_config_keys(
        raw_config,
        ("--strategy-file", "strategy_file"),
        ("--grid", "grid"),
        ("--experiment-id", "experiment_id"),
    )
    if "dataset_paths" not in raw_config and not (
        raw_config.get("catalog_path") and raw_config.get("dataset_ids")
    ):
        raise ValueError(
            "Missing required option(s): --dataset-path or --catalog-path/--dataset-id"
        )


def apply_event_batch_defaults(raw_config: dict[str, object]) -> None:
    raw_config.setdefault("from_date", parse_date("0001-01-01"))
    raw_config.setdefault("schedule", "every-bar")
    raw_config.setdefault("timing", "signal_close_to_next_bar_close")
    raw_config.setdefault("allow_noncausal_same_close", False)
    raw_config.setdefault("end_of_data_position_policy", "reject")
    raw_config.setdefault("decision_times", [])
    raw_config.setdefault("decision_tolerance_minutes", None)
    raw_config.setdefault("missing_decision_policy", "carry")
    raw_config.setdefault("commission_bps", 0.0)
    raw_config.setdefault("slippage_bps", 0.0)
    raw_config.setdefault("bid_ask_spread_bps", 0.0)
    raw_config.setdefault("bid_ask_spread_ticks", 0.0)
    raw_config.setdefault("volume_participation_limit", None)
    raw_config.setdefault("volume_to_contracts_multiplier", 1.0)
    raw_config.setdefault("partial_fill_policy", "allow")
    raw_config.setdefault("intrabar_ambiguity_policy", "conservative")
    raw_config.setdefault("output_root", Path("data/backtests"))
