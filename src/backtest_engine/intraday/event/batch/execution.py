from __future__ import annotations

from pathlib import Path
from typing import Any

from backtest_engine.core.run_identity import case_identity
from backtest_engine.intraday.event.batch.config import IntradayEventBatchConfig
from backtest_engine.intraday.event.batch.parsing import (
    decision_time_text,
    optional_float,
    optional_int,
    parse_date,
    parse_intrabar_ambiguity_policy,
    parse_missing_decision_policy,
    parse_partial_fill_policy,
    parse_schedule,
    parse_time,
    parse_timing,
    positive_float,
    required_float,
    safe_path_part,
    safe_value_part,
)
from backtest_engine.intraday.event.engine import IntradayEventExecutionConfig


def generate_intraday_event_run_id(
    *,
    experiment_id: str,
    dataset_validation: dict[str, Any],
    strategy_file: Path,
    engine_params: dict[str, Any],
    plugin_params: dict[str, Any],
    dataset_path: Path,
) -> str:
    param_parts = {
        **{key: value for key, value in engine_params.items() if value is not None},
        **plugin_params,
    }
    param_text = "_".join(
        f"{safe_path_part(key)}-{safe_value_part(value)}"
        for key, value in sorted(param_parts.items())
    )
    base = "_".join(
        part
        for part in (
            safe_path_part(experiment_id),
            safe_path_part(str(dataset_validation["continuous_symbol"])),
            safe_path_part(str(dataset_validation["timeframe"])),
            safe_path_part(strategy_file.stem),
            param_text,
        )
        if part
    )
    digest = case_identity(
        dataset_path,
        strategy_file,
        {"dataset": dataset_validation, "engine": engine_params, "strategy": plugin_params},
    )
    return f"{base[:163].rstrip('_')}_{digest}"


def execution_config(
    config: IntradayEventBatchConfig,
    engine_params: dict[str, Any],
) -> IntradayEventExecutionConfig:
    decision_times = resolve_decision_times(config, engine_params)
    return IntradayEventExecutionConfig(
        timing=parse_timing(engine_params.get("timing", config.timing)),
        allow_noncausal_same_close=config.allow_noncausal_same_close,
        end_of_data_position_policy=config.end_of_data_position_policy,
        from_date=parse_date(engine_params.get("from_date", config.from_date)),
        schedule=parse_schedule(engine_params.get("schedule", config.schedule)),
        decision_times=decision_times,
        decision_tolerance_minutes=optional_int(
            engine_params.get(
                "decision_tolerance_minutes",
                config.decision_tolerance_minutes,
            )
        ),
        missing_decision_policy=parse_missing_decision_policy(
            engine_params.get("missing_decision_policy", config.missing_decision_policy)
        ),
        commission_bps=required_float(
            engine_params.get("commission_bps", config.commission_bps),
            "commission_bps",
        ),
        slippage_bps=required_float(
            engine_params.get("slippage_bps", config.slippage_bps),
            "slippage_bps",
        ),
        slippage_ticks=required_float(
            engine_params.get("slippage_ticks", config.slippage_ticks),
            "slippage_ticks",
        ),
        bid_ask_spread_bps=required_float(
            engine_params.get("bid_ask_spread_bps", config.bid_ask_spread_bps),
            "bid_ask_spread_bps",
        ),
        bid_ask_spread_ticks=required_float(
            engine_params.get("bid_ask_spread_ticks", config.bid_ask_spread_ticks),
            "bid_ask_spread_ticks",
        ),
        volume_participation_limit=optional_float(
            engine_params.get(
                "volume_participation_limit",
                config.volume_participation_limit,
            ),
            "volume_participation_limit",
        ),
        volume_to_contracts_multiplier=positive_float(
            engine_params.get(
                "volume_to_contracts_multiplier",
                config.volume_to_contracts_multiplier,
            ),
            "volume_to_contracts_multiplier",
        ),
        partial_fill_policy=parse_partial_fill_policy(
            engine_params.get("partial_fill_policy", config.partial_fill_policy)
        ),
        intrabar_ambiguity_policy=parse_intrabar_ambiguity_policy(
            engine_params.get("intrabar_ambiguity_policy", config.intrabar_ambiguity_policy)
        ),
        roll_cost_bps=required_float(
            engine_params.get("roll_cost_bps", config.roll_cost_bps),
            "roll_cost_bps",
        ),
        roll_slippage_ticks=required_float(
            engine_params.get("roll_slippage_ticks", config.roll_slippage_ticks),
            "roll_slippage_ticks",
        ),
        roll_pnl_policy=str(engine_params.get("roll_pnl_policy", config.roll_pnl_policy)),
    )


def resolve_decision_times(
    config: IntradayEventBatchConfig,
    engine_params: dict[str, Any],
) -> list[Any]:
    if "decision_time" in engine_params:
        return [parse_time(engine_params["decision_time"])]
    return config.decision_times


def resolve_execution_for_dataset(
    execution: IntradayEventExecutionConfig,
    dataset_validation: dict[str, Any],
) -> IntradayEventExecutionConfig:
    if execution.schedule != "times" or execution.decision_tolerance_minutes is not None:
        return execution
    return execution.model_copy(
        update={"decision_tolerance_minutes": int(dataset_validation["bar_minutes"])}
    )


def run_id_engine_params(execution: IntradayEventExecutionConfig) -> dict[str, Any]:
    return {
        "from_date": execution.from_date,
        "schedule": execution.schedule,
        "timing": execution.timing,
        "decision_time": decision_time_text(execution.decision_times),
        "decision_tolerance_minutes": execution.decision_tolerance_minutes,
        "missing_decision_policy": execution.missing_decision_policy,
        "commission_bps": execution.commission_bps,
        "slippage_bps": execution.slippage_bps,
        "slippage_ticks": execution.slippage_ticks,
        "bid_ask_spread_bps": execution.bid_ask_spread_bps,
        "bid_ask_spread_ticks": execution.bid_ask_spread_ticks,
        "volume_participation_limit": execution.volume_participation_limit,
        "volume_to_contracts_multiplier": execution.volume_to_contracts_multiplier,
        "partial_fill_policy": execution.partial_fill_policy,
        "intrabar_ambiguity_policy": execution.intrabar_ambiguity_policy,
        "roll_cost_bps": execution.roll_cost_bps,
        "roll_slippage_ticks": execution.roll_slippage_ticks,
        "roll_pnl_policy": execution.roll_pnl_policy,
    }
