from __future__ import annotations

import hashlib
import itertools
import json
import re
import tomllib
from datetime import date, time
from pathlib import Path
from typing import Any

from backtest_engine.intraday.event.batch.config import (
    ENGINE_PARAM_NAMES,
    IntradayEventBatchConfig,
    default_max_workers,
)
from backtest_engine.intraday.event.engine import (
    INTRADAY_EVENT_INTRABAR_AMBIGUITY_POLICIES,
    INTRADAY_EVENT_MISSING_DECISION_POLICIES,
    INTRADAY_EVENT_PARTIAL_FILL_POLICIES,
    INTRADAY_EVENT_SCHEDULES,
    INTRADAY_EVENT_TIMINGS,
)


def load_intraday_event_batch_config(config_path: Path) -> IntradayEventBatchConfig:
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    return IntradayEventBatchConfig(
        experiment_id=raw_config["experiment_id"],
        output_root=raw_config.get("output_root", Path("data/backtests")),
        dataset_paths=[Path(value) for value in raw_config.get("dataset_paths", [])],
        catalog_path=raw_config.get("catalog_path"),
        dataset_ids=[str(value) for value in raw_config.get("dataset_ids", [])],
        catalog_enforcement=raw_config.get("catalog_enforcement", "off"),
        quality_approval_path=raw_config.get("quality_approval_path"),
        quality_approval_required=bool(raw_config.get("quality_approval_required", True)),
        min_usable_decision_day_pct=float(raw_config.get("min_usable_decision_day_pct", 0.95)),
        allow_legacy_bar_timestamps=bool(raw_config.get("allow_legacy_bar_timestamps", False)),
        strategy_file=raw_config["strategy_file"],
        timing=parse_timing(raw_config.get("timing", "signal_close_to_next_bar_close")),
        allow_noncausal_same_close=bool(raw_config.get("allow_noncausal_same_close", False)),
        end_of_data_position_policy=str(raw_config.get("end_of_data_position_policy", "reject")),
        from_date=parse_date(raw_config.get("from_date", "0001-01-01")),
        schedule=parse_schedule(raw_config.get("schedule", "every-bar")),
        decision_times=[parse_time(value) for value in raw_config.get("decision_times", [])],
        decision_tolerance_minutes=raw_config.get("decision_tolerance_minutes"),
        missing_decision_policy=parse_missing_decision_policy(
            raw_config.get("missing_decision_policy", "carry")
        ),
        commission_bps=float(raw_config.get("commission_bps", 0.0)),
        slippage_bps=float(raw_config.get("slippage_bps", 0.0)),
        slippage_ticks=float(raw_config.get("slippage_ticks", 0.0)),
        bid_ask_spread_bps=float(raw_config.get("bid_ask_spread_bps", 0.0)),
        bid_ask_spread_ticks=float(raw_config.get("bid_ask_spread_ticks", 0.0)),
        volume_participation_limit=optional_float(
            raw_config.get("volume_participation_limit"),
            "volume_participation_limit",
        ),
        volume_to_contracts_multiplier=float(raw_config.get("volume_to_contracts_multiplier", 1.0)),
        partial_fill_policy=parse_partial_fill_policy(
            raw_config.get("partial_fill_policy", "allow")
        ),
        intrabar_ambiguity_policy=parse_intrabar_ambiguity_policy(
            raw_config.get("intrabar_ambiguity_policy", "conservative")
        ),
        roll_cost_bps=float(raw_config.get("roll_cost_bps", 0.0)),
        roll_slippage_ticks=float(raw_config.get("roll_slippage_ticks", 0.0)),
        roll_pnl_policy=str(raw_config.get("roll_pnl_policy", "reject")),
        instrument=raw_config.get("instrument", {}),
        portfolio=raw_config.get("portfolio", {}),
        session=parse_session_config(raw_config.get("session", {})),
        grid=raw_config.get("grid", {}),
        continue_on_error=bool(raw_config.get("continue_on_error", True)),
        max_workers=int(raw_config.get("max_workers", default_max_workers())),
        resume_completed=bool(raw_config.get("resume_completed", True)),
        verify_cached_output_hashes=bool(raw_config.get("verify_cached_output_hashes", True)),
    )


def parse_intraday_event_grid_options(grid_options: list[str]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for raw_grid in grid_options:
        if "=" not in raw_grid:
            msg = f"Invalid --grid value, expected key=v1,v2: {raw_grid}"
            raise ValueError(msg)
        key, raw_values = raw_grid.split("=", 1)
        key = key.strip()
        if not key:
            msg = f"Invalid --grid value, empty key: {raw_grid}"
            raise ValueError(msg)
        if key in grid:
            msg = f"Duplicate --grid key: {key}"
            raise ValueError(msg)
        values = [
            parse_grid_value(value.strip()) for value in raw_values.split(",") if value.strip()
        ]
        if not values:
            msg = f"Invalid --grid value, no values for key: {key}"
            raise ValueError(msg)
        grid[key] = values
    return grid


def expand_intraday_event_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        msg = "Intraday event batch grid must contain at least one parameter"
        raise ValueError(msg)
    keys = sorted(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def split_intraday_event_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    engine_params = {key: value for key, value in params.items() if key in ENGINE_PARAM_NAMES}
    plugin_params = {key: value for key, value in params.items() if key not in ENGINE_PARAM_NAMES}
    return engine_params, plugin_params


def parse_grid_value(raw_value: str) -> int | float | str:
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def parse_schedule(raw_value: Any) -> str:
    return parse_choice(raw_value, INTRADAY_EVENT_SCHEDULES, "schedule")


def parse_timing(raw_value: Any) -> str:
    return parse_choice(raw_value, INTRADAY_EVENT_TIMINGS, "timing")


def parse_missing_decision_policy(raw_value: Any) -> str:
    return parse_choice(
        raw_value,
        INTRADAY_EVENT_MISSING_DECISION_POLICIES,
        "missing_decision_policy",
    )


def parse_partial_fill_policy(raw_value: Any) -> str:
    return parse_choice(raw_value, INTRADAY_EVENT_PARTIAL_FILL_POLICIES, "partial_fill_policy")


def parse_intrabar_ambiguity_policy(raw_value: Any) -> str:
    return parse_choice(
        raw_value,
        INTRADAY_EVENT_INTRABAR_AMBIGUITY_POLICIES,
        "intrabar_ambiguity_policy",
    )


def parse_choice(raw_value: Any, choices: frozenset[str], name: str) -> str:
    value = str(raw_value).strip().lower()
    if value not in choices:
        msg = f"{name} must be one of: {', '.join(sorted(choices))}"
        raise ValueError(msg)
    return value


def parse_date(raw_value: Any) -> date:
    if isinstance(raw_value, date):
        return raw_value
    try:
        return date.fromisoformat(str(raw_value))
    except ValueError as exc:
        msg = f"Invalid date: {raw_value}. Expected YYYY-MM-DD"
        raise ValueError(msg) from exc


def parse_time(raw_value: Any) -> time:
    if isinstance(raw_value, time):
        return raw_value
    try:
        parsed = time.fromisoformat(str(raw_value))
    except ValueError as exc:
        msg = f"Invalid time: {raw_value}. Expected HH:MM"
        raise ValueError(msg) from exc
    if parsed.second != 0 or parsed.microsecond != 0:
        msg = f"Invalid time: {raw_value}. Expected minute precision HH:MM"
        raise ValueError(msg)
    return parsed


def parse_session_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    parsed = dict(raw_config)
    for key in (
        "main_session_start",
        "main_session_end",
        "evening_session_start",
        "evening_session_end",
    ):
        if key in parsed and parsed[key] is not None:
            parsed[key] = parse_time(parsed[key])
    if "holiday_dates" in parsed:
        parsed["holiday_dates"] = [parse_date(value) for value in parsed["holiday_dates"]]
    if "short_day_dates" in parsed:
        parsed["short_day_dates"] = [parse_date(value) for value in parsed["short_day_dates"]]
    return parsed


def optional_int(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    return required_int(raw_value, "decision_tolerance_minutes")


def optional_float(raw_value: Any, name: str) -> float | None:
    if raw_value is None:
        return None
    return required_float(raw_value, name)


def required_int(raw_value: Any, name: str) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be an integer"
        raise ValueError(msg) from exc
    if value < 0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)
    return value


def required_float(raw_value: Any, name: str) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be numeric"
        raise ValueError(msg) from exc
    if value < 0.0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)
    return value


def positive_float(raw_value: Any, name: str) -> float:
    value = required_float(raw_value, name)
    if value <= 0.0:
        msg = f"{name} must be positive"
        raise ValueError(msg)
    return value


def decision_time_text(decision_times: list[time]) -> str | None:
    if not decision_times:
        return None
    return ",".join(value.isoformat(timespec="minutes") for value in decision_times)


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def safe_value_part(value: Any) -> str:
    if isinstance(value, time):
        return value.isoformat(timespec="minutes").replace(":", "")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return safe_path_part(str(int(value)))
    return safe_path_part(str(value).replace(":", ""))


def safe_path_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("._")
    return safe or "value"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
