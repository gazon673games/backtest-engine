from __future__ import annotations

import hashlib
import itertools
import re
import tomllib
from datetime import date, time
from pathlib import Path
from typing import Any

from backtest_engine.intraday.batch.config import ENGINE_PARAM_NAMES, IntradayBatchConfig
from backtest_engine.strategies.plugins.loader import strategy_file_sha256

strategy_sha256 = strategy_file_sha256


def load_intraday_batch_config(config_path: Path) -> IntradayBatchConfig:
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    return IntradayBatchConfig(
        experiment_id=raw_config["experiment_id"],
        output_root=raw_config.get("output_root", Path("data/backtests")),
        dataset_path=raw_config["dataset_path"],
        quality_approval_path=raw_config.get("quality_approval_path"),
        quality_approval_required=bool(raw_config.get("quality_approval_required", True)),
        min_usable_decision_day_pct=float(raw_config.get("min_usable_decision_day_pct", 0.95)),
        allow_legacy_bar_timestamps=bool(raw_config.get("allow_legacy_bar_timestamps", False)),
        strategy_file=raw_config["strategy_file"],
        from_date=parse_date(raw_config.get("from_date", "0001-01-01")),
        decision_time=parse_time(raw_config["decision_time"])
        if raw_config.get("decision_time")
        else None,
        decision_tolerance_minutes=raw_config.get("decision_tolerance_minutes"),
        minimum_bars_since_session_start=int(raw_config.get("minimum_bars_since_session_start", 0)),
        commission_bps=float(raw_config.get("commission_bps", 0.0)),
        slippage_bps=float(raw_config.get("slippage_bps", 0.0)),
        grid=raw_config.get("grid", {}),
        continue_on_error=bool(raw_config.get("continue_on_error", True)),
    )


def parse_intraday_grid_options(grid_options: list[str]) -> dict[str, list[Any]]:
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


def expand_intraday_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        msg = "Intraday batch grid must contain at least one parameter"
        raise ValueError(msg)
    keys = sorted(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def split_intraday_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    engine_params = {key: value for key, value in params.items() if key in ENGINE_PARAM_NAMES}
    plugin_params = {key: value for key, value in params.items() if key not in ENGINE_PARAM_NAMES}
    return engine_params, plugin_params


def generate_intraday_run_id(
    *,
    experiment_id: str,
    dataset_validation: dict[str, Any],
    strategy_file: Path,
    engine_params: dict[str, Any],
    plugin_params: dict[str, Any],
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
    if len(base) <= 180:
        return base
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]
    return f"{base[:167].rstrip('_')}_{digest}"


def parse_grid_value(raw_value: str) -> int | float | str:
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        return raw_value


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


def optional_int(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    return required_int(raw_value, "decision_tolerance_minutes")


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
