from __future__ import annotations

from datetime import date, time
from pathlib import Path

from backtest_engine.portfolio.multi.engine import MultiInstrumentLegConfig


def parse_params(params: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_param in params:
        if "=" not in raw_param:
            msg = f"Invalid --param value, expected key=value: {raw_param}"
            raise ValueError(msg)
        key, value = raw_param.split("=", 1)
        key = key.strip()
        if not key:
            msg = f"Invalid --param value, empty key: {raw_param}"
            raise ValueError(msg)
        if key in parsed:
            msg = f"Duplicate --param key: {key}"
            raise ValueError(msg)
        parsed[key] = value
    return parsed


def parse_multi_instrument_leg(raw_value: str) -> MultiInstrumentLegConfig:
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not parts or "=" not in parts[0]:
        raise ValueError(
            f"Invalid --leg value, expected asset_id=path[,contract_multiplier=...]: {raw_value}"
        )
    asset_id, path_value = parts[0].split("=", 1)
    raw_options: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"Invalid --leg option, expected key=value: {part}")
        key, value = part.split("=", 1)
        raw_options[key.strip()] = value.strip()
    allowed = {
        "allow_legacy_bar_timestamps",
        "calendar_path",
        "catalog_enforcement",
        "catalog_path",
        "contract_multiplier",
        "dataset_id",
        "initial_margin",
        "instrument_registry_path",
        "lot_size",
        "pnl_currency",
        "price_currency",
        "quality_approval_path",
        "quality_approval_required",
        "session_filter",
        "short_day_policy",
        "tick_size",
        "tick_value",
    }
    unknown = sorted(set(raw_options).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown --leg option(s): {', '.join(unknown)}")
    dataset_path = Path(path_value.strip())
    return MultiInstrumentLegConfig(
        asset_id=asset_id.strip(),
        dataset_path=dataset_path if str(dataset_path) != "-" else None,
        catalog_path=optional_cli_path(raw_options.get("catalog_path")),
        dataset_id=raw_options.get("dataset_id"),
        catalog_enforcement=raw_options.get("catalog_enforcement", "off"),
        quality_approval_path=optional_cli_path(raw_options.get("quality_approval_path")),
        quality_approval_required=optional_cli_bool(
            raw_options.get("quality_approval_required"),
            True,
        ),
        allow_legacy_bar_timestamps=optional_cli_bool(
            raw_options.get("allow_legacy_bar_timestamps"),
            False,
        ),
        instrument_registry_path=optional_cli_path(raw_options.get("instrument_registry_path")),
        calendar_path=optional_cli_path(raw_options.get("calendar_path")),
        session_filter=raw_options.get("session_filter", "all"),
        short_day_policy=raw_options.get("short_day_policy", "allow"),
        price_currency=raw_options.get("price_currency"),
        pnl_currency=raw_options.get("pnl_currency"),
        contract_multiplier=optional_cli_float(raw_options.get("contract_multiplier"), 1.0),
        lot_size=optional_cli_float(raw_options.get("lot_size"), 1.0),
        tick_size=optional_cli_float(raw_options.get("tick_size"), None),
        tick_value=optional_cli_float(raw_options.get("tick_value"), None),
        initial_margin=optional_cli_float(raw_options.get("initial_margin"), None),
    )


def optional_cli_path(raw_value: str | None) -> Path | None:
    if raw_value is None or raw_value == "":
        return None
    return Path(raw_value)


def optional_cli_bool(raw_value: str | None, default: bool) -> bool:
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got: {raw_value}")


def optional_cli_float(raw_value: str | None, default: float | None) -> float | None:
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Expected numeric value, got: {raw_value}") from exc


def parse_date(raw_value: str) -> date:
    try:
        return date.fromisoformat(raw_value)
    except ValueError as exc:
        msg = f"Invalid date: {raw_value}. Expected YYYY-MM-DD"
        raise ValueError(msg) from exc


def parse_time(raw_value: str) -> time:
    try:
        parsed = time.fromisoformat(raw_value)
    except ValueError as exc:
        msg = f"Invalid time: {raw_value}. Expected HH:MM"
        raise ValueError(msg) from exc
    if parsed.second != 0 or parsed.microsecond != 0:
        msg = f"Invalid time: {raw_value}. Expected minute precision HH:MM"
        raise ValueError(msg)
    return parsed


def parse_intraday_event_schedule(raw_value: str) -> str:
    normalized = raw_value.strip().lower()
    allowed = {"every-bar", "times"}
    if normalized not in allowed:
        msg = f"Invalid schedule: {raw_value}. Expected one of: {', '.join(sorted(allowed))}"
        raise ValueError(msg)
    return normalized


def parse_intraday_event_timing(raw_value: str) -> str:
    normalized = raw_value.strip().lower()
    allowed = {
        "signal_close_to_same_close",
        "signal_close_to_next_bar_close",
        "signal_close_to_next_bar_open_to_close",
        "signal_close_to_next_bar_close_to_following_close",
    }
    if normalized not in allowed:
        msg = f"Invalid timing: {raw_value}. Expected one of: {', '.join(sorted(allowed))}"
        raise ValueError(msg)
    return normalized


def parse_intraday_event_missing_decision_policy(raw_value: str) -> str:
    normalized = raw_value.strip().lower()
    allowed = {"carry", "close", "skip"}
    if normalized not in allowed:
        msg = (
            f"Invalid missing decision policy: {raw_value}. "
            f"Expected one of: {', '.join(sorted(allowed))}"
        )
        raise ValueError(msg)
    return normalized


def parse_output_format(output_format: str) -> str:
    normalized = output_format.strip().lower()
    allowed = {"all", "md", "json", "csv", "parquet"}
    if normalized not in allowed:
        msg = (
            f"Invalid output format: {output_format}. Expected one of: {', '.join(sorted(allowed))}"
        )
        raise ValueError(msg)
    return normalized
