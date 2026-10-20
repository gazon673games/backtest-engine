from __future__ import annotations

from datetime import date, datetime
from typing import Any

import polars as pl

from backtest_engine.core.exceptions import BacktestError
from backtest_engine.data.asset_schema import normalize_intraday_asset_frame
from backtest_engine.data.catalog import resolve_data_catalog_entry, validate_catalog_binding
from backtest_engine.data.dataset import load_dataset
from backtest_engine.data.instrument_registry import build_instrument_specs, select_instrument_spec
from backtest_engine.data.instrument_registry import (
    parse_instrument_spec_dates as registry_parse_instrument_spec_dates,
)
from backtest_engine.data.market_identity import identity_updates
from backtest_engine.data.session_model import (
    row_session_eligible,
    session_label,
    session_updates_from_calendar,
    time_in_range,
)
from backtest_engine.intraday.engine import validate_intraday_dataset
from backtest_engine.intraday.event.row_source import FrameRows
from backtest_engine.intraday.validation import enforce_causal_bar_timestamps
from backtest_engine.portfolio.multi.config import (
    MultiInstrumentInstrumentSpec,
    MultiInstrumentLegConfig,
)
from backtest_engine.portfolio.multi.quality import evaluate_leg_quality_approval
from backtest_engine.portfolio.multi.utils import read_json_or_toml

row_session_eligible_for_leg = row_session_eligible
session_label_for_leg = session_label
time_in_range_for_leg = time_in_range


def load_legs(
    legs: list[MultiInstrumentLegConfig],
    from_date: date | None = None,
    till_date: date | None = None,
) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for leg in legs:
        resolved_leg, catalog_binding = resolve_leg_config(leg)
        if resolved_leg.dataset_path is None:
            raise BacktestError(f"dataset_path could not be resolved for {resolved_leg.asset_id}")
        frame = normalize_intraday_asset_frame(load_dataset(resolved_leg.dataset_path)).sort("ts")
        validation = validate_intraday_dataset(frame, resolved_leg.dataset_path)
        enforce_causal_bar_timestamps(
            validation,
            allow_legacy=resolved_leg.allow_legacy_bar_timestamps,
            label=resolved_leg.asset_id,
        )
        resolved_leg = infer_leg_identity(resolved_leg, validation)
        quality_approval = evaluate_leg_quality_approval(
            resolved_leg,
            validation,
            from_date=from_date,
        )
        if till_date is not None:
            # Keep pre-start bars for causal warmup; the selected end date is inclusive.
            time_column = "available_at" if "available_at" in frame.columns else "ts"
            frame = frame.filter(pl.col(time_column).dt.date() <= till_date)
            if frame.is_empty():
                raise BacktestError(f"No finalized data for {leg.asset_id} through {till_date}")
            validation = {**validation, "selected_till_date": till_date.isoformat()}
        loaded[resolved_leg.asset_id] = {
            "config": resolved_leg,
            "frame": frame,
            "rows": FrameRows(frame),
            "validation": validation,
            "catalog_binding": catalog_binding,
            "quality_approval": quality_approval,
        }
    return loaded


def resolve_leg_config(
    leg: MultiInstrumentLegConfig,
) -> tuple[MultiInstrumentLegConfig, dict[str, Any]]:
    resolved = leg
    catalog_binding: dict[str, Any] = {
        "status": "not_required",
        "enforcement": leg.catalog_enforcement,
        "dataset_id": leg.dataset_id,
        "violations": [],
    }
    catalog_entry = None
    if leg.catalog_path is not None and leg.dataset_id is not None:
        catalog_entry = resolve_data_catalog_entry(leg.catalog_path, leg.dataset_id)
        resolved = apply_catalog_entry(resolved, catalog_entry)
    elif leg.catalog_enforcement != "off":
        raise BacktestError(
            "catalog_path and dataset_id are required when catalog_enforcement is enabled"
        )
    if resolved.dataset_path is None:
        raise BacktestError(f"dataset_path is required for {resolved.asset_id}")
    if catalog_entry is not None:
        catalog_binding = validate_catalog_binding(
            entry=catalog_entry,
            dataset_path=resolved.dataset_path,
            enforcement=resolved.catalog_enforcement,
            quality_approval_path=resolved.quality_approval_path,
            instrument_registry_path=resolved.instrument_registry_path,
            calendar_path=resolved.calendar_path,
        )
    return resolve_leg_instrument_specs(resolve_leg_calendar(resolved)), catalog_binding


def apply_catalog_entry(
    leg: MultiInstrumentLegConfig,
    catalog_entry: Any,
) -> MultiInstrumentLegConfig:
    updates: dict[str, Any] = {}
    if leg.dataset_path is None:
        updates["dataset_path"] = catalog_entry.dataset_path
    if leg.quality_approval_path is None and catalog_entry.quality_approval_path is not None:
        updates["quality_approval_path"] = catalog_entry.quality_approval_path
    if leg.instrument_registry_path is None and catalog_entry.instrument_registry_path is not None:
        updates["instrument_registry_path"] = catalog_entry.instrument_registry_path
    if leg.calendar_path is None and catalog_entry.calendar_path is not None:
        updates["calendar_path"] = catalog_entry.calendar_path
    return leg.model_copy(update=updates) if updates else leg


def resolve_leg_calendar(leg: MultiInstrumentLegConfig) -> MultiInstrumentLegConfig:
    if leg.calendar_path is None:
        return leg
    raw_calendar = read_json_or_toml(leg.calendar_path, "Session calendar")
    updates = session_updates_from_calendar(raw_calendar)
    updates.pop("trading_calendar", None)
    return leg.model_copy(update=updates) if updates else leg


def resolve_leg_instrument_specs(leg: MultiInstrumentLegConfig) -> MultiInstrumentLegConfig:
    if leg.instrument_registry_path is None:
        return leg
    raw_registry = read_json_or_toml(leg.instrument_registry_path, "Instrument registry")
    missing = raw_registry.get("historical_margin", {}).get("missing_coverage", [])
    if missing:
        raise ValueError("Historical margin coverage is incomplete: " + "; ".join(missing[:5]))
    specs = build_instrument_specs(raw_registry, MultiInstrumentInstrumentSpec)
    if (leg.quality_approval_required or leg.catalog_enforcement == "strict") and any(
        spec.margin_estimate_schedule for spec in specs
    ):
        raise ValueError(
            f"Exact dated initial_margin rules are required for {leg.asset_id}; "
            "research margin estimates require diagnostic mode"
        )
    if "instruments" in raw_registry:
        combined = list(leg.specs)
        combined.extend(spec for spec in specs if spec not in combined)
        return leg.model_copy(update={"specs": combined})

    parsed = registry_parse_instrument_spec_dates(raw_registry)
    spec = MultiInstrumentInstrumentSpec(**parsed)
    updates = instrument_spec_updates(parsed, spec, leg)
    return leg.model_copy(update=updates)


def instrument_spec_updates(
    parsed: dict[str, Any],
    spec: MultiInstrumentInstrumentSpec,
    leg: MultiInstrumentLegConfig,
) -> dict[str, Any]:
    combined = list(leg.specs)
    if spec not in combined:
        combined.append(spec)
    updates: dict[str, Any] = {"specs": combined}
    for field_name in (
        "price_currency",
        "pnl_currency",
        "contract_multiplier",
        "lot_size",
        "tick_size",
        "tick_value",
        "initial_margin",
    ):
        if field_name in parsed:
            updates[field_name] = parsed[field_name]
    return updates


def instrument_spec_for_row(
    row: dict[str, Any],
    current_ts: datetime,
    leg: MultiInstrumentLegConfig,
) -> MultiInstrumentInstrumentSpec:
    fallback = MultiInstrumentInstrumentSpec(
        symbol=leg.asset_id,
        price_currency=leg.price_currency,
        pnl_currency=leg.pnl_currency,
        contract_multiplier=leg.contract_multiplier,
        lot_size=leg.lot_size,
        tick_size=leg.tick_size,
        tick_value=leg.tick_value,
        initial_margin=leg.initial_margin,
        exchange_fee_per_contract=leg.exchange_fee_per_contract,
        asset_class=leg.asset_class,
        instrument_type=leg.instrument_type,
        base_currency=leg.base_currency,
        quote_currency=leg.quote_currency,
        settlement_currency=leg.settlement_currency,
        contract_type=leg.contract_type,
        quantity_step=leg.quantity_step,
        minimum_quantity=leg.minimum_quantity,
        minimum_notional=leg.minimum_notional,
        maker_fee_bps=leg.maker_fee_bps,
        taker_fee_bps=leg.taker_fee_bps,
        maintenance_margin_rate=leg.maintenance_margin_rate,
        liquidation_fee_bps=leg.liquidation_fee_bps,
        margin_tiers=leg.margin_tiers,
    )
    return select_instrument_spec(
        row=row,
        current_ts=current_ts,
        specs=leg.specs,
        fallback=fallback,
        extra_symbols=[leg.asset_id],
    )


def infer_leg_identity(
    leg: MultiInstrumentLegConfig,
    validation: dict[str, Any],
) -> MultiInstrumentLegConfig:
    updates = identity_updates(leg, validation)
    return leg.model_copy(update=updates) if updates else leg
