from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.quality_approval import (
    explicit_usable_decision_day_pct,
    is_session_gap_approval,
)
from backtest_engine.data.catalog import resolve_data_catalog_entry, validate_catalog_binding
from backtest_engine.data.instrument_registry import build_instrument_specs, select_instrument_spec
from backtest_engine.data.instrument_registry import (
    parse_instrument_spec_dates as registry_parse_instrument_spec_dates,
)
from backtest_engine.data.market_identity import (
    account_currency,
    annualization_days,
    identity_updates,
)
from backtest_engine.data.session_model import session_updates_from_calendar
from backtest_engine.execution.calibration import calibration_updates
from backtest_engine.intraday.event.config import (
    IntradayEventBacktestConfig,
    IntradayEventInstrumentConfig,
    IntradayEventInstrumentSpec,
    IntradayEventSessionConfig,
)
from backtest_engine.intraday.event.quality import quality_approval_violations
from backtest_engine.intraday.event.utils import (
    file_sha256,
    read_json_or_toml,
    unique_sorted_times,
)


def evaluate_intraday_event_catalog_approval(
    config: IntradayEventBacktestConfig,
) -> dict[str, Any]:
    if config.dataset.catalog_enforcement == "off" and not config.dataset.catalog_path:
        return catalog_not_required(config)
    if config.dataset.catalog_path is None or config.dataset.dataset_id is None:
        if config.dataset.catalog_enforcement == "off":
            return catalog_not_required(config)
        raise DatasetValidationError(
            "catalog_path and dataset_id are required when catalog_enforcement is enabled"
        )
    entry = resolve_data_catalog_entry(config.dataset.catalog_path, config.dataset.dataset_id)
    return validate_catalog_binding(
        entry=entry,
        dataset_path=config.dataset.path,
        enforcement=config.dataset.catalog_enforcement,
        quality_approval_path=config.dataset.quality_approval_path,
        instrument_registry_path=config.instrument.registry_path or config.instrument.spec_path,
        calendar_path=config.session.calendar_path,
    )


def catalog_not_required(config: IntradayEventBacktestConfig) -> dict[str, Any]:
    return {
        "status": "not_required",
        "enforcement": "off",
        "dataset_id": config.dataset.dataset_id,
        "violations": [],
    }


def evaluate_intraday_event_quality_approval(
    config: IntradayEventBacktestConfig,
    dataset_validation: dict[str, Any],
) -> dict[str, Any]:
    approval_path = config.dataset.quality_approval_path
    if approval_path is None:
        if config.dataset.quality_approval_required:
            raise DatasetValidationError("quality approval is required but no path was provided")
        return {
            "status": "not_required",
            "required": False,
            "approval_path": None,
            "approval_sha256": None,
            "violations": [],
        }
    if not approval_path.exists():
        raise DatasetValidationError(f"Quality approval file does not exist: {approval_path}")
    with approval_path.open("rb") as approval_file:
        approval_payload = json.load(approval_file)

    violations = quality_approval_violations(config, dataset_validation, approval_payload)
    approval = {
        "status": "approved" if not violations else "rejected",
        "required": config.dataset.quality_approval_required,
        "approval_path": str(approval_path),
        "approval_sha256": file_sha256(approval_path),
        "min_usable_decision_day_pct": config.dataset.min_usable_decision_day_pct,
        "decision_coverage_basis": (
            "runtime_schedule"
            if explicit_usable_decision_day_pct(approval_payload) is None
            and is_session_gap_approval(approval_payload)
            else "approval_metric"
        ),
        "violations": violations,
        "source": approval_payload,
    }
    if violations:
        raise DatasetValidationError("Quality approval failed: " + "; ".join(violations))
    return approval


def resolve_event_config(
    config: IntradayEventBacktestConfig,
    dataset_validation: dict[str, Any],
) -> IntradayEventBacktestConfig:
    dataset = config.dataset
    instrument = config.instrument
    session = config.session
    if config.dataset.catalog_path is not None and config.dataset.dataset_id is not None:
        catalog_entry = resolve_data_catalog_entry(
            config.dataset.catalog_path,
            config.dataset.dataset_id,
        )
        if (
            (
                config.dataset.quality_approval_required
                or config.dataset.catalog_enforcement == "strict"
            )
            and config.dataset.quality_approval_path is None
            and catalog_entry.quality_approval_path
        ):
            dataset = dataset.model_copy(
                update={"quality_approval_path": catalog_entry.quality_approval_path}
            )
        if should_use_catalog_registry(config, catalog_entry):
            instrument = instrument.model_copy(
                update={"registry_path": catalog_entry.instrument_registry_path}
            )
        if config.session.calendar_path is None and catalog_entry.calendar_path is not None:
            session = session.model_copy(update={"calendar_path": catalog_entry.calendar_path})

    execution_updates: dict[str, Any] = {
        "decision_times": unique_sorted_times(config.execution.decision_times)
    }
    if config.execution.calibration_path is not None:
        execution_updates.update(
            calibration_updates(
                config.execution.calibration_path,
                int(dataset_validation["bar_minutes"]),
            )
        )
    if config.execution.schedule == "times" and config.execution.decision_tolerance_minutes is None:
        execution_updates["decision_tolerance_minutes"] = int(dataset_validation["bar_minutes"])
    resolved_instrument = infer_dataset_instrument(
        resolve_instrument_config(instrument), dataset_validation
    )
    if (
        config.dataset.quality_approval_required or config.dataset.catalog_enforcement == "strict"
    ) and any(spec.margin_estimate_schedule for spec in resolved_instrument.specs):
        raise DatasetValidationError(
            "Research margin estimates cannot be used with Require approved data; "
            "supply exact dated initial_margin rules or select diagnostic mode"
        )
    execution_updates["trading_days_per_year"] = annualization_days(
        config.execution.trading_days_per_year, [session.calendar_path]
    )
    currency = account_currency(
        config.portfolio.base_currency,
        [resolved_instrument.pnl_currency]
        + [
            spec.pnl_currency or resolved_instrument.pnl_currency
            for spec in resolved_instrument.specs
        ],
    )
    return config.model_copy(
        update={
            "dataset": dataset,
            "execution": config.execution.model_copy(update=execution_updates),
            "instrument": resolved_instrument,
            "portfolio": config.portfolio.model_copy(update={"base_currency": currency}),
            "session": resolve_session_config(session),
        }
    )


def infer_dataset_instrument(
    instrument: IntradayEventInstrumentConfig,
    validation: dict[str, Any],
) -> IntradayEventInstrumentConfig:
    """Use canonical dataset identity only for fields the user did not configure."""
    updates = identity_updates(instrument, validation)
    return instrument.model_copy(update=updates) if updates else instrument


def should_use_catalog_registry(config: IntradayEventBacktestConfig, catalog_entry: Any) -> bool:
    return (
        config.instrument.registry_path is None
        and config.instrument.spec_path is None
        and not config.instrument.specs
        and catalog_entry.instrument_registry_path is not None
    )


def resolve_session_config(session: IntradayEventSessionConfig) -> IntradayEventSessionConfig:
    if session.calendar_path is None:
        return session
    raw_calendar = read_json_or_toml(session.calendar_path, "Session calendar")
    updates = session_updates_from_calendar(raw_calendar)
    return session.model_copy(update=updates)


def resolve_instrument_config(
    instrument: IntradayEventInstrumentConfig,
) -> IntradayEventInstrumentConfig:
    spec_paths = [path for path in (instrument.spec_path, instrument.registry_path) if path]
    if not spec_paths:
        return instrument
    resolved = instrument
    for spec_path in spec_paths:
        raw_spec = read_json_or_toml(spec_path, "Instrument spec")
        missing = raw_spec.get("historical_margin", {}).get("missing_coverage", [])
        if missing:
            raise DatasetValidationError(
                "Historical margin coverage is incomplete: "
                + "; ".join(missing[:5])
                + "; rebuild with historical risk data or supply exact dated rules"
            )
        specs = build_instrument_specs(raw_spec, IntradayEventInstrumentSpec)
        if "instruments" in raw_spec:
            resolved = resolved.model_copy(update={"specs": [*resolved.specs, *specs]})
            continue
        raw_spec = registry_parse_instrument_spec_dates(raw_spec)
        raw_spec.setdefault("spec_path", spec_path)
        resolved = IntradayEventInstrumentConfig(**{**resolved.model_dump(), **raw_spec})
    return resolved


def instrument_spec_for_row(
    row: dict[str, Any],
    current_ts: datetime,
    instrument: IntradayEventInstrumentConfig,
) -> IntradayEventInstrumentSpec:
    fallback = IntradayEventInstrumentSpec(
        symbol=instrument.symbol or str(row.get("active_contract", "*")),
        price_currency=instrument.price_currency,
        pnl_currency=instrument.pnl_currency,
        contract_multiplier=instrument.contract_multiplier,
        lot_size=instrument.lot_size,
        tick_size=instrument.tick_size,
        tick_value=instrument.tick_value,
        initial_margin=instrument.initial_margin,
        exchange_fee_per_contract=instrument.exchange_fee_per_contract,
        asset_class=instrument.asset_class,
        instrument_type=instrument.instrument_type,
        base_currency=instrument.base_currency,
        quote_currency=instrument.quote_currency,
        settlement_currency=instrument.settlement_currency,
        contract_type=instrument.contract_type,
        quantity_step=instrument.quantity_step,
        minimum_quantity=instrument.minimum_quantity,
        minimum_notional=instrument.minimum_notional,
        maker_fee_bps=instrument.maker_fee_bps,
        taker_fee_bps=instrument.taker_fee_bps,
        maintenance_margin_rate=instrument.maintenance_margin_rate,
        liquidation_fee_bps=instrument.liquidation_fee_bps,
        margin_tiers=instrument.margin_tiers,
    )
    return select_instrument_spec(
        row=row,
        current_ts=current_ts,
        specs=instrument.specs,
        fallback=fallback,
        extra_symbols=[instrument.symbol or ""],
    )
