from __future__ import annotations

from datetime import date, datetime
from fnmatch import fnmatch
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backtest_engine.accounting.contract_support import validate_accounting_contract
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.margin_estimates import MarginEstimateInterval, resolve_margin_estimate
from backtest_engine.data.session_model import parse_date_value


class MarginTierSpec(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    notional_floor: float = Field(default=0.0, ge=0.0)
    notional_cap: float | None = Field(default=None, gt=0.0)
    maintenance_margin_rate: float = Field(ge=0.0, lt=1.0)
    maintenance_amount: float = Field(default=0.0, ge=0.0)
    max_leverage: float | None = Field(default=None, ge=1.0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> MarginTierSpec:
        if self.notional_cap is not None and self.notional_cap <= self.notional_floor:
            raise ValueError("margin tier notional_cap must exceed notional_floor")
        return self


class InstrumentSpec(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    symbol: str = "*"
    venue: str | None = None
    rule_source: str | None = None
    source_as_of: datetime | None = None
    asset_class: str = "generic"
    instrument_type: str = "future"
    valid_from: date | None = None
    valid_to: date | None = None
    price_currency: str | None = None
    pnl_currency: str | None = None
    contract_multiplier: float = Field(default=1.0, gt=0.0)
    lot_size: float = Field(default=1.0, gt=0.0)
    tick_size: float | None = Field(default=None, gt=0.0)
    tick_value: float | None = Field(default=None, gt=0.0)
    initial_margin: float | None = Field(default=None, ge=0.0)
    maximum_quantity: float | None = Field(default=None, gt=0.0)
    margin_estimate_schedule: list[MarginEstimateInterval] = Field(default_factory=list)
    exchange_fee_per_contract: float = Field(default=0.0, ge=0.0)
    base_currency: str | None = None
    quote_currency: str | None = None
    settlement_currency: str | None = None
    contract_type: str = "linear"
    quantity_step: float | None = Field(default=None, gt=0.0)
    minimum_quantity: float | None = Field(default=None, ge=0.0)
    minimum_notional: float | None = Field(default=None, ge=0.0)
    maker_fee_bps: float = Field(default=0.0, ge=0.0)
    taker_fee_bps: float = Field(default=0.0, ge=0.0)
    maintenance_margin_rate: float | None = Field(default=None, ge=0.0, lt=1.0)
    liquidation_fee_bps: float = Field(default=0.0, ge=0.0)
    margin_tiers: list[MarginTierSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_margin_estimates(self) -> InstrumentSpec:
        for previous, current in zip(
            self.margin_estimate_schedule, self.margin_estimate_schedule[1:], strict=False
        ):
            if current.valid_from < previous.valid_to:
                raise ValueError("Margin estimate intervals must be ordered and nonoverlapping")
        return self

    @field_validator("asset_class")
    @classmethod
    def _validate_asset_class(cls, value: str) -> str:
        return normalized_choice(
            value,
            {"futures", "crypto", "equity", "fx", "commodity", "generic"},
            "asset_class",
        )

    @field_validator("instrument_type")
    @classmethod
    def _validate_instrument_type(cls, value: str) -> str:
        return normalized_choice(value, {"future", "spot", "perpetual"}, "instrument_type")

    @field_validator("contract_type")
    @classmethod
    def _validate_contract_type(cls, value: str) -> str:
        return normalized_choice(value, {"linear", "inverse", "quanto", "spot"}, "contract_type")

    @model_validator(mode="after")
    def _validate_margin_tiers(self) -> InstrumentSpec:
        if not self.margin_tiers:
            return self
        ordered = sorted(self.margin_tiers, key=lambda tier: tier.notional_floor)
        if ordered[0].notional_floor != 0.0:
            raise ValueError("first margin tier must start at zero")
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.notional_cap != current.notional_floor:
                raise ValueError("margin tiers must be ordered and contiguous")
        if ordered != self.margin_tiers:
            raise ValueError("margin tiers must be ordered by notional_floor")
        return self


def build_instrument_specs[SpecT: InstrumentSpec](
    raw_payload: dict[str, Any],
    spec_type: type[SpecT],
) -> list[SpecT]:
    if "instruments" in raw_payload:
        return [
            spec_type(**parse_instrument_spec_dates(raw_item))
            for raw_item in raw_payload["instruments"]
        ]
    return [spec_type(**parse_instrument_spec_dates(raw_payload))]


def parse_instrument_spec_dates(raw_spec: dict[str, Any]) -> dict[str, Any]:
    parsed = dict(raw_spec)
    for field_name in ("valid_from", "valid_to"):
        if field_name in parsed and parsed[field_name] is not None:
            parsed[field_name] = parse_date_value(parsed[field_name])
    return parsed


def select_instrument_spec[SpecT: InstrumentSpec](
    *,
    row: dict[str, Any],
    current_ts: datetime,
    specs: list[SpecT],
    fallback: SpecT,
    extra_symbols: list[str] | None = None,
) -> SpecT:
    symbols = [
        str(row.get("active_contract", "")),
        str(row.get("source_contract_symbol", "")),
        str(row.get("continuous_symbol", "")),
        *(extra_symbols or []),
    ]
    for spec in specs:
        if spec.valid_from is not None and current_ts.date() < spec.valid_from:
            continue
        if spec.valid_to is not None and current_ts.date() > spec.valid_to:
            continue
        if any(fnmatch(symbol, spec.symbol) for symbol in symbols if symbol):
            return resolved_spec(spec, fallback, row, current_ts)
    return resolved_spec(fallback, fallback, row, current_ts)


def resolved_spec[SpecT: InstrumentSpec](
    spec: SpecT, fallback: SpecT, row: dict[str, Any], current_ts: datetime
) -> SpecT:
    updates = {
        field: getattr(fallback, field)
        for field in (
            "asset_class",
            "instrument_type",
            "contract_type",
            "base_currency",
            "quote_currency",
            "settlement_currency",
        )
        if field not in spec.model_fields_set
    }
    price = (
        spec.price_currency
        or spec.quote_currency
        or fallback.price_currency
        or row.get("quote_currency")
        or row.get("currency")
    )
    pnl = (
        spec.pnl_currency
        or spec.settlement_currency
        or fallback.pnl_currency
        or row.get("settlement_currency")
        or price
    )
    if not price or not pnl:
        raise DatasetValidationError(
            f"Instrument {spec.symbol}: price_currency and pnl_currency are required"
        )
    updates.update(price_currency=price, pnl_currency=pnl)
    resolved = resolve_margin_estimate(spec.model_copy(update=updates), row, current_ts)
    validate_accounting_contract(resolved)
    return resolved


def normalized_choice(value: str, choices: set[str], field: str) -> str:
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"Invalid {field}: {value}; expected one of: {', '.join(sorted(choices))}")
    return normalized
