"""Reject simulated crypto fills that violate the historical instrument filters."""

from __future__ import annotations

import math
from typing import Any

from backtest_engine.core.exceptions import DatasetValidationError


def validate_crypto_fill(
    instrument: Any,
    previous: float,
    current: float,
    price: float,
    leverage: float,
) -> None:
    delta = abs(current - previous)
    if delta > 1e-12:
        step = instrument.quantity_step
        if step and not math.isclose(delta / step, round(delta / step), abs_tol=1e-8):
            raise DatasetValidationError(
                f"{instrument.symbol} fill quantity {delta} violates quantity_step={step}"
            )
        reducing = instrument.instrument_type == "perpetual" and abs(current) < abs(previous)
        if instrument.minimum_quantity and delta + 1e-12 < instrument.minimum_quantity:
            raise DatasetValidationError(
                f"{instrument.symbol} fill below minimum_quantity={instrument.minimum_quantity}"
            )
        notional = delta * price * instrument.contract_multiplier * instrument.lot_size
        if (
            not reducing
            and instrument.minimum_notional
            and notional + 1e-9 < instrument.minimum_notional
        ):
            raise DatasetValidationError(
                f"{instrument.symbol} fill notional {notional} below "
                f"minimum_notional={instrument.minimum_notional}"
            )
    if instrument.instrument_type == "perpetual" and current:
        notional = abs(current) * price * instrument.contract_multiplier * instrument.lot_size
        for tier in instrument.margin_tiers:
            if tier.notional_cap is None or notional <= tier.notional_cap:
                if tier.max_leverage and leverage > tier.max_leverage:
                    raise DatasetValidationError(
                        f"{instrument.symbol} leverage={leverage} exceeds historical "
                        f"max_leverage={tier.max_leverage}"
                    )
                break
