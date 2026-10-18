"""Exchange-neutral spot and linear perpetual accounting primitives."""

from __future__ import annotations

from dataclasses import dataclass

from backtest_engine.core.exceptions import DatasetValidationError


@dataclass(frozen=True)
class CryptoSettlement:
    trading_fee: float
    funding_payment: float
    maintenance_margin: float
    liquidation_fee: float
    liquidation_price: float | None
    liquidated: bool
    effective_exit_price: float
    bankruptcy_price: float | None = None
    liquidation_quantity: float = 0.0
    remaining_quantity: float = 0.0
    liquidation_status: str = "none"


def settle_crypto_interval(
    *,
    instrument_type: str,
    contract_type: str,
    quantity: float,
    quantity_delta: float,
    point_value: float,
    fill_price: float,
    exit_price: float,
    low_price: float,
    high_price: float,
    equity: float,
    fee_bps: float,
    funding_rate: float,
    maintenance_margin_rate: float | None,
    liquidation_fee_bps: float,
) -> CryptoSettlement:
    """Settle one interval; positive funding is paid by longs and received by shorts."""
    if contract_type not in {"linear", "spot"}:
        raise DatasetValidationError(
            f"Crypto event accounting supports linear/spot contracts, got {contract_type}"
        )
    if instrument_type == "spot" and quantity < 0.0:
        raise DatasetValidationError("Spot instruments cannot hold a negative quantity")
    trading_fee = abs(quantity_delta) * fill_price * point_value * fee_bps / 10_000.0
    if instrument_type != "perpetual" or quantity == 0.0:
        return CryptoSettlement(
            trading_fee=trading_fee,
            funding_payment=0.0,
            maintenance_margin=0.0,
            liquidation_fee=0.0,
            liquidation_price=None,
            liquidated=False,
            effective_exit_price=exit_price,
        )

    funding_payment = quantity * exit_price * point_value * funding_rate
    maintenance_rate = float(maintenance_margin_rate or 0.0)
    maintenance_margin = abs(quantity) * exit_price * point_value * maintenance_rate
    liquidation_price = linear_liquidation_price(
        quantity=quantity,
        point_value=point_value,
        entry_price=fill_price,
        equity=equity,
        maintenance_margin_rate=maintenance_rate,
    )
    liquidated = liquidation_crossed(
        quantity=quantity,
        liquidation_price=liquidation_price,
        low_price=low_price,
        high_price=high_price,
    )
    effective_exit = liquidation_price if liquidated else exit_price
    liquidation_fee = (
        abs(quantity) * effective_exit * point_value * liquidation_fee_bps / 10_000.0
        if liquidated
        else 0.0
    )
    return CryptoSettlement(
        trading_fee=trading_fee,
        funding_payment=funding_payment,
        maintenance_margin=maintenance_margin,
        liquidation_fee=liquidation_fee,
        liquidation_price=liquidation_price if liquidated else None,
        liquidated=liquidated,
        effective_exit_price=effective_exit,
    )


def linear_liquidation_price(
    *,
    quantity: float,
    point_value: float,
    entry_price: float,
    equity: float,
    maintenance_margin_rate: float,
) -> float | None:
    if quantity == 0.0 or maintenance_margin_rate <= 0.0:
        return None
    absolute_quantity = abs(quantity)
    equity_per_unit = equity / (absolute_quantity * point_value)
    if quantity > 0.0:
        price = (entry_price - equity_per_unit) / (1.0 - maintenance_margin_rate)
    else:
        price = (entry_price + equity_per_unit) / (1.0 + maintenance_margin_rate)
    return max(price, 0.0)


def liquidation_crossed(
    *,
    quantity: float,
    liquidation_price: float | None,
    low_price: float,
    high_price: float,
) -> bool:
    if liquidation_price is None:
        return False
    if quantity > 0.0:
        return low_price <= liquidation_price
    return high_price >= liquidation_price
