"""Stateful linear-perpetual margin and liquidation accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass

from backtest_engine.core.exceptions import DatasetValidationError


@dataclass(frozen=True)
class MarginTier:
    notional_floor: float
    notional_cap: float | None
    maintenance_margin_rate: float
    maintenance_amount: float = 0.0


@dataclass(frozen=True)
class FillAccounting:
    realized_pnl: float
    fee: float
    wallet_delta: float


@dataclass(frozen=True)
class LiquidationResult:
    liquidated: bool
    status: str
    price: float | None
    bankruptcy_price: float | None
    closed_quantity: float
    remaining_quantity: float
    realized_pnl: float
    fee: float


class TieredMarginSchedule:
    def __init__(self, tiers: list[MarginTier]) -> None:
        if not tiers:
            raise DatasetValidationError("Perpetual margin schedule must contain a tier")
        ordered = sorted(tiers, key=lambda tier: tier.notional_floor)
        if ordered[0].notional_floor != 0.0:
            raise DatasetValidationError("First perpetual margin tier must start at zero")
        for index, tier in enumerate(ordered):
            if tier.maintenance_margin_rate < 0.0 or tier.maintenance_margin_rate >= 1.0:
                raise DatasetValidationError("maintenance_margin_rate must be in [0, 1)")
            if tier.notional_cap is not None and tier.notional_cap <= tier.notional_floor:
                raise DatasetValidationError("Margin tier cap must exceed its floor")
            if index and ordered[index - 1].notional_cap != tier.notional_floor:
                raise DatasetValidationError("Perpetual margin tiers must be contiguous")
        self.tiers = tuple(ordered)

    def tier(self, notional: float) -> MarginTier:
        value = abs(notional)
        for tier in self.tiers:
            if tier.notional_cap is None or value <= tier.notional_cap:
                return tier
        raise DatasetValidationError(f"Perpetual notional exceeds margin tiers: {value}")

    def maintenance_margin(self, notional: float) -> float:
        tier = self.tier(notional)
        return max(abs(notional) * tier.maintenance_margin_rate - tier.maintenance_amount, 0.0)


class LinearPerpetualAccount:
    def __init__(
        self,
        *,
        wallet_balance: float,
        point_value: float,
        leverage: float,
        margin_schedule: TieredMarginSchedule,
        liquidation_fee_bps: float,
    ) -> None:
        if point_value <= 0.0 or leverage < 1.0:
            raise DatasetValidationError("Perpetual point_value and leverage are invalid")
        self.wallet_balance = float(wallet_balance)
        self.point_value = float(point_value)
        self.leverage = float(leverage)
        self.margin_schedule = margin_schedule
        self.liquidation_fee_bps = float(liquidation_fee_bps)
        self.quantity = 0.0
        self.entry_price = 0.0

    def apply_fill(self, quantity_delta: float, price: float, fee_bps: float) -> FillAccounting:
        if not math.isfinite(price) or price <= 0.0:
            raise DatasetValidationError("Perpetual fill price must be finite and positive")
        if quantity_delta == 0.0:
            return FillAccounting(0.0, 0.0, 0.0)
        previous = self.quantity
        realized = realized_pnl(previous, self.entry_price, quantity_delta, price, self.point_value)
        updated = previous + quantity_delta
        self.entry_price = updated_entry_price(previous, self.entry_price, quantity_delta, price)
        self.quantity = updated
        fee = abs(quantity_delta) * price * self.point_value * fee_bps / 10_000.0
        wallet_delta = realized - fee
        self.wallet_balance += wallet_delta
        return FillAccounting(realized, fee, wallet_delta)

    def apply_funding(self, mark_price: float, funding_rate: float) -> float:
        payment = self.quantity * mark_price * self.point_value * funding_rate
        return self.apply_funding_payment(payment)

    def apply_funding_payment(self, payment: float) -> float:
        """Apply a priced funding cash flow; positive means paid by this account."""
        self.wallet_balance -= payment
        return payment

    def unrealized_pnl(self, mark_price: float) -> float:
        return self.quantity * (mark_price - self.entry_price) * self.point_value

    def equity(self, mark_price: float) -> float:
        return self.wallet_balance + self.unrealized_pnl(mark_price)

    def notional(self, mark_price: float) -> float:
        return abs(self.quantity) * mark_price * self.point_value

    def maintenance_margin(self, mark_price: float) -> float:
        return self.margin_schedule.maintenance_margin(self.notional(mark_price))

    def initial_margin(self, mark_price: float) -> float:
        return self.notional(mark_price) / self.leverage

    def liquidation_price(self) -> float | None:
        return self._critical_price(include_maintenance=True, include_liquidation_fee=True)

    def bankruptcy_price(self) -> float | None:
        return self._critical_price(include_maintenance=False, include_liquidation_fee=False)

    def liquidate_if_crossed(
        self,
        *,
        mark_low: float,
        mark_high: float,
        quantity_step: float | None,
        target_buffer: float = 1.05,
        minimum_close_fraction: float = 0.25,
    ) -> LiquidationResult:
        price = self.liquidation_price()
        bankruptcy = self.bankruptcy_price()
        crossed = price is not None and (
            (self.quantity > 0.0 and mark_low <= price)
            or (self.quantity < 0.0 and mark_high >= price)
        )
        if not crossed or price is None:
            return LiquidationResult(False, "none", None, bankruptcy, 0.0, self.quantity, 0.0, 0.0)
        close_quantity = self._partial_close_quantity(
            price,
            quantity_step,
            target_buffer,
            minimum_close_fraction,
        )
        delta = -math.copysign(close_quantity, self.quantity)
        fill = self.apply_fill(delta, price, self.liquidation_fee_bps)
        status = "full" if self.quantity == 0.0 else "partial"
        return LiquidationResult(
            True,
            status,
            price,
            bankruptcy,
            close_quantity,
            self.quantity,
            fill.realized_pnl,
            fill.fee,
        )

    def _partial_close_quantity(
        self,
        price: float,
        quantity_step: float | None,
        target_buffer: float,
        minimum_fraction: float,
    ) -> float:
        absolute = abs(self.quantity)
        step = quantity_step or absolute * minimum_fraction
        minimum = max(step, absolute * minimum_fraction)
        # Minimum liquidation fractions do not override the executable lot step.
        # Round the whole reduction upward so its residual remains executable.
        if quantity_step:
            minimum = math.ceil(minimum / quantity_step) * quantity_step
        step = min(minimum, absolute)
        close = step
        while close < absolute:
            remaining = absolute - close
            fee = close * price * self.point_value * self.liquidation_fee_bps / 10_000.0
            trial_equity = self.equity(price) - fee
            maintenance = self.margin_schedule.maintenance_margin(
                remaining * price * self.point_value
            )
            if trial_equity >= maintenance * target_buffer:
                return close
            close = min(close + step, absolute)
        return absolute

    def _critical_price(
        self,
        *,
        include_maintenance: bool,
        include_liquidation_fee: bool,
    ) -> float | None:
        if self.quantity == 0.0:
            return None

        tier = self.margin_schedule.tiers[0]
        rate = tier.maintenance_margin_rate if include_maintenance else 0.0
        rate += self.liquidation_fee_bps / 10_000.0 if include_liquidation_fee else 0.0
        if (
            len(self.margin_schedule.tiers) == 1
            and tier.notional_cap is None
            and tier.maintenance_amount == 0.0
            and rate < 1.0
        ):
            # With a single uncapped rate the buffer is linear in price. Keep
            # tiered/deducted schedules on the existing general numerical path.
            signed_value = self.quantity * self.point_value
            offset = self.wallet_balance - signed_value * self.entry_price
            slope = signed_value - abs(signed_value) * rate
            price = -offset / slope
            if self.quantity > 0.0 and price < 0.0:
                return None
            if price >= 0.0 and (self.quantity > 0.0 or price <= self.entry_price * 1_000_000):
                return price

        def buffer(price: float) -> float:
            result = self.equity(price)
            if include_maintenance:
                result -= self.maintenance_margin(price)
            if include_liquidation_fee:
                result -= self.notional(price) * self.liquidation_fee_bps / 10_000.0
            return result

        if self.quantity > 0.0:
            low, high = 0.0, max(self.entry_price, 1.0)
            if buffer(low) > 0.0:
                return None
        else:
            low, high = 0.0, max(self.entry_price, 1.0)
            while buffer(high) > 0.0 and high < self.entry_price * 1_000_000.0:
                high *= 2.0
            if buffer(high) > 0.0:
                return None
        low_buffer = buffer(low)
        for _ in range(100):
            midpoint = (low + high) / 2.0
            midpoint_buffer = buffer(midpoint)
            if low_buffer * midpoint_buffer <= 0.0:
                high = midpoint
            else:
                low = midpoint
                low_buffer = midpoint_buffer
        return (low + high) / 2.0


def realized_pnl(
    quantity: float,
    entry_price: float,
    quantity_delta: float,
    price: float,
    point_value: float,
) -> float:
    if quantity == 0.0 or quantity * quantity_delta >= 0.0:
        return 0.0
    closing = min(abs(quantity), abs(quantity_delta))
    return closing * (price - entry_price) * math.copysign(1.0, quantity) * point_value


def updated_entry_price(
    quantity: float,
    entry_price: float,
    quantity_delta: float,
    price: float,
) -> float:
    updated = quantity + quantity_delta
    if updated == 0.0:
        return 0.0
    if quantity == 0.0 or quantity * quantity_delta > 0.0:
        return (entry_price * abs(quantity) + price * abs(quantity_delta)) / abs(updated)
    if quantity * updated < 0.0:
        return price
    return entry_price
