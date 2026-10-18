"""Shared-collateral accounting for several linear perpetual positions."""

from __future__ import annotations

from dataclasses import dataclass

from backtest_engine.accounting.perpetual import (
    TieredMarginSchedule,
    realized_pnl,
    updated_entry_price,
)
from backtest_engine.core.exceptions import DatasetValidationError


@dataclass
class CrossMarginPosition:
    point_value: float
    leverage: float
    margin_schedule: TieredMarginSchedule
    liquidation_fee_bps: float
    quantity: float = 0.0
    entry_price: float = 0.0

    def unrealized_pnl(self, mark: float) -> float:
        return self.quantity * (mark - self.entry_price) * self.point_value

    def notional(self, mark: float) -> float:
        return abs(self.quantity) * mark * self.point_value


@dataclass(frozen=True)
class CrossMarginTransition:
    symbol: str
    realized_pnl: float
    trading_fee: float
    funding_payment: float
    additional_cost: float
    ending_quantity: float


@dataclass(frozen=True)
class CrossMarginLiquidation:
    symbol: str
    price: float
    closed_quantity: float
    realized_pnl: float
    liquidation_fee: float


class CrossMarginAccount:
    def __init__(self, wallet_balance: float) -> None:
        self.wallet_balance = float(wallet_balance)
        self.positions: dict[str, CrossMarginPosition] = {}

    def register(
        self,
        symbol: str,
        *,
        point_value: float,
        leverage: float,
        margin_schedule: TieredMarginSchedule,
        liquidation_fee_bps: float,
    ) -> None:
        if symbol in self.positions:
            position = self.positions[symbol]
            if position.point_value != point_value:
                raise DatasetValidationError(f"Point value changed for open cross-margin {symbol}")
            position.leverage = leverage
            position.margin_schedule = margin_schedule
            position.liquidation_fee_bps = liquidation_fee_bps
            return
        self.positions[symbol] = CrossMarginPosition(
            point_value,
            leverage,
            margin_schedule,
            liquidation_fee_bps,
        )

    def apply_batch(
        self,
        orders: list[dict[str, float | str]],
        marks: dict[str, float],
    ) -> list[CrossMarginTransition]:
        snapshot = self._snapshot()
        transitions = []
        risk_increased = False
        for order in orders:
            symbol = str(order["symbol"])
            position = self._position(symbol)
            delta = float(order["quantity_delta"])
            price = _positive(float(order["fill_price"]), "fill_price")
            previous_quantity = position.quantity
            realized = realized_pnl(
                previous_quantity,
                position.entry_price,
                delta,
                price,
                position.point_value,
            )
            position.entry_price = updated_entry_price(
                previous_quantity, position.entry_price, delta, price
            )
            position.quantity += delta
            fee = abs(delta) * price * position.point_value * float(order["fee_bps"]) / 10_000.0
            mark = _positive(marks[symbol], f"mark[{symbol}]")
            funding = (
                float(order["funding_payment"])
                if "funding_payment" in order
                else (
                    position.quantity * mark * position.point_value * float(order["funding_rate"])
                )
            )
            additional_cost = float(order.get("additional_cost", 0.0))
            self.wallet_balance += realized - fee - additional_cost
            risk_increased |= abs(position.quantity) > abs(previous_quantity)
            transitions.append(
                CrossMarginTransition(
                    symbol,
                    realized,
                    fee,
                    funding,
                    additional_cost,
                    position.quantity,
                )
            )
        if risk_increased and self.initial_margin(marks) > self.equity(marks):
            self._restore(snapshot)
            raise DatasetValidationError("Perpetual batch exceeds shared cross-margin collateral")
        # Future funding income cannot finance the initial margin of a fill.
        self.wallet_balance -= sum(item.funding_payment for item in transitions)
        return transitions

    def equity(self, marks: dict[str, float]) -> float:
        return self.wallet_balance + sum(
            position.unrealized_pnl(marks[symbol])
            for symbol, position in self.positions.items()
            if position.quantity
        )

    def maintenance_margin(self, marks: dict[str, float]) -> float:
        return sum(
            position.margin_schedule.maintenance_margin(position.notional(marks[symbol]))
            for symbol, position in self.positions.items()
            if position.quantity
        )

    def initial_margin(self, marks: dict[str, float]) -> float:
        return sum(
            position.notional(marks[symbol]) / position.leverage
            for symbol, position in self.positions.items()
            if position.quantity
        )

    def liquidation_fee_reserve(self, marks: dict[str, float]) -> float:
        return sum(
            position.notional(marks[symbol]) * position.liquidation_fee_bps / 10_000.0
            for symbol, position in self.positions.items()
            if position.quantity
        )

    def liquidate_if_required(
        self,
        adverse_marks: dict[str, float],
        *,
        target_buffer: float = 1.05,
    ) -> list[CrossMarginLiquidation]:
        required = self.maintenance_margin(adverse_marks) + self.liquidation_fee_reserve(
            adverse_marks
        )
        if self.equity(adverse_marks) >= required:
            return []
        liquidations = []
        for symbol in self._liquidation_order(adverse_marks):
            position = self.positions[symbol]
            price = adverse_marks[symbol]
            quantity = abs(position.quantity)
            delta = -position.quantity
            realized = realized_pnl(
                position.quantity,
                position.entry_price,
                delta,
                price,
                position.point_value,
            )
            fee = quantity * price * position.point_value * position.liquidation_fee_bps / 10_000.0
            self.wallet_balance += realized - fee
            position.quantity = 0.0
            position.entry_price = 0.0
            liquidations.append(CrossMarginLiquidation(symbol, price, quantity, realized, fee))
            required = self.maintenance_margin(adverse_marks) * target_buffer
            if self.equity(adverse_marks) >= required:
                break
        return liquidations

    def _liquidation_order(self, marks: dict[str, float]) -> list[str]:
        return sorted(
            (symbol for symbol, position in self.positions.items() if position.quantity),
            key=lambda symbol: (
                self.positions[symbol].unrealized_pnl(marks[symbol]),
                -self.positions[symbol].notional(marks[symbol]),
            ),
        )

    def _position(self, symbol: str) -> CrossMarginPosition:
        if symbol not in self.positions:
            raise DatasetValidationError(f"Unregistered cross-margin symbol: {symbol}")
        return self.positions[symbol]

    def _snapshot(self) -> tuple[float, dict[str, tuple[float, float]]]:
        return self.wallet_balance, {
            symbol: (position.quantity, position.entry_price)
            for symbol, position in self.positions.items()
        }

    def _restore(self, snapshot: tuple[float, dict[str, tuple[float, float]]]) -> None:
        self.wallet_balance = snapshot[0]
        for symbol, (quantity, entry_price) in snapshot[1].items():
            self.positions[symbol].quantity = quantity
            self.positions[symbol].entry_price = entry_price


def _positive(value: float, label: str) -> float:
    if value <= 0.0:
        raise DatasetValidationError(f"{label} must be positive")
    return value
