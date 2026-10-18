"""Validated L2 public order-book state."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.execution.models import OrderRequest


class L2OrderBook:
    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_sequence: int | None = None

    def apply_batch(self, rows: list[dict[str, Any]]) -> dict[tuple[str, float], float]:
        if not rows:
            return {}
        sequence = _same_int(rows, "sequence")
        update_type = _same_text(rows, "update_type")
        if update_type == "snapshot":
            self.bids.clear()
            self.asks.clear()
        elif update_type == "delta":
            self._validate_delta(sequence, rows)
        else:
            raise DatasetValidationError(f"Unsupported book update_type: {update_type}")
        removed: dict[tuple[str, float], float] = defaultdict(float)
        for row in rows:
            side = str(row["side"]).lower()
            levels = self._levels(side)
            price = _positive_float(row["price"], "book price")
            size = _non_negative_float(row["size"], "book size")
            previous = levels.get(price, 0.0)
            if size < previous:
                removed[(side, price)] += previous - size
            if size == 0.0:
                levels.pop(price, None)
            else:
                levels[price] = size
        self.last_sequence = sequence
        self._validate_spread()
        return dict(removed)

    def best_bid(self) -> float | None:
        return max(self.bids, default=None)

    def best_ask(self) -> float | None:
        return min(self.asks, default=None)

    def crosses(self, request: OrderRequest) -> bool:
        if request.order_type == "market":
            return True
        if request.limit_price is None:
            return False
        opposite = self.best_ask() if request.side == "buy" else self.best_bid()
        if opposite is None:
            return False
        if request.side == "buy":
            return request.limit_price >= opposite
        return request.limit_price <= opposite

    def available(self, side: str, limit_price: float | None = None) -> float:
        return sum(size for _, size in self._sweep_levels(side, limit_price))

    def sweep(
        self,
        side: str,
        quantity: float,
        limit_price: float | None = None,
    ) -> list[tuple[float, float]]:
        fills: list[tuple[float, float]] = []
        remaining = quantity
        levels = self.asks if side == "buy" else self.bids
        for price, size in self._sweep_levels(side, limit_price):
            filled = min(remaining, size)
            if filled <= 0.0:
                continue
            fills.append((filled, price))
            remaining -= filled
            updated = size - filled
            if updated <= 0.0:
                levels.pop(price, None)
            else:
                levels[price] = updated
            if remaining <= 0.0:
                break
        return fills

    def level_size(self, side: str, price: float) -> float:
        return self._levels(side).get(price, 0.0)

    def _sweep_levels(self, side: str, limit_price: float | None) -> list[tuple[float, float]]:
        levels = self.asks if side == "buy" else self.bids
        prices = sorted(levels, reverse=side == "sell")
        if limit_price is not None:
            prices = [
                price
                for price in prices
                if (price <= limit_price if side == "buy" else price >= limit_price)
            ]
        return [(price, levels[price]) for price in prices]

    def _levels(self, side: str) -> dict[float, float]:
        if side == "bid":
            return self.bids
        if side == "ask":
            return self.asks
        raise DatasetValidationError(f"Book side must be bid or ask: {side}")

    def _validate_delta(self, sequence: int, rows: list[dict[str, Any]]) -> None:
        if self.last_sequence is None:
            raise DatasetValidationError("L2 delta received before a snapshot")
        if sequence != self.last_sequence + 1:
            raise DatasetValidationError(
                f"L2 sequence gap: expected {self.last_sequence + 1}, got {sequence}"
            )
        previous = {row.get("previous_sequence") for row in rows}
        previous.discard(None)
        if previous and previous != {self.last_sequence}:
            raise DatasetValidationError("L2 previous_sequence does not match book state")

    def _validate_spread(self) -> None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is not None and ask is not None and bid >= ask:
            raise DatasetValidationError(f"Crossed L2 book after update: bid={bid}, ask={ask}")


def _same_int(rows: list[dict[str, Any]], key: str) -> int:
    values = {int(row[key]) for row in rows}
    if len(values) != 1:
        raise DatasetValidationError(f"L2 batch must contain one {key}")
    return values.pop()


def _same_text(rows: list[dict[str, Any]], key: str) -> str:
    values = {str(row[key]).lower() for row in rows}
    if len(values) != 1:
        raise DatasetValidationError(f"L2 batch must contain one {key}")
    return values.pop()


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(f"{label} must be finite and positive") from exc
    if not isfinite(parsed) or parsed <= 0.0:
        raise DatasetValidationError(f"{label} must be finite and positive")
    return parsed


def _non_negative_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(f"{label} must be finite and non-negative") from exc
    if not isfinite(parsed) or parsed < 0.0:
        raise DatasetValidationError(f"{label} must be finite and non-negative")
    return parsed
