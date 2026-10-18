"""Sparse spot execution; every cash transfer is booked by the portfolio ledger."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from strategy_sdk import AccountSnapshot, Decision

from backtest_engine.accounting.ledger import MultiCurrencyLedger
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.execution.models import OrderRequest
from backtest_engine.execution.simulator import ExecutionSimulatorV3


@dataclass(frozen=True)
class SpotPrices:
    opened: Sequence[datetime]
    available: Sequence[datetime]
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    symbol: str
    base: str
    quote: str


@dataclass
class SpotScheduleResult:
    metrics: dict
    equity: np.ndarray
    fills: list[dict]
    cash_flows: list[dict]


def run_spot_schedule(
    prices: SpotPrices,
    decisions: Sequence[tuple[int, bool]],
    entry_decision: Callable[[AccountSnapshot, float], Decision],
    *,
    initial_capital: float = 1000,
    fee_bps: float = 10,
) -> SpotScheduleResult:
    """Execute causal targets on the next observed bar's open, with no borrowing.

    Decisions contain completed-bar indices. Between fills balances do not
    change, so valuations consume ledger balances without replaying empty orders.
    No second cash/PnL calculation or synthetic candle interpolation is used.
    """
    ledger = MultiCurrencyLedger(base_currency=prices.quote)
    ledger.deposit(ts=prices.opened[0], currency=prices.quote, amount=initial_capital)
    simulator = ExecutionSimulatorV3()
    equity = np.full(len(prices.closes), initial_capital, dtype=np.float64)
    fills, segment_start, previous_index = [], 0, -1
    for index, want_long in decisions:
        if not previous_index < index < len(prices.closes) - 1:
            raise DatasetValidationError("Spot decisions must increase and have a following bar")
        previous_index = index
        quantity = ledger.cash.get(prices.base, 0.0)
        if (want_long and quantity > 0) or (not want_long and quantity == 0):
            continue
        if want_long:
            account = AccountSnapshot(
                prices.quote,
                ledger.equity({(prices.base, prices.quote): float(prices.closes[index])}),
                ledger.cash.get(prices.quote, 0.0),
            )
            decision = entry_decision(account, float(prices.closes[index]))
            if decision.kind in {"HOLD", "SKIP"}:
                continue
            if decision.target != 1 or decision.contracts is None:
                raise DatasetValidationError(
                    "Spot entry callback must return a sized long decision"
                )
            delta = float(decision.contracts)
        else:
            delta = -quantity
        fill_index = index + 1
        if prices.available[index] > prices.opened[fill_index]:
            raise DatasetValidationError("Spot execution cannot precede signal availability")
        order_id = f"spot-{len(fills)}"
        state = simulator.submit(
            OrderRequest(order_id, "buy" if delta > 0 else "sell", abs(delta)),
            bar_index=index,
            created_ts=prices.available[index],
            position_quantity=quantity,
        )
        if state.status != "active":
            raise DatasetValidationError(f"Spot order was rejected: {order_id}")
        produced = simulator.process_bar(
            {
                "ts": prices.opened[fill_index],
                "open": float(prices.opens[fill_index]),
                "high": float(prices.highs[fill_index]),
                "low": float(prices.lows[fill_index]),
                "close": float(prices.closes[fill_index]),
                "volume": float(prices.volumes[fill_index]),
            },
            bar_index=fill_index,
        )
        if len(produced) != 1 or produced[0].quantity != abs(delta):
            raise DatasetValidationError(f"Spot research requires a complete fill: {order_id}")
        fill = produced[0]
        fee = fill.quantity * fill.price * fee_bps / 10_000
        # A rejected cash/inventory transfer fails this candidate; never borrow.
        old_cash = ledger.cash.get(prices.quote, 0.0)
        equity[segment_start:fill_index] = (
            old_cash + quantity * prices.closes[segment_start:fill_index]
        )
        ledger.apply_spot_trade(
            ts=prices.opened[fill_index],
            symbol=prices.symbol,
            side=fill.side,
            quantity=fill.quantity,
            price=fill.price,
            base_currency=prices.base,
            quote_currency=prices.quote,
            fee_amount=fee,
        )
        segment_start = fill_index
        fills.append(
            {
                "decision_ts": prices.available[index],
                "fill_ts": prices.opened[fill_index],
                "fill_index": fill_index,
                "side": fill.side,
                "quantity": fill.quantity,
                "price": fill.price,
                "fee": fee,
                "quote_cash": ledger.cash.get(prices.quote, 0.0),
                "base_quantity": ledger.cash.get(prices.base, 0.0),
                "execution_delay_seconds": (
                    prices.opened[fill_index] - prices.available[index]
                ).total_seconds(),
            }
        )
    cash, quantity = ledger.cash.get(prices.quote, 0.0), ledger.cash.get(prices.base, 0.0)
    equity[segment_start:] = cash + quantity * prices.closes[segment_start:]
    final = ledger.equity({(prices.base, prices.quote): float(prices.closes[-1])})
    cash_reconstructed: dict[str, float] = {}
    flows = []
    for flow in ledger.cash_flows:
        cash_reconstructed[flow.currency] = cash_reconstructed.get(flow.currency, 0) + flow.amount
        flows.append(vars(flow))
    if ledger.reconcile_cash(cash_reconstructed, tolerance=1e-9) or abs(final - equity[-1]) > 1e-9:
        raise DatasetValidationError("Spot ledger reconciliation failed")
    metrics = {
        "currency": prices.quote,
        "initial_capital": initial_capital,
        "final_equity_cash": final,
        "return_pct": (final / initial_capital - 1) * 100,
        "max_drawdown_pct": float(np.min(equity / np.maximum.accumulate(equity) - 1) * 100),
        "fills": len(fills),
        "fees_cash": sum(fill["fee"] for fill in fills),
        "quote_cash": cash,
        "base_quantity": quantity,
        "reconciliation_error": abs(final - equity[-1]),
        "max_execution_delay_seconds": max(
            (fill["execution_delay_seconds"] for fill in fills), default=0
        ),
    }
    return SpotScheduleResult(metrics, equity, fills, flows)
