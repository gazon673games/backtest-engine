"""Causal sparse candidate sweeps through production fills, accounts and cash ledger."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from heapq import heapify, heappop, heappush

import numpy as np
from strategy_sdk import AccountSnapshot

from backtest_engine.accounting.fees import trading_fee_bps
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.execution.models import OrderRequest
from backtest_engine.execution.simulator import ExecutionSimulatorV3
from backtest_engine.intraday.event.crypto_rules import validate_crypto_fill
from backtest_engine.research.candidate_schedule_state import ScheduleAccount


@dataclass
class CandidateScheduleResult:
    metrics: dict
    equity: np.ndarray
    quantities: np.ndarray
    fills: list[dict]
    cash_flows: list[dict]
    liquidations: list[dict]


def run_candidate_schedule(
    prices,
    spec,
    decisions,
    entry_quantity,
    *,
    initial_capital=1000,
    fee_bps=10,
    friction_bps=0,
    latency_bars=0,
):
    """Desired directions; trade next observed open, never resize a held direction.

    Full fills are an explicit screening assumption. The ordinary Workbench runner
    verifies selected cases with its complete runtime and liquidity constraints.
    This path accepts one immutable specification, native settlement currency,
    spot or linear perpetuals, and the production 1x cross-margin model.
    """
    indices = [index for index, _ in decisions]
    if not isinstance(latency_bars, int) or latency_bars < 0:
        raise DatasetValidationError("Candidate latency_bars must be a non-negative integer")
    if indices != sorted(set(indices)) or any(
        i < 0 or i >= len(prices.closes) - 1 for i in indices
    ):
        raise DatasetValidationError(
            "Candidate decisions require increasing completed bars with a following bar"
        )
    if latency_bars and any(
        right[0] - left[0] <= latency_bars and right[1] != left[1]
        for left, right in zip(decisions, decisions[1:], strict=False)
    ):
        raise DatasetValidationError(
            "Candidate delayed direction changes overlap pending fills; use the ordinary runtime"
        )
    account = ScheduleAccount(prices, spec, initial_capital)
    simulator = ExecutionSimulatorV3(market_impact_bps=friction_bps)
    fees = trading_fee_bps(spec, "taker", fee_bps)
    decision_map = dict(decisions)
    funding_indices = (
        set(np.flatnonzero(prices.funding_unit)) if account.perpetual is not None else set()
    )
    pending, fills = {}, []
    scheduled = set(indices) | funding_indices
    timeline = list(scheduled)
    heapify(timeline)
    while timeline:
        index = heappop(timeline)
        scheduled.remove(index)
        account.advance(index)
        if index in pending:
            decision_index, target = pending.pop(index)
            delta = target - account.quantity
            if delta:
                if prices.available[decision_index] > prices.opened[index]:
                    raise DatasetValidationError("Candidate fill precedes signal availability")
                order = OrderRequest(
                    f"candidate-{len(fills)}", "buy" if delta > 0 else "sell", abs(delta)
                )
                simulator.submit(
                    order,
                    bar_index=decision_index,
                    created_ts=prices.available[decision_index],
                    position_quantity=account.quantity,
                )
                produced = simulator.process_bar(
                    {
                        "ts": prices.opened[index],
                        "open": float(prices.opens[index]),
                        "high": float(prices.highs[index]),
                        "low": float(prices.lows[index]),
                        "close": float(prices.closes[index]),
                        "volume": float(prices.volumes[index]),
                    },
                    bar_index=index,
                )
                if len(produced) != 1 or produced[0].quantity != abs(delta):
                    raise DatasetValidationError(
                        "Candidate screening requires a confirmed complete fill"
                    )
                fill = produced[0]
                validate_crypto_fill(
                    spec, account.quantity, account.quantity + delta, fill.price, 1
                )
                if account.perpetual is not None:
                    fee = account.post_perpetual_fill(index, delta, fill.price, fees)
                else:
                    fee = fill.quantity * fill.price * fees / 10_000
                    account.ledger.apply_spot_trade(
                        ts=prices.opened[index],
                        symbol=spec.symbol,
                        side=fill.side,
                        quantity=fill.quantity,
                        price=fill.price,
                        base_currency=spec.base_currency,
                        quote_currency=account.currency,
                        fee_amount=fee,
                    )
                fills.append(
                    {
                        "decision_index": decision_index,
                        "fill_index": index,
                        "decision_ts": prices.available[decision_index],
                        "fill_ts": prices.opened[index],
                        "side": fill.side,
                        "quantity": fill.quantity,
                        "price": fill.price,
                        "fee": fee,
                        "position_after": account.quantity,
                    }
                )
        if index in funding_indices:
            account.funding(index)
        account.advance(index + 1)
        if index in decision_map:
            direction = decision_map[index]
            if direction not in {-1, 0, 1} or (spec.instrument_type == "spot" and direction < 0):
                raise DatasetValidationError(
                    "Candidate direction is unavailable on the declared market"
                )
            previous_direction = (account.quantity > 0) - (account.quantity < 0)
            if direction == previous_direction:
                continue
            quantity = 0.0
            if direction:
                snapshot = AccountSnapshot(
                    account.currency,
                    account.value(index),
                    account.ledger.cash.get(account.currency, 0),
                )
                quantity = float(entry_quantity(snapshot, Decimal(str(prices.closes[index]))))
                if quantity <= 0:
                    continue
            fill_index = index + 1 + latency_bars
            if fill_index >= len(prices.closes):
                continue
            if fill_index in pending:
                raise DatasetValidationError(
                    "Candidate delayed orders overlap; use the ordinary runtime"
                )
            pending[fill_index] = (index, direction * quantity)
            if fill_index not in scheduled:
                heappush(timeline, fill_index)
                scheduled.add(fill_index)
    account.advance(len(prices.closes))
    error = account.reconcile()
    metrics = {
        "initial_capital": initial_capital,
        "currency": account.currency,
        "final_equity_cash": float(account.equity[-1]),
        "return_pct": (float(account.equity[-1]) / initial_capital - 1) * 100,
        "max_drawdown_pct": float(
            np.min(account.equity / np.maximum.accumulate(account.equity) - 1) * 100
        ),
        "fills": len(fills),
        "fees_cash": sum(fill["fee"] for fill in fills),
        "funding_cash": account.funding_paid,
        "liquidations": len(account.liquidations),
        "reconciliation_error": error,
    }
    return CandidateScheduleResult(
        metrics,
        account.equity,
        account.quantities,
        fills,
        [vars(flow) for flow in account.ledger.cash_flows],
        account.liquidations,
    )
