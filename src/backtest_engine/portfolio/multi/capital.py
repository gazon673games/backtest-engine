"""Sizing budgets attributed exclusively from settled native ledger PnL."""

import math
from dataclasses import dataclass
from datetime import datetime

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.portfolio.multi.utils import as_datetime


@dataclass
class CapitalBudget:
    equity: float
    basis: float
    weight: float


class SleeveCapital:
    def __init__(self, capitals: dict[str, float], initial: float, policy: str) -> None:
        active = sum(capitals.values())
        if active > initial:
            raise StrategyPluginError("SDK sleeve capitals exceed portfolio initial_capital")
        self.reserve = initial - active
        self.reserve_weight = self.reserve / initial
        self.budgets = {
            asset: CapitalBudget(capital, capital, capital / active)
            for asset, capital in capitals.items()
        }
        self.policy = policy
        self.month: tuple[int, int] | None = None
        self.transfers: list[dict] = []
        self.settled_at: datetime | None = None

    def reconcile(self, equity: float) -> None:
        attributed = self.reserve + sum(b.equity for b in self.budgets.values())
        if not math.isclose(attributed, equity, abs_tol=1e-7, rel_tol=1e-9):
            raise StrategyPluginError(
                f"SDK sleeve attribution does not reconcile: {attributed} != {equity}"
            )

    def settle(self, rows: list[dict], equity: float) -> None:
        at = max(as_datetime(row["next_ts"]) for row in rows)
        if self.settled_at is not None and at <= self.settled_at:
            raise StrategyPluginError("SDK sleeve settlement must advance monotonically")
        for row in rows:
            self.budgets[row["asset_id"]].equity += float(row["net_pnl_base"])
        self.settled_at = at
        self.reconcile(equity)

    def redistribute(self, at: datetime, positions: dict[str, float], equity: float) -> None:
        if self.settled_at is not None and self.settled_at > at:
            raise StrategyPluginError("SDK capital cannot observe a future settlement")
        self.reconcile(equity)
        month = (at.year, at.month)
        previous, self.month = self.month, month
        if previous is None or previous == month or self.policy == "independent":
            return
        before = {asset: budget.equity for asset, budget in self.budgets.items()}
        reserve_before = self.reserve
        if self.policy == "monthly_profit":
            # Transfer only completed profits. Open positions never donate unrealized gains.
            pool = 0.0
            for asset, budget in self.budgets.items():
                profit = max(0.0, budget.equity - budget.basis) if positions[asset] == 0 else 0.0
                budget.equity -= profit
                pool += profit
            for budget in self.budgets.values():
                budget.equity += pool * budget.weight
                budget.basis += pool * budget.weight
        elif self.policy in {"monthly_rebalance", "monthly_rebalance_total"}:
            if self.policy == "monthly_rebalance_total":
                self.reserve = equity * self.reserve_weight
            active = equity - self.reserve
            for budget in self.budgets.values():
                budget.equity = active * budget.weight
                budget.basis = budget.equity
        for asset, budget in self.budgets.items():
            delta = budget.equity - before[asset]
            if abs(delta) > 1e-10:
                self.transfers.append({"ts": at.isoformat(), "asset_id": asset, "amount": delta})
        reserve_delta = self.reserve - reserve_before
        if abs(reserve_delta) > 1e-10:
            self.transfers.append(
                {"ts": at.isoformat(), "asset_id": "__reserve__", "amount": reserve_delta}
            )
        self.reconcile(equity)
