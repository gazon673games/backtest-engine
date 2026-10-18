"""Attribute native position PnL and priced costs to shared-margin leg rows."""

from typing import Any

from backtest_engine.accounting.cross_margin import CrossMarginAccount
from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime, amount_in_base_currency
from backtest_engine.portfolio.multi.utils import as_datetime


def update_shared_margin_rows(
    account: CrossMarginAccount,
    ledger: BacktestLedgerRuntime,
    currency: str,
    marks: dict[str, float],
    rows: list[dict[str, Any]],
    transitions: list[Any],
    liquidations: list[Any],
    current_contracts: dict[str, float],
    current_targets: dict[str, float],
    previous_unrealized: dict[str, float],
) -> None:
    transitions_by_symbol = {item.symbol: item for item in transitions}
    liquidations_by_symbol = {item.symbol: item for item in liquidations}
    for row in rows:
        symbol = str(row["asset_id"])
        transition = transitions_by_symbol[symbol]
        liquidation = liquidations_by_symbol.get(symbol)
        position = account.positions[symbol]
        row["gross_pnl"] = (
            transition.realized_pnl
            + (liquidation.realized_pnl if liquidation else 0.0)
            + position.unrealized_pnl(marks[symbol])
            - previous_unrealized.get(symbol, 0.0)
        )
        row["trading_fee_cash"] = transition.trading_fee
        row["commission_cost_cash"] = transition.trading_fee
        row["funding_cost_cash"] = transition.funding_payment
        row["liquidation_fee_cash"] = liquidation.liquidation_fee if liquidation else 0.0
        row["cost_cash"] = sum(
            float(row[key])
            for key in (
                "trading_fee_cash",
                "slippage_cost_cash",
                "spread_cost_cash",
                "roll_cost_cash",
                "funding_cost_cash",
                "liquidation_fee_cash",
            )
        )
        row["net_pnl"] = float(row["gross_pnl"]) - float(row["cost_cash"])
        fill_rate = (
            ledger.rate_book.rates_at(as_datetime(row["fill_ts"]), ledger.static_rates)
            if ledger.rate_book is not None
            else ledger.static_rates
        )
        fill_cost = transition.trading_fee + transition.additional_cost
        row["cost_cash_base"] = amount_in_base_currency(
            fill_cost, currency, ledger.base_currency, fill_rate
        ) + ledger.amount_in_base(float(row["cost_cash"]) - fill_cost, currency)
        row["gross_pnl_base"] = ledger.amount_in_base(row["gross_pnl"], currency)
        row["net_pnl_base"] = row["gross_pnl_base"] - row["cost_cash_base"]
        row["maintenance_margin"] = position.margin_schedule.maintenance_margin(
            position.notional(marks[symbol])
        )
        row["liquidated"] = liquidation is not None
        row["liquidation_price"] = liquidation.price if liquidation else None
        row["liquidation_status"] = "full" if liquidation else "none"
        if liquidation:
            row["target_contracts"] = 0.0
    for liquidation in liquidations:
        current_contracts[liquidation.symbol] = 0.0
        current_targets[liquidation.symbol] = 0.0
