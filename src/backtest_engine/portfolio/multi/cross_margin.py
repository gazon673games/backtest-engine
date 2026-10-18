"""Settles synchronized perpetual legs against one shared collateral wallet."""

from __future__ import annotations

from typing import Any

from backtest_engine.accounting.cross_margin import CrossMarginAccount
from backtest_engine.accounting.fees import trading_fee_bps
from backtest_engine.accounting.funding import funding_payment_for_row
from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime
from backtest_engine.accounting.perpetual import TieredMarginSchedule
from backtest_engine.accounting.perpetual_inputs import (
    perpetual_margin_schedule,
    perpetual_mark_prices,
)
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig
from backtest_engine.portfolio.multi.cross_margin_rows import update_shared_margin_rows
from backtest_engine.portfolio.multi.utils import as_datetime


class SharedCrossMarginRuntime:
    def __init__(self, ledger: BacktestLedgerRuntime) -> None:
        self.ledger = ledger
        self.account: CrossMarginAccount | None = None
        self.currency: str | None = None
        self.marks: dict[str, float] = {}

    def settle(
        self,
        rows: list[dict[str, Any]],
        config: MultiInstrumentBacktestConfig,
        previous_equity: float,
        current_contracts: dict[str, float],
        current_targets: dict[str, float],
    ) -> float:
        currency = self._common_currency(rows)
        if self.account is None:
            self.account = CrossMarginAccount(self.ledger.ledger.cash.get(currency, 0.0))
            self.currency = currency
        elif self.currency != currency:
            raise DatasetValidationError(
                "Shared cross-margin legs must use one collateral currency"
            )
        else:
            self.account.wallet_balance = self.ledger.ledger.cash.get(currency, 0.0)
        previous_unrealized = {
            symbol: position.unrealized_pnl(self.marks[symbol])
            for symbol, position in self.account.positions.items()
        }
        orders, adverse_marks = self._prepare(rows, config)
        fill_marks = {
            **self.marks,
            **{str(row["asset_id"]): float(row["fill_price"]) for row in rows},
        }
        transitions = self.account.apply_batch(orders, fill_marks)
        self._post_transitions(rows, transitions, currency)
        fill_ts = max(as_datetime(row["fill_ts"]) for row in rows)
        self.ledger.activate_rates(fill_ts)
        fill_unrealized = sum(
            position.unrealized_pnl(fill_marks.get(symbol, self.marks[symbol]))
            for symbol, position in self.account.positions.items()
        )
        self.ledger.record_unrealized(fill_ts, "CROSS_MARGIN", currency, fill_unrealized)
        liquidations = self.account.liquidate_if_required(
            adverse_marks,
            target_buffer=config.portfolio.liquidation_target_buffer,
        )
        settlement_ts = max(as_datetime(row["next_ts"]) for row in rows)
        self.ledger.activate_rates(settlement_ts)
        by_symbol = {str(row["asset_id"]): row for row in rows}
        for item in transitions:
            self._post_cost(
                as_datetime(by_symbol[item.symbol]["next_ts"]),
                item.symbol,
                currency,
                item.funding_payment,
            )
        self._post_liquidations(rows, liquidations, currency, settlement_ts)
        equity = self.ledger.equity_cash + self.ledger.amount_in_base(
            sum(
                position.unrealized_pnl(self.marks[symbol])
                for symbol, position in self.account.positions.items()
                if position.quantity
            ),
            currency,
        )
        update_shared_margin_rows(
            self.account,
            self.ledger,
            currency,
            self.marks,
            rows,
            transitions,
            liquidations,
            current_contracts,
            current_targets,
            previous_unrealized,
        )
        self.ledger.record_unrealized(
            settlement_ts,
            "CROSS_MARGIN",
            currency,
            sum(
                position.unrealized_pnl(self.marks[symbol])
                for symbol, position in self.account.positions.items()
            ),
        )
        if self.ledger.rate_book is not None:
            fx_pnl = equity - previous_equity - sum(row["net_pnl_base"] for row in rows)
            rows[0]["fx_pnl_base"] = fx_pnl
            rows[0]["gross_pnl_base"] += fx_pnl
            rows[0]["net_pnl_base"] += fx_pnl
        self._append_snapshot(rows, currency, equity, previous_equity)
        return equity

    def _prepare(
        self,
        rows: list[dict[str, Any]],
        config: MultiInstrumentBacktestConfig,
    ) -> tuple[list[dict[str, float | str]], dict[str, float]]:
        orders = []
        adverse = dict(self.marks)
        for row in rows:
            instrument = row["_instrument"]
            symbol = str(row["asset_id"])
            self.account.register(
                symbol,
                point_value=instrument.contract_multiplier * instrument.lot_size,
                leverage=config.portfolio.leverage,
                margin_schedule=margin_schedule(instrument, config),
                liquidation_fee_bps=instrument.liquidation_fee_bps,
            )
            mark, low, high = row_marks(row, config)
            self.marks[symbol] = mark
            ending = float(row["target_contracts"])
            adverse[symbol] = low if ending >= 0.0 else high
            fee_bps = trading_fee_bps(
                instrument, config.execution.liquidity_role, config.execution.commission_bps
            )
            orders.append(
                {
                    "symbol": symbol,
                    "quantity_delta": float(row["contract_delta"]),
                    "fill_price": float(row["fill_price"]),
                    "fee_bps": fee_bps,
                    "funding_payment": funding_payment_for_row(
                        row["_exit_row"],
                        quantity=ending,
                        point_value=instrument.contract_multiplier * instrument.lot_size,
                        mark_price=mark,
                        rate_column=config.execution.funding_rate_column,
                        default_rate=config.execution.default_funding_rate,
                        event_column=config.execution.funding_event_column,
                        require_event_flag=config.execution.require_funding_event_flag,
                    ),
                    "additional_cost": sum(
                        float(row[key])
                        for key in ("slippage_cost_cash", "spread_cost_cash", "roll_cost_cash")
                    ),
                }
            )
        return orders, adverse

    def _post_transitions(
        self,
        rows: list[dict[str, Any]],
        transitions: list[Any],
        currency: str,
    ) -> None:
        by_symbol = {str(row["asset_id"]): row for row in rows}
        for item in transitions:
            row = by_symbol[item.symbol]
            ts = as_datetime(row["fill_ts"])
            self.ledger.ledger.apply_variation_margin(
                ts=ts, symbol=item.symbol, amount=item.realized_pnl, currency=currency
            )
            self._post_cost(ts, item.symbol, currency, item.trading_fee)
            self._post_cost(ts, item.symbol, currency, item.additional_cost)

    def _post_liquidations(
        self,
        rows: list[dict[str, Any]],
        items: list[Any],
        currency: str,
        fallback_ts: Any,
    ) -> None:
        by_symbol = {str(row["asset_id"]): row for row in rows}
        for item in items:
            row = by_symbol.get(item.symbol)
            ts = fallback_ts if row is None else as_datetime(row["next_ts"])
            self.ledger.ledger.apply_variation_margin(
                ts=ts, symbol=item.symbol, amount=item.realized_pnl, currency=currency
            )
            self._post_cost(ts, item.symbol, currency, item.liquidation_fee)

    def _post_cost(self, ts: Any, symbol: str, currency: str, amount: float) -> None:
        if amount >= 0.0:
            self.ledger.ledger.apply_fee(ts=ts, symbol=symbol, amount=amount, currency=currency)
        else:
            self.ledger.ledger.apply_variation_margin(
                ts=ts, symbol=symbol, amount=-amount, currency=currency
            )

    def _append_snapshot(
        self, rows: list[dict[str, Any]], currency: str, equity: float, previous_equity: float
    ) -> None:
        costs = sum(float(row["cost_cash_base"]) for row in rows)
        self.ledger.snapshots.append(
            {
                "ts": max(as_datetime(row["next_ts"]) for row in rows),
                "symbol": "CROSS_MARGIN",
                "currency": currency,
                "gross_pnl": equity - previous_equity + costs,
                "cost_cash": costs,
                "net_cash_flow": equity - previous_equity,
                "cash_balance": self.ledger.ledger.cash.get(currency, 0.0),
                "equity_cash": equity,
                "reported_equity_cash": equity,
                "reconciliation_diff": 0.0,
            }
        )

    def _common_currency(self, rows: list[dict[str, Any]]) -> str:
        currencies = {
            row["_instrument"].settlement_currency or row["_instrument"].pnl_currency
            for row in rows
        }
        if len(currencies) != 1:
            raise DatasetValidationError("Shared cross-margin requires one settlement currency")
        return str(currencies.pop())


def margin_schedule(instrument: Any, config: MultiInstrumentBacktestConfig) -> TieredMarginSchedule:
    return perpetual_margin_schedule(
        instrument, require_margin_tiers=config.portfolio.require_margin_tiers
    )


def row_marks(
    row: dict[str, Any], config: MultiInstrumentBacktestConfig
) -> tuple[float, float, float]:
    source = row["_exit_row"]
    return perpetual_mark_prices(
        source,
        float(row["exit_price"]),
        require_mark_price=config.portfolio.require_mark_price,
        mark_price_column=config.execution.mark_price_column,
        mark_low_column=config.execution.mark_low_column,
        mark_high_column=config.execution.mark_high_column,
    )
