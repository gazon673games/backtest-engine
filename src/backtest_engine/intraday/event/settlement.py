"""Routes event cash flows through the correct asset accounting model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from backtest_engine.accounting.crypto import CryptoSettlement
from backtest_engine.accounting.futures_marks import FuturesMarkBook
from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime, symbol_from_row
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig
from backtest_engine.intraday.event.data import instrument_spec_for_row
from backtest_engine.intraday.event.perpetual_settlement import PerpetualEventAccounting


@dataclass(frozen=True)
class EventSettlementResult:
    equity_cash: float
    crypto_settlement: CryptoSettlement | None
    ending_quantity: float
    effective_exit_price: float


class EventAccountingRuntime:
    def __init__(self, ledger: BacktestLedgerRuntime) -> None:
        self.ledger = ledger
        self.futures_marks = FuturesMarkBook()
        self.perpetual = PerpetualEventAccounting(ledger)

    def settle_interval(self, **values: Any) -> EventSettlementResult:
        instrument = values["instrument"]
        if instrument.instrument_type == "perpetual":
            result = self.perpetual.settle(**values)
            return EventSettlementResult(**result)
        if instrument.instrument_type == "future":
            config = values["config"]
            previous_equity = values["previous_equity_cash"]
            previous_quantity = values["quantity"] - values["quantity_delta"]
            units = config.portfolio.sizing_mode == "signal_units"
            exit_row = values["exit_row"] or values["fill_row"]
            exit_spec = instrument_spec_for_row(exit_row, values["valuation_ts"], config.instrument)
            gap = self.futures_marks.advance(
                key="event",
                previous_quantity=previous_quantity,
                fill_symbol=symbol_from_row(values["fill_row"]),
                fill_price=values["fill_price"],
                exit_symbol=symbol_from_row(exit_row),
                exit_price=values["mark_price"],
                exit_instrument=exit_spec,
                ledger=self.ledger,
                equity_fraction=previous_quantity * previous_equity if units else None,
            )
            pnl = values["pnl_state"]
            adjustment = gap + (pnl["gross_pnl"] * gap / previous_equity if units else 0.0)
            pnl["gross_pnl"] += adjustment
            pnl["net_pnl"] += adjustment
            pnl["gross_return"] = pnl["gross_pnl"] / previous_equity
            pnl["net_return"] = pnl["net_pnl"] / previous_equity
            values["opening_pnl"] = gap
        equity = settle_event_interval(ledger=self.ledger, **values)
        return EventSettlementResult(
            equity,
            values.get("crypto_settlement"),
            float(values["quantity"]),
            float(values["mark_price"]),
        )


def settle_event_interval(
    *,
    ledger: BacktestLedgerRuntime,
    config: IntradayEventBacktestConfig,
    instrument: Any,
    fill_row: dict[str, Any],
    fill_ts: datetime,
    valuation_ts: datetime,
    fill_price: float,
    mark_price: float,
    quantity_delta: float,
    previous_equity_cash: float,
    pnl_state: dict[str, float],
    cost_parts: dict[str, float],
    exit_row: dict[str, Any] | None = None,
    quantity: float = 0.0,
    crypto_settlement: CryptoSettlement | None = None,
    opening_pnl: float = 0.0,
) -> float:
    del exit_row, quantity, crypto_settlement
    if instrument.instrument_type != "spot":
        currency = (
            config.portfolio.base_currency
            if config.portfolio.sizing_mode == "signal_units"
            else instrument.pnl_currency
        )
        ledger.activate_rates(fill_ts)
        rate = ledger.amount_in_base(1.0, currency)
        equity = ledger.apply_event_cash_flow(
            ts=valuation_ts,
            fill_ts=fill_ts,
            symbol=symbol_from_row(fill_row),
            currency=currency,
            gross_pnl=pnl_state["gross_pnl"] / rate,
            cost_cash=pnl_state["cost_cash"] / rate,
            opening_pnl=opening_pnl / rate,
            previous_equity_cash=previous_equity_cash,
        )
        pnl_state["net_pnl"] = equity - previous_equity_cash
        pnl_state["gross_pnl"] = pnl_state["net_pnl"] + pnl_state["cost_cash"]
        pnl_state["net_return"] = pnl_state["net_pnl"] / previous_equity_cash
        pnl_state["gross_return"] = pnl_state["gross_pnl"] / previous_equity_cash
        return equity
    base = instrument.base_currency
    quote = instrument.quote_currency
    if not base or not quote:
        raise DatasetValidationError("Spot accounting requires base_currency and quote_currency")
    quote_to_base = ledger.amount_in_base(1.0, quote)
    trading_fee_quote = cost_parts["trading_fee_cash"] / quote_to_base
    additional_cost_quote = (
        cost_parts["total_cost_cash"] - cost_parts["trading_fee_cash"]
    ) / quote_to_base
    equity_cash = ledger.apply_spot_interval(
        fill_ts=fill_ts,
        valuation_ts=valuation_ts,
        symbol=symbol_from_row(fill_row),
        base_currency=base,
        quote_currency=quote,
        quantity_delta=quantity_delta * instrument.contract_multiplier * instrument.lot_size,
        fill_price=fill_price,
        mark_price=mark_price,
        trading_fee_quote=trading_fee_quote,
        additional_cost_quote=additional_cost_quote,
        previous_equity_cash=previous_equity_cash,
    )
    pnl_state["net_pnl"] = equity_cash - previous_equity_cash
    pnl_state["gross_pnl"] = pnl_state["net_pnl"] + pnl_state["cost_cash"]
    pnl_state["net_return"] = pnl_state["net_pnl"] / previous_equity_cash
    pnl_state["gross_return"] = pnl_state["gross_pnl"] / previous_equity_cash
    return equity_cash
