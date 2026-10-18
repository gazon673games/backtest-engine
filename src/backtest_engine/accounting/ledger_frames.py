from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from itertools import chain
from typing import Any

import polars as pl

from backtest_engine.accounting.ledger import MultiCurrencyLedger
from backtest_engine.accounting.ledger_runtime import parse_fx_rates
from backtest_engine.core.exceptions import DatasetValidationError


def ledger_frame_from_event_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    initial_capital: float,
    base_currency: str,
    fx_rates: dict[str, float] | None = None,
    default_symbol: str | None = None,
) -> pl.DataFrame:
    iterator = iter(rows)
    first = next(iterator, None)
    if first is None:
        return _empty_ledger_frame()
    ledger = MultiCurrencyLedger(base_currency=base_currency)
    parsed_fx_rates = parse_fx_rates(fx_rates or {})
    ledger.deposit(
        ts=_event_cash_ts(first),
        currency=base_currency,
        amount=initial_capital,
        reason="initial_capital",
    )
    snapshots = []
    for row in chain((first,), iterator):
        ts = _event_cash_ts(row)
        currency = str(row.get("pnl_currency") or base_currency)
        symbol = str(row.get("active_contract") or row.get("asset_id") or default_symbol or "")
        gross_pnl = _float(row.get("gross_pnl"))
        cost_cash = _float(row.get("cost_cash"))
        ledger.apply_variation_margin(
            ts=ts,
            symbol=symbol,
            amount=gross_pnl,
            currency=currency,
        )
        if cost_cash >= 0.0:
            ledger.apply_fee(ts=ts, symbol=symbol, amount=cost_cash, currency=currency)
        else:
            ledger.apply_variation_margin(
                ts=ts,
                symbol=symbol,
                amount=-cost_cash,
                currency=currency,
            )
        reported = _optional_float(row.get("equity_cash"))
        equity_cash = ledger.equity(parsed_fx_rates)
        snapshots.append(
            {
                "ts": ts,
                "symbol": symbol,
                "currency": currency,
                "gross_pnl": gross_pnl,
                "cost_cash": cost_cash,
                "net_cash_flow": gross_pnl - cost_cash,
                "cash_balance": ledger.cash.get(currency, 0.0),
                "equity_cash": equity_cash,
                "reported_equity_cash": reported,
                "reconciliation_diff": None if reported is None else reported - equity_cash,
            }
        )
    return pl.DataFrame(snapshots, orient="row")


def ledger_frame_from_snapshots(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return _empty_ledger_frame()
    return pl.DataFrame(rows, schema=_empty_ledger_frame().schema, orient="row")


def _event_cash_ts(row: Mapping[str, Any]) -> datetime:
    return _as_datetime(row.get("next_ts") or row["ts"])


def _empty_ledger_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [],
        schema={
            "ts": pl.Datetime,
            "symbol": pl.String,
            "currency": pl.String,
            "gross_pnl": pl.Float64,
            "cost_cash": pl.Float64,
            "net_cash_flow": pl.Float64,
            "cash_balance": pl.Float64,
            "equity_cash": pl.Float64,
            "reported_equity_cash": pl.Float64,
            "reconciliation_diff": pl.Float64,
        },
        orient="row",
    )


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    raise DatasetValidationError(f"ts must contain datetime values, got {value!r}")


def _float(value: Any) -> float:
    parsed = _optional_float(value)
    return 0.0 if parsed is None else parsed


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
