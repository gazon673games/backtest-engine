"""Asset-neutral settlement for synchronized portfolio leg events."""

from __future__ import annotations

from typing import Any

from backtest_engine.accounting.fees import trading_fee_bps
from backtest_engine.accounting.ledger_runtime import BacktestLedgerRuntime, symbol_from_row
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.intraday.event.settlement import EventAccountingRuntime
from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig
from backtest_engine.portfolio.multi.cross_margin import SharedCrossMarginRuntime
from backtest_engine.portfolio.multi.utils import as_datetime


def validate_crypto_accounting_topology(
    loaded_legs: dict[str, dict[str, Any]],
    config: MultiInstrumentBacktestConfig,
) -> bool:
    perpetual_legs = []
    for asset_id, payload in loaded_legs.items():
        leg = payload["config"]
        instrument_types = {leg.instrument_type} | {spec.instrument_type for spec in leg.specs}
        if "perpetual" in instrument_types:
            perpetual_legs.append(asset_id)
    if len(perpetual_legs) <= 1:
        return False
    if config.portfolio.margin_mode != "cross":
        raise DatasetValidationError("Multiple perpetual legs require portfolio.margin_mode=cross")
    currencies = set()
    for asset_id in perpetual_legs:
        leg = loaded_legs[asset_id]["config"]
        currencies.add(leg.settlement_currency or leg.pnl_currency)
        currencies.update(spec.settlement_currency or spec.pnl_currency for spec in leg.specs)
    if len(currencies) != 1:
        raise DatasetValidationError(
            "Shared cross-margin perpetual legs require one settlement currency"
        )
    return True


def settle_mixed_leg_rows(
    *,
    rows: list[dict[str, Any]],
    config: MultiInstrumentBacktestConfig,
    ledger: BacktestLedgerRuntime,
    accounting: EventAccountingRuntime,
    shared_margin: SharedCrossMarginRuntime | None,
    previous_equity_cash: float,
    current_contracts: dict[str, float],
    current_targets: dict[str, float],
) -> float:
    equity = previous_equity_cash
    perpetual_rows = [row for row in rows if row["_instrument"].instrument_type == "perpetual"]
    regular_rows = [row for row in rows if row["_instrument"].instrument_type != "perpetual"]
    for row in regular_rows:
        instrument = row["_instrument"]
        if instrument.instrument_type == "spot":
            equity = settle_crypto_leg(
                row,
                instrument,
                config,
                ledger,
                accounting,
                equity,
                current_contracts,
                current_targets,
            )
        else:
            ledger.activate_rates(as_datetime(row["fill_ts"]))
            native_to_base = ledger.amount_in_base(1.0, str(row["pnl_currency"]))
            opening_pnl = (
                accounting.futures_marks.advance(
                    key=str(row["asset_id"]),
                    previous_quantity=float(row["previous_contracts"]),
                    fill_symbol=symbol_from_row(row["_fill_row"]),
                    fill_price=float(row["fill_price"]),
                    exit_symbol=symbol_from_row(row["_exit_row"]),
                    exit_price=float(row["exit_price"]),
                    exit_instrument=row["_exit_instrument"],
                    ledger=ledger,
                )
                / native_to_base
            )
            row["gross_pnl"] += opening_pnl
            row["net_pnl"] += opening_pnl
            equity = ledger.apply_event_cash_flow(
                ts=as_datetime(row["next_ts"]),
                symbol=symbol_from_row(row, default=str(row.get("asset_id", ""))),
                currency=str(row["pnl_currency"]),
                gross_pnl=float(row["gross_pnl"]),
                cost_cash=float(row["cost_cash"]),
                fill_ts=as_datetime(row["fill_ts"]),
                opening_pnl=opening_pnl,
                previous_equity_cash=equity,
            )
            row["cost_cash_base"] = float(row["cost_cash"]) * native_to_base
            row["gross_pnl_base"] = ledger.snapshots[-1]["gross_pnl"]
            row["net_pnl_base"] = ledger.snapshots[-1]["net_cash_flow"]
    if shared_margin is not None and perpetual_rows:
        equity = shared_margin.settle(
            perpetual_rows,
            config,
            equity,
            current_contracts,
            current_targets,
        )
    elif perpetual_rows:
        for row in perpetual_rows:
            equity = settle_crypto_leg(
                row,
                row["_instrument"],
                config,
                ledger,
                accounting,
                equity,
                current_contracts,
                current_targets,
            )
    for row in rows:
        for key in (
            "_instrument",
            "_exit_instrument",
            "_fill_row",
            "_exit_row",
            "_reference_fill_price",
        ):
            row.pop(key, None)
    return equity


def settle_crypto_leg(
    row: dict[str, Any],
    instrument: Any,
    config: MultiInstrumentBacktestConfig,
    ledger: BacktestLedgerRuntime,
    accounting: EventAccountingRuntime,
    previous_equity: float,
    current_contracts: dict[str, float],
    current_targets: dict[str, float],
) -> float:
    currency = instrument.settlement_currency or instrument.pnl_currency
    ledger.activate_rates(as_datetime(row["fill_ts"]))
    native_to_base = ledger.amount_in_base(1.0, currency)
    fee_bps = trading_fee_bps(
        instrument, config.execution.liquidity_role, config.execution.commission_bps
    )
    trading_fee_native = (
        abs(float(row["contract_delta"]))
        * float(row["fill_price"])
        * instrument.contract_multiplier
        * instrument.lot_size
        * fee_bps
        / 10_000.0
    )
    row["commission_cost_cash"] = trading_fee_native
    row["trading_fee_cash"] = trading_fee_native
    row["cost_cash"] = sum(
        float(row[key])
        for key in (
            "trading_fee_cash",
            "slippage_cost_cash",
            "spread_cost_cash",
            "roll_cost_cash",
        )
    )
    cost_parts = {
        key: float(row[key]) * native_to_base
        for key in (
            "commission_cost_cash",
            "slippage_cost_cash",
            "spread_cost_cash",
            "roll_cost_cash",
            "trading_fee_cash",
            "funding_cost_cash",
            "liquidation_fee_cash",
        )
    }
    cost_parts["total_cost_cash"] = float(row["cost_cash"]) * native_to_base
    cost_parts["total_cost_return"] = cost_parts["total_cost_cash"] / previous_equity
    pnl = {
        "gross_pnl": float(row["gross_pnl"]) * native_to_base,
        "cost_cash": cost_parts["total_cost_cash"],
        "net_pnl": float(row["net_pnl"]) * native_to_base,
        "gross_return": float(row["gross_pnl"]) * native_to_base / previous_equity,
        "net_return": float(row["net_pnl"]) * native_to_base / previous_equity,
        "cost": cost_parts["total_cost_return"],
    }
    result = accounting.settle_interval(
        config=config,
        instrument=instrument,
        fill_row=row["_fill_row"],
        exit_row=row["_exit_row"],
        fill_ts=as_datetime(row["fill_ts"]),
        valuation_ts=as_datetime(row["next_ts"]),
        fill_price=float(row["fill_price"]),
        mark_price=float(row["exit_price"]),
        quantity_delta=float(row["contract_delta"]),
        quantity=float(row["target_contracts"]),
        previous_equity_cash=previous_equity,
        pnl_state=pnl,
        cost_parts=cost_parts,
        crypto_settlement=None,
    )
    apply_crypto_result(
        row,
        result,
        pnl,
        cost_parts,
        native_to_base,
        ledger.amount_in_base(1.0, currency),
        current_contracts,
        current_targets,
    )
    return result.equity_cash


def apply_crypto_result(
    row: dict[str, Any],
    result: Any,
    pnl: dict[str, float],
    costs: dict[str, float],
    native_to_base: float,
    valuation_rate: float,
    current_contracts: dict[str, float],
    current_targets: dict[str, float],
) -> None:
    for field in ("gross_pnl", "cost_cash", "net_pnl"):
        row[field] = pnl[field] / valuation_rate
        row[field + "_base"] = pnl[field]
    for field in (
        "commission_cost_cash",
        "trading_fee_cash",
        "funding_cost_cash",
        "liquidation_fee_cash",
    ):
        rate = (
            native_to_base
            if field in {"commission_cost_cash", "trading_fee_cash"}
            else valuation_rate
        )
        row[field] = costs[field] / rate
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
    row["gross_pnl"] = row["net_pnl"] + row["cost_cash"]
    settlement = result.crypto_settlement
    if settlement is not None:
        row["maintenance_margin"] = settlement.maintenance_margin
        row["liquidated"] = settlement.liquidated
        row["liquidation_price"] = settlement.liquidation_price
        row["bankruptcy_price"] = settlement.bankruptcy_price
        row["liquidation_status"] = settlement.liquidation_status
    asset_id = str(row["asset_id"])
    previous_quantity = float(row["target_contracts"])
    if result.ending_quantity != previous_quantity:
        row["target_contracts"] = result.ending_quantity
        row["contract_delta"] = result.ending_quantity - float(row["previous_contracts"])
        current_contracts[asset_id] = result.ending_quantity
        current_targets[asset_id] = (
            0.0
            if previous_quantity == 0.0
            else float(row["target"]) * result.ending_quantity / previous_quantity
        )
