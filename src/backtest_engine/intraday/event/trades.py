"""Attributes event PnL to contract-specific completed trades."""

from __future__ import annotations

from typing import Any

import polars as pl

from backtest_engine.intraday.event.schemas import trades_schema
from backtest_engine.intraday.event.state import ActiveTrade
from backtest_engine.intraday.event.utils import as_datetime, ratio_float


def attribute_intraday_event_trades(
    events: pl.DataFrame,
    dataset_validation: dict[str, Any],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    ts_dtype = events.schema.get("ts", pl.Datetime)
    active: ActiveTrade | None = None
    trade_id = 1

    for row in events.iter_rows(named=True):
        target_position = float(row["target_position"])
        strategy_cost_cash, roll_cost_cash = split_costs(row)
        closing_cost_cash, opening_cost_cash = transition_costs(
            previous_position=float(row["previous_position"]),
            target_position=target_position,
            previous_contracts=float(row["previous_contracts"]),
            target_contracts=float(row["target_contracts"]),
            total_cost_cash=strategy_cost_cash,
        )

        if active is not None and target_position != active.position:
            active.cost_cash += closing_cost_cash
            rows.append(close_trade_row(active, dataset_validation, row, False))
            trade_id += 1
            active = None

        if active is None and target_position != 0.0:
            active = open_active_trade(
                trade_id,
                target_position,
                row,
                opening_cost_cash,
            )
        elif active is not None:
            # Contract-count rebalances can incur costs without changing signal.
            active.cost_cash += strategy_cost_cash
            active.current_contracts = float(row["target_contracts"])
            active.current_point_value = float(row["point_value"])

        if active is not None:
            # Event PnL is already expressed in the run's accounting currency.
            # Accumulating it keeps trade attribution aligned with signal-unit
            # compounding, contract rebalances, and the portfolio ledger.
            active.gross_pnl_cash += float(row["gross_pnl"])

        if active is not None and bool(row["contract_roll"]):
            active.cost_cash += roll_cost_cash
            rows.append(close_trade_row(active, dataset_validation, row, False))
            trade_id += 1
            active = open_active_trade_at_exit(trade_id, target_position, row)
        elif active is not None:
            active.holding_bars += 1
            active.price_path.append(float(row["exit_price"]))

    if active is not None and events.height > 0:
        rows.append(close_trade_row(active, dataset_validation, events.row(-1, named=True), True))

    return pl.DataFrame(rows, schema=trades_schema(ts_dtype), orient="row")


def split_costs(row: dict[str, Any]) -> tuple[float, float]:
    total_cost_cash = float(row["cost_cash"])
    roll_cost_cash = float(row["roll_cost_cash"])
    return total_cost_cash - roll_cost_cash, roll_cost_cash


def transition_costs(
    *,
    previous_position: float,
    target_position: float,
    previous_contracts: float,
    target_contracts: float,
    total_cost_cash: float,
) -> tuple[float, float]:
    if previous_position == target_position:
        return 0.0, 0.0
    if previous_position == 0.0:
        return 0.0, total_cost_cash
    if target_position == 0.0:
        return total_cost_cash, 0.0
    if previous_position * target_position > 0.0:
        if abs(target_position) > abs(previous_position):
            return 0.0, total_cost_cash
        return total_cost_cash, 0.0
    total_contracts = abs(previous_contracts) + abs(target_contracts)
    closing_share = (
        abs(previous_position) / (abs(previous_position) + abs(target_position))
        if total_contracts == 0.0
        else abs(previous_contracts) / total_contracts
    )
    closing_cost = total_cost_cash * closing_share
    return closing_cost, total_cost_cash - closing_cost


def open_active_trade(
    trade_id: int,
    target_position: float,
    row: dict[str, Any],
    opening_cost_cash: float,
) -> ActiveTrade:
    return ActiveTrade(
        trade_id=trade_id,
        position=target_position,
        entry_contracts=float(row["target_contracts"]),
        current_contracts=float(row["target_contracts"]),
        entry_ts=as_datetime(row["fill_ts"]),
        entry_price=float(row["fill_price"]),
        entry_notional=float(row["position_notional"]),
        entry_active_contract=str(row["active_contract"]),
        entry_source_contract_symbol=str(row["source_contract_symbol"]),
        entry_equity_cash=equity_before_event(row),
        cost_cash=opening_cost_cash,
        gross_pnl_cash=0.0,
        entry_point_value=float(row["point_value"]),
        current_point_value=float(row["point_value"]),
        holding_bars=0,
        price_path=[float(row["fill_price"])],
    )


def open_active_trade_at_exit(
    trade_id: int,
    target_position: float,
    row: dict[str, Any],
) -> ActiveTrade:
    return ActiveTrade(
        trade_id=trade_id,
        position=target_position,
        entry_contracts=float(row["target_contracts"]),
        current_contracts=float(row["target_contracts"]),
        entry_ts=as_datetime(row["next_ts"]),
        entry_price=float(row["exit_price"]),
        entry_notional=(
            abs(float(row["target_contracts"]))
            * float(row["exit_price"])
            * float(row["point_value"])
        ),
        entry_active_contract=str(row["next_active_contract"]),
        entry_source_contract_symbol=str(row["next_source_contract_symbol"]),
        entry_equity_cash=float(row["equity_cash"]),
        cost_cash=0.0,
        gross_pnl_cash=0.0,
        entry_point_value=float(row["point_value"]),
        current_point_value=float(row["point_value"]),
        holding_bars=0,
        price_path=[float(row["exit_price"])],
    )


def close_trade_row(
    active: ActiveTrade,
    dataset_validation: dict[str, Any],
    row: dict[str, Any],
    closed_by_end_of_data: bool,
) -> dict[str, Any]:
    return event_trade_row(
        active,
        dataset_validation,
        exit_ts=as_datetime(row["next_ts"] if closed_by_end_of_data else row["fill_ts"]),
        exit_price=float(row["exit_price"] if closed_by_end_of_data else row["fill_price"]),
        exit_active_contract=str(
            row["next_active_contract"] if closed_by_end_of_data else row["active_contract"]
        ),
        exit_source_contract_symbol=str(
            row["next_source_contract_symbol"]
            if closed_by_end_of_data
            else row["source_contract_symbol"]
        ),
        closed_by_end_of_data=closed_by_end_of_data,
    )


def event_trade_row(
    active: ActiveTrade,
    dataset_validation: dict[str, Any],
    *,
    exit_ts: Any,
    exit_price: float,
    exit_active_contract: str,
    exit_source_contract_symbol: str,
    closed_by_end_of_data: bool,
) -> dict[str, Any]:
    direction = "long" if active.position > 0.0 else "short"
    direction_sign = 1.0 if active.position > 0.0 else -1.0
    excursions = [
        direction_sign * (price / active.entry_price - 1.0) for price in active.price_path
    ]
    gross_pnl = active.gross_pnl_cash
    net_pnl = gross_pnl - active.cost_cash
    gross_return = ratio_float(gross_pnl, active.entry_equity_cash)
    cost_return = ratio_float(active.cost_cash, active.entry_equity_cash)
    net_return = ratio_float(net_pnl, active.entry_equity_cash)
    return {
        "trade_id": active.trade_id,
        "continuous_symbol": str(dataset_validation["continuous_symbol"]),
        "adjustment_mode": str(dataset_validation["adjustment_mode"]),
        "direction": direction,
        "entry_ts": active.entry_ts,
        "exit_ts": exit_ts,
        "entry_price": active.entry_price,
        "exit_price": exit_price,
        "entry_contracts": active.entry_contracts,
        "exit_contracts": active.current_contracts,
        "point_value": active.entry_point_value,
        "entry_notional": active.entry_notional,
        "exit_notional": (abs(active.current_contracts) * exit_price * active.current_point_value),
        "entry_active_contract": active.entry_active_contract,
        "exit_active_contract": exit_active_contract,
        "entry_source_contract_symbol": active.entry_source_contract_symbol,
        "exit_source_contract_symbol": exit_source_contract_symbol,
        "holding_bars": active.holding_bars,
        "gross_return": gross_return,
        "cost": cost_return,
        "net_return": net_return,
        "gross_pnl": gross_pnl,
        "cost_cash": active.cost_cash,
        "net_pnl": net_pnl,
        "pnl_contribution": net_return,
        "max_favorable_excursion": max(excursions),
        "max_adverse_excursion": min(excursions),
        "winning_trade": net_pnl > 0.0,
        "closed_by_end_of_data": closed_by_end_of_data,
    }


def equity_before_event(row: dict[str, Any]) -> float:
    return float(row["equity_cash"]) - float(row["net_pnl"])
