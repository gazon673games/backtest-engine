from __future__ import annotations

from typing import Any

import polars as pl


def attribute_completed_trades(equity: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    positions = [float(value) for value in equity.get_column("position").to_list()]
    active_start_index: int | None = None
    trade_id = 1

    for index, position in enumerate(positions):
        previous_position = positions[index - 1] if index > 0 else 0.0

        if previous_position == 0.0 and position > 0.0:
            active_start_index = index
        elif previous_position > 0.0 and position == 0.0 and active_start_index is not None:
            exit_index = index - 1
            rows.append(
                _build_long_trade(
                    equity,
                    trade_id=trade_id,
                    active_start_index=active_start_index,
                    exit_index=exit_index,
                    exit_cost_index=index,
                    closed_by_end_of_data=False,
                )
            )
            trade_id += 1
            active_start_index = None

    if active_start_index is not None:
        rows.append(
            _build_long_trade(
                equity,
                trade_id=trade_id,
                active_start_index=active_start_index,
                exit_index=equity.height - 1,
                exit_cost_index=None,
                closed_by_end_of_data=True,
            )
        )

    return pl.DataFrame(rows, schema=_completed_trade_schema(equity), orient="row")


def _build_long_trade(
    equity: pl.DataFrame,
    *,
    trade_id: int,
    active_start_index: int,
    exit_index: int,
    exit_cost_index: int | None,
    closed_by_end_of_data: bool,
) -> dict[str, Any]:
    entry_index = max(active_start_index - 1, 0)
    entry_price = float(equity.get_column("close")[entry_index])
    exit_price = float(equity.get_column("close")[exit_index])
    gross_return = _compounded_return(
        equity,
        "strategy_return",
        active_start_index,
        exit_index,
    )

    cost_start = active_start_index
    cost_end = exit_cost_index if exit_cost_index is not None else exit_index
    cost = _sum_cost(equity, cost_start, cost_end)
    net_return = _compounded_return(
        equity,
        "strategy_return_after_cost",
        active_start_index,
        cost_end,
    )
    start_equity = (
        1.0
        if active_start_index == 0
        else float(equity.get_column("equity")[active_start_index - 1])
    )
    end_equity = float(equity.get_column("equity")[cost_end])
    closes = [
        float(value)
        for value in equity.get_column("close").slice(entry_index, exit_index - entry_index + 1)
    ]
    excursions = [close / entry_price - 1.0 for close in closes]

    return {
        "trade_id": trade_id,
        "continuous_symbol": equity.get_column("continuous_symbol")[entry_index],
        "adjustment_mode": equity.get_column("adjustment_mode")[entry_index],
        "direction": "long",
        "entry_ts": equity.get_column("ts")[entry_index],
        "exit_ts": equity.get_column("ts")[exit_index],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_active_contract": equity.get_column("active_contract")[entry_index],
        "exit_active_contract": equity.get_column("active_contract")[exit_index],
        "entry_source_contract_symbol": equity.get_column("source_contract_symbol")[entry_index],
        "exit_source_contract_symbol": equity.get_column("source_contract_symbol")[exit_index],
        "holding_bars": exit_index - active_start_index + 1,
        "gross_return": gross_return,
        "cost": cost,
        "net_return": net_return,
        "pnl_contribution": end_equity - start_equity,
        "max_favorable_excursion": max(excursions),
        "max_adverse_excursion": min(excursions),
        "winning_trade": net_return > 0.0,
        "closed_by_end_of_data": closed_by_end_of_data,
    }


def _sum_cost(equity: pl.DataFrame, start_index: int, end_index: int) -> float:
    if end_index < start_index:
        return 0.0
    return float(equity.get_column("cost").slice(start_index, end_index - start_index + 1).sum())


def _compounded_return(
    equity: pl.DataFrame,
    column: str,
    start_index: int,
    end_index: int,
) -> float:
    result = 1.0
    for value in equity.get_column(column).slice(start_index, end_index - start_index + 1):
        result *= 1.0 + float(value)
    return result - 1.0


def _completed_trade_schema(equity: pl.DataFrame) -> dict[str, pl.DataType]:
    ts_dtype = equity.schema["ts"]
    return {
        "trade_id": pl.Int64,
        "continuous_symbol": pl.String,
        "adjustment_mode": pl.String,
        "direction": pl.String,
        "entry_ts": ts_dtype,
        "exit_ts": ts_dtype,
        "entry_price": pl.Float64,
        "exit_price": pl.Float64,
        "entry_active_contract": pl.String,
        "exit_active_contract": pl.String,
        "entry_source_contract_symbol": pl.String,
        "exit_source_contract_symbol": pl.String,
        "holding_bars": pl.Int64,
        "gross_return": pl.Float64,
        "cost": pl.Float64,
        "net_return": pl.Float64,
        "pnl_contribution": pl.Float64,
        "max_favorable_excursion": pl.Float64,
        "max_adverse_excursion": pl.Float64,
        "winning_trade": pl.Boolean,
        "closed_by_end_of_data": pl.Boolean,
    }
