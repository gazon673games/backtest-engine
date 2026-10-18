from __future__ import annotations

from typing import Any

import polars as pl


def generate_trade_events(equity: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    positions = [float(value) for value in equity.get_column("position").to_list()]
    old_position = 0.0
    for index, new_position in enumerate(positions):
        if old_position != new_position:
            rows.append(position_event(equity, index, old_position, new_position))
        old_position = new_position
    rows.extend(roll_events(equity, positions))
    if equity.height and positions[-1] != 0.0:
        rows.append(end_of_data_event(equity, positions[-1]))
    rows.sort(key=lambda row: row["ts"])
    return pl.DataFrame(rows, schema=trade_event_schema(equity))


def position_event(
    equity: pl.DataFrame,
    index: int,
    old_position: float,
    new_position: float,
) -> dict[str, Any]:
    execution_index = max(index - 1, 0)
    total_turnover = float(equity.get_column("position_turnover")[index])
    end_turnover = float(equity.get_column("end_of_data_turnover")[index])
    transition_turnover = total_turnover - end_turnover
    total_cost = float(equity.get_column("position_cost")[index])
    transition_cost = (
        0.0 if total_turnover == 0.0 else total_cost * transition_turnover / total_turnover
    )
    return {
        "ts": equity.get_column("ts")[execution_index],
        "continuous_symbol": equity.get_column("continuous_symbol")[execution_index],
        "event": trade_event_name(old_position, new_position),
        "old_position": old_position,
        "new_position": new_position,
        "price": float(equity.get_column("close")[execution_index]),
        "turnover": transition_turnover,
        "cost": transition_cost,
        "previous_active_contract": equity.get_column("active_contract")[execution_index],
        "active_contract": equity.get_column("active_contract")[execution_index],
        "source_contract_symbol": equity.get_column("source_contract_symbol")[execution_index],
    }


def roll_events(equity: pl.DataFrame, positions: list[float]) -> list[dict[str, Any]]:
    rows = []
    for index in range(1, equity.height):
        turnover = float(equity.get_column("roll_turnover")[index])
        if turnover == 0.0:
            continue
        position = positions[index]
        rows.append(
            {
                "ts": equity.get_column("ts")[index],
                "continuous_symbol": equity.get_column("continuous_symbol")[index],
                "event": "roll_long" if position > 0.0 else "roll_short",
                "old_position": position,
                "new_position": position,
                "price": float(equity.get_column("close")[index]),
                "turnover": turnover,
                "cost": float(equity.get_column("roll_cost")[index]),
                "previous_active_contract": equity.get_column("active_contract")[index - 1],
                "active_contract": equity.get_column("active_contract")[index],
                "source_contract_symbol": equity.get_column("source_contract_symbol")[index],
            }
        )
    return rows


def end_of_data_event(equity: pl.DataFrame, position: float) -> dict[str, Any]:
    end_turnover = float(equity.get_column("end_of_data_turnover")[-1])
    total_turnover = float(equity.get_column("position_turnover")[-1])
    total_cost = float(equity.get_column("position_cost")[-1])
    end_cost = 0.0 if total_turnover == 0.0 else total_cost * end_turnover / total_turnover
    return {
        "ts": equity.get_column("ts")[-1],
        "continuous_symbol": equity.get_column("continuous_symbol")[-1],
        "event": "exit_long_end_of_data",
        "old_position": position,
        "new_position": 0.0,
        "price": float(equity.get_column("close")[-1]),
        "turnover": end_turnover,
        "cost": end_cost,
        "previous_active_contract": equity.get_column("active_contract")[-1],
        "active_contract": equity.get_column("active_contract")[-1],
        "source_contract_symbol": equity.get_column("source_contract_symbol")[-1],
    }


def trade_event_schema(equity: pl.DataFrame) -> dict[str, pl.DataType]:
    return {
        "ts": equity.schema["ts"],
        "continuous_symbol": pl.String,
        "event": pl.String,
        "old_position": pl.Float64,
        "new_position": pl.Float64,
        "price": pl.Float64,
        "turnover": pl.Float64,
        "cost": pl.Float64,
        "previous_active_contract": pl.String,
        "active_contract": pl.String,
        "source_contract_symbol": pl.String,
    }


def trade_event_name(old_position: float, new_position: float) -> str:
    if old_position == 0.0 and new_position > 0.0:
        return "enter_long"
    if old_position > 0.0 and new_position == 0.0:
        return "exit_long"
    return "rebalance"
