from __future__ import annotations

from typing import Any

import polars as pl

from backtest_engine.intraday.config import IntradayExecutionConfig
from backtest_engine.intraday.utils import cost_rate, validate_interval_prices


def build_intraday_equity_curve(
    decision_points: pl.DataFrame,
    execution: IntradayExecutionConfig,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    current_equity = 1.0
    previous_position = 0.0

    for index in range(max(decision_points.height - 1, 0)):
        current = decision_points.row(index, named=True)
        next_decision = decision_points.row(index + 1, named=True)
        decision_close = float(current["decision_close"])
        next_decision_close = float(next_decision["decision_close"])
        validate_interval_prices(decision_close, next_decision_close)

        target_position = float(current["target_position"])
        turnover = abs(target_position - previous_position)
        gross_return = target_position * (next_decision_close / decision_close - 1.0)
        cost = turnover * cost_rate(execution.commission_bps, execution.slippage_bps)
        net_return = gross_return - cost
        current_equity *= 1.0 + net_return

        rows.append(
            {
                "decision_ts": current["decision_ts"],
                "next_decision_ts": next_decision["decision_ts"],
                "decision_close": decision_close,
                "next_decision_close": next_decision_close,
                "target_position": target_position,
                "active_contract": current["active_contract"],
                "next_active_contract": next_decision["active_contract"],
                "source_contract_symbol": current["source_contract_symbol"],
                "next_source_contract_symbol": next_decision["source_contract_symbol"],
                "turnover": turnover,
                "gross_return": gross_return,
                "cost": cost,
                "net_return": net_return,
                "equity": current_equity,
            }
        )
        previous_position = target_position

    if rows and rows[-1]["target_position"] != 0.0:
        closing_turnover = abs(float(rows[-1]["target_position"]))
        closing_cost = closing_turnover * cost_rate(
            execution.commission_bps,
            execution.slippage_bps,
        )
        rows[-1]["turnover"] += closing_turnover
        rows[-1]["cost"] += closing_cost
        rows[-1]["net_return"] -= closing_cost
        previous_equity = 1.0 if len(rows) == 1 else float(rows[-2]["equity"])
        rows[-1]["equity"] = previous_equity * (1.0 + float(rows[-1]["net_return"]))

    ts_dtype = (
        decision_points.schema["decision_ts"]
        if "decision_ts" in decision_points.schema
        else pl.Datetime
    )
    return pl.DataFrame(rows, schema=equity_schema(ts_dtype), orient="row")


def attribute_intraday_trades(
    equity: pl.DataFrame,
    dataset_validation: dict[str, Any],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    positions = [float(value) for value in equity.get_column("target_position").to_list()]
    active_start_index: int | None = None
    trade_id = 1

    for index, position in enumerate(positions):
        previous_position = positions[index - 1] if index > 0 else 0.0
        if previous_position == 0.0 and position > 0.0:
            active_start_index = index
        elif previous_position > 0.0 and position == 0.0 and active_start_index is not None:
            rows.append(
                build_intraday_long_trade(
                    equity,
                    dataset_validation,
                    trade_id=trade_id,
                    active_start_index=active_start_index,
                    exit_index=index,
                    closed_by_end_of_data=False,
                )
            )
            trade_id += 1
            active_start_index = None

    if active_start_index is not None and equity.height > 0:
        rows.append(
            build_intraday_long_trade(
                equity,
                dataset_validation,
                trade_id=trade_id,
                active_start_index=active_start_index,
                exit_index=equity.height - 1,
                closed_by_end_of_data=True,
            )
        )

    ts_dtype = equity.schema["decision_ts"] if "decision_ts" in equity.schema else pl.Datetime
    return pl.DataFrame(rows, schema=trades_schema(ts_dtype), orient="row")


def build_intraday_long_trade(
    equity: pl.DataFrame,
    dataset_validation: dict[str, Any],
    *,
    trade_id: int,
    active_start_index: int,
    exit_index: int,
    closed_by_end_of_data: bool,
) -> dict[str, Any]:
    entry_price = float(equity.get_column("decision_close")[active_start_index])
    exit_price, exit_ts, holding_intervals, price_path = _trade_exit_state(
        equity,
        entry_price,
        active_start_index,
        exit_index,
        closed_by_end_of_data,
    )
    gross_return = exit_price / entry_price - 1.0
    cost = sum_cost(equity, active_start_index, exit_index)
    net_return = gross_return - cost
    excursions = [price / entry_price - 1.0 for price in price_path]

    return {
        "trade_id": trade_id,
        "continuous_symbol": str(dataset_validation["continuous_symbol"]),
        "adjustment_mode": str(dataset_validation["adjustment_mode"]),
        "direction": "long",
        "entry_ts": equity.get_column("decision_ts")[active_start_index],
        "exit_ts": exit_ts,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_active_contract": equity.get_column("active_contract")[active_start_index],
        "exit_active_contract": exit_active_contract(equity, exit_index, closed_by_end_of_data),
        "entry_source_contract_symbol": equity.get_column("source_contract_symbol")[
            active_start_index
        ],
        "exit_source_contract_symbol": exit_source_contract(
            equity,
            exit_index,
            closed_by_end_of_data,
        ),
        "holding_intervals": holding_intervals,
        "gross_return": gross_return,
        "cost": cost,
        "net_return": net_return,
        "pnl_contribution": net_return,
        "max_favorable_excursion": max(excursions),
        "max_adverse_excursion": min(excursions),
        "winning_trade": net_return > 0.0,
        "closed_by_end_of_data": closed_by_end_of_data,
    }


def equity_schema(ts_dtype: pl.DataType) -> dict[str, pl.DataType]:
    return {
        "decision_ts": ts_dtype,
        "next_decision_ts": ts_dtype,
        "decision_close": pl.Float64,
        "next_decision_close": pl.Float64,
        "target_position": pl.Float64,
        "active_contract": pl.String,
        "next_active_contract": pl.String,
        "source_contract_symbol": pl.String,
        "next_source_contract_symbol": pl.String,
        "turnover": pl.Float64,
        "gross_return": pl.Float64,
        "cost": pl.Float64,
        "net_return": pl.Float64,
        "equity": pl.Float64,
    }


def trades_schema(ts_dtype: pl.DataType) -> dict[str, pl.DataType]:
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
        "holding_intervals": pl.Int64,
        "gross_return": pl.Float64,
        "cost": pl.Float64,
        "net_return": pl.Float64,
        "pnl_contribution": pl.Float64,
        "max_favorable_excursion": pl.Float64,
        "max_adverse_excursion": pl.Float64,
        "winning_trade": pl.Boolean,
        "closed_by_end_of_data": pl.Boolean,
    }


def sum_cost(equity: pl.DataFrame, start_index: int, end_index: int) -> float:
    if end_index < start_index:
        return 0.0
    return float(equity.get_column("cost").slice(start_index, end_index - start_index + 1).sum())


def exit_active_contract(
    equity: pl.DataFrame,
    exit_index: int,
    closed_by_end_of_data: bool,
) -> Any:
    if closed_by_end_of_data:
        return equity.get_column("next_active_contract")[exit_index]
    return equity.get_column("active_contract")[exit_index]


def exit_source_contract(
    equity: pl.DataFrame,
    exit_index: int,
    closed_by_end_of_data: bool,
) -> Any:
    if closed_by_end_of_data:
        return equity.get_column("next_source_contract_symbol")[exit_index]
    return equity.get_column("source_contract_symbol")[exit_index]


def _trade_exit_state(
    equity: pl.DataFrame,
    entry_price: float,
    active_start_index: int,
    exit_index: int,
    closed_by_end_of_data: bool,
) -> tuple[float, Any, int, list[float]]:
    if closed_by_end_of_data:
        exit_price = float(equity.get_column("next_decision_close")[exit_index])
        exit_ts = equity.get_column("next_decision_ts")[exit_index]
        holding_intervals = exit_index - active_start_index + 1
    else:
        exit_price = float(equity.get_column("decision_close")[exit_index])
        exit_ts = equity.get_column("decision_ts")[exit_index]
        holding_intervals = exit_index - active_start_index
    price_path = _price_path(equity, entry_price, active_start_index, holding_intervals)
    return exit_price, exit_ts, holding_intervals, price_path


def _price_path(
    equity: pl.DataFrame,
    entry_price: float,
    active_start_index: int,
    holding_intervals: int,
) -> list[float]:
    prices = [
        float(value)
        for value in equity.get_column("next_decision_close").slice(
            active_start_index,
            holding_intervals,
        )
    ]
    prices.insert(0, entry_price)
    return prices
