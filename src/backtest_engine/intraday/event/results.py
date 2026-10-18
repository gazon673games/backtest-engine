from __future__ import annotations

import polars as pl

from backtest_engine.intraday.event.schemas import (
    decision_actions_schema,
    equity_schema,
)


def build_intraday_event_equity(events: pl.DataFrame) -> pl.DataFrame:
    if events.height == 0:
        ts_dtype = events.schema.get("ts", pl.Datetime)
        return pl.DataFrame([], schema=equity_schema(ts_dtype), orient="row")
    columns = list(equity_schema(events.schema["ts"]).keys())
    return events.select(
        [pl.col("next_ts").alias("ts"), *(pl.col(name) for name in columns if name != "ts")]
    )


def build_intraday_event_decision_actions(events: pl.DataFrame) -> pl.DataFrame:
    ts_dtype = events.schema.get("ts", pl.Datetime)
    if events.height == 0:
        return pl.DataFrame([], schema=decision_actions_schema(ts_dtype), orient="row")
    selected = events.filter(
        pl.col("strategy_called")
        | pl.col("missing_decision")
        | (pl.col("turnover") > 0.0)
        | pl.col("contract_roll")
    ).with_columns(
        [
            decision_action_expr().alias("action"),
            decision_reason_expr().alias("reason"),
        ]
    )
    return selected.select(list(decision_actions_schema(ts_dtype))).cast(
        decision_actions_schema(ts_dtype)
    )


def decision_action_expr() -> pl.Expr:
    previous = pl.col("previous_position")
    target = pl.col("target_position")
    return (
        pl.when(pl.col("contract_roll") & (target != 0.0) & (previous == target))
        .then(pl.when(target > 0.0).then(pl.lit("roll_long")).otherwise(pl.lit("roll_short")))
        .when(pl.col("missing_decision"))
        .then(
            pl.when(previous != 0.0)
            .then(pl.lit("missing_decision_close"))
            .otherwise(pl.lit("missing_decision_stay_flat"))
        )
        .when(previous == target)
        .then(
            pl.when(target > 0.0)
            .then(pl.lit("hold_long"))
            .when(target < 0.0)
            .then(pl.lit("hold_short"))
            .otherwise(pl.lit("stay_flat"))
        )
        .when((previous == 0.0) & (target > 0.0))
        .then(pl.lit("open_long"))
        .when((previous == 0.0) & (target < 0.0))
        .then(pl.lit("open_short"))
        .when((previous > 0.0) & (target == 0.0))
        .then(pl.lit("close_long"))
        .when((previous < 0.0) & (target == 0.0))
        .then(pl.lit("close_short"))
        .when((previous > 0.0) & (target < 0.0))
        .then(pl.lit("reverse_long_to_short"))
        .when((previous < 0.0) & (target > 0.0))
        .then(pl.lit("reverse_short_to_long"))
        .otherwise(pl.lit("rebalance"))
    )


def decision_reason_expr() -> pl.Expr:
    strategy_reason = pl.col("strategy_reason").replace("", None)
    signal_reason = pl.col("signal_reason").replace("", None)
    return pl.coalesce(
        strategy_reason,
        signal_reason,
        pl.when(pl.col("missing_decision")).then(pl.lit("missing_decision")),
        pl.when(pl.col("strategy_called")).then(pl.lit("strategy_signal")),
        pl.when(pl.col("turnover") > 0.0).then(pl.lit("engine_position_change")),
        pl.when(pl.col("contract_roll")).then(pl.lit("contract_roll")),
        pl.lit("carried_position"),
    )
