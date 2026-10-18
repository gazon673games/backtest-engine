"""Translate visible replay rows into a bounded, validated SDK snapshot."""

from datetime import date, datetime, time, timedelta

from strategy_sdk import (
    AccountSnapshot,
    Bar,
    Instrument,
    Position,
    StrategyContext,
    resolve_timezone,
)
from strategy_sdk.time import timeframe_minutes

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.data.bar_availability import row_decision_boundary
from backtest_engine.strategies.sdk_numbers import native_quantity


def localized(value, zone):
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, time())
    if not isinstance(value, datetime):
        raise StrategyPluginError("SDK strategy requires datetime bar timestamps")
    return value.replace(tzinfo=zone) if value.tzinfo is None else value.astimezone(zone)


def sdk_bar(row, requirements):
    zone = resolve_timezone(requirements.timezone)
    duration = timedelta(minutes=timeframe_minutes(requirements.timeframe))
    explicit_close = row.get("available_at") or row.get("bar_close_ts")
    if explicit_close is None:
        # Legacy replay datasets treat ts as the decision/close timestamp. Their
        # acceptance remains controlled by the host's dataset quality policy.
        closed = localized(row["ts"], zone)
        opened = closed - duration
    else:
        opened = localized(row.get("bar_open_ts") or row["ts"], zone)
        closed = localized(explicit_close, zone)
    close = row["close"]
    return Bar(
        opened,
        closed,
        row.get("open", close),
        row.get("high", close),
        row.get("low", close),
        close,
        row.get("volume", 0),
    )


def sdk_context(ctx, strategy, state):
    requirements = strategy.requirements
    history = ctx.history_tail(requirements.history_bars)
    bars = tuple(sdk_bar(row, requirements) for row in history.iter_rows(named=True))
    row = ctx.current_bar()
    symbol = ctx.active_contract or row.get("symbol") or row.get("continuous_symbol") or "SERIES"
    multiplier = ctx.point_value
    if multiplier is None and {"leverage_percent", "initial_percent"}.intersection(
        strategy.parameters
    ):
        raise StrategyPluginError("SDK leverage sizing requires the host's instrument point value")
    quantity = ctx.position_quantity
    if quantity is None:
        quantity = state.get("quantity", ctx.previous_position) if state else ctx.previous_position
    zone = resolve_timezone(requirements.timezone)
    instrument = ctx.instrument_spec
    currency = ctx.base_currency
    as_of = row.get("available_at") or row.get("bar_close_ts") or row["ts"]
    if requirements.decision_times:
        as_of = row_decision_boundary(row, timeframe_minutes(requirements.timeframe))
    return StrategyContext(
        localized(as_of, zone),
        Instrument(
            str(symbol),
            currency,
            multiplier=multiplier if multiplier is not None else 1,
            quantity_step=(instrument.quantity_step or 1) if instrument is not None else 1,
            price_tick=instrument.tick_size if instrument is not None else None,
        ),
        ctx.timeframe,
        bars,
        timezone=requirements.timezone,
        position=Position(
            str(symbol),
            native_quantity(
                quantity,
                (instrument.quantity_step or 1) if instrument is not None else 1,
            ),
        ),
        account=AccountSnapshot(currency, ctx.equity_cash, ctx.available_funds)
        if ctx.equity_cash is not None
        else None,
        state=state,
    )
