"""Deterministic OHLC fill rules with explicit gap and path assumptions."""

from __future__ import annotations

from dataclasses import replace

from backtest_engine.execution.models import ExecutionBar, OrderRequest, OrderState


def impact_price(request: OrderRequest, price: float, market_impact_bps: float) -> float:
    impact = market_impact_bps / 10_000.0
    if impact == 0.0:
        return price
    impacted = price * (1.0 + impact) if request.side == "buy" else price * (1.0 - impact)
    if request.order_type in {"limit", "stop_limit"}:
        limit = float(request.limit_price)
        return min(impacted, limit) if request.side == "buy" else max(impacted, limit)
    return impacted


def bar_fill_price(
    request: OrderRequest,
    bar: ExecutionBar,
    intrabar_ambiguity_policy: str,
    *,
    stop_limit_previously_triggered: bool = False,
) -> float | None:
    if request.order_type == "market":
        return market_price(request, bar)
    if request.order_type == "limit":
        return limit_price(request, bar)
    if request.order_type in {"stop", "trailing_stop"}:
        return stop_price(request, bar)
    if stop_limit_previously_triggered:
        return limit_price(request, bar)
    return stop_limit_price(request, bar, intrabar_ambiguity_policy)


def market_price(request: OrderRequest, bar: ExecutionBar) -> float:
    if request.side == "buy":
        if bar.ask is not None:
            return bar.ask
    elif bar.bid is not None:
        return bar.bid
    return bar.market_reference_price if bar.market_reference_price is not None else bar.open


def limit_price(request: OrderRequest, bar: ExecutionBar) -> float | None:
    limit = float(request.limit_price)
    if request.side == "buy":
        if bar.open <= limit:
            return bar.open
        return limit if bar.low <= limit else None
    if bar.open >= limit:
        return bar.open
    return limit if bar.high >= limit else None


def stop_price(request: OrderRequest, bar: ExecutionBar) -> float | None:
    stop = float(request.stop_price)
    if request.side == "buy":
        if bar.open >= stop:
            return bar.open
        return stop if bar.high >= stop else None
    if bar.open <= stop:
        return bar.open
    return stop if bar.low <= stop else None


def stop_limit_price(
    request: OrderRequest,
    bar: ExecutionBar,
    policy: str,
) -> float | None:
    stop = float(request.stop_price)
    limit = float(request.limit_price)
    triggered_at_open = bar.open >= stop if request.side == "buy" else bar.open <= stop
    if triggered_at_open:
        return limit_price(request, bar)
    stop_touched = bar.high >= stop if request.side == "buy" else bar.low <= stop
    limit_touched = bar.low <= limit if request.side == "buy" else bar.high >= limit
    if not stop_touched or not limit_touched:
        return None
    immediately_marketable = limit >= stop if request.side == "buy" else limit <= stop
    if immediately_marketable:
        return stop
    return limit if policy in {"optimistic", "stop_first"} else None


def stop_limit_triggered(request: OrderRequest, bar: ExecutionBar) -> bool:
    stop = float(request.stop_price)
    return bar.high >= stop if request.side == "buy" else bar.low <= stop


def initialize_trailing_state(
    state: OrderState,
    bar: ExecutionBar,
    bar_index: int,
) -> OrderState:
    request = state.request
    if request.order_type != "trailing_stop" or request.stop_price is not None:
        return state
    watermark = bar.open
    if request.side == "sell":
        request = replace(request, stop_price=watermark - float(request.trailing_offset))
        return replace(
            state,
            request=request,
            high_watermark=watermark,
            updated_bar=bar_index,
        )
    request = replace(request, stop_price=watermark + float(request.trailing_offset))
    return replace(
        state,
        request=request,
        low_watermark=watermark,
        updated_bar=bar_index,
    )


def advance_trailing_state(
    state: OrderState,
    bar: ExecutionBar,
    bar_index: int,
) -> OrderState:
    request = state.request
    if request.order_type != "trailing_stop":
        return state
    if request.side == "sell":
        watermark = max(float(state.high_watermark), bar.high)
        request = replace(request, stop_price=watermark - float(request.trailing_offset))
        return replace(
            state,
            request=request,
            high_watermark=watermark,
            updated_bar=bar_index,
        )
    watermark = min(float(state.low_watermark), bar.low)
    request = replace(request, stop_price=watermark + float(request.trailing_offset))
    return replace(
        state,
        request=request,
        low_watermark=watermark,
        updated_bar=bar_index,
    )


def oco_priority(request: OrderRequest, policy: str) -> int:
    stop_like = request.order_type in {"stop", "trailing_stop", "stop_limit"}
    prefer_stop = policy in {"conservative", "stop_first"}
    return 0 if stop_like == prefer_stop else 1


def ordered_open_order_ids(
    orders: dict[str, OrderState],
    open_order_ids: set[str],
    policy: str,
) -> list[str]:
    return sorted(
        open_order_ids,
        key=lambda order_id: (
            orders[order_id].created_bar,
            orders[order_id].request.oco_group or order_id,
            oco_priority(orders[order_id].request, policy),
            order_id,
        ),
    )
