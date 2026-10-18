"""Build first-class bracket exit orders with a shared OCO group."""

from __future__ import annotations

from backtest_engine.execution.models import OrderRequest


def bracket_exit_orders(
    *,
    parent_order_id: str,
    side: str,
    quantity: float,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
    oco_group: str | None = None,
) -> list[OrderRequest]:
    group = oco_group or f"bracket:{parent_order_id}"
    orders: list[OrderRequest] = []
    if take_profit_price is not None:
        orders.append(
            OrderRequest(
                order_id=f"{parent_order_id}:take_profit",
                parent_order_id=parent_order_id,
                side=side,
                quantity=quantity,
                order_type="limit",
                limit_price=take_profit_price,
                time_in_force="gtc",
                oco_group=group,
            )
        )
    if stop_loss_price is not None:
        orders.append(
            OrderRequest(
                order_id=f"{parent_order_id}:stop_loss",
                parent_order_id=parent_order_id,
                side=side,
                quantity=quantity,
                order_type="stop",
                stop_price=stop_loss_price,
                time_in_force="gtc",
                oco_group=group,
            )
        )
    return orders
