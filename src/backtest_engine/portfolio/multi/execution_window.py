"""Causal UTC restrictions on increasing risk, independent of strategy signals."""

from datetime import UTC, datetime

from backtest_engine.portfolio.multi.strategy_config import SDKSleeveConfig


def apply_increase_window(
    decision: dict, current_quantity: float, at: datetime, sleeve: SDKSleeveConfig
) -> None:
    if not sleeve.risk_increase_blocked_weekdays and sleeve.risk_increase_allowed_hours is None:
        return
    # Native naive timeline timestamps represent UTC, never the machine's local zone.
    boundary = at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)
    blocked = boundary.weekday() in sleeve.risk_increase_blocked_weekdays
    if sleeve.risk_increase_allowed_hours is not None:
        blocked |= boundary.hour not in sleeve.risk_increase_allowed_hours
    if not blocked:
        return
    target_quantity = decision["target"] * decision["contracts"]
    if target_quantity * current_quantity < 0:
        # A reversal contains a close and a fresh entry. Permit only the close.
        decision.update(target=0, contracts=0.0, reason="risk_increase_window_close_only")
    elif abs(target_quantity) > abs(current_quantity):
        decision.update(
            target=(current_quantity > 0) - (current_quantity < 0),
            contracts=abs(current_quantity),
            reason="risk_increase_window_blocked",
        )
