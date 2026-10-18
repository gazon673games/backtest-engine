"""The same fee schedule applies to single and portfolio executions."""

from typing import Protocol


class FeeSchedule(Protocol):
    maker_fee_bps: float
    taker_fee_bps: float


def trading_fee_bps(schedule: FeeSchedule, liquidity_role: str, fallback_bps: float) -> float:
    # A declared schedule may intentionally give makers a zero fee. Only an
    # entirely zero schedule falls back to the run's explicit commission.
    if schedule.maker_fee_bps or schedule.taker_fee_bps:
        return schedule.maker_fee_bps if liquidity_role == "maker" else schedule.taker_fee_bps
    return fallback_bps
