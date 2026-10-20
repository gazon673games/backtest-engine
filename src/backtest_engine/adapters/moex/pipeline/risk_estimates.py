"""Build timestamped, explicitly diagnostic margin schedules from NCC risk rates."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

from backtest_engine.adapters.moex.iss.risk_history import RiskHistory
from backtest_engine.data.margin_estimates import MarginEstimateInterval


def margin_schedule(
    history: RiskHistory,
    dates: list[date],
    multiplier: float,
    *,
    carry_weekend_rates: bool = False,
) -> list[dict[str, Any]]:
    if multiplier <= 0:
        raise ValueError(f"Positive contract multiplier is required for {history.asset}")
    result = []
    for day in sorted(dates):
        todays = [row for row in history.rows if row["tradedate"] == day.isoformat()]
        if not todays and not (carry_weekend_rates and day.weekday() >= 5):
            # Missing weekdays remain gaps; only the declared research weekend policy carries.
            continue
        start = datetime.combine(day, time())
        end = start + timedelta(days=1)
        earlier = [
            row
            for row in history.rows
            if row["tradedate"] <= day.isoformat()
            and datetime.fromisoformat(row["updatetime"]) <= start
        ]
        changes = {}
        if earlier:
            changes[start] = max(earlier, key=lambda row: (row["updatetime"], row["tradedate"]))
        for row in sorted(todays, key=lambda value: value["updatetime"]):
            at = max(start, datetime.fromisoformat(row["updatetime"]))
            if at < end:
                changes[at] = row
        boundaries = sorted(changes)
        for index, at in enumerate(boundaries):
            row = changes[at]
            record = MarginEstimateInterval(
                valid_from=at,
                valid_to=boundaries[index + 1] if index + 1 < len(boundaries) else end,
                available_at=datetime.fromisoformat(row["updatetime"]),
                rate=row["mr1"],
                maximum_contracts=float(Decimal(str(row["lk1"])) / Decimal(str(multiplier))),
                source_sha256=row["source_sha256"],
                source_trade_date=date.fromisoformat(row["tradedate"]),
                carried_forward=row["tradedate"] < day.isoformat(),
            )
            result.append(record.model_dump(mode="json"))
    return result
