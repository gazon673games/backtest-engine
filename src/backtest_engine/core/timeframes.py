from __future__ import annotations

import re

MINUTE_TIMEFRAME_PATTERN = re.compile(r"^[1-9]\d*m$")


def parse_timeframe_minutes(timeframe: str) -> int:
    """Parse strict minute timeframe strings such as 1m, 5m, or 60m."""

    if not MINUTE_TIMEFRAME_PATTERN.fullmatch(timeframe):
        msg = (
            f"Invalid intraday timeframe: {timeframe!r}. "
            "Expected a positive integer followed by 'm', for example 1m, 5m, or 60m."
        )
        raise ValueError(msg)
    return int(timeframe[:-1])
