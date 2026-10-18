"""Plan missing query intervals without inventing bars inside observed coverage."""

from datetime import date, datetime, timedelta

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.candle_transport import live_tail_requested
from backtest_engine.adapters.moex.iss.raw_snapshot_cache import CachedCandleSnapshot
from backtest_engine.adapters.moex.sessions import MOEX_TIMEZONE


def missing_intervals(
    request: DatasetBuildRequest,
    snapshots: list[CachedCandleSnapshot],
) -> list[tuple[str, str]]:
    ranges = []
    for snapshot in snapshots:
        through = snapshot.confirmed_till or snapshot.data_till
        if through is None:
            continue
        # Legacy snapshots can end partway through a day; requery that boundary once.
        if not snapshot.confirmed_till and through < request.till_date:
            through = shift_date(through, -1)
        if live_tail_requested(request):
            through = min(through, shift_date(datetime.now(MOEX_TIMEZONE).date().isoformat(), -1))
        if snapshot.filename_from <= through:
            ranges.append((snapshot.filename_from, through))
    cursor = request.from_date
    missing = []
    for first, last in sorted(ranges):
        if last < cursor or first > request.till_date:
            continue
        if first > cursor:
            missing.append((cursor, min(shift_date(first, -1), request.till_date)))
        if last >= request.till_date:
            return missing
        cursor = max(cursor, shift_date(last, 1))
    if cursor <= request.till_date:
        missing.append((cursor, request.till_date))
    return missing


def shift_date(value: str, days: int) -> str:
    return (date.fromisoformat(value) + timedelta(days=days)).isoformat()
