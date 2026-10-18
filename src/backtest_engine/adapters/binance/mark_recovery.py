"""Explicit recovery of absent archive marks from preserved REST observations."""

from backtest_engine.adapters.binance.download import download_mark_klines
from backtest_engine.adapters.binance.mark_minutes import complete_minute_marks
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.core.timeframes import parse_timeframe_minutes


def recover_missing_marks(request, transport, bars, marks, sources):
    if request.history_source != "auto":
        raise DatasetValidationError("Missing-mark REST recovery requires history_source=auto")
    observed = {int(row[0]): row for row in marks}
    missing = [row for row in bars if int(row[0]) not in observed]
    if not missing:
        return marks
    # Fetch only contiguous missing runs; a distant second gap must not trigger
    # years of needless REST requests. Retain all existing archive observations.
    runs = []
    step = parse_timeframe_minutes(request.timeframe) * 60_000
    for row in missing:
        if runs and int(row[0]) == int(runs[-1][-1][0]) + step:
            runs[-1].append(row)
        else:
            runs.append([row])
    recovered = []
    for run in runs:
        recovered.extend(
            download_mark_klines(
                transport,
                symbol=request.security,
                timeframe=request.timeframe,
                start_ms=int(run[0][0]),
                end_ms=int(run[-1][6]),
            )
        )
    wanted = {int(row[0]) for row in missing}
    added = {int(row[0]): row for row in recovered if int(row[0]) in wanted}
    for run in runs:
        unresolved = [row for row in run if int(row[0]) not in added]
        added.update(complete_minute_marks(transport, request.security, unresolved, step, sources))
    observed.update(added)
    sources.append(
        {
            "resource": "markPriceKlines",
            "policy": "missing_archive_marks_from_rest",
            "requested_missing_timestamps": sorted(wanted),
            "recovered_timestamps": sorted(added),
            "unresolved_timestamps": sorted(wanted - added.keys()),
        }
    )
    return [observed[index] for index in sorted(observed)]
