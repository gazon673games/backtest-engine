from types import SimpleNamespace

from backtest_engine.adapters.binance.mark_recovery import recover_missing_marks


def bar(stamp, minutes=1):
    return [
        stamp,
        "100",
        "102",
        "99",
        "101",
        "0",
        stamp + minutes * 60_000 - 1,
        "0",
        0,
        "0",
        "0",
        "0",
    ]


class MinuteTransport:
    def __init__(self, missing=None):
        self.missing = missing
        self.calls = []

    def get(self, product, path, params):
        self.calls.append(params)
        if params["interval"] != "1m":
            return []
        return [
            bar(stamp)
            for stamp in range(params["startTime"], params["endTime"] + 1, 60_000)
            if stamp != self.missing
        ]


def recover(transport, timeframe="5m", bars=None):
    sources = []
    request = SimpleNamespace(history_source="auto", security="ETHUSDT", timeframe=timeframe)
    existing = bar(300_000, 5)
    result = recover_missing_marks(
        request, transport, bars or [bar(0, 5), existing], [existing], sources
    )
    return result, sources, existing


def test_complete_minutes_recover_mark_without_changing_existing_archive():
    result, sources, existing = recover(MinuteTransport())
    assert result == [bar(0, 5), existing]
    assert sources[0]["policy"] == "aggregate_complete_observed_1m_marks"
    assert sources[-1]["recovered_timestamps"] == [0]
    assert sources[-1]["unresolved_timestamps"] == []


def test_partial_minutes_leave_missing_mark_unresolved():
    result, sources, existing = recover(MinuteTransport(missing=120_000))
    assert result == [existing]
    assert sources[-1]["unresolved_timestamps"] == [0]


def test_large_native_outage_does_not_trigger_unbounded_minute_requests():
    transport = MinuteTransport()
    recover(transport, "60m", [bar(0, 60), bar(3_600_000, 60)])
    assert all(call["interval"] == "1h" for call in transport.calls)


def test_invalid_minute_prices_cannot_be_hidden_by_aggregation():
    class Invalid(MinuteTransport):
        def get(self, product, path, params):
            rows = super().get(product, path, params)
            if rows:
                rows[2][2] = "98"  # Below both the opening and closing mark.
            return rows

    result, sources, existing = recover(Invalid())
    assert result == [existing]
    assert sources[-1]["unresolved_timestamps"] == [0]
