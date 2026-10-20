from __future__ import annotations

import hashlib
import json
from datetime import date, datetime

import pytest
from moex_risk_fixture import history, payload, rate_row, request

from backtest_engine.adapters.moex.iss.risk_history import download_risk_history, validate_rate_row
from backtest_engine.adapters.moex.pipeline.risk_estimates import margin_schedule
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.instrument_registry import InstrumentSpec, select_instrument_spec


def test_recorded_response_is_cached_and_pinned_for_offline_builds(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.cache.fetch_network_json",
        lambda url, params: calls.append((url, params)) or payload(),
    )
    first = download_risk_history("CNY", request(tmp_path), tmp_path / "reports")
    source = first.sources[0]
    raw = json.loads(__import__("pathlib").Path(source["path"]).read_text("utf-8"))
    digest = hashlib.sha256(
        json.dumps(raw["payload"], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    assert digest == source["sha256"]
    assert first.rows[0]["mr1"] == 0.08
    assert "/rms/engines/futures/objects/limits/CNY.json" in source["url"]
    second = download_risk_history(
        "CNY", request(tmp_path, cache_policy="offline"), tmp_path / "offline"
    )
    assert second.rows == first.rows
    assert len(calls) == 1


def test_history_pages_are_all_read_and_preserved(tmp_path, monkeypatch):
    starts = []

    def fetch(url, params, **_):
        starts.append(params["start"])
        result = payload()
        result["limits"]["data"][0][0] = f"2026-01-{12 + int(params['start'])}"
        result["limits.cursor"]["data"] = [[int(params["start"]), 2, 1]]
        return result

    monkeypatch.setattr("backtest_engine.adapters.moex.iss.risk_history.fetch_iss_json", fetch)
    result = download_risk_history(
        "CNY", request(tmp_path, till_date="2026-01-13"), tmp_path / "reports"
    )
    assert starts == ["0", "1"]
    assert len(result.rows) == len(result.sources) == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"mr1": 0},
        {"mr1": float("nan")},
        {"mr1": 0.2},
        {"lk1": -1},
        {"updatetime": "2026-01-12T15:00:00+03:00"},
    ],
)
def test_invalid_rates_are_not_used(overrides):
    with pytest.raises(ValueError, match="Invalid MOEX risk"):
        validate_rate_row(rate_row(**overrides), "CNY")


def test_wrong_asset_is_not_used():
    with pytest.raises(ValueError, match="asset mismatch"):
        validate_rate_row(rate_row(assetcode="USD"), "CNY")


def test_cold_offline_cache_does_not_use_network(tmp_path, monkeypatch):
    def forbidden(*_):
        pytest.fail("offline build attempted to use the network")

    monkeypatch.setattr("backtest_engine.adapters.moex.iss.cache.fetch_network_json", forbidden)
    with pytest.raises(FileNotFoundError, match="offline"):
        download_risk_history("CNY", request(tmp_path, cache_policy="offline"), tmp_path / "out")


def test_empty_page_with_remaining_rows_is_rejected(tmp_path, monkeypatch):
    data = payload()
    data["limits"]["data"] = []
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.risk_history.fetch_iss_json", lambda *_, **__: data
    )
    with pytest.raises(ValueError, match="Truncated"):
        download_risk_history("CNY", request(tmp_path), tmp_path / "reports")


def spec(rows):
    return InstrumentSpec(
        symbol="CRH6",
        price_currency="RUB",
        pnl_currency="RUB",
        contract_multiplier=1000,
        margin_estimate_schedule=margin_schedule(history(*rows), [date(2026, 1, 12)], 1000),
    )


def resolve(instrument, timestamp, price=12.5):
    return select_instrument_spec(
        row={"active_contract": "CRH6", "close": price},
        current_ts=timestamp,
        specs=[instrument],
        fallback=InstrumentSpec(),
    )


def test_future_intraday_update_cannot_change_an_earlier_margin():
    earlier = rate_row()
    later = rate_row(mr1=0.1, updatetime="2026-01-12 18:50:00")
    instrument = spec([earlier, later])
    assert resolve(instrument, datetime(2026, 1, 12, 15)).initial_margin == 1000
    assert resolve(instrument, datetime(2026, 1, 12, 18, 50)).initial_margin == 1250
    poisoned = spec([earlier, {**later, "mr1": 0.12}])
    assert resolve(poisoned, datetime(2026, 1, 12, 15)).initial_margin == 1000
    assert resolve(instrument, datetime(2026, 1, 12, 15), price=12.50001).initial_margin == 1000.01


def test_missing_trade_date_is_not_filled_from_previous_regime():
    instrument = spec([rate_row()])
    with pytest.raises(DatasetValidationError, match="unavailable.*2026-01-13"):
        resolve(instrument, datetime(2026, 1, 13, 15))


def test_missing_opening_regime_does_not_use_the_evening_rate():
    instrument = spec([rate_row(updatetime="2026-01-12 18:50:00")])
    with pytest.raises(DatasetValidationError, match="unavailable"):
        resolve(instrument, datetime(2026, 1, 12, 15))


def test_weekend_carry_is_explicit_and_never_uses_monday_rates():
    friday = rate_row(tradedate="2026-07-17", updatetime="2026-07-17 05:00:00")
    monday = rate_row(tradedate="2026-07-20", mr1=0.1, updatetime="2026-07-20 18:50:00")
    day = date(2026, 7, 18)
    assert margin_schedule(history(friday, monday), [day], 1000) == []
    schedule = margin_schedule(history(friday, monday), [day], 1000, carry_weekend_rates=True)
    assert schedule[0]["rate"] == 0.08
    assert schedule[0]["source_trade_date"] == "2026-07-17"
    assert schedule[0]["carried_forward"] is True


def test_weekend_policy_still_rejects_missing_weekdays():
    day = date(2026, 1, 13)
    assert margin_schedule(history(), [day], 1000, carry_weekend_rates=True) == []


def test_margin_estimate_does_not_depend_on_process_decimal_precision():
    from decimal import localcontext

    instrument = spec([rate_row()])
    with localcontext(prec=2):
        assert (
            resolve(instrument, datetime(2026, 1, 12, 15), price=12.50001).initial_margin == 1000.01
        )


def test_invalid_asset_code_cannot_escape_raw_snapshot_directory(tmp_path):
    with pytest.raises(ValueError, match="Invalid MOEX risk asset code"):
        download_risk_history("../CNY", request(tmp_path), tmp_path / "reports")
