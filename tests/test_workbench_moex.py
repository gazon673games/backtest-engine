from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.cache import cache_path_for, fetch_iss_json_cached
from backtest_engine.adapters.moex.iss.candles import (
    build_moex_dataset,
    download_network_candles,
    iss_table_rows,
    list_moex_futures_roots,
    list_moex_instruments,
)
from backtest_engine.adapters.moex.iss.contract_selection import nearest_rows
from backtest_engine.adapters.moex.pipeline.raw_storage import (
    write_raw_snapshot,
)
from backtest_engine.intraday.engine import validate_intraday_dataset


def test_moex_adapter_lists_instruments_from_iss_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch(
        _url: str,
        _params: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> dict[str, object]:
        return {
            "securities": {
                "columns": ["SECID", "SHORTNAME", "SECNAME", "LASTTRADEDATE"],
                "data": [
                    ["CRU4", "CNYRUB", "CNY/RUB futures", "2024-09-19"],
                    ["CRZ4", "CNYRUB", "CNY/RUB futures", "2024-12-19"],
                    ["SiU4", "Si", "USD/RUB", "2024-09-19"],
                ],
            }
        }

    monkeypatch.setattr("backtest_engine.adapters.moex.iss.catalog.fetch_iss_json", fake_fetch)

    instruments = list_moex_instruments(query="cny")

    assert instruments[0]["symbol"] == "CRU4"
    assert instruments[0]["short_name"] == "CNYRUB"
    roots = list_moex_futures_roots(query="cr")
    assert roots[0]["symbol"] == "CR"
    assert roots[0]["contract_count"] == 2


def test_moex_dataset_builder_writes_backtest_ready_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.candles.download_moex_candles",
        lambda _request, **_kwargs: moex_raw_rows(),
    )
    calendar = tmp_path / "expected-calendar.json"
    calendar.write_text(
        json.dumps(
            {
                "trading_dates": ["2024-01-02"],
                "bar_minutes": 5,
                "session_windows": [{"start": "10:00", "end": "10:10"}],
            }
        ),
        encoding="utf-8",
    )
    result = build_moex_dataset(
        DatasetBuildRequest(
            adapter_id="moex",
            engine="futures",
            market="forts",
            board="RFUD",
            security="CRU4",
            calendar_path=calendar,
            from_date="2024-01-02",
            till_date="2024-01-02",
            timeframe="5m",
            output_root=tmp_path / "datasets",
        )
    )
    frame = pl.read_parquet(result.dataset_path)
    validation = validate_intraday_dataset(frame, result.dataset_path)

    assert result.row_count == 2
    assert result.approval_status == "approved"
    assert result.quality["approval_status"] == "approved"
    assert validation["validation_status"] == "accepted"
    assert frame.get_column("engine").to_list() == ["futures", "futures"]
    assert frame.get_column("board").to_list() == ["RFUD", "RFUD"]
    assert frame.get_column("is_final").to_list() == [True, True]
    assert frame.get_column("available_at").to_list() == frame.get_column("bar_close_ts").to_list()
    assert result.raw_snapshot_path.parent.name == "raw"
    assert result.raw_snapshot_path.name.endswith(".json.gz")
    assert len(list(result.raw_snapshot_path.parent.glob("*.json.gz"))) == 1
    with gzip.open(result.raw_snapshot_path, "rt", encoding="utf-8") as source:
        assert json.load(source)["rows"]
    assert not (result.output_root / "adapter_raw").exists()
    calendar = json.loads(result.calendar_path.read_text(encoding="utf-8"))
    assert calendar["session_windows"] == [{"start": "10:00", "end": "10:10"}]
    specs = json.loads(result.instrument_registry_path.read_text(encoding="utf-8"))["instruments"]
    assert specs[0]["quantity_step"] == 1.0
    assert specs[0]["minimum_quantity"] == 1.0


def test_raw_snapshot_publication_preserves_previous_versions(tmp_path: Path) -> None:
    output_root = tmp_path / "dataset"
    raw_dir = output_root / "raw"
    old = write_raw_snapshot(raw_dir, {"rows": [{"close": 1.0}]})
    current = write_raw_snapshot(raw_dir, {"rows": [{"close": 2.0}]})
    legacy = output_root / "adapter_raw" / "candles_raw.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{}", encoding="utf-8")

    assert current.exists()
    assert old.exists()
    assert legacy.exists()


def test_live_tail_bypasses_prefer_cache_for_last_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policies = []

    def fake_fetch(_url, _params, *, cache_policy, **_kwargs):
        policies.append(cache_policy)
        close = 2.0 if cache_policy == "refresh" else 1.0
        return {
            "candles": {
                "columns": ["begin", "open", "high", "low", "close", "volume", "value"],
                "data": [["2026-07-19T10:00:00", close, close, close, close, 1.0, 1.0]],
            },
            "candles.cursor": {
                "columns": ["INDEX", "TOTAL", "PAGESIZE"],
                "data": [[0, 1, 500]],
            },
        }

    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.candle_transport.fetch_iss_json",
        fake_fetch,
    )
    request = DatasetBuildRequest(
        adapter_id="moex",
        engine="futures",
        market="forts",
        board="RFUD",
        security="CRU6",
        from_date="2026-07-19",
        till_date="9999-12-31",
        cache_policy="prefer",
        cache_dir=tmp_path,
    )

    rows = download_network_candles(request)

    assert policies == ["prefer", "refresh"]
    assert rows[0]["close"] == 2.0


def test_iss_table_rows_handles_standard_table_payload() -> None:
    payload = {"items": {"columns": ["id", "name"], "data": [["a", "Alpha"]]}}

    assert iss_table_rows(payload, "items") == [{"id": "a", "name": "Alpha"}]


def test_moex_cache_supports_reproducible_offline_reads(tmp_path: Path) -> None:
    url = "https://example.test/iss/items.json"
    params = {"iss.meta": "off", "start": "0"}
    path = cache_path_for(url, params, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "url": url,
                "params": params,
                "fetched_at": "2024-01-01T00:00:00+00:00",
                "payload": {"items": {"columns": ["id"], "data": [["cached"]]}},
            }
        ),
        encoding="utf-8",
    )

    payload = fetch_iss_json_cached(url, params, cache_policy="offline", cache_dir=tmp_path)

    assert payload["items"]["data"] == [["cached"]]


def test_nearest_rows_selects_earliest_live_contract() -> None:
    contracts = [
        {"SECID": "CRU4", "LASTTRADEDATE": "2024-09-19"},
        {"SECID": "CRZ4", "LASTTRADEDATE": "2024-12-19"},
    ]
    rows = [
        {"begin": "2024-10-01T10:00:00", "SECID": "CRU4", "close": 1.0},
        {"begin": "2024-10-01T10:00:00", "SECID": "CRZ4", "close": 2.0},
        {"begin": "2024-08-01T10:00:00", "SECID": "CRU4", "close": 3.0},
        {"begin": "2024-08-01T10:00:00", "SECID": "CRZ4", "close": 4.0},
    ]

    selected = nearest_rows(rows, contracts)

    assert [row["SECID"] for row in selected] == ["CRU4", "CRZ4"]


def test_nearest_rows_does_not_fill_front_contract_gaps_from_next_contract() -> None:
    contracts = [
        {"SECID": "CRU4", "LASTTRADEDATE": "2024-09-19"},
        {"SECID": "CRZ4", "LASTTRADEDATE": "2024-12-19"},
    ]
    rows = [
        {"begin": "2024-08-01T10:00:00", "SECID": "CRU4", "close": 1.0},
        {"begin": "2024-08-01T10:00:00", "SECID": "CRZ4", "close": 2.0},
        {"begin": "2024-08-01T10:01:00", "SECID": "CRZ4", "close": 2.1},
    ]

    selected = nearest_rows(rows, contracts)

    assert selected == [rows[0]]


def test_volume_roll_uses_daily_leader_without_rolling_back() -> None:
    contracts = [
        {"SECID": "CRU4", "LASTTRADEDATE": "2024-09-19"},
        {"SECID": "CRZ4", "LASTTRADEDATE": "2024-12-19"},
    ]
    rows = [
        {"begin": "2024-08-01T10:00:00", "SECID": "CRU4", "volume": 100.0},
        {"begin": "2024-08-01T10:00:00", "SECID": "CRZ4", "volume": 300.0},
        {"begin": "2024-08-01T10:05:00", "SECID": "CRU4", "volume": 100.0},
        {"begin": "2024-08-01T10:05:00", "SECID": "CRZ4", "volume": 300.0},
        {"begin": "2024-08-02T10:00:00", "SECID": "CRU4", "volume": 1000.0},
        {"begin": "2024-08-02T10:00:00", "SECID": "CRZ4", "volume": 100.0},
    ]

    selected = nearest_rows(rows, contracts, rule="volume")

    assert [row["SECID"] for row in selected] == ["CRU4", "CRU4", "CRZ4"]


def moex_raw_rows() -> list[dict[str, object]]:
    start = datetime(2024, 1, 2, 10, 0)
    rows = []
    for index in range(10):
        close = 10.0 + index * 0.1
        rows.append(
            {
                "begin": (start + timedelta(minutes=index)).isoformat(),
                "open": close,
                "high": close + 0.01,
                "low": close - 0.01,
                "close": close,
                "volume": 1000.0,
                "value": 100000.0,
                "SECID": "CRU4",
                "source": "moex_iss",
                "engine": "futures",
                "market": "forts",
                "board": "RFUD",
                "currency": "RUB",
            }
        )
    return rows
