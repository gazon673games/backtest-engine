from __future__ import annotations

import json
from pathlib import Path

import pytest

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.candles import (
    contracts_for_root,
    download_moex_candles,
)
from backtest_engine.adapters.moex.iss.raw_snapshot_cache import (
    cached_candle_rows,
    cached_candle_snapshot,
    snapshot_contract_rows,
)


@pytest.fixture(autouse=True)
def isolate_candle_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.raw_snapshot_cache.DEFAULT_CACHE_DIR",
        tmp_path / "cache",
    )


def test_raw_snapshot_cache_slices_requested_dates(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path / "candles"
    directory = (
        root / "engine=futures" / "market=forts" / "board=RFUD" / "symbol=CRU6" / "timeframe=5m"
    )
    directory.mkdir(parents=True)
    path = directory / "from=2026-01-01_till=2026-12-31.json"
    path.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "candles": {
                            "columns": ["open", "close", "high", "low", "begin"],
                            "data": [
                                [1, 1, 1, 1, "2026-06-30 10:00:00"],
                                [2, 2, 2, 2, "2026-07-01 10:00:00"],
                                [3, 3, 3, 3, "2026-07-03 10:00:00"],
                            ],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WORKBENCH_MOEX_RAW_ROOT", str(root))
    request = DatasetBuildRequest(
        adapter_id="moex",
        engine="futures",
        market="forts",
        board="RFUD",
        security="CRU6",
        instrument_mode="security",
        from_date="2026-07-01",
        till_date="2026-07-02",
        timeframe="5m",
    )

    assert cached_candle_rows(request) == [
        {"open": 2, "close": 2, "high": 2, "low": 2, "begin": "2026-07-01 10:00:00"}
    ]
    snapshot = cached_candle_snapshot(request)
    assert snapshot is not None
    assert snapshot.filename_till == "2026-12-31"
    assert snapshot.data_till == "2026-07-03"


def test_preferred_snapshot_downloads_and_merges_only_stale_tail(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path / "candles"
    directory = contract_directory(root, "CRU6")
    directory.mkdir(parents=True)
    (directory / "from=2026-01-01_till=2026-09-17.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "candles": {
                            "columns": ["open", "close", "high", "low", "begin"],
                            "data": [
                                [1, 1, 1, 1, "2026-07-01 10:00:00"],
                                [2, 2, 2, 2, "2026-07-03 10:00:00"],
                            ],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WORKBENCH_MOEX_RAW_ROOT", str(root))
    calls = []

    def fake_fetch(
        _url: str,
        params: dict[str, str],
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append((params, kwargs))
        return {
            "candles": {
                "columns": ["open", "close", "high", "low", "begin"],
                "data": [
                    [20, 20, 20, 20, "2026-07-03 10:00:00"],
                    [3, 3, 3, 3, "2026-07-06 10:00:00"],
                ],
            }
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
        from_date="2026-07-01",
        till_date="2026-07-13",
        timeframe="5m",
        cache_policy="prefer",
    )

    rows = download_moex_candles(request)

    assert [row["begin"] for row in rows] == [
        "2026-07-01 10:00:00",
        "2026-07-03 10:00:00",
        "2026-07-06 10:00:00",
    ]
    assert rows[1]["close"] == 20
    assert calls[0][0]["from"] == "2026-07-03"
    assert calls[0][0]["till"] == "2026-07-13"
    assert calls[0][1]["cache_policy"] == "prefer"


def test_expired_contract_snapshot_can_overlap_a_long_request(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path / "candles"
    directory = contract_directory(root, "CRU2")
    directory.mkdir(parents=True)
    (directory / "from=2022-01-01_till=2022-09-15.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "candles": {
                            "columns": ["open", "close", "high", "low", "begin"],
                            "data": [[1, 1, 1, 1, "2022-09-01 10:00:00"]],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WORKBENCH_MOEX_RAW_ROOT", str(root))
    request = DatasetBuildRequest(
        adapter_id="moex",
        engine="futures",
        market="forts",
        board="RFUD",
        security="CRU2",
        from_date="2022-09-01",
        till_date="2026-07-02",
        timeframe="5m",
    )

    assert len(cached_candle_rows(request) or []) == 1


def test_live_tail_refresh_failure_does_not_publish_stale_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "candles"
    directory = contract_directory(root, "CRU6")
    directory.mkdir(parents=True)
    (directory / "from=2026-01-01_till=9999-12-31.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "candles": {
                            "columns": ["open", "close", "high", "low", "begin"],
                            "data": [[1, 1, 1, 1, "2026-08-31 23:49:00"]],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WORKBENCH_MOEX_RAW_ROOT", str(root))

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise OSError("ISS unavailable")

    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.security_download.download_network_candles",
        unavailable,
    )
    request = DatasetBuildRequest(
        adapter_id="moex",
        engine="futures",
        market="forts",
        board="RFUD",
        security="CRU6",
        from_date="2026-08-01",
        till_date="9999-12-31",
        timeframe="5m",
        cache_policy="prefer",
    )

    with pytest.raises(OSError, match="ISS unavailable"):
        download_moex_candles(request)


def test_historical_contracts_are_discovered_and_merged(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    root = tmp_path / "candles"
    for symbol, expiry in (("CRU2", "2022-09-15"), ("CRZ2", "2022-12-15"), ("CRH3", "2023-03-16")):
        directory = contract_directory(root, symbol)
        directory.mkdir(parents=True)
        (directory / f"from=2022-01-01_till={expiry}.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("WORKBENCH_MOEX_RAW_ROOT", str(root))
    request = DatasetBuildRequest(
        adapter_id="moex",
        engine="futures",
        market="forts",
        board="RFUD",
        security="CR",
        instrument_mode="nearest_futures",
        from_date="2022-09-01",
        till_date="2022-12-20",
        timeframe="5m",
        cache_policy="offline",
        cache_dir=tmp_path / "iss-cache",
    )
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.contract_catalog.snapshot_instrument_rows",
        lambda **_: [
            {
                "SECID": "CRH3",
                "LASTTRADEDATE": "2023-03-16",
                "LOTSIZE": 1000,
                "MINSTEP": 0.001,
            }
        ],
    )

    discovered = snapshot_contract_rows(request, "CR")
    assert sorted(row["SECID"] for row in discovered) == ["CRH3", "CRU2", "CRZ2"]
    assert all("LASTTRADEDATE" not in row and "LOTSIZE" not in row for row in discovered)
    with pytest.raises(ValueError, match="expiry metadata is missing"):
        contracts_for_root(request)


def contract_directory(root: Path, symbol: str) -> Path:
    return (
        root
        / "engine=futures"
        / "market=forts"
        / "board=RFUD"
        / f"symbol={symbol}"
        / "timeframe=5m"
    )
