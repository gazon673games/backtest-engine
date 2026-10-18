from __future__ import annotations

import json
from pathlib import Path

import pytest

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.cache import cache_path_for, read_cached_payload
from backtest_engine.adapters.moex.iss.contract_catalog import contracts_for_root
from backtest_engine.adapters.moex.iss.contract_metadata import DESCRIPTION_PARAMS
from backtest_engine.adapters.moex.iss.contract_selection import chain_audit


@pytest.fixture
def request_and_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DatasetBuildRequest:
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.contract_catalog.snapshot_instrument_rows",
        lambda **_: [{"SECID": "CRH3", "LASTTRADEDATE": "2023-03-16"}],
    )
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.contract_catalog.snapshot_contract_rows",
        lambda *_: [
            {"SECID": "CRU2", "snapshot_till": "9999-12-31"},
            {"SECID": "CRZ2", "snapshot_till": "9999-12-31"},
        ],
    )
    return DatasetBuildRequest(
        adapter_id="moex",
        engine="futures",
        market="forts",
        board="RFUD",
        security="CR",
        instrument_mode="nearest_futures",
        roll_rule="expiry",
        from_date="2022-09-14",
        till_date="2022-12-20",
        cache_dir=tmp_path / "iss-cache",
    )


def description(symbol: str, expiry: str | None) -> dict:
    return {
        "description": {
            "columns": ["value", "name"],
            "data": [
                [symbol, "SECID"],
                [expiry, "LSTTRADE"],
                [expiry, "LSTDELDATE"],
                ["1000", "LOTSIZE"],
                ["CNY", "ASSETCODE"],
            ],
        }
    }


def test_historical_expiries_are_loaded_once_then_available_offline(
    request_and_registry: DatasetBuildRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fetch(url: str, params: dict) -> dict:
        calls.append(url)
        assert params == DESCRIPTION_PARAMS
        symbol = url.rsplit("/", 1)[-1].removesuffix(".json")
        return description(symbol, {"CRU2": "2022-09-15", "CRZ2": "2022-12-15"}[symbol])

    monkeypatch.setattr("backtest_engine.adapters.moex.iss.cache.fetch_network_json", fetch)
    contracts = contracts_for_root(request_and_registry)
    assert [row["SECID"] for row in contracts] == ["CRU2", "CRZ2", "CRH3"]
    assert len(calls) == 2
    assert contracts[0]["LASTTRADEDATE"] == "2022-09-15"
    assert contracts[0]["LOTSIZE"] == "1000"
    source = contracts[0]["_EXPIRY_METADATA"]
    path = cache_path_for(source["source_url"], DESCRIPTION_PARAMS, request_and_registry.cache_dir)
    assert read_cached_payload(path) == description("CRU2", "2022-09-15")
    assert len(source["source_sha256"]) == 64
    offline = request_and_registry.model_copy(update={"cache_policy": "offline"})
    assert contracts_for_root(offline) == contracts
    assert len(calls) == 2, "complete cached metadata must not cause network requests"
    audit = chain_audit(contracts, roll_rule="expiry")
    assert audit["contract_expiries"][0]["source_sha256"] == source["source_sha256"]
    assert audit["contract_expiries"][0]["selection_expiry"] == "2022-09-15"
    assert audit["contract_expiries"][0]["source_payload"] == description("CRU2", "2022-09-15")


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("missing_expiry", "expiry metadata is missing for CRU2, CRZ2"),
        ("bad_date", "CRU2: invalid LSTTRADE"),
        ("wrong_symbol", "metadata mismatch for CRU2"),
        ("missing_symbol", "metadata mismatch for CRU2"),
    ],
)
def test_untrusted_expiries_remain_blocked(
    request_and_registry: DatasetBuildRequest,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    expected: str,
) -> None:
    def fetch(url: str, _params: dict) -> dict:
        symbol = url.rsplit("/", 1)[-1].removesuffix(".json")
        if kind == "wrong_symbol":
            symbol = "OTHER"
        if kind == "missing_symbol":
            symbol = ""
        expiry = None if kind == "missing_expiry" else "bad-date"
        return description(symbol, expiry)

    monkeypatch.setattr("backtest_engine.adapters.moex.iss.cache.fetch_network_json", fetch)
    with pytest.raises(ValueError, match=expected):
        contracts_for_root(request_and_registry)


def test_cold_offline_cache_does_not_contact_moex(
    request_and_registry: DatasetBuildRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_: object) -> None:
        pytest.fail("offline mode must not fetch metadata")

    monkeypatch.setattr("backtest_engine.adapters.moex.iss.cache.fetch_network_json", forbidden)
    with pytest.raises(ValueError, match="CRU2.*offline mode.*load once"):
        contracts_for_root(request_and_registry.model_copy(update={"cache_policy": "offline"}))


def test_moex_download_failure_identifies_contract_and_source(
    request_and_registry: DatasetBuildRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_: object) -> None:
        raise OSError("ISS unavailable")

    monkeypatch.setattr("backtest_engine.adapters.moex.iss.cache.fetch_network_json", unavailable)
    with pytest.raises(ValueError, match=r"CRU2.*securities/CRU2.json.*ISS unavailable"):
        contracts_for_root(request_and_registry)


def test_recorded_security_description_uses_contractual_dates(
    request_and_registry: DatasetBuildRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backtest_engine.adapters.moex.iss.contract_metadata import load_contract_metadata

    payload = json.loads(
        (Path(__file__).parent / "fixtures/moex_crh3_description.json").read_text(
            encoding="utf-8",
        )
    )
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.cache.fetch_network_json",
        lambda *_: payload,
    )
    row = load_contract_metadata(
        {"SECID": "CRH3", "snapshot_till": "9999-12-31"}, request_and_registry
    )
    assert row["LASTTRADEDATE"] == "2023-03-16"
    assert row["MATDATE"] == "2023-03-16"
