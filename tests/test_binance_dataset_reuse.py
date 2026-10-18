"""A refreshed dataset must preserve its prefix and request only the missing bars."""

import gzip
import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from binance_fixture import exchange_info

from backtest_engine.adapters.binance import normalize, stream
from backtest_engine.adapters.binance.adapter import BinanceAdapter
from backtest_engine.adapters.binance.reuse import finalized_prefix
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.core.exceptions import DatasetValidationError

START = datetime(2024, 1, 31, tzinfo=UTC)


class HistoryTransport:
    def __init__(self):
        self.calls = []

    def get(self, product, path, params=None):
        params = params or {}
        self.calls.append((path, params))
        if path.endswith("exchangeInfo"):
            return exchange_info(product)
        rows = []
        for hour in range(48):
            stamp = int((START + timedelta(hours=hour)).timestamp() * 1000)
            if not params["startTime"] <= stamp <= params["endTime"]:
                continue
            price = 40000 + hour * 10
            if path.endswith("fundingRate"):
                if hour % 8 == 0:
                    rows.append(
                        {
                            "symbol": "BTCUSDT",
                            "fundingTime": stamp,
                            "fundingRate": "0.0001",
                            "markPrice": str(price),
                        }
                    )
            else:
                rows.append(
                    [
                        stamp,
                        str(price),
                        str(price + 20),
                        str(price - 20),
                        str(price + 5),
                        "10",
                        stamp + 3_600_000 - 1,
                        "400000",
                        100,
                        "5",
                        "200000",
                        "0",
                    ]
                )
        return rows


def request(root, product, **changes):
    return DatasetBuildRequest(
        adapter_id="binance",
        engine=product,
        security="BTCUSDT",
        from_date="2024-01-31",
        till_date="2024-01-31",
        timeframe="60m",
        history_source="rest",
        output_root=root,
        **changes,
    )


@pytest.mark.parametrize("product", ["spot", "perpetual"])
def test_refresh_preserves_prefix_and_matches_full_build(tmp_path, product):
    transport = HistoryTransport()
    adapter = BinanceAdapter(transport)
    initial = request(tmp_path / "incremental", product)
    before = adapter.build_dataset(initial)
    prefix = pl.read_parquet(before.dataset_path)
    transport.calls.clear()
    after = adapter.build_dataset(initial.model_copy(update={"till_date": "2024-02-01"}))
    history_calls = [
        params for path, params in transport.calls if not path.endswith("exchangeInfo")
    ]
    assert history_calls
    assert all(
        params["startTime"] >= int((START + timedelta(days=1)).timestamp() * 1000)
        for params in history_calls
    )
    combined = pl.read_parquet(after.dataset_path)
    assert combined.height == 48
    assert combined.head(24).equals(prefix)
    assert combined["ts"].n_unique() == 48
    full = adapter.build_dataset(
        initial.model_copy(
            update={
                "output_root": tmp_path / "full",
                "till_date": "2024-02-01",
                "cache_policy": "network",
            }
        )
    )
    assert combined.equals(pl.read_parquet(full.dataset_path))
    for field in ("gap_count", "missing_bars", "missing_mark_bars", "unpriced_funding_bars"):
        assert after.quality["validation"].get(field, 0) == full.quality["validation"].get(field, 0)
    with gzip.open(after.raw_snapshot_path, "rt", encoding="utf-8") as source:
        raw = json.load(source)
    assert raw["archives"][0]["source"] == "finalized_dataset_prefix"
    assert raw["archives"][0]["dataset_sha256"] == before.quality["dataset_sha256"]
    assert before.dataset_path.is_file()
    assert after.approval_status == "warning"


def test_reuse_rejects_modified_prefix(tmp_path):
    adapter = BinanceAdapter(HistoryTransport())
    initial = request(tmp_path, "spot")
    before = adapter.build_dataset(initial)
    before.dataset_path.write_bytes(b"changed")
    with pytest.raises(DatasetValidationError, match="reusable dataset hash mismatch"):
        adapter.build_dataset(initial.model_copy(update={"till_date": "2024-02-01"}))


def test_network_policy_bypasses_prefix_and_changed_timeframe(tmp_path):
    transport = HistoryTransport()
    adapter = BinanceAdapter(transport)
    initial = request(tmp_path, "spot")
    adapter.build_dataset(initial)
    transport.calls.clear()
    adapter.build_dataset(initial.model_copy(update={"cache_policy": "network"}))
    assert any(
        params.get("startTime") == int(START.timestamp() * 1000) for _, params in transport.calls
    )
    assert (
        finalized_prefix(
            initial.model_copy(update={"timeframe": "5m"}),
            tmp_path / "binance-spot-btcusdt-60m",
            exchange_info("spot")["symbols"][0],
            int((START + timedelta(days=1)).timestamp() * 1000),
        )
        is None
    )


@pytest.mark.parametrize("product", ["spot", "perpetual"])
def test_same_day_refresh_fetches_only_new_finalized_bars(tmp_path, monkeypatch, product):
    cutoff = START + timedelta(hours=12)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cutoff.astimezone(tz) if tz else cutoff.replace(tzinfo=None)

    monkeypatch.setattr(stream, "datetime", Clock)
    monkeypatch.setattr(normalize, "datetime", Clock)
    transport = HistoryTransport()
    adapter = BinanceAdapter(transport)
    initial = request(tmp_path, product)
    before = adapter.build_dataset(initial)
    assert before.row_count == 12
    transport.calls.clear()
    cutoff = START + timedelta(hours=16)
    after = adapter.build_dataset(initial)
    assert after.row_count == 16
    assert pl.read_parquet(after.dataset_path).head(12).equals(pl.read_parquet(before.dataset_path))
    assert all(
        params["startTime"] >= int((START + timedelta(hours=12)).timestamp() * 1000)
        for path, params in transport.calls
        if not path.endswith("exchangeInfo")
    )


def test_offline_rebuild_uses_verified_prefix_without_http(tmp_path):
    transport = HistoryTransport()
    adapter = BinanceAdapter(transport)
    initial = request(tmp_path, "spot")
    before = adapter.build_dataset(initial)
    transport.calls.clear()
    after = adapter.build_dataset(initial.model_copy(update={"cache_policy": "only"}))
    assert not transport.calls
    assert pl.read_parquet(after.dataset_path).equals(pl.read_parquet(before.dataset_path))


def test_incomplete_marks_are_not_reused_as_a_verified_prefix(tmp_path):
    initial = request(tmp_path, "perpetual")
    result = BinanceAdapter(HistoryTransport()).build_dataset(initial)
    manifest = json.loads(result.manifest_path.read_text())
    manifest["validation"]["missing_mark_bars"] = 1
    result.manifest_path.write_text(json.dumps(manifest))
    assert (
        finalized_prefix(
            initial,
            result.output_root.parent,
            exchange_info("perpetual")["symbols"][0],
            int((START + timedelta(days=1)).timestamp() * 1000),
        )
        is None
    )
