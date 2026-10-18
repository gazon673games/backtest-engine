from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from binance_fixture import FixtureTransport

from backtest_engine.adapters.binance.adapter import BinanceAdapter
from backtest_engine.adapters.conformance import assert_adapter_conforms
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.core.exceptions import DatasetValidationError


@pytest.mark.parametrize(
    ("product", "dataset_type", "funding_rate"),
    [("spot", "spot", 0.0), ("perpetual", "perpetual", 0.0001)],
)
def test_binance_builds_spot_then_perpetual_dataset(
    tmp_path: Path,
    product: str,
    dataset_type: str,
    funding_rate: float,
) -> None:
    adapter = BinanceAdapter(FixtureTransport())
    result = adapter.build_dataset(
        DatasetBuildRequest(
            adapter_id="binance",
            engine=product,
            market="crypto",
            board="SPOT" if product == "spot" else "USD_M_PERPETUAL",
            security="BTCUSDT",
            asset_class="crypto",
            from_date="2024-01-01",
            till_date="2024-01-01",
            timeframe="1m",
            output_root=tmp_path,
            cache_policy="network",
            source_delay_minutes=0,
        )
    )

    frame = pl.read_parquet(result.dataset_path)
    assert result.approval_status == "warning"
    assert frame.get_column("dataset_type").to_list() == [dataset_type]
    assert frame.get_column("funding_rate").to_list() == [pytest.approx(funding_rate)]
    if product == "perpetual":
        assert frame.get_column("mark_price").to_list() == [pytest.approx(40040.0)]
    assert result.instrument_registry_path is not None
    assert result.manifest_path.is_file()


def test_binance_catalog_and_contract_conform() -> None:
    adapter = BinanceAdapter(FixtureTransport())

    assert_adapter_conforms(adapter)
    assert adapter.info().capabilities.default_engine == "perpetual"
    assert adapter.engines()[1].name == "USD-M Futures (perpetual)"
    instruments = adapter.instruments(
        engine="spot",
        market="crypto",
        board="SPOT",
        query="btc",
    )

    assert instruments[0]["symbol"] == "BTCUSDT"


def test_binance_reports_paginated_download_progress(tmp_path: Path) -> None:
    class Observer:
        preview_timeframe = "5m"

        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def progress(self, **event: object) -> None:
            self.events.append(event)

        def preview(self, _rows: list[dict[str, object]], *, timeframe: str) -> None:
            assert timeframe == "5m"

    observer = Observer()
    BinanceAdapter(FixtureTransport()).build_dataset(
        DatasetBuildRequest(
            adapter_id="binance",
            engine="spot",
            market="crypto",
            board="SPOT",
            security="BTCUSDT",
            from_date="2024-01-01",
            till_date="2024-01-01",
            timeframe="5m",
            output_root=tmp_path,
        ),
        observer=observer,
    )

    assert any("candles: page 1" in str(event["message"]) for event in observer.events)


def test_binance_builds_historical_trade_dataset(tmp_path: Path) -> None:
    result = BinanceAdapter(FixtureTransport()).build_dataset(
        DatasetBuildRequest(
            adapter_id="binance",
            engine="spot",
            market="crypto",
            board="SPOT",
            security="BTCUSDT",
            data_kind="trades",
            from_date="2024-01-01",
            till_date="2024-01-01",
            output_root=tmp_path,
        )
    )

    frame = pl.read_parquet(result.dataset_path)
    assert frame.get_column("data_kind").to_list() == ["trades"]
    assert frame.get_column("trade_price").to_list() == [40050.0]
    assert frame.get_column("aggressor_side").to_list() == ["buy"]


def test_binance_imports_historical_quotes_with_provenance(tmp_path: Path) -> None:
    source = tmp_path / "quotes.csv"
    source.write_text(
        "ts,bid,ask,bid_size,ask_size\n2024-01-01T00:00:00+00:00,40000,40002,2,3\n",
        encoding="utf-8",
    )
    result = BinanceAdapter(FixtureTransport()).build_dataset(
        DatasetBuildRequest(
            adapter_id="binance",
            engine="spot",
            market="crypto",
            board="SPOT",
            security="BTCUSDT",
            data_kind="quotes",
            from_date="2024-01-01",
            till_date="2024-01-01",
            output_root=tmp_path / "output",
            source_path=source,
        )
    )

    frame = pl.read_parquet(result.dataset_path)
    with gzip.open(result.raw_snapshot_path, "rt", encoding="utf-8") as source_file:
        raw = json.load(source_file)
    assert frame.get_column("data_kind").to_list() == ["quotes"]
    assert frame.get_column("close").to_list() == [40001.0]
    assert raw["source_path"] == str(source)
    assert len(raw["source_sha256"]) == 64
    assert result.raw_snapshot_path.name == "adapter_raw.json.gz"


def test_binance_strict_rules_build_versioned_registry(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "venue": "binance",
                "product": "perpetual",
                "symbol": "BTCUSDT",
                "snapshots": [
                    {
                        "valid_from": "2024-01-01",
                        "valid_to": "2024-01-01",
                        "tick_size": 0.1,
                        "quantity_step": 0.001,
                        "minimum_quantity": 0.001,
                        "minimum_notional": 5.0,
                        "maker_fee_bps": 2.0,
                        "taker_fee_bps": 5.0,
                        "liquidation_fee_bps": 1.25,
                        "margin_tiers": [
                            {
                                "notional_floor": 0.0,
                                "notional_cap": None,
                                "maintenance_margin_rate": 0.004,
                                "max_leverage": 125.0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = BinanceAdapter(FixtureTransport()).build_dataset(
        DatasetBuildRequest(
            adapter_id="binance",
            engine="perpetual",
            market="crypto",
            board="USD_M_PERPETUAL",
            security="BTCUSDT",
            from_date="2024-01-01",
            till_date="2024-01-01",
            timeframe="1m",
            output_root=tmp_path / "output",
            rules_path=rules_path,
            rules_enforcement="strict",
        )
    )

    registry = json.loads(Path(result.instrument_registry_path).read_text(encoding="utf-8"))
    spec = registry["instruments"][0]
    assert result.approval_status == "warning"  # Rules cannot supply missing funding event prices.
    assert result.quality["validation"]["unpriced_funding_bars"] == 1
    assert spec["maker_fee_bps"] == 2.0
    assert spec["margin_tiers"][0]["max_leverage"] == 125.0
    assert spec["valid_from"] == "2024-01-01"


def test_binance_strict_rules_fail_closed_without_archive(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="point-in-time rule archive missing"):
        BinanceAdapter(FixtureTransport()).build_dataset(
            DatasetBuildRequest(
                adapter_id="binance",
                engine="spot",
                market="crypto",
                board="SPOT",
                security="BTCUSDT",
                from_date="2024-01-01",
                till_date="2024-01-01",
                output_root=tmp_path,
                rules_enforcement="strict",
            )
        )


def test_binance_imports_l2_archive(tmp_path: Path) -> None:
    source = tmp_path / "book.parquet"
    pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
            "sequence": [1, 1],
            "previous_sequence": [None, None],
            "update_type": ["snapshot", "snapshot"],
            "side": ["bid", "ask"],
            "price": [40000.0, 40002.0],
            "size": [2.0, 3.0],
        }
    ).write_parquet(source)
    result = BinanceAdapter(FixtureTransport()).build_dataset(
        DatasetBuildRequest(
            adapter_id="binance",
            engine="spot",
            market="crypto",
            board="SPOT",
            security="BTCUSDT",
            data_kind="book",
            from_date="2024-01-01",
            till_date="2024-01-01",
            output_root=tmp_path / "output",
            source_path=source,
        )
    )

    frame = pl.read_parquet(result.dataset_path)
    assert frame.get_column("data_kind").unique().to_list() == ["book"]
    assert frame.get_column("bid").unique().to_list() == [40000.0]
    assert frame.get_column("ask").unique().to_list() == [40002.0]
