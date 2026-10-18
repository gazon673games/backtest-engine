"""Saved research observations are pinned and only their missing suffix is fetched."""

import gzip
import json
from datetime import timedelta

import polars as pl
import pytest
from test_binance_dataset_reuse import START, HistoryTransport, request

from backtest_engine.adapters.binance.adapter import BinanceAdapter
from backtest_engine.adapters.binance.quality import quality_payload
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.data.store import file_sha256
from backtest_engine.workbench import dataset_jobs
from backtest_engine.workbench.dataset_reuse import bind_reusable_dataset
from backtest_engine.workbench.datasets import catalog_path_for, register_workbench_dataset


def saved_prefix(tmp_path, product):
    transport = HistoryTransport()
    adapter = BinanceAdapter(transport)
    initial = request(tmp_path / "original", product)
    result = adapter.build_dataset(initial)
    frame = pl.read_parquet(result.dataset_path)
    external = tmp_path / "research" / "market_data.parquet"
    external.parent.mkdir()
    frame.select(list(reversed(frame.columns))).write_parquet(external)
    result = result.model_copy(update={"dataset_path": external, "dataset_id": "saved-research"})
    output_root = tmp_path / "workbench" / "runs"
    register_workbench_dataset(output_root, initial, result)
    catalog_path = catalog_path_for(output_root)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["entries"][0]["metadata"]["manifest_path"] = None
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    extended = initial.model_copy(
        update={
            "output_root": tmp_path / "workbench" / "datasets",
            "till_date": "2024-02-01",
        }
    )
    extended = bind_reusable_dataset(output_root, extended, adapter)
    transport.calls.clear()
    return output_root, adapter, extended, frame


@pytest.mark.parametrize("product", ["spot", "perpetual"])
def test_catalog_reuse_pins_research_file_and_fetches_only_suffix(tmp_path, product):
    _, adapter, extended, original = saved_prefix(tmp_path, product)
    prefix = extended.reuse_dataset
    assert prefix is not None and prefix.dataset_sha256 == file_sha256(prefix.dataset_path)
    assert prefix.dataset_id == "saved-research" and prefix.quality_status == "warning"
    result = adapter.build_dataset(extended)
    combined = pl.read_parquet(result.dataset_path).select(original.columns)
    assert combined.height == 48
    assert combined.head(24).equals(original)
    assert all(
        params["startTime"] >= int((START + timedelta(days=1)).timestamp() * 1000)
        for path, params in adapter.transport.calls
        if not path.endswith("exchangeInfo")
    )
    with gzip.open(result.raw_snapshot_path, "rt", encoding="utf-8") as source:
        raw = json.load(source)
    assert raw["archives"][0]["source"] == "catalog_dataset_prefix"
    assert raw["archives"][0]["dataset_sha256"] == prefix.dataset_sha256
    assert result.approval_status == "warning"
    rules_approved = {"status": "approved", "violations": []}
    quality = quality_payload(None, extended, result.quality["validation"], rules_approved)
    assert quality["approval_status"] == "warning", "appending cannot approve a diagnostic prefix"
    assert quality["reused_dataset"]["dataset_id"] == prefix.dataset_id


def test_queued_build_freezes_catalog_prefix_before_worker_runs(tmp_path, monkeypatch):
    output_root, adapter, extended, _ = saved_prefix(tmp_path, "perpetual")
    captured = {}
    monkeypatch.setattr(dataset_jobs, "get_adapter", lambda _: adapter)

    def submit(root, job_id, job_type, payload):
        captured.update(payload)
        return {"status": "queued"}

    monkeypatch.setattr(dataset_jobs, "submit_job", submit)
    payload = extended.model_copy(update={"reuse_dataset": None}).model_dump(mode="json")
    dataset_jobs.submit_dataset_build(output_root, payload)
    assert (
        captured["request"]["reuse_dataset"]["dataset_sha256"]
        == extended.reuse_dataset.dataset_sha256
    )
    extended.reuse_dataset.dataset_path.write_bytes(b"tampered after queuing")
    with pytest.raises(DatasetValidationError, match="reusable dataset hash mismatch"):
        adapter.build_dataset(extended)


@pytest.mark.parametrize("problem", ["mark", "gap", "currency", "partial", "identity"])
def test_catalog_prefix_validation_fails_closed(tmp_path, problem):
    _, adapter, extended, _ = saved_prefix(tmp_path, "perpetual")
    prefix = extended.reuse_dataset
    frame = pl.read_parquet(prefix.dataset_path)
    if problem == "mark":
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("mark_price"))
    elif problem == "gap":
        frame = frame.slice(0, 10).vstack(frame.slice(11))
    elif problem == "currency":
        frame = frame.with_columns(pl.lit("RUB").alias("settlement_currency"))
    elif problem == "partial":
        frame = frame.with_columns(pl.col("available_at") - pl.duration(minutes=10))
    else:
        prefix = prefix.model_copy(update={"symbol": "ETHUSDT"})
    frame.write_parquet(prefix.dataset_path)
    prefix = prefix.model_copy(update={"dataset_sha256": file_sha256(prefix.dataset_path)})
    with pytest.raises(
        DatasetValidationError, match="(incomplete finalized bars|identity mismatch)"
    ):
        adapter.build_dataset(extended.model_copy(update={"reuse_dataset": prefix}))
    assert all(path.endswith("exchangeInfo") for path, _ in adapter.transport.calls)


def test_catalog_binding_respects_network_and_adapter_capability(tmp_path):
    output_root, adapter, extended, _ = saved_prefix(tmp_path, "spot")
    fresh = extended.model_copy(update={"reuse_dataset": None, "cache_policy": "network"})
    assert bind_reusable_dataset(output_root, fresh, adapter).reuse_dataset is None
    fresh = extended.model_copy(update={"reuse_dataset": None})
    info = adapter.info()
    info.capabilities.supports_dataset_prefix_reuse = False
    adapter.info = lambda: info
    assert bind_reusable_dataset(output_root, fresh, adapter).reuse_dataset is None


@pytest.mark.parametrize("changed", ["manifest", "raw", "quality_approval"])
def test_native_catalog_reuse_verifies_original_manifest_and_audits(tmp_path, changed):
    adapter = BinanceAdapter(HistoryTransport())
    initial = request(tmp_path / "workbench/datasets", "spot")
    result = adapter.build_dataset(initial)
    output_root = tmp_path / "workbench/runs"
    register_workbench_dataset(output_root, initial, result)
    extended = bind_reusable_dataset(
        output_root, initial.model_copy(update={"till_date": "2024-02-01"}), adapter
    )
    assert extended.reuse_dataset.provenance["manifest_sha256"] == file_sha256(result.manifest_path)
    paths = {
        "manifest": result.manifest_path,
        "raw": result.raw_snapshot_path,
        "quality_approval": result.approval_path,
    }
    paths[changed].write_bytes(b"changed after queue submission")
    with pytest.raises(DatasetValidationError, match="hash mismatch"):
        adapter.build_dataset(extended)
