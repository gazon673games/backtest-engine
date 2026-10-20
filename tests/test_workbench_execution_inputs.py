"""Catalog previews expose execution blockers separately from candle quality."""

import json

from backtest_engine.data.catalog import DataCatalogEntry
from backtest_engine.workbench.datasets import catalog_item


def test_legacy_warning_dataset_exposes_missing_margin_without_rewriting_inputs(tmp_path):
    audit = tmp_path / "instrument_registry_audit.json"
    original = json.dumps(
        {
            "historical_margin": {
                "missing_coverage": ["TEST: historical margin unavailable on 2024-01-02"]
            }
        }
    )
    audit.write_text(original, encoding="utf-8")
    entry = DataCatalogEntry(
        dataset_id="legacy-strict",
        dataset_path=tmp_path / "bars.parquet",
        quality_status="warning",
        instrument_registry_audit_path=audit,
    )
    item = catalog_item(entry, tmp_path / "catalog.json")
    assert item["quality_status"] == "warning"
    assert item["metadata"]["execution_input_issues"] == [
        "TEST: historical margin unavailable on 2024-01-02"
    ]
    assert entry.metadata == {}
    assert audit.read_text(encoding="utf-8") == original


def test_complete_research_rules_have_no_execution_blockers(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"historical_margin": {"missing_coverage": []}}), encoding="utf-8")
    entry = DataCatalogEntry(
        dataset_id="research",
        dataset_path=tmp_path / "bars.parquet",
        quality_status="warning",
        instrument_registry_audit_path=audit,
    )
    assert (
        catalog_item(entry, tmp_path / "catalog.json")["metadata"]["execution_input_issues"] == []
    )
