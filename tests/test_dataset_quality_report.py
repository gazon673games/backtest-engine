import json

from backtest_engine.workbench.report import build_workbench_report


def test_report_preserves_unapproved_status_without_loading_candles(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    # Quality must be available even when the caller intentionally omits chart candles.
    market = dataset / "market_data.parquet"
    market.write_bytes(b"candles must not be read")
    quality = {"approval_status": "warning", "violations": ["historical rules missing"]}
    (dataset / "quality_approval.json").write_text(json.dumps(quality))
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"dataset": {"path": str(market)}}))
    report = build_workbench_report(run)
    assert report["dataset_quality"] == quality
    assert report["series"]["candles"] == []
