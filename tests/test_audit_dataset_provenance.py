"""Inputs and published results remain independently inspectable after refresh/rerun."""

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from test_engine_support import write_dataset, write_manifest
from test_workbench_moex import moex_raw_rows
from typer.testing import CliRunner

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.moex.iss.candles import build_moex_dataset
from backtest_engine.cli import app
from backtest_engine.core.exceptions import BacktestError
from backtest_engine.daily.batch import BatchConfig, run_batch
from backtest_engine.data.store import file_sha256
from backtest_engine.portfolio.combine import (
    PortfolioCombineConfig,
    combine_intraday_event_portfolio,
)


def test_moex_refresh_preserves_old_dataset_and_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.candles.download_moex_candles",
        lambda *args, **kwargs: moex_raw_rows(),
    )
    request = DatasetBuildRequest(
        adapter_id="moex",
        engine="futures",
        market="forts",
        board="RFUD",
        security="CRU4",
        from_date="2024-01-02",
        till_date="2024-01-02",
        timeframe="5m",
        output_root=tmp_path / "datasets",
    )
    first = build_moex_dataset(request)
    digest = file_sha256(first.dataset_path)
    changed = [
        {**row, "open": 20.0, "high": 21.0, "low": 19.0, "close": 20.0} for row in moex_raw_rows()
    ]
    monkeypatch.setattr(
        "backtest_engine.adapters.moex.iss.candles.download_moex_candles",
        lambda *args, **kwargs: changed,
    )
    second = build_moex_dataset(request)
    assert first.dataset_id != second.dataset_id
    assert first.dataset_path != second.dataset_path
    assert file_sha256(first.dataset_path) == digest
    assert first.raw_snapshot_path.is_file() and second.raw_snapshot_path.is_file()
    assert first.approval_status == second.approval_status == "warning"


def test_daily_batch_distinguishes_content_and_repeats_idempotently(tmp_path):
    datasets = []
    entries = []
    for index, closes in enumerate(([100.0, 110.0, 120.0], [100.0, 90.0, 80.0])):
        directory = tmp_path / str(index)
        directory.mkdir()
        path = write_dataset(directory, closes=list(closes))
        manifest = write_manifest(directory, dataset_path=path, status="approved")
        entries.extend(json.loads(manifest.read_text())["datasets"])
        datasets.append(path)
    quality = tmp_path / "quality.json"
    quality.write_text(json.dumps({"datasets": entries}), encoding="utf-8")
    strategy = tmp_path / "long.py"
    strategy.write_text(
        'def generate_signals(ctx, params):\n    return [1.0] * len(ctx.series("close"))\n',
        encoding="utf-8",
    )
    config = BatchConfig(
        experiment_id="identity",
        output_root=tmp_path / "outputs",
        dataset_paths=datasets,
        quality_manifest=quality,
        strategy_file=strategy,
        grid={"n": [1]},
    )
    first = run_batch(config)
    assert first.successful_run_count == 2, first.failed_rows
    paths = [row["run_path"] for row in first.summary_rows]
    assert len(set(paths)) == 2
    assert [row["total_return"] for row in first.summary_rows] == pytest.approx([0.2, -0.2])
    assert [
        json.loads((Path(path) / "metrics.json").read_text())["total_return"] for path in paths
    ] == pytest.approx([0.2, -0.2])
    second = run_batch(config)
    assert second.successful_run_count == 2, second.failed_rows
    assert [row["run_path"] for row in second.summary_rows] == paths


def test_combine_rejects_mixed_currencies_before_adding_cash(tmp_path):
    runs = []
    for currency in ("RUB", "USDT"):
        run = tmp_path / currency
        run.mkdir()
        (run / "config.json").write_text(
            json.dumps({"portfolio": {"initial_capital": 1000, "base_currency": currency}}),
            encoding="utf-8",
        )
        pl.DataFrame({"ts": [datetime(2024, 1, 1)], "net_pnl": [10.0]}).write_parquet(
            run / "events.parquet"
        )
        runs.append(run)
    with pytest.raises(BacktestError, match="base currency"):
        combine_intraday_event_portfolio(
            PortfolioCombineConfig(run_paths=runs, output_path=tmp_path / "combined")
        )


def test_public_js_cli_executes_a_real_offline_strategy(tmp_path):
    from test_intraday_event_workflow_support import write_intraday_dataset

    dataset = write_intraday_dataset(tmp_path)
    script = tmp_path / "flat.js"
    script.write_text("function onDecision(ctx) { return 0; }", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "run-js-strategy",
            "--dataset-path",
            str(dataset),
            "--script",
            str(script),
            "--run-id",
            "js-audit",
            "--output-root",
            str(tmp_path / "runs"),
            "--no-strict-runtime",
            "--no-require-quality-approval",
            "--allow-legacy-bar-timestamps",
            "--schedule",
            "every-bar",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "runs/js-audit/metrics.json").is_file()
