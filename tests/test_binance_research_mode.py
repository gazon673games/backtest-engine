"""Research defaults permit strategy runs and preserve the unapproved status."""

import json

import polars as pl
import pytest
from test_binance_workbench import SeriesTransport, build_request, run_request

from backtest_engine.adapters.binance.adapter import BinanceAdapter
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.workbench.datasets import register_workbench_dataset
from backtest_engine.workbench.report import build_workbench_report
from backtest_engine.workbench.runner import run_workbench_backtest


@pytest.mark.parametrize("product", ["spot", "perpetual"])
def test_research_mode_runs_strategy_without_a_historical_rule_archive(tmp_path, product):
    adapter = BinanceAdapter(SeriesTransport())
    defaults = adapter.info().capabilities
    request = build_request(
        tmp_path, product, rules_path=None, rules_enforcement=defaults.default_rules_enforcement
    )
    result = adapter.build_dataset(request)
    result = register_workbench_dataset(tmp_path / "runs", request, result)
    assert result.approval_status == "warning"
    assert result.quality["rule_audit"]["enforcement"] == "audit"
    response = run_workbench_backtest(
        run_request(
            result,
            tmp_path,
            calendar_path=result.calendar_path,
            catalog_path=result.catalog_path,
            dataset_id=result.dataset_id,
            catalog_enforcement="audit",
            quality_approval_required=defaults.default_quality_approval_required,
            quality_approval_path=None,
            commission_bps=5.0,
        )
    )
    assert response.status == "completed"
    report = build_workbench_report(response.output_dir)
    assert report["dataset_quality"]["approval_status"] == "warning"
    assert "point-in-time rule archive missing" in report["dataset_quality"]["violations"]
    config = json.loads((response.output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["dataset"]["quality_approval_required"] is False
    assert config["dataset"]["catalog_enforcement"] == "audit"
    assert config["execution"]["commission_bps"] == 5.0
    assert config["portfolio"]["base_currency"] == "USDT"
    events = pl.read_parquet(response.output_dir / "events.parquet")
    assert events["trading_fee_cash"].sum() > 0
    assert pl.read_parquet(response.output_dir / "portfolio_ledger.parquet").height > 0
    with pytest.raises(DatasetValidationError, match="not approved"):
        run_workbench_backtest(run_request(result, tmp_path, run_id="strict_research_rejected"))
