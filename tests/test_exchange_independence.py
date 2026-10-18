"""Guard the provider boundary with a complete third-adapter financial run."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from neutral_adapter_fixture import NeutralAdapter, run_request

from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.adapters.registry import register_adapter_factory, unregister_adapter_factory
from backtest_engine.workbench.api import adapter_payload, dataset_build_payload
from backtest_engine.workbench.datasets import list_workbench_datasets
from backtest_engine.workbench.runner import run_workbench_backtest


@pytest.mark.parametrize(
    ("product", "expected"), [("cash", 1049.45), ("dated", 1049.5), ("swap", 1049.45)]
)
def test_third_adapter_build_to_report_and_catalog_recovery(tmp_path, product, expected):
    register_adapter_factory(NeutralAdapter)
    try:
        assert adapter_payload(NeutralAdapter.id, "/markets", {})["items"][0]["id"] == "test-market"
        payload = dataset_build_payload(
            tmp_path / "runs",
            {
                "adapter_id": NeutralAdapter.id,
                "engine": product,
                "security": "GOLDUSD",
                "from_date": "2024-01-02",
                "till_date": "2024-01-02",
                "timeframe": "5m",
            },
        )
    finally:
        unregister_adapter_factory(NeutralAdapter)
    from backtest_engine.adapters.models import DatasetBuildResult

    dataset = DatasetBuildResult.model_validate(payload)
    request = run_request(tmp_path, dataset)
    result = run_workbench_backtest(request)
    ledger = pl.read_parquet(result.output_dir / "portfolio_ledger.parquet")
    events = pl.read_parquet(result.output_dir / "events.parquet")
    fills = events.filter(pl.col("contract_delta") != 0)
    assert fills["fill_price"].to_list() == [100.0, 120.0]
    assert fills["contract_delta"].to_list() == [2.5, -2.5]
    assert ledger["equity_cash"][-1] == pytest.approx(expected, abs=1e-9)
    assert set(ledger["currency"]) == {"USD"}
    assert result.report["metrics"]["total_return"] == pytest.approx(expected / 1000 - 1)
    config = json.loads((result.output_dir / "config.json").read_text())
    assert config["portfolio"]["base_currency"] == "USD"
    assert config["execution"]["trading_days_per_year"] == 260
    assert config["session"]["main_session_start"] == "00:30:00"
    assert not (result.output_dir / "forts_accounting.parquet").exists()
    dataset.catalog_path.unlink()
    assert (
        list_workbench_datasets(tmp_path / "runs")["items"][0]["dataset_id"] == dataset.dataset_id
    )


def test_common_request_has_no_selected_provider():
    with pytest.raises(ValueError, match="adapter_id"):
        DatasetBuildRequest(security="A", from_date="2024-01-01", till_date="2024-01-01")


def test_missing_annualization_preserves_nav_without_annualized_metrics(tmp_path):
    adapter = NeutralAdapter()
    request = DatasetBuildRequest(
        adapter_id=adapter.id,
        security="GOLDUSD",
        from_date="2024-01-02",
        till_date="2024-01-02",
        output_root=tmp_path / "datasets",
    )
    dataset = adapter.build_dataset(request)
    calendar = json.loads(dataset.calendar_path.read_text())
    del calendar["trading_days_per_year"]
    dataset.calendar_path.write_text(json.dumps(calendar))
    from backtest_engine.workbench.datasets import register_workbench_dataset

    dataset = register_workbench_dataset(tmp_path / "runs", request, dataset)
    result = run_workbench_backtest(run_request(tmp_path, dataset))
    metrics = result.report["metrics"]
    assert metrics["final_equity_cash"] == pytest.approx(1049.45)
    assert metrics["annualization_status"] == "unavailable: trading_days_per_year missing"
    assert metrics["annualized_volatility"] is None


def test_core_has_no_provider_imports_or_literals():
    source = Path(__file__).parents[1] / "src/backtest_engine"
    domains = (
        "core",
        "data",
        "daily",
        "intraday",
        "execution",
        "accounting",
        "portfolio",
        "reporting",
        "workbench",
        "research",
        "strategies",
    )
    forbidden = (
        "backtest_engine.adapters.moex",
        "backtest_engine.adapters.binance",
        "moex_iss_adapter",
        "continuous_futures_builder",
    )
    violations = []
    for domain in domains:
        for path in (source / domain).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                imports = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else (
                        [name.name for name in node.names] if isinstance(node, ast.Import) else []
                    )
                )
                if any(name.startswith(forbidden) for name in imports):
                    violations.append(f"{path.name}:{node.lineno}: provider import")
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value
                    in {
                        "RUB",
                        "USDT",
                        "moex",
                        "MOEX",
                        "binance",
                        "RFUD",
                        "forts",
                        "Europe/Moscow",
                        "SECID",
                        "LASTTRADEDATE",
                        "lasttradedate",
                        "MATDATE",
                        "matdate",
                        "BOARDID",
                        "STEPPRICE",
                        "BUYDEPO",
                        "LOTVOLUME",
                        "LOT_SIZE",
                        "MIN_NOTIONAL",
                        "PRICE_FILTER",
                        "baseAsset",
                        "quoteAsset",
                        "fundingTime",
                    }
                ):
                    violations.append(f"{path.name}:{node.lineno}: {node.value}")
    assert not violations, "\n".join(violations)


def test_public_engine_and_cli_run_with_provider_imports_blocked(tmp_path):
    script = """import importlib.abc, sys
from pathlib import Path
class BlockProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("backtest_engine.adapters.moex", "backtest_engine.adapters.binance",
                                "moex_iss_adapter", "continuous_futures_builder")):
            raise ImportError("Provider leaked into neutral runtime: " + fullname)
sys.meta_path.insert(0, BlockProviders())
sys.path.insert(0, sys.argv[1])
from neutral_adapter_fixture import NeutralAdapter, run_request
from backtest_engine.adapters.registry import register_adapter_factory
from backtest_engine.workbench.api import dataset_build_payload
from backtest_engine.adapters.models import DatasetBuildResult
from backtest_engine.workbench.runner import run_workbench_backtest
from backtest_engine.cli import app
from backtest_engine.paper import PaperTradingEngine, ParquetPaperFeed
root = Path(sys.argv[2])
register_adapter_factory(NeutralAdapter)
payload = dataset_build_payload(root / "runs", {"adapter_id": NeutralAdapter.id,
    "security": "GOLDUSD", "from_date": "2024-01-02", "till_date": "2024-01-02", "timeframe": "5m"})
result = run_workbench_backtest(run_request(root, DatasetBuildResult.model_validate(payload)))
assert abs(result.report["metrics"]["final_equity_cash"] - 1049.45) < 1e-9
assert not any(name.startswith(("moex_iss_adapter", "backtest_engine.adapters.moex",
    "backtest_engine.adapters.binance", "continuous_futures_builder")) for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).parent), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
