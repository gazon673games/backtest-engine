"""Native spot/perpetual results, fail-closed audit, immutable resume and causal selection."""

from pathlib import Path

import polars as pl
import pytest
from robustness_fixture import protocol_fixture

from backtest_engine.research.robustness.metrics import audit_run
from backtest_engine.research.robustness.planning import load_protocol, prepare_plan
from backtest_engine.research.robustness.provenance import read_json, write_json
from backtest_engine.research.robustness.runner import run_protocol, selected_fresh_case
from backtest_engine.research.robustness.statistics import select_on_training


@pytest.mark.parametrize(
    "product,expected", [("spot", 997.3609075), ("perpetual", 997.2511410322657)]
)
def test_real_native_protocol_preserves_golden_nav_inactive_cases_and_resume(
    tmp_path, product, expected
):
    config = protocol_fixture(tmp_path, product)
    result = run_protocol(config)
    assert result["status"] == "completed"
    assert result["case_count"] == 3
    root = tmp_path / "out"
    rows = {r["run_id"]: r for r in read_json(root / "summary.json")}
    assert rows["baseline"]["final_nav"] == pytest.approx(expected, abs=1e-9)
    assert rows["baseline"]["base_currency"] == "USDT"
    assert rows["high_cost"]["final_nav"] < rows["baseline"]["final_nav"]
    assert rows["inactive"]["active"] is False
    assert rows["inactive"]["final_nav"] == 1000
    family = read_json(root / "families.json")[0]
    assert family["planned_count"] == family["completed_count"] == 2
    assert family["inactive_count"] == 1
    assert family["selected_on_train"] == "baseline"
    assert family["positive_all_count"] == 0
    assert read_json(root / "block_bootstrap.json")[0]["status"] == "skipped_insufficient_days"
    native_path = root / "runs/baseline/events.parquet"
    before = native_path.stat().st_mtime_ns
    assert run_protocol(config, resume=True)["fingerprint"] == result["fingerprint"]
    assert native_path.stat().st_mtime_ns == before
    assert (root / "runtime_sources.zip").is_file()


def test_resume_rejects_changed_configuration_inputs_and_native_results(tmp_path):
    config = protocol_fixture(tmp_path)
    run_protocol(config)
    payload = read_json(config)
    payload["seed"] = 42
    write_json(config, payload)
    with pytest.raises(ValueError, match="another plan or runtime"):
        run_protocol(config, resume=True)
    payload.pop("seed")
    write_json(config, payload)
    request = read_json(tmp_path / "request.json")
    registry = Path(request["instrument_registry_path"])
    original = registry.read_bytes()
    registry.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="another plan or runtime"):
        run_protocol(config, resume=True)
    registry.write_bytes(original)
    path = tmp_path / "out/runs/baseline/metrics.json"
    native = read_json(path)
    native["final_equity_cash"] += 100
    write_json(path, native)
    with pytest.raises(ValueError, match="native result changed"):
        run_protocol(config, resume=True)


@pytest.mark.parametrize("change", ["typo", "catalog_off", "unknown_param"])
def test_preflight_blocks_bad_requests_before_any_native_run(tmp_path, change):
    config = protocol_fixture(tmp_path)
    payload = read_json(config)
    overrides = payload["cases"][0].setdefault("request_overrides", {})
    if change == "typo":
        overrides["comission_bps"] = 2
    elif change == "catalog_off":
        overrides["catalog_enforcement"] = "off"
    else:
        payload["cases"][0]["params"] = {"undeclared_window": 50}
    write_json(config, payload)
    with pytest.raises(Exception, match="unknown Workbench|strict quality|Unknown.*param"):
        run_protocol(config)
    assert not (tmp_path / "out/runs").exists()


def test_grid_is_bounded_and_plan_freezes_each_request_before_execution(tmp_path):
    config = protocol_fixture(tmp_path)
    payload = read_json(config)
    payload["grids"] = [{"name": "windows", "params": {"timeframe": ["M60", "M60"]}}]
    payload["max_runs"] = 4
    write_json(config, payload)
    with pytest.raises(ValueError, match="exceeds max_runs"):
        load_protocol(config)
    payload["max_runs"] = 5
    write_json(config, payload)
    result = run_protocol(config, plan_only=True)
    assert result["status"] == "planned" and result["case_count"] == 5
    plan = read_json(tmp_path / "out/manifest.json")["plan"]
    assert all(
        item["request"]["script_source"].startswith("SDK_CATALOG_SNAPSHOT")
        for item in plan["cases"]
    )
    assert not (tmp_path / "out/runs").exists()


def test_training_selection_ignores_future_returns_and_starts_with_original_capital(tmp_path):
    config = protocol_fixture(tmp_path)
    protocol, cases = load_protocol(config)
    protocol.fresh_selected_group = "neighbors"
    plan = prepare_plan(protocol, cases, existing=False)
    rows = [
        {
            "run_id": "baseline",
            "group": "neighbors",
            "status": "completed",
            "train_total_return": 0.2,
            "check_total_return": -0.99,
            "train_fill_rows": 4,
        },
        {
            "run_id": "inactive",
            "group": "neighbors",
            "status": "completed",
            "train_total_return": 0.1,
            "check_total_return": 100,
            "train_fill_rows": 1,
        },
    ]
    assert select_on_training(rows, "neighbors")["run_id"] == "baseline"
    rows[0]["check_total_return"], rows[1]["check_total_return"] = 1000, -1000
    chosen = selected_fresh_case(protocol, plan["cases"], rows)
    assert chosen["request"]["from_date"] == "2024-04-02"
    assert chosen["request"]["initial_capital"] == 1000
    assert read_json(tmp_path / "out/training_selection.json")["selected_run"] == "baseline"


def test_existing_audit_uses_native_config_and_preserves_failed_trial(tmp_path):
    config = protocol_fixture(tmp_path)
    run_protocol(config)
    root = tmp_path / "out"
    bad_path = root / "runs/high_cost/portfolio_ledger.parquet"
    ledger = pl.read_parquet(bad_path).with_columns(pl.col("equity_cash") + 1)
    ledger.write_parquet(bad_path)
    with pytest.raises(ValueError, match="accounting reconciliation failed"):
        audit_run(root / "runs/high_cost")
    payload = read_json(config)
    payload["output_root"] = "imported"
    payload.pop("base_request")
    payload["cases"] = [
        {
            "name": "baseline",
            "group": "neighbors",
            "existing_run": "out/runs/baseline",
            "params": {"ignored_research_label": "native config is authoritative"},
        },
        {"name": "broken", "group": "neighbors", "existing_run": "out/runs/high_cost"},
    ]
    write_json(config, payload)
    result = run_protocol(config, existing=True)
    assert result["status"] == "failed" and result["case_count"] == 2
    assert result["failed_count"] == 1
    family = read_json(tmp_path / "imported/families.json")[0]
    assert family["planned_count"] == 2 and family["failed_count"] == 1
    assert (
        read_json(tmp_path / "imported/selection_bias.json")[0]["status"]
        == "skipped_incomplete_family"
    )
    assert not (tmp_path / "imported/runs").exists()
    assert Path(result["report_path"]).is_file()
