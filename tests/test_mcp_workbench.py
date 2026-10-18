"""Golden NAV through MCP -> HTTP -> queue -> native worker, with immutable preparation."""

import asyncio
from pathlib import Path

import pytest
from mcp_workbench_fixture import await_job, call, with_mcp, workbench_fixture

from backtest_engine.research.robustness.provenance import read_json, write_json
from backtest_engine.workbench.datasets import catalog_path_for, get_workbench_dataset
from backtest_engine.workbench.http_client import WorkbenchClient


@pytest.mark.parametrize(
    "product,expected", [("spot", 997.3609075), ("perpetual", 997.2511410322657)]
)
def test_real_mcp_backtest_and_robustness_preserve_native_golden_and_pinned_strategy(
    tmp_path, product, expected
):
    with workbench_fixture(tmp_path, product) as (url, output_root, spec):

        async def workflow(session):
            catalog = await call(session, "list_datasets", {"limit": 1})
            assert catalog["items"][0]["dataset_id"] == spec["dataset_id"]
            assert catalog["items"][0]["quality_status"] == "approved"
            strategies = await call(session, "list_strategies", {"query": "Recorded SOL"})
            assert strategies["items"][0]["id"] == "recorded_sol"
            assert "source" not in strategies["items"][0]
            contract = await call(session, "get_strategy_contract", {"strategy_id": "recorded_sol"})
            assert contract["metadata"]["timeframe"] == "60m"
            submitted = await call(
                session, "submit_backtest", {"run": spec, "run_id": "native_sol"}
            )
            assert submitted["status"] == "accepted"
            assert (await await_job(session, "native_sol"))["status"] == "completed"
            result = await call(session, "get_results", {"result_id": "native_sol"})
            assert result["metrics"]["final_equity_cash"] == pytest.approx(expected, abs=1e-9)
            ledger = await call(
                session,
                "get_artifact",
                {
                    "result_id": "native_sol",
                    "name": "portfolio_ledger",
                    "offset": 20,
                    "limit": 3,
                    "columns": ["ts", "equity_cash", "net_cash_flow"],
                },
            )
            assert ledger["total"] == 23 and ledger["next_offset"] is None
            assert ledger["items"][-1]["equity_cash"] == pytest.approx(expected, abs=1e-9)
            protocol = {
                "name": "MCP real SOL",
                "split_date": "2024-04-02",
                "workers": 2,
                "bootstrap_iterations": 20,
                "selection_bias_iterations": 20,
                "cases": [
                    {"name": "baseline", "group": "neighbors"},
                    {"name": "inactive", "group": "neighbors", "params": {"targets_json": "{}"}},
                    {"name": "cost", "group": "cost", "request_overrides": {"slippage_bps": 10}},
                ],
            }
            prepared = await call(
                session,
                "prepare_experiment",
                {
                    "run": spec,
                    "protocol": protocol,
                    "experiment_id": "frozen_sol",
                },
            )
            assert prepared["status"] == "prepared" and prepared["case_count"] == 3
            assert not (output_root.parent / "research/protocols/frozen_sol/runs").exists()
            # A later live strategy change must not affect an already prepared plan.
            path = output_root.parent / "strategies/recorded_sol.json"
            record = read_json(path)
            record["source"] = "raise RuntimeError('live source changed after preparation')\n"
            write_json(path, record)
            await call(session, "submit_robustness", {"experiment_id": "frozen_sol"})
            finished = await await_job(session, "frozen_sol")
            assert finished["status"] == "completed" and finished["case_count"] == 3
            first = await call(
                session,
                "get_results",
                {
                    "result_id": "frozen_sol",
                    "kind": "robustness",
                    "limit": 1,
                },
            )
            assert first["total"] == 3 and first["next_offset"] == 1
            assert first["items"][0]["final_nav"] == pytest.approx(expected, abs=1e-9)
            family = first["families"][0]
            assert family["planned_count"] == 2 and family["inactive_count"] == 1
            await call(
                session, "submit_robustness", {"experiment_id": "frozen_sol", "resume": True}
            )
            assert (await await_job(session, "frozen_sol"))["status"] == "completed"

        asyncio.run(with_mcp(url, workflow))


def test_agent_rejects_quality_bypass_bad_fields_and_arbitrary_artifact_paths(tmp_path):
    with workbench_fixture(tmp_path) as (url, _output_root, spec):
        client = WorkbenchClient(url)
        for overrides in (
            {"catalog_enforcement": "off"},
            {"quality_approval_required": False},
            {"comission_bps": 1},
            {"allow_noncausal_same_close": True},
        ):
            with pytest.raises(ValueError, match="invalid execution|strict runtime"):
                client.call(
                    "/api/agent/backtests",
                    payload={"run": {**spec, "execution_overrides": overrides}},
                )
        with pytest.raises(ValueError, match="invalid result|unknown artifact"):
            client.call(
                "/api/agent/artifacts", query={"result_id": "../private", "name": "../secret"}
            )
        with pytest.raises(ValueError, match="limit"):
            client.call("/api/agent/datasets", query={"limit": 101})


def test_queued_case_failure_is_visible_in_shared_progress_and_family(tmp_path):
    with workbench_fixture(tmp_path) as (url, _output_root, spec):

        async def workflow(session):
            prepared = await call(
                session,
                "prepare_experiment",
                {
                    "run": spec,
                    "protocol": {
                        "name": "Failure audit",
                        "split_date": "2024-04-02",
                        "workers": 2,
                        "bootstrap_iterations": 0,
                        "selection_bias_iterations": 0,
                        "cases": [
                            {"name": "baseline", "group": "neighbors"},
                            {
                                "name": "future",
                                "group": "neighbors",
                                "request_overrides": {"from_date": "2030-01-01"},
                            },
                        ],
                    },
                },
            )
            experiment_id = prepared["experiment_id"]
            await call(session, "submit_robustness", {"experiment_id": experiment_id})
            job = await await_job(session, experiment_id)
            assert job["status"] == "failed" and job["failed_count"] == 1
            report = await call(
                session, "get_results", {"result_id": experiment_id, "kind": "robustness"}
            )
            assert report["total"] == 2 and report["families"][0]["failed_count"] == 1
            assert [r for r in report["items"] if r["status"] == "failed"][0]["error"]

        asyncio.run(with_mcp(url, workflow))


@pytest.mark.parametrize(
    "url", ["https://example.com", "http://example.com", "http://localhost/private"]
)
def test_mcp_client_requires_a_loopback_workbench_origin(url):
    with pytest.raises(ValueError, match="loopback"):
        WorkbenchClient(url)


def test_agent_quality_rejection_explains_actual_missing_market_inputs(tmp_path):
    with workbench_fixture(tmp_path, "perpetual") as (url, output_root, spec):
        dataset = get_workbench_dataset(output_root, spec["dataset_id"])
        approval_path = Path(dataset["quality_approval_path"])
        approval = read_json(approval_path)
        approval.update(
            approval_status="warning",
            violations=["historical fee schedule", "3 funding bars use opening mark estimates"],
        )
        write_json(approval_path, approval)
        path = catalog_path_for(output_root)
        catalog = read_json(path)
        for entry in catalog["entries"]:
            entry.update(quality_status="warning", approved_for_research=False)
        write_json(path, catalog)
        with pytest.raises(ValueError, match="historical fee schedule.*opening mark estimates"):
            WorkbenchClient(url).call("/api/agent/backtests", payload={"run": spec})
