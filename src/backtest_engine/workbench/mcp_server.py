"""Expose compact research tools over the existing local Workbench HTTP services."""

import argparse
import os
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from backtest_engine.workbench.agent_models import RunSpec
from backtest_engine.workbench.http_client import WorkbenchClient
from backtest_engine.workbench.settings import load_workbench_settings

INSTRUCTIONS = (
    "Research through Workbench. List approved datasets and inspect strategy contracts first. "
    "Prepare a frozen experiment, then submit it; poll get_job and read compact get_results. "
    "Capital and currency are explicit. Preserve strict gates and causal execution. "
    "portfolio ledger is the accounting source of truth. Keep failed/inactive trials. "
    "Read large artifacts by columns and pages. Shared SDK edits use the existing catalog editor."
)


def create_server(url: str) -> FastMCP:
    client = WorkbenchClient(url)
    server = FastMCP("ohlcv-replay", instructions=INSTRUCTIONS, log_level="WARNING")
    read = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    write = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

    @server.tool(annotations=read)
    def list_datasets(
        query: str = "", approved_only: bool = True, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """List dataset IDs, market, timeframe, dates and approval. Pages: 1..100 rows."""
        return client.call(
            "/api/agent/datasets",
            query={
                "query": query,
                "approved_only": approved_only,
                "offset": offset,
                "limit": limit,
            },
        )

    @server.tool(annotations=read)
    def list_strategies(query: str = "", offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """List saved presets and configured SDK strategies with IDs and defaults."""
        return client.call(
            "/api/agent/strategies", query={"query": query, "offset": offset, "limit": limit}
        )

    @server.tool(annotations=read)
    def get_strategy_contract(
        strategy_id: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Inspect parameter bounds, scheduling, history and timeframe for supplied params."""
        return client.call(
            "/api/agent/strategy-contract", payload={"strategy_id": strategy_id, "params": params}
        )

    @server.tool(annotations=write)
    def submit_backtest(run: RunSpec, run_id: str | None = None) -> dict[str, Any]:
        """Queue a strict Workbench backtest by dataset/strategy IDs. Returns job ID."""
        return client.call(
            "/api/agent/backtests", payload={"run": run.model_dump(mode="json"), "run_id": run_id}
        )

    @server.tool(annotations=write)
    def prepare_experiment(
        run: RunSpec, protocol: dict[str, Any], experiment_id: str | None = None
    ) -> dict[str, Any]:
        """Freeze a robustness plan without trading. Protocol requires name and split_date;
        optional cases/grids, baseline, family_groups, fresh_selected_group, seed, workers (1..4),
        max_runs (<=500), bootstrap settings and assumptions. Workbench allocates output paths.
        Cases: name/group/params/request_overrides. Grids: name/group/params lists.
        Call submit_robustness next.
        """
        return client.call(
            "/api/agent/experiments/prepare",
            payload={
                "run": run.model_dump(mode="json"),
                "protocol": protocol,
                "experiment_id": experiment_id,
            },
        )

    @server.tool(annotations=write)
    def submit_robustness(experiment_id: str, resume: bool = False) -> dict[str, Any]:
        """Queue a prepared immutable experiment. Resume reuses only verified completed cases."""
        return client.call(
            "/api/research/protocols/submit",
            payload={"experiment_id": experiment_id, "resume": resume},
        )

    @server.tool(annotations=read)
    def get_job(job_id: str) -> dict[str, Any]:
        """Read shared job status/progress. Poll this tool for long runs."""
        return client.call("/api/progress/" + job_id)

    @server.tool(
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)
    )
    def cancel_job(job_id: str) -> dict[str, Any]:
        """Cancel a queued/running Workbench job and terminate its owned worker process tree."""
        return client.call("/api/jobs/" + job_id + "/cancel", payload={})

    @server.tool(annotations=read)
    def get_results(
        result_id: str,
        kind: Literal["backtest", "robustness"] = "backtest",
        offset: int = 0,
        limit: int = 20,
        group: str | None = None,
        sort_by: str = "run_id",
        descending: bool = False,
    ) -> dict[str, Any]:
        """Read native run metrics or paged experiment summaries, failures and family diagnostics.
        Returns artifact references without loading full candles or recomputing accounting.
        """
        return client.call(
            "/api/agent/results",
            query={
                "result_id": result_id,
                "kind": kind,
                "offset": offset,
                "limit": limit,
                "group": group,
                "sort_by": sort_by,
                "descending": descending,
            },
        )

    @server.tool(annotations=read)
    def get_artifact(
        result_id: str,
        name: str,
        kind: Literal["backtest", "robustness"] = "backtest",
        offset: int = 0,
        limit: int = 20,
        columns: list[str] | None = None,
        case_name: str | None = None,
    ) -> dict[str, Any]:
        """Read allowlisted metrics/config/audit JSON or 1..100 rows of events, trades, equity,
        portfolio_ledger, summary, annual, families and statistical tables. Select <=20 columns.
        Specify case_name for a native run inside an experiment.
        Large JSON returns a file reference.
        """
        return client.call(
            "/api/agent/artifacts",
            query={
                "result_id": result_id,
                "name": name,
                "kind": kind,
                "offset": offset,
                "limit": limit,
                "columns": ",".join(columns) if columns is not None else None,
                "case_name": case_name,
            },
        )

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    settings = load_workbench_settings()
    default_url = os.environ.get("WORKBENCH_URL") or f"http://127.0.0.1:{settings.server.port}"
    parser.add_argument("--url", default=default_url, help="running local Workbench origin")
    args = parser.parse_args()
    create_server(args.url).run(transport="stdio")


if __name__ == "__main__":
    main()
