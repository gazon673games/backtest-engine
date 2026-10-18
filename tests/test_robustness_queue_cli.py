"""CLI requests use the same frozen plans, workers and result readers as MCP."""

import json
import time

from mcp_workbench_fixture import workbench_fixture
from typer.testing import CliRunner

from backtest_engine.cli.composition import app
from backtest_engine.workbench.http_client import WorkbenchClient


def test_cli_prepare_submit_status_execute_recorded_sol(tmp_path):
    with workbench_fixture(tmp_path) as (url, output_root, _spec):
        runner = CliRunner()
        prepared = runner.invoke(
            app,
            [
                "robustness-prepare",
                "--config",
                str(tmp_path / "protocol.json"),
                "--experiment-id",
                "cli_sol",
                "--url",
                url,
            ],
        )
        assert prepared.exit_code == 0, prepared.output
        assert json.loads(prepared.output)["case_count"] == 3
        submitted = runner.invoke(app, ["robustness-submit", "cli_sol", "--url", url])
        assert submitted.exit_code == 0, submitted.output
        client = WorkbenchClient(url)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            job = client.call("/api/progress/cli_sol")
            if job["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        assert job["status"] == "completed", job
        result = runner.invoke(app, ["robustness-status", "cli_sol", "--url", url])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["total"] == 3
        assert (output_root.parent / "research/protocols/cli_sol/manifest.json").is_file()
