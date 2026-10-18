"""Real recorded SOL data served through the actual HTTP and MCP transports."""

import asyncio
import os
import sys
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from robustness_fixture import protocol_fixture

from backtest_engine.research.robustness.provenance import read_json
from backtest_engine.workbench.jobs import MANAGERS
from backtest_engine.workbench.server import handler_class
from backtest_engine.workbench.strategies import save_strategy, strategies_root


@contextmanager
def workbench_fixture(tmp_path, product="spot"):
    protocol_fixture(tmp_path, product)
    native = read_json(tmp_path / "request.json")
    output_root = tmp_path / "runs"
    save_strategy(
        strategies_root(output_root),
        {
            "id": "recorded_sol",
            "name": "Recorded SOL",
            "source": native["script_source"],
            "language": "python",
            "params": native["params"],
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class(output_root))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    spec = {
        "dataset_id": native["dataset_id"],
        "strategy_id": "recorded_sol",
        "from_date": "2024-04-01",
        "initial_capital": 1000,
        "base_currency": "USDT",
        "execution_overrides": {
            "leverage": native["leverage"],
            "execution_timing": native["execution_timing"],
            "end_of_data_position_policy": "reject",
        },
    }
    try:
        yield f"http://127.0.0.1:{server.server_port}", output_root, spec
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        manager = MANAGERS.pop(str(output_root.resolve()), None)
        if manager is not None:
            manager.executor.shutdown(wait=True, cancel_futures=True)


async def call(session, name, args=None):
    result = await session.call_tool(name, args or {})
    assert not result.isError, result.content
    return result.structuredContent


async def await_job(session, job_id):
    deadline = asyncio.get_running_loop().time() + 60
    while asyncio.get_running_loop().time() < deadline:
        result = await call(session, "get_job", {"job_id": job_id})
        if result["status"] in {"completed", "failed", "cancelled"}:
            return result
        await asyncio.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within 60 seconds")


def mcp_process(url):
    return StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "backtest_engine.workbench.mcp_server",
            "--url",
            url,
        ],
        env={**os.environ, "WORKBENCH_SDK_CATALOG_PATH": ""},
    )


async def with_mcp(url, workflow):
    async with stdio_client(mcp_process(url)) as (read, write):
        async with ClientSession(read, write) as session:
            initialized = await session.initialize()
            assert initialized.serverInfo.name == "ohlcv-replay"
            assert "ledger" in initialized.instructions
            await workflow(session)
