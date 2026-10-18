"""Cold bridge startup and strategy response deadlines are independent."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.strategies.javascript import runtime


def delayed_bridge(tmp_path: Path, monkeypatch, delay: float) -> list[subprocess.Popen[str]]:
    bridge = tmp_path / "delayed_bridge.py"
    bridge.write_text(
        "import json, sys, time\n"
        f"time.sleep({delay})\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "for index, line in enumerate(sys.stdin):\n"
        "    if index: time.sleep(5)\n"
        "    request = json.loads(line)\n"
        "    print(json.dumps({'id': request['id'], 'ok': True, 'result': None}), flush=True)\n",
        encoding="utf-8",
    )
    original_popen = subprocess.Popen
    processes: list[subprocess.Popen[str]] = []

    def launch(args, **kwargs):
        process = original_popen([sys.executable, "-u", str(bridge)], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    return processes


def test_cold_start_does_not_use_strategy_response_deadline(tmp_path: Path, monkeypatch) -> None:
    processes = delayed_bridge(tmp_path, monkeypatch, delay=0.8)
    strategy = runtime.JsIntradayStrategy(
        tmp_path / "flat.js", node_binary=sys.executable, timeout_ms=1
    )
    try:
        assert strategy.metadata() is None
        # A later stalled response must still use the small strategy deadline.
        with pytest.raises(StrategyPluginError, match="timed out after 1 ms"):
            strategy.metadata()
        assert processes[0].poll() is not None
    finally:
        strategy.close()


def test_cold_start_has_a_bounded_deadline(tmp_path: Path, monkeypatch) -> None:
    processes = delayed_bridge(tmp_path, monkeypatch, delay=5)
    monkeypatch.setattr(runtime, "DEFAULT_STARTUP_TIMEOUT_MS", 1)
    strategy = runtime.JsIntradayStrategy(tmp_path / "flat.js", node_binary=sys.executable)
    try:
        with pytest.raises(StrategyPluginError, match="timed out after 1 ms"):
            strategy.metadata()
        assert processes[0].poll() is not None
    finally:
        strategy.close()
