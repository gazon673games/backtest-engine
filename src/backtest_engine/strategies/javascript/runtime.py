"""Runs JavaScript strategies in an isolated Node-compatible bridge process."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.javascript.context import (
    DEFAULT_HISTORY_BARS,
    JsContextEncoder,
    json_default,
)
from backtest_engine.strategies.javascript.context import (
    js_context_payload as js_context_payload,
)
from backtest_engine.strategies.runtime import (
    StrategyMetadata,
    StrategyParamSpec,
    strategy_fingerprint,
)

DEFAULT_NODE_BINARY = "node"
DEFAULT_TIMEOUT_MS = 1000
DEFAULT_STARTUP_TIMEOUT_MS = 10_000
MAX_STRATEGY_SOURCE_BYTES = 256 * 1024


class JsIntradayStrategy:
    def __init__(
        self,
        strategy_file: Path,
        *,
        node_binary: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        history_bars: int = DEFAULT_HISTORY_BARS,
    ) -> None:
        self.strategy_file = strategy_file
        self.node_binary = node_binary or os.environ.get(
            "BACKTEST_NODE_BINARY",
            DEFAULT_NODE_BINARY,
        )
        self.timeout_ms = timeout_ms
        self.history_bars = history_bars
        self._process: subprocess.Popen[str] | None = None
        self._closed = False
        self._reader = ThreadPoolExecutor(max_workers=1)
        self._request_id = 0
        self._process_generation = 0
        self._context_generation = -1
        self._started_generation = -1
        self._context_encoder = JsContextEncoder()

    def __call__(self, ctx: IntradayStrategyContext, params: dict[str, Any] | None = None) -> Any:
        return self.execute(ctx, params or {})

    def metadata(self) -> StrategyMetadata | None:
        payload = self._request({"type": "metadata"})
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise StrategyPluginError("JS STRATEGY_METADATA must be an object")
        params = payload.get("params", ())
        return StrategyMetadata(
            name=str(payload.get("name", self.strategy_file.stem)),
            version=str(payload.get("version", "1")),
            description=payload.get("description"),
            warmup_bars=int(payload.get("warmup_bars", 0)),
            params=tuple(js_param_spec(item) for item in params),
            timeframe=optional_text(payload.get("timeframe")),
            schedule=str(payload.get("schedule", "every-bar")),
            decision_times=tuple(str(value) for value in payload.get("decision_times", ())),
            decision_tolerance_minutes=optional_int(payload.get("decision_tolerance_minutes")),
            missing_decision_policy=str(payload.get("missing_decision_policy", "carry")),
            history_bars=int(payload.get("history_bars", DEFAULT_HISTORY_BARS)),
            data_columns=tuple(str(value) for value in payload.get("data_columns", ())),
            decision_time_param=optional_text(payload.get("decision_time_param")),
        )

    def execute(self, ctx: IntradayStrategyContext, params: dict[str, Any]) -> Any:
        metadata = getattr(self, "__strategy_metadata__", None)
        history_bars = metadata.history_bars if metadata is not None else self.history_bars
        data_columns = metadata.data_columns if metadata is not None else ()
        self._ensure_process()
        if self._started_generation != self._process_generation:
            self._request({"type": "start", "params": params})
            self._started_generation = self._process_generation
        if self._context_generation != self._process_generation:
            self._context_encoder.reset()
            self._context_generation = self._process_generation
        return self._request(
            {
                "type": "decision",
                "ctx": self._context_encoder.encode(
                    ctx,
                    history_bars=history_bars,
                    data_columns=data_columns,
                ),
                "params": params,
                "timeoutMs": self.timeout_ms,
            }
        )

    def notify_fill(self, event: dict[str, Any]) -> None:
        if self._process is not None:
            self._request({"type": "fill", "event": event})

    def finish(self, summary: dict[str, Any]) -> None:
        if self._process is not None:
            self._request({"type": "end", "summary": summary})

    def close(self) -> None:
        self._closed = True
        process = self._process
        self._process = None
        self._context_encoder.reset()
        if process is not None:
            self._terminate_process(process)
        self._reader.shutdown(wait=True, cancel_futures=True)

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()

    def _request(self, payload: dict[str, Any]) -> Any:
        process = self._ensure_process()
        self._request_id += 1
        request = {"id": self._request_id, **payload}
        if process.stdin is None or process.stdout is None:
            raise StrategyPluginError("JS strategy process pipes are unavailable")
        try:
            process.stdin.write(json.dumps(request, default=json_default) + "\n")
            process.stdin.flush()
        except OSError as exc:
            self.close()
            raise StrategyPluginError("JS strategy process is not writable") from exc
        line = self._read_response_line(process, self.timeout_ms)
        if not line:
            raise StrategyPluginError("JS strategy process exited without response")
        response = json.loads(line)
        if response.get("id") != self._request_id:
            raise StrategyPluginError("JS strategy process returned mismatched response id")
        if not response.get("ok"):
            raise StrategyPluginError(f"JS strategy failed: {response.get('error')}")
        return response.get("result")

    def _read_response_line(
        self,
        process: subprocess.Popen[str],
        timeout_ms: int,
    ) -> str:
        if process.stdout is None:
            raise StrategyPluginError("JS strategy stdout is unavailable")
        future = self._reader.submit(process.stdout.readline)
        try:
            return future.result(timeout=max(0.1, (timeout_ms + 500) / 1000.0))
        except TimeoutError as exc:
            self._process = None
            self._terminate_process(process)
            raise StrategyPluginError(f"JS strategy timed out after {timeout_ms} ms") from exc

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._closed:
            raise StrategyPluginError("JS strategy runtime is closed")
        if self._process is not None and self._process.poll() is None:
            return self._process
        configured_path = Path(self.node_binary)
        node_path = (
            str(configured_path) if configured_path.exists() else shutil.which(self.node_binary)
        )
        if node_path is None:
            raise StrategyPluginError(f"Node.js binary not found: {self.node_binary}")
        bridge_path = Path(__file__).with_name("bridge.mjs")
        process_env = os.environ.copy()
        if process_env.get("BACKTEST_NODE_RUN_AS_NODE") == "1":
            process_env["ELECTRON_RUN_AS_NODE"] = "1"
        self._process = subprocess.Popen(
            [node_path, str(bridge_path), str(self.strategy_file.resolve())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=process_env,
        )
        self._process_generation += 1
        # Cold Node startup has its own deadline; strategy calls keep their budget.
        line = self._read_response_line(self._process, DEFAULT_STARTUP_TIMEOUT_MS)
        if json.loads(line or "null") != {"ready": True}:
            process, self._process = self._process, None
            self._terminate_process(process)
            raise StrategyPluginError("JS strategy process did not become ready")
        return self._process

    def __del__(self) -> None:
        self.close()


def load_js_intraday_strategy_plugin(
    strategy_file: Path,
    *,
    require_metadata: bool = False,
) -> JsIntradayStrategy:
    validate_js_strategy_file(strategy_file)
    strategy = JsIntradayStrategy(strategy_file)
    metadata = strategy.metadata()
    if metadata is None and require_metadata:
        raise StrategyPluginError("Strict strategy runtime requires STRATEGY_METADATA")
    if metadata is not None:
        strategy.__strategy_metadata__ = metadata
        strategy.__strategy_param_specs__ = metadata.params
        strategy.__strategy_warmup_bars__ = metadata.warmup_bars
    strategy.__strategy_fingerprint__ = strategy_fingerprint(
        strategy_path=strategy_file,
        metadata=metadata,
        params=None,
    )
    return strategy


def validate_js_strategy_file(strategy_file: Path) -> None:
    if "://" in str(strategy_file) or strategy_file.suffix.lower() != ".js":
        raise StrategyPluginError(f"JS strategy must be a local .js file: {strategy_file}")
    if not strategy_file.exists():
        raise StrategyPluginError(f"JS strategy file does not exist: {strategy_file}")
    if not strategy_file.is_file():
        raise StrategyPluginError(f"JS strategy path is not a file: {strategy_file}")
    if strategy_file.stat().st_size > MAX_STRATEGY_SOURCE_BYTES:
        raise StrategyPluginError("JS strategy file is too large")


def js_param_spec(raw: Any) -> StrategyParamSpec:
    if isinstance(raw, StrategyParamSpec):
        return raw
    if not isinstance(raw, dict):
        raise StrategyPluginError("JS strategy param specs must be objects")
    return StrategyParamSpec(**raw)


def optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
