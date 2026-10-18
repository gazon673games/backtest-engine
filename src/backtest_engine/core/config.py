from __future__ import annotations

import tomllib
from pathlib import Path

from backtest_engine.core.models import (
    BacktestConfig,
    DatasetConfig,
    ExecutionConfig,
    OutputConfig,
    QualityGateConfig,
    StrategyConfig,
)


def load_config_file(config_path: Path) -> BacktestConfig:
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    return BacktestConfig(
        dataset=DatasetConfig(
            path=raw_config["dataset"]["path"],
            quality_manifest=raw_config["dataset"]["quality_manifest"],
        ),
        quality_gate=QualityGateConfig(**raw_config.get("quality_gate", {})),
        strategy=StrategyConfig(**raw_config["strategy"]),
        execution=ExecutionConfig(**raw_config.get("execution", {})),
        output=OutputConfig(
            root=raw_config.get("output", {}).get("root", "data/backtests"),
            run_id=raw_config["output"]["run_id"],
            write_csv=raw_config.get("output", {}).get("write_csv", False),
        ),
    )


def merge_cli_overrides(
    config: BacktestConfig,
    *,
    dataset_path: Path | None = None,
    quality_manifest: Path | None = None,
    strategy_name: str | None = None,
    fast_window: int | None = None,
    slow_window: int | None = None,
    commission_bps: float | None = None,
    slippage_bps: float | None = None,
    output_root: Path | None = None,
    run_id: str | None = None,
    write_csv: bool | None = None,
) -> BacktestConfig:
    raw = config.model_dump(mode="python")

    overrides = {
        "dataset": {"path": dataset_path, "quality_manifest": quality_manifest},
        "strategy": {"name": strategy_name, "fast_window": fast_window, "slow_window": slow_window},
        "execution": {"commission_bps": commission_bps, "slippage_bps": slippage_bps},
        "output": {"root": output_root, "run_id": run_id, "write_csv": write_csv},
    }
    for section, values in overrides.items():
        raw[section].update({key: value for key, value in values.items() if value is not None})
    return BacktestConfig.model_validate(raw)
