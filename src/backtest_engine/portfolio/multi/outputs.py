from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.accounting.reconciliation import assert_reconciled, reconcile_ledger_path
from backtest_engine.core.provenance import runtime_provenance
from backtest_engine.core.run_output import publish_run_output
from backtest_engine.data.store import build_dataset_version
from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig
from backtest_engine.portfolio.multi.utils import file_sha256, write_json
from backtest_engine.research.experiment_identity import artifact_identity


def save_multi_instrument_outputs(
    config: MultiInstrumentBacktestConfig,
    loaded_legs: dict[str, dict[str, Any]],
    portfolio_legs: pl.DataFrame,
    portfolio_equity: pl.DataFrame,
    portfolio_ledger: pl.DataFrame,
    metrics: dict[str, Any],
    order_history: list[dict[str, Any]] | None = None,
    sdk_runtime: Any = None,
) -> Path:
    def write(output_dir: Path, published_dir: Path) -> None:
        del published_dir
        _write_multi_outputs(
            output_dir,
            config,
            loaded_legs,
            portfolio_legs,
            portfolio_equity,
            portfolio_ledger,
            metrics,
            order_history,
            sdk_runtime,
        )

    return publish_run_output(config.output.root, config.output.run_id, write)


def _write_multi_outputs(
    output_dir: Path,
    config: MultiInstrumentBacktestConfig,
    loaded_legs: dict[str, dict[str, Any]],
    portfolio_legs: pl.DataFrame,
    portfolio_equity: pl.DataFrame,
    portfolio_ledger: pl.DataFrame,
    metrics: dict[str, Any],
    order_history: list[dict[str, Any]] | None = None,
    sdk_runtime: Any = None,
) -> None:
    write_json(output_dir / "config.json", config.model_dump(mode="json"))
    write_json(
        output_dir / "dataset_validations.json",
        {asset_id: payload["validation"] for asset_id, payload in loaded_legs.items()},
    )
    write_json(
        output_dir / "catalog_bindings.json",
        {asset_id: payload["catalog_binding"] for asset_id, payload in loaded_legs.items()},
    )
    write_json(
        output_dir / "quality_approvals.json",
        {asset_id: payload["quality_approval"] for asset_id, payload in loaded_legs.items()},
    )
    portfolio_legs.write_parquet(output_dir / "portfolio_legs.parquet")
    portfolio_legs.write_csv(output_dir / "portfolio_legs.csv")
    portfolio_equity.write_parquet(output_dir / "portfolio_equity.parquet")
    portfolio_equity.write_csv(output_dir / "portfolio_equity.csv")
    portfolio_ledger.write_parquet(output_dir / "portfolio_ledger.parquet")
    portfolio_ledger.write_csv(output_dir / "portfolio_ledger.csv")
    reconciliation = {"status": "not_requested"}
    if config.portfolio.reconciliation_statement_path is not None:
        reconciliation = reconcile_ledger_path(
            portfolio_ledger,
            config.portfolio.reconciliation_statement_path,
            tolerance=config.portfolio.reconciliation_tolerance,
            fail_on_break=False,
        )
    write_json(output_dir / "accounting_reconciliation.json", reconciliation)
    if reconciliation["status"] != "not_requested":
        assert_reconciled(reconciliation)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "execution_orders.json", order_history or [])
    if sdk_runtime is not None:
        write_json(output_dir / "sdk_sources.json", sdk_runtime.snapshots)
        write_json(output_dir / "capital_transfers.json", sdk_runtime.capital.transfers)
        write_json(
            output_dir / "sleeve_capital.json",
            {
                "reserve": sdk_runtime.capital.reserve,
                "equity": {asset: b.equity for asset, b in sdk_runtime.capital.budgets.items()},
            },
        )
    write_json(
        output_dir / "experiment_manifest.json",
        experiment_manifest(config, loaded_legs, metrics, output_dir),
    )


def experiment_manifest(
    config: MultiInstrumentBacktestConfig,
    loaded_legs: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "runtime": runtime_provenance(),
        "engine": "multi_instrument_event_v1",
        "strategy": {
            "path": str(config.strategy.strategy_file),
            "sha256": file_sha256(config.strategy.strategy_file)
            if config.strategy.strategy_file
            else None,
            "params": config.strategy.strategy_params,
            "sdk_sleeves": [s.model_dump(mode="json") for s in config.strategy.sdk_sleeves],
            "capital_policy": config.strategy.capital_policy,
        },
        "datasets": {
            asset_id: dataset_manifest(payload) for asset_id, payload in loaded_legs.items()
        },
        "portfolio": config.portfolio.model_dump(mode="json"),
        "execution": config.execution.model_dump(mode="json"),
        "input_artifacts": multi_input_artifacts(config),
        "metrics": metrics,
        "outputs": output_hashes(output_dir),
    }


def dataset_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    dataset_path = Path(str(config.dataset_path))
    dataset_version = build_dataset_version(
        dataset_id=config.dataset_id or str(payload["validation"].get("continuous_symbol")),
        dataset_path=dataset_path,
    )
    return {
        "path": str(dataset_path),
        "sha256": file_sha256(dataset_path),
        "version_id": dataset_version.version_id,
        "schema_hash": dataset_version.schema_hash,
        "schema_version": dataset_version.schema_version,
        "row_count": dataset_version.row_count,
        "validation": payload["validation"],
        "catalog": payload["catalog_binding"],
        "quality_approval": {
            "status": payload["quality_approval"].get("status"),
            "path": payload["quality_approval"].get("approval_path"),
            "sha256": payload["quality_approval"].get("approval_sha256"),
            "required": payload["quality_approval"].get("required"),
        },
        "instrument_registry_path": (
            str(config.instrument_registry_path)
            if config.instrument_registry_path is not None
            else None
        ),
        "calendar_path": str(config.calendar_path) if config.calendar_path is not None else None,
    }


def multi_input_artifacts(config: MultiInstrumentBacktestConfig) -> dict[str, Any]:
    result = {
        "valuation_rates": artifact_identity(config.portfolio.valuation_rates_path),
        "execution_calibration": artifact_identity(config.execution.calibration_path),
        "reconciliation_statement": artifact_identity(
            config.portfolio.reconciliation_statement_path
        ),
        "point_in_time_universe": artifact_identity(config.universe.path),
    }
    result["legs"] = {
        leg.asset_id: {
            "catalog": artifact_identity(leg.catalog_path),
            "quality_approval": artifact_identity(leg.quality_approval_path),
            "instrument_registry": artifact_identity(leg.instrument_registry_path),
            "calendar": artifact_identity(leg.calendar_path),
        }
        for leg in config.legs
    }
    return result


def output_hashes(output_dir: Path) -> dict[str, str]:
    names = [
        "portfolio_legs.parquet",
        "portfolio_equity.parquet",
        "portfolio_ledger.parquet",
        "metrics.json",
        "accounting_reconciliation.json",
        "execution_orders.json",
    ]
    names += [
        name
        for name in ("sdk_sources.json", "capital_transfers.json", "sleeve_capital.json")
        if (output_dir / name).exists()
    ]
    return {name: file_sha256(output_dir / name) for name in names}
