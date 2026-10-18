from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from backtest_engine.accounting.reconciliation import assert_reconciled, reconcile_ledger_path
from backtest_engine.core.provenance import runtime_provenance
from backtest_engine.core.run_output import publish_run_output
from backtest_engine.data.store import build_dataset_version
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig
from backtest_engine.intraday.event.utils import file_sha256, write_json
from backtest_engine.research.experiment_identity import canonical_fingerprint, event_run_identity
from backtest_engine.strategies.plugins.loader import strategy_file_sha256


def save_intraday_event_outputs(
    config: IntradayEventBacktestConfig,
    dataset_validation: dict[str, Any],
    quality_approval: dict[str, Any],
    catalog_approval: dict[str, Any],
    events: pl.DataFrame,
    decision_actions: pl.DataFrame,
    equity: pl.DataFrame,
    trades: pl.DataFrame,
    portfolio_ledger: pl.DataFrame,
    metrics: dict[str, Any],
) -> Path:
    def write(output_dir: Path, published_dir: Path) -> None:
        _write_intraday_event_outputs(
            output_dir,
            published_dir,
            config,
            dataset_validation,
            quality_approval,
            catalog_approval,
            events,
            decision_actions,
            equity,
            trades,
            portfolio_ledger,
            metrics,
        )

    return publish_run_output(config.output.root, config.output.run_id, write)


def _write_intraday_event_outputs(
    output_dir: Path,
    published_dir: Path,
    config: IntradayEventBacktestConfig,
    dataset_validation: dict[str, Any],
    quality_approval: dict[str, Any],
    catalog_approval: dict[str, Any],
    events: pl.DataFrame,
    decision_actions: pl.DataFrame,
    equity: pl.DataFrame,
    trades: pl.DataFrame,
    portfolio_ledger: pl.DataFrame,
    metrics: dict[str, Any],
) -> None:
    write_json(output_dir / "config.json", config_output_payload(config, dataset_validation))
    write_json(output_dir / "dataset_validation.json", dataset_validation)
    write_json(output_dir / "catalog_approval.json", catalog_approval)
    write_json(output_dir / "quality_approval.json", quality_approval)
    write_frame(events, output_dir, "events", write_csv=config.output.write_csv)
    write_frame(
        decision_actions,
        output_dir,
        "decision_actions",
        write_csv=config.output.write_csv,
    )
    write_frame(equity, output_dir, "equity", write_csv=config.output.write_csv)
    write_frame(trades, output_dir, "trades", write_csv=config.output.write_csv)
    write_frame(
        portfolio_ledger,
        output_dir,
        "portfolio_ledger",
        write_csv=config.output.write_csv,
    )
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
    write_json(
        output_dir / "experiment_manifest.json",
        experiment_manifest_payload(
            config,
            dataset_validation,
            quality_approval,
            catalog_approval,
            metrics,
            output_dir,
            published_dir,
        ),
    )


def write_frame(frame: pl.DataFrame, output_dir: Path, name: str, *, write_csv: bool) -> None:
    frame.write_parquet(output_dir / f"{name}.parquet")
    if write_csv:
        frame.write_csv(output_dir / f"{name}.csv")


def config_output_payload(
    config: IntradayEventBacktestConfig,
    dataset_validation: dict[str, Any],
) -> dict[str, Any]:
    from strategy_sdk import __version__ as sdk_version

    payload = config.model_dump(mode="json")
    payload["strategy_sdk_version"] = sdk_version
    payload["strategy_type"] = "intraday_event_plugin"
    payload["strategy_file"] = str(config.strategy.strategy_file)
    payload["strategy_file_sha256"] = strategy_file_sha256(config.strategy.strategy_file)
    payload["strategy_params"] = dict(config.strategy.strategy_params)
    payload["execution_timing"] = config.execution.timing
    payload["quality_gate_required"] = config.dataset.quality_approval_required
    payload["timeframe"] = dataset_validation["timeframe"]
    payload["bar_minutes"] = dataset_validation["bar_minutes"]
    payload["dataset_type"] = dataset_validation["dataset_type"]
    payload["adjustment_mode"] = dataset_validation["adjustment_mode"]
    return payload


def experiment_manifest_payload(
    config: IntradayEventBacktestConfig,
    dataset_validation: dict[str, Any],
    quality_approval: dict[str, Any],
    catalog_approval: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
    published_dir: Path | None = None,
) -> dict[str, Any]:
    dataset_version = build_dataset_version(
        dataset_id=config.dataset.dataset_id
        or str(dataset_validation.get("continuous_symbol", config.dataset.path.stem)),
        dataset_path=config.dataset.path,
    )
    run_identity = event_run_identity(config, dataset_validation)
    return {
        "manifest_version": 2,
        "run_fingerprint": canonical_fingerprint(run_identity),
        "run_identity": run_identity,
        "runtime": runtime_provenance(),
        "run_id": config.output.run_id,
        "dataset": {
            "path": str(config.dataset.path),
            "sha256": file_sha256(config.dataset.path),
            "version_id": dataset_version.version_id,
            "schema_hash": dataset_version.schema_hash,
            "schema_version": dataset_version.schema_version,
            "row_count": dataset_version.row_count,
            "validation": dataset_validation,
        },
        "catalog_approval": catalog_approval,
        "quality_approval": {
            "status": quality_approval.get("status"),
            "path": quality_approval.get("approval_path"),
            "sha256": quality_approval.get("approval_sha256"),
            "required": quality_approval.get("required"),
        },
        "strategy": {
            "path": str(config.strategy.strategy_file),
            "sha256": strategy_file_sha256(config.strategy.strategy_file),
            "params": dict(config.strategy.strategy_params),
        },
        "instrument": config.instrument.model_dump(mode="json"),
        "portfolio": config.portfolio.model_dump(mode="json"),
        "session": config.session.model_dump(mode="json"),
        "execution": config.execution.model_dump(mode="json"),
        "input_artifacts": run_identity["input_artifacts"],
        "metrics": metrics,
        "outputs": output_manifest(output_dir, published_dir or output_dir),
    }


def output_manifest(
    output_dir: Path,
    published_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    published_dir = published_dir or output_dir
    output_files = [
        "config.json",
        "dataset_validation.json",
        "catalog_approval.json",
        "quality_approval.json",
        "events.parquet",
        "events.csv",
        "decision_actions.parquet",
        "decision_actions.csv",
        "equity.parquet",
        "equity.csv",
        "trades.parquet",
        "trades.csv",
        "portfolio_ledger.parquet",
        "portfolio_ledger.csv",
        "metrics.json",
        "accounting_reconciliation.json",
    ]
    return {
        file_name: {
            "path": str(published_dir / file_name),
            "sha256": file_sha256(output_dir / file_name),
        }
        for file_name in output_files
        if (output_dir / file_name).exists()
    }
