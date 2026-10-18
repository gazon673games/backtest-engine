"""Bounded result reads by run identity and allowlisted artifacts, never arbitrary files."""

import re
from pathlib import Path
from typing import Literal

import polars as pl

from backtest_engine.research.robustness.provenance import read_json
from backtest_engine.workbench.agent_catalog import page
from backtest_engine.workbench.robustness_jobs import experiment_root, experiment_status

ResultKind = Literal["backtest", "robustness"]
JSON_ARTIFACTS = {
    "metrics",
    "config",
    "quality_approval",
    "catalog_approval",
    "reconciliation",
    "manifest",
    "training_selection",
    "result",
    "families",
    "block_bootstrap",
    "selection_bias",
    "deflated_sharpe",
    "summary",
    "annual",
}
TABLE_ARTIFACTS = {
    "events",
    "equity",
    "trades",
    "portfolio_ledger",
    "decision_actions",
    "summary",
    "annual",
    "families",
    "block_bootstrap",
    "selection_bias",
    "deflated_sharpe",
    "reconciliation",
}


def result_root(output_root: Path, result_id: str, kind: ResultKind) -> Path:
    if kind == "robustness":
        return experiment_root(output_root, result_id)
    if kind != "backtest" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", result_id):
        raise ValueError("invalid result kind or run_id")
    return output_root.resolve() / result_id


def results(
    output_root: Path,
    result_id: str,
    kind: ResultKind,
    offset=0,
    limit=20,
    group: str | None = None,
    sort_by="run_id",
    descending=False,
):
    root = result_root(output_root, result_id, kind)
    if kind == "backtest":
        return {
            "run_id": result_id,
            "metrics": read_json(root / "metrics.json"),
            "quality_approval": read_json(root / "quality_approval.json"),
            "catalog_approval": read_json(root / "catalog_approval.json"),
            "artifacts": sorted(p.name for p in root.iterdir() if p.is_file()),
        }
    status = experiment_status(output_root, result_id)
    rows = read_json(root / "summary.json") if (root / "summary.json").exists() else []
    if group is not None:
        rows = [row for row in rows if row["group"] == group]
    allowed = {"run_id", "total_return", "max_drawdown", "train_total_return", "check_total_return"}
    if sort_by not in allowed:
        raise ValueError(f"sort_by must be one of {sorted(allowed)}")
    valid = sorted(
        (r for r in rows if r.get(sort_by) is not None),
        key=lambda row: row[sort_by],
        reverse=descending,
    )
    rows = [*valid, *(r for r in rows if r.get(sort_by) is None)]
    fields = (
        "run_id",
        "group",
        "status",
        "active",
        "total_return",
        "max_drawdown",
        "train_total_return",
        "check_total_return",
        "final_nav",
        "base_currency",
        "error",
    )
    return {
        **status,
        **page([{k: row.get(k) for k in fields} for row in rows], offset, limit),
        "families": read_json(root / "families.json") if (root / "families.json").exists() else [],
        "report_path": str(root / "report.html"),
    }


def artifact(
    output_root: Path,
    result_id: str,
    kind: ResultKind,
    name: str,
    offset=0,
    limit=20,
    columns: list[str] | None = None,
    case_name: str | None = None,
):
    root = result_root(output_root, result_id, kind)
    if case_name is not None:
        if kind != "robustness" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", case_name):
            raise ValueError("case_name requires a valid robustness case identity")
        manifest = read_json(root / "manifest.json")
        names = {item["name"] for item in manifest["plan"]["cases"]}
        if case_name == "selected_fresh" and manifest["plan"]["protocol"]["fresh_selected_group"]:
            names.add(case_name)
        if case_name not in names:
            raise ValueError(f"case not in frozen plan: {case_name}")
        root = root / "runs" / case_name
    if name not in JSON_ARTIFACTS | TABLE_ARTIFACTS:
        raise ValueError(f"unknown artifact: {name}")
    table = root / (name + ".parquet")
    if name in TABLE_ARTIFACTS and table.exists():
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("offset must be >= 0; limit must be 1..100")
        scan = pl.scan_parquet(table)
        schema = scan.collect_schema()
        preferred = [
            name
            for name in (
                "ts",
                "next_ts",
                "fill_ts",
                "symbol",
                "equity_cash",
                "net_cash_flow",
                "gross_pnl",
                "cost_cash",
                "net_pnl",
                "net_return",
                "target_contracts",
                "contract_delta",
                "fill_price",
                "order_status",
                "reconciliation_diff",
            )
            if name in schema
        ]
        chosen = columns if columns is not None else (preferred or list(schema))[:20]
        if not chosen or len(chosen) > 20 or not set(chosen) <= set(schema):
            raise ValueError("choose 1..20 existing artifact columns")
        total = scan.select(pl.len()).collect().item()
        rows = scan.select(chosen).slice(offset, limit).collect().to_dicts()
        return {
            "artifact": name,
            "columns": chosen,
            "available_columns": list(schema),
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": rows,
            "next_offset": offset + limit if offset + limit < total else None,
        }
    path = root / (name + ".json")
    if not path.exists():
        raise FileNotFoundError(f"artifact {name} not available for {result_id}")
    if path.stat().st_size > 64 * 1024:
        return {
            "artifact": name,
            "path": str(path),
            "status": "file_reference",
            "bytes": path.stat().st_size,
        }
    value = read_json(path)
    return {
        "artifact": name,
        "data": page(value, offset, limit) if isinstance(value, list) else value,
    }
