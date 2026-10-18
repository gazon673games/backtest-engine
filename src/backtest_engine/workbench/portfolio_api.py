"""Local portfolio presets and frozen native runs on the shared job queue."""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backtest_engine.data.store import file_sha256 as digest
from backtest_engine.portfolio.multi.config import MultiInstrumentBacktestConfig
from backtest_engine.portfolio.multi.data import resolve_leg_config
from backtest_engine.strategies.sdk_catalog_sources import catalog_snapshot
from backtest_engine.workbench.jobs import submit_job
from backtest_engine.workbench.runner import safe_run_id
from backtest_engine.workbench.sdk_catalog import freeze_catalog_source


class PortfolioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    diagnostic: bool = False
    config: MultiInstrumentBacktestConfig
    sources: dict[str, str] = Field(default_factory=dict)
    preset_id: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        sleeves = self.config.strategy.sdk_sleeves
        if not sleeves or {s.asset_id for s in sleeves} != {s.asset_id for s in self.config.legs}:
            raise ValueError("Portfolio SDK sleeves must match the dataset asset IDs exactly")
        if sum(s.initial_capital for s in sleeves) > self.config.portfolio.initial_capital:
            raise ValueError("Sleeve budgets exceed portfolio initial_capital")
        if set(self.sources) - {s.asset_id for s in sleeves}:
            raise ValueError("Unknown asset_id in sources")
        if not self.diagnostic and any(
            not leg.quality_approval_required
            or leg.catalog_enforcement != "strict"
            or leg.catalog_path is None
            or leg.dataset_id is None
            for leg in self.config.legs
        ):
            raise ValueError("Unapproved inputs require explicit diagnostic=true")
        if self.config.execution.allow_noncausal_same_close:
            raise ValueError("Workbench portfolio runs require causal execution")
        return self


def portfolio_list_payload(output_root: Path) -> dict[str, Any]:
    root = output_root.parent / "portfolios"
    items = []
    for path in sorted(root.glob("*.json")):
        items.append(json.loads(path.read_text(encoding="utf-8")))
    return {"items": items}


def portfolio_save_payload(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    request = PortfolioRequest.model_validate(payload)
    preset_id = safe_run_id(request.preset_id or f"portfolio-{uuid4().hex}")
    request.preset_id = preset_id
    root = output_root.parent / "portfolios"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{preset_id}.json"
    temporary = root / f".{preset_id}-{uuid4().hex}.tmp"
    temporary.write_text(request.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return request.model_dump(mode="json")


def freeze_portfolio(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    request = PortfolioRequest.model_validate(payload)
    config = request.config
    run_id = safe_run_id(config.output.run_id or f"portfolio-{uuid4().hex}")
    config.output.root = output_root
    config.output.run_id = run_id
    pins = {}
    for index, leg in enumerate(config.legs):
        resolved, _binding = resolve_leg_config(leg)
        config.legs[index] = resolved
        for field in (
            "dataset_path",
            "catalog_path",
            "quality_approval_path",
            "instrument_registry_path",
            "calendar_path",
        ):
            path = getattr(resolved, field)
            if path is not None:
                pins[str(path.resolve())] = digest(path)
    for path in (
        config.universe.path,
        config.execution.calibration_path,
        config.portfolio.valuation_rates_path,
        config.portfolio.reconciliation_statement_path,
    ):
        if path is not None:
            pins[str(path.resolve())] = digest(path)
    for sleeve in config.strategy.sdk_sleeves:
        source = request.sources.get(sleeve.asset_id)
        if source is not None:
            frozen = freeze_catalog_source(source, "python")
            tree = ast.parse(frozen)
            if (
                len(tree.body) != 1
                or not isinstance(tree.body[0], ast.Assign)
                or len(tree.body[0].targets) != 1
                or not isinstance(tree.body[0].targets[0], ast.Name)
                or tree.body[0].targets[0].id != "SDK_CATALOG_SNAPSHOT"
            ):
                raise ValueError(f"{sleeve.asset_id}: select a shared SDK strategy")
            sleeve.source_snapshot = ast.literal_eval(tree.body[0].value)
        elif sleeve.source_snapshot is None:
            path = sleeve.strategy_file
            sleeve.source_snapshot = catalog_snapshot(path.parent, path.stem)
    plan = request.model_dump(mode="json")
    fingerprint = hashlib.sha256(
        json.dumps(
            {"plan": plan, "inputs": pins},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    return {
        "output_root": str(output_root),
        "request": plan,
        "input_pins": pins,
        "fingerprint": fingerprint,
        "created_at": datetime.now(UTC).isoformat(),
    }


def portfolio_run_payload(output_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    frozen = freeze_portfolio(output_root, payload)
    run_id = frozen["request"]["config"]["output"]["run_id"]
    submit_job(output_root, run_id, "portfolio", frozen)
    return {"run_id": run_id, "fingerprint": frozen["fingerprint"]}
