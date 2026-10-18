"""Freeze all SDK sources and explicit inputs before starting any case."""

from itertools import product
from pathlib import Path
from typing import Any

from backtest_engine.strategies.runtime import validate_strategy_params
from backtest_engine.workbench.models import WorkbenchRunRequest
from backtest_engine.workbench.sdk_catalog import freeze_run_request
from backtest_engine.workbench.strategy_contract import load_workbench_strategy_metadata

from .models import Case, Protocol
from .provenance import (
    INPUT_FIELDS,
    artifact_hashes,
    digest,
    file_hash,
    read_json,
    runtime_identity,
)


def resolve_path(value: str | Path, parent: Path) -> Path:
    path = Path(value)
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


def load_protocol(path: Path) -> tuple[Protocol, list[Case]]:
    protocol = Protocol.model_validate(read_json(path))
    parent = path.resolve().parent
    protocol.output_root = resolve_path(protocol.output_root, parent)
    if protocol.base_request is not None:
        protocol.base_request = resolve_path(protocol.base_request, parent)
    cases = list(protocol.cases)
    for grid in protocol.grids:
        names = list(grid.params)
        for index, values in enumerate(product(*(grid.params[n] for n in names)), start=1):
            cases.append(
                Case(
                    name=f"{grid.name}_{index:03d}",
                    group=grid.group,
                    params=dict(zip(names, values, strict=True)),
                )
            )
    names = [case.name for case in cases]
    if len(names) != len(set(names)) or "selected_fresh" in names:
        raise ValueError("case names must be unique; selected_fresh is reserved")
    if protocol.baseline not in names:
        raise ValueError(f"baseline case missing: {protocol.baseline}")
    if len(cases) + bool(protocol.fresh_selected_group) > protocol.max_runs:
        raise ValueError(f"protocol exceeds max_runs={protocol.max_runs}")
    for case in cases:
        if case.existing_run is not None:
            case.existing_run = resolve_path(case.existing_run, parent)
    return protocol, cases


def request_payload(raw: dict[str, Any], parent: Path) -> dict[str, Any]:
    unknown = set(raw) - set(WorkbenchRunRequest.model_fields)
    if unknown:
        raise ValueError(f"unknown Workbench request fields: {sorted(unknown)}")
    result = dict(raw)
    for key in (*INPUT_FIELDS, "script_path", "output_root"):
        if result.get(key) is not None:
            result[key] = str(resolve_path(result[key], parent))
    return result


def frozen_request(raw: dict[str, Any], root: Path, name: str) -> WorkbenchRunRequest:
    request = WorkbenchRunRequest.model_validate(raw)
    if not request.quality_approval_required or request.catalog_enforcement != "strict":
        raise ValueError(f"{name}: robustness requires strict quality and dataset catalog gates")
    if request.allow_noncausal_same_close or not request.strict_runtime:
        raise ValueError(f"{name}: causal execution and strict runtime are required")
    request = freeze_run_request(request)
    if request.script_path is not None:
        request = request.model_copy(
            update={
                "script_source": request.script_path.read_text(encoding="utf-8"),
                "script_language": "python"
                if request.script_path.suffix == ".py"
                else "javascript",
                "script_path": None,
            }
        )
    suffix = ".py" if request.script_language == "python" else ".js"
    script = root / "preflight" / (name + "_" + digest(request.script_source)[:12] + suffix)
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(request.script_source, encoding="utf-8")
    metadata = load_workbench_strategy_metadata(script, request.params)
    validate_strategy_params(request.params, metadata.params)
    return request.model_copy(update={"run_id": name, "output_root": root / "runs"})


def prepare_plan(protocol: Protocol, cases: list[Case], *, existing: bool) -> dict[str, Any]:
    raw = {}
    if not existing:
        if protocol.base_request is None:
            raise ValueError("base_request is required for execution")
        raw = request_payload(read_json(protocol.base_request), protocol.base_request.parent)
        # One catalog freeze for the entire base family, including imported helper files.
        raw = frozen_request(raw, protocol.output_root, "base_request").model_dump(mode="json")
    inputs, prepared = {}, []
    for case in cases:
        item = case.model_dump(mode="json")
        if existing:
            if case.existing_run is None:
                raise ValueError(f"{case.name}: existing_run is required for analysis-only mode")
            hashes = artifact_hashes(case.existing_run)
            for name in (
                "config.json",
                "metrics.json",
                "events.parquet",
                "portfolio_ledger.parquet",
                "trades.parquet",
                "quality_approval.json",
                "catalog_approval.json",
            ):
                if name not in hashes:
                    raise ValueError(f"{case.name}: native artifact missing: {name}")
            item["artifacts"] = hashes
        else:
            if case.existing_run is not None:
                raise ValueError(
                    "existing_run requires --analyze-existing; never mix new and old runs"
                )
            parent = protocol.base_request.parent
            overrides = request_payload(case.request_overrides, parent)
            if {"run_id", "output_root", "params"} & overrides.keys():
                raise ValueError("use case.name/params and protocol.output_root")
            case_raw = {**raw, **overrides, "params": {**raw.get("params", {}), **case.params}}
            if "script_source" in overrides:
                case_raw["script_path"] = None
            if "script_path" in overrides:
                case_raw["script_source"] = None
            request = frozen_request(case_raw, protocol.output_root, case.name)
            item["request"] = request.model_dump(mode="json")
            for key in INPUT_FIELDS:
                path = getattr(request, key)
                if path is not None and str(path) not in inputs:
                    inputs[str(path)] = file_hash(path)
        prepared.append(item)
    if existing and protocol.fresh_selected_group:
        raise ValueError("fresh_selected_group requires new execution")
    if protocol.fresh_selected_group and not any(
        c.group == protocol.fresh_selected_group for c in cases
    ):
        raise ValueError("fresh_selected_group does not identify any case")
    return {
        "protocol": protocol.model_dump(mode="json"),
        "mode": "existing" if existing else "execute",
        "cases": prepared,
        "input_hashes": inputs,
        "runtime": runtime_identity(),
    }
