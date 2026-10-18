"""Execute plans exclusively through Workbench; resume only verified native results."""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from backtest_engine.research.case_execution import execute_case_record
from backtest_engine.workbench.models import WorkbenchRunRequest
from backtest_engine.workbench.runner import run_workbench_backtest

from .metrics import analyze_run, audit_run
from .planning import frozen_request, load_protocol, prepare_plan
from .provenance import (
    artifact_hashes,
    digest,
    read_json,
    verify_hashes,
    write_json,
)
from .reporting import write_report
from .statistics import select_on_training


def execute_case(
    item: dict, root: Path, fingerprint: str, inputs: dict[str, str], existing: bool, resume: bool
) -> dict:
    name = item["name"]
    record_path = root / "checkpoints" / (name + ".json")
    if record_path.exists():
        if not resume:
            raise ValueError(f"{name}: checkpoint exists; use --resume")
        record = read_json(record_path)
        if record["fingerprint"] != fingerprint:
            raise ValueError(f"{name}: checkpoint belongs to another plan")
        if record["status"] == "completed":
            verify_hashes(record["artifacts"], Path(record["run_path"]))
            return record
        raise ValueError(
            f"{name}: previous failure requires a new output_root; {record.get('error')}"
        )

    def operation():
        verify_hashes(inputs)
        if existing:
            path = Path(item["existing_run"])
            verify_hashes(item["artifacts"], path)
        else:
            path = root / "runs" / name
            if path.exists():
                raise ValueError(
                    f"{name}: native run exists without a checkpoint; refusing overwrite"
                )
            request = WorkbenchRunRequest.model_validate(item["request"])
            result = run_workbench_backtest(request)
            path = result.output_dir
        audit_run(path)
        verify_hashes(inputs)
        return {
            "name": name,
            "status": "completed",
            "run_path": str(path),
            "fingerprint": fingerprint,
            "artifacts": artifact_hashes(path),
        }

    return execute_case_record(
        name=name, fingerprint=fingerprint, checkpoint=record_path, operation=operation
    )


def capture_runtime(root: Path) -> None:
    from .provenance import RUNTIME_PACKAGES

    with ZipFile(root / "runtime_sources.zip", "w", ZIP_DEFLATED) as archive:
        for package in RUNTIME_PACKAGES:
            source = Path(package.__file__).resolve().parent
            for path in sorted(source.rglob("*.py")):
                archive.write(path, f"{package.__name__}/{path.relative_to(source).as_posix()}")


def evaluate_records(protocol, items, records):
    rows, annual, audits, daily = [], [], [], {}
    for item in items:
        record = records[item["name"]]
        if record["status"] != "completed":
            rows.append(
                {
                    "run_id": item["name"],
                    "group": item["group"],
                    "status": "failed",
                    "error": record["error"],
                }
            )
            continue
        path = Path(record["run_path"])
        try:
            row, years, audit, frame = analyze_run(
                path, item["name"], item["group"], protocol.split_date
            )
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(protocol.output_root / "checkpoints" / f"{item['name']}.json", record)
            rows.append(
                {
                    "run_id": item["name"],
                    "group": item["group"],
                    "status": "failed",
                    "error": record["error"],
                }
            )
            continue
        row["params_json"] = json.dumps(
            read_json(path / "config.json")["strategy"]["strategy_params"], sort_keys=True
        )
        rows.append(row)
        annual.extend(years)
        audits.append({"run_id": item["name"], **audit})
        daily[item["name"]] = frame
    return rows, annual, audits, daily


def selected_fresh_case(protocol, items, rows):
    group = protocol.fresh_selected_group
    if any(r["status"] != "completed" for r in rows if r["group"] == group):
        raise ValueError("training selection requires the complete planned family")
    selected = select_on_training(rows, group)
    source = next(item for item in items if item["name"] == selected["run_id"])
    raw = {**source["request"], "from_date": str(protocol.split_date)}
    request = frozen_request(raw, protocol.output_root, "selected_fresh")
    choice = {
        "selected_run": source["name"],
        "criterion": "train_total_return",
        "train_end_exclusive": str(protocol.split_date),
        "train_value": selected["train_total_return"],
        "request": request.model_dump(mode="json"),
    }
    path = protocol.output_root / "training_selection.json"
    if path.exists() and read_json(path) != choice:
        raise ValueError("training selection changed since checkpoint")
    write_json(path, choice)
    return {"name": "selected_fresh", "group": "selected_fresh", "request": choice["request"]}


def freeze_protocol(config: Path, *, resume=False, existing=False) -> dict:
    protocol, cases = load_protocol(config)
    root = protocol.output_root
    root.mkdir(parents=True, exist_ok=True)
    plan = prepare_plan(protocol, cases, existing=existing)
    fingerprint = digest(plan)
    manifest = root / "manifest.json"
    if manifest.exists():
        saved = read_json(manifest)
        if not resume or saved["fingerprint"] != fingerprint:
            raise ValueError(
                "output_root has another plan or runtime; use a new root or matching --resume"
            )
        plan = saved["plan"]
    else:
        write_json(manifest, {"fingerprint": fingerprint, "plan": plan})
        capture_runtime(root)
    verify_hashes(plan["input_hashes"])
    return {"fingerprint": fingerprint, "plan": plan}


def execute_protocol(manifest: dict, *, resume=False, progress_callback=None) -> dict:
    """Execute an already frozen plan without reading live strategy/catalog sources."""
    from .models import Protocol
    from .provenance import runtime_identity

    plan, fingerprint = manifest["plan"], manifest["fingerprint"]
    if digest(plan) != fingerprint or digest(runtime_identity()) != digest(plan["runtime"]):
        raise ValueError("frozen protocol or runtime changed since preparation")
    protocol = Protocol.model_validate(plan["protocol"])
    root = protocol.output_root
    existing = plan["mode"] == "existing"
    verify_hashes(plan["input_hashes"])
    records = {}
    planned_count = len(plan["cases"]) + bool(protocol.fresh_selected_group)
    with ProcessPoolExecutor(mp_context=get_context("spawn"), max_workers=protocol.workers) as pool:
        futures = {
            pool.submit(
                execute_case, item, root, fingerprint, plan["input_hashes"], existing, resume
            ): item["name"]
            for item in plan["cases"]
        }
        for future in as_completed(futures):
            name = futures[future]
            records[name] = future.result()
            print(f"[{len(records)}/{len(futures)}] {name}: {records[name]['status']}", flush=True)
            if progress_callback is not None:
                progress_callback(len(records), planned_count, name, records[name]["status"])
    items = list(plan["cases"])
    rows, annual, audits, daily = evaluate_records(protocol, items, records)
    if protocol.fresh_selected_group:
        try:
            item = selected_fresh_case(protocol, items, rows)
            records[item["name"]] = execute_case(
                item, root, fingerprint, plan["input_hashes"], False, resume
            )
        except Exception as error:
            item = {"name": "selected_fresh", "group": "selected_fresh"}
            records[item["name"]] = {"status": "failed", "error": str(error)}
        items.append(item)
        if progress_callback is not None:
            progress_callback(
                len(records), planned_count, item["name"], records[item["name"]]["status"]
            )
        rows, annual, audits, daily = evaluate_records(protocol, items, records)
    verify_hashes(plan["input_hashes"])
    return write_report(protocol, rows, annual, audits, daily, fingerprint)


def run_protocol(config: Path, *, resume=False, existing=False, plan_only=False) -> dict:
    manifest = freeze_protocol(config, resume=resume, existing=existing)
    if plan_only:
        return {
            "status": "planned",
            "case_count": len(manifest["plan"]["cases"]),
            "fingerprint": manifest["fingerprint"],
        }
    return execute_protocol(manifest, resume=resume)
