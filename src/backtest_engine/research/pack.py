from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from backtest_engine.core.exceptions import ReportingError
from backtest_engine.reporting.tearsheet.report import (
    IntradayEventTearsheetConfig,
    generate_intraday_event_tearsheet,
)


class ResearchPackConfig(BaseModel):
    path: Path
    output_path: Path | None = None
    top: int = Field(default=10, gt=0)
    generate_tearsheet: bool = True


class ResearchPackResult(BaseModel):
    output_path: Path
    markdown_path: Path
    html_path: Path
    manifest_path: Path
    pack_type: str


def generate_research_pack(config: ResearchPackConfig) -> ResearchPackResult:
    if not config.path.exists():
        raise ReportingError(f"Research pack path does not exist: {config.path}")
    pack_type = _detect_pack_type(config.path)
    output_path = config.output_path or config.path / "research_pack"
    output_path.mkdir(parents=True, exist_ok=True)
    if pack_type == "experiment":
        payload = _experiment_payload(config)
    else:
        payload = _run_payload(config, output_path)
    markdown = _render_markdown(payload)
    markdown_path = output_path / "research_pack.md"
    html_path = output_path / "research_pack.html"
    manifest_path = output_path / "research_pack_manifest.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(_render_html(markdown), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return ResearchPackResult(
        output_path=output_path,
        markdown_path=markdown_path,
        html_path=html_path,
        manifest_path=manifest_path,
        pack_type=pack_type,
    )


def _detect_pack_type(path: Path) -> str:
    if _table_exists(path / "summary"):
        return "experiment"
    if (path / "events.parquet").exists() or (path / "metrics.json").exists():
        return "run"
    raise ReportingError(f"Cannot detect research pack type for path: {path}")


def _experiment_payload(config: ResearchPackConfig) -> dict[str, Any]:
    summary = _read_table(config.path / "summary")
    failed_runs = _read_optional_table(config.path / "failed_runs")
    manifest = _read_optional_json(config.path / "experiment_manifest.json")
    top_runs = _top_runs(summary, config.top)
    return {
        "pack_type": "experiment",
        "source_path": str(config.path),
        "manifest": manifest,
        "summary": {
            "run_count": summary.height,
            "failed_run_count": failed_runs.height,
            "columns": summary.columns,
        },
        "top_runs": top_runs,
        "failed_runs": failed_runs.head(config.top).to_dicts() if failed_runs.height else [],
        "linked_reports": _linked_reports(config.path),
    }


def _run_payload(config: ResearchPackConfig, output_path: Path) -> dict[str, Any]:
    manifest = _read_optional_json(config.path / "experiment_manifest.json")
    metrics = _read_optional_json(config.path / "metrics.json")
    dataset_validation = _read_optional_json(config.path / "dataset_validation.json")
    quality_approval = _read_optional_json(config.path / "quality_approval.json")
    catalog_approval = _read_optional_json(config.path / "catalog_approval.json")
    tearsheet_path = None
    if config.generate_tearsheet and (config.path / "events.parquet").exists():
        tearsheet = generate_intraday_event_tearsheet(
            IntradayEventTearsheetConfig(
                run_path=config.path,
                output_path=output_path / "tearsheet",
            )
        )
        tearsheet_path = str(tearsheet.report_path)
    return {
        "pack_type": "run",
        "source_path": str(config.path),
        "manifest": manifest,
        "metrics": metrics,
        "dataset_validation": dataset_validation,
        "quality_approval": quality_approval,
        "catalog_approval": catalog_approval,
        "tearsheet_path": tearsheet_path,
        "available_outputs": sorted(path.name for path in config.path.iterdir() if path.is_file()),
    }


def _top_runs(summary: pl.DataFrame, top: int) -> list[dict[str, Any]]:
    if summary.height == 0:
        return []
    sort_column = _best_sort_column(summary)
    if sort_column is None:
        return summary.head(top).to_dicts()
    return summary.sort(sort_column, descending=True, nulls_last=True).head(top).to_dicts()


def _best_sort_column(summary: pl.DataFrame) -> str | None:
    for column in ("sharpe", "total_return", "net_pnl"):
        if column in summary.columns and summary.schema[column].is_numeric():
            return column
    return None


def _linked_reports(path: Path) -> dict[str, str]:
    reports = {}
    for relative in (
        "report.md",
        "research_validation/report.md",
        "compare_report.md",
        "pipeline_manifest.json",
    ):
        candidate = path / relative
        if candidate.exists():
            reports[relative] = str(candidate)
    return reports


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Research Pack",
        "",
        f"- pack_type: `{payload['pack_type']}`",
        f"- source_path: `{payload['source_path']}`",
        "",
    ]
    if payload["pack_type"] == "experiment":
        summary = payload["summary"]
        lines.extend(
            [
                "## Experiment Summary",
                "",
                f"- run_count: {summary['run_count']}",
                f"- failed_run_count: {summary['failed_run_count']}",
                "",
                "## Top Runs",
                "",
                _markdown_table(payload["top_runs"]),
                "",
            ]
        )
        if payload["failed_runs"]:
            lines.extend(["## Failed Runs", "", _markdown_table(payload["failed_runs"]), ""])
        if payload["linked_reports"]:
            lines.extend(["## Linked Reports", ""])
            lines.extend(
                f"- `{name}`: `{path}`" for name, path in payload["linked_reports"].items()
            )
            lines.append("")
    else:
        metrics = payload.get("metrics") or {}
        lines.extend(
            [
                "## Run Metrics",
                "",
                _markdown_kv(metrics),
                "",
            ]
        )
        if payload.get("tearsheet_path"):
            lines.extend(["## Tearsheet", "", f"- `{payload['tearsheet_path']}`", ""])
        quality = payload.get("quality_approval") or {}
        if quality:
            lines.extend(["## Quality Approval", "", _markdown_kv(quality), ""])
        catalog = payload.get("catalog_approval") or {}
        if catalog:
            lines.extend(["## Catalog Approval", "", _markdown_kv(catalog), ""])
        lines.extend(["## Available Outputs", ""])
        lines.extend(f"- `{name}`" for name in payload.get("available_outputs", []))
        lines.append("")
    manifest = payload.get("manifest")
    if manifest:
        lines.extend(["## Manifest Snapshot", "", "```json", _json_preview(manifest), "```", ""])
    return "\n".join(lines)


def _render_html(markdown: str) -> str:
    escaped = html.escape(markdown)
    return (
        "<!doctype html>\n"
        '<html><head><meta charset="utf-8"><title>Research Pack</title>'
        "<style>body{font-family:system-ui,sans-serif;max-width:1080px;margin:32px auto;"
        "line-height:1.45;padding:0 18px}pre{background:#f5f5f5;padding:12px;overflow:auto}"
        "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:4px 8px}</style>"
        "</head><body><pre>"
        f"{escaped}"
        "</pre></body></html>\n"
    )


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def _markdown_kv(payload: dict[str, Any]) -> str:
    if not payload:
        return "_No data._"
    return "\n".join(f"- {key}: `{_format_cell(value)}`" for key, value in sorted(payload.items()))


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str).replace("|", "\\|")
    return str(value).replace("|", "\\|")


def _json_preview(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if len(text) <= 4000:
        return text
    return text[:4000] + "\n..."


def _table_exists(base_path: Path) -> bool:
    return any(base_path.with_suffix(suffix).exists() for suffix in (".parquet", ".csv", ".json"))


def _read_table(base_path: Path) -> pl.DataFrame:
    for suffix in (".parquet", ".csv", ".json"):
        path = base_path.with_suffix(suffix)
        if not path.exists():
            continue
        if suffix == ".parquet":
            return pl.read_parquet(path)
        if suffix == ".csv":
            return pl.read_csv(path, infer_schema_length=1000)
        rows = json.loads(path.read_text(encoding="utf-8"))
        return pl.DataFrame(rows, orient="row") if rows else pl.DataFrame()
    raise ReportingError(f"Missing table: {base_path}.[parquet|csv|json]")


def _read_optional_table(base_path: Path) -> pl.DataFrame:
    return _read_table(base_path) if _table_exists(base_path) else pl.DataFrame()


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
