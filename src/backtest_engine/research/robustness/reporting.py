"""Portable overview plus complete machine-readable tables, including failures."""

from html import escape

import polars as pl

from backtest_engine.research.validation.io import write_table

from .models import Protocol
from .provenance import write_json
from .statistics import baseline_bootstrap, family_diagnostics


def number(value, percent=False):
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%" if percent else f"{float(value):,.2f}"


def write_report(
    protocol: Protocol,
    rows: list[dict],
    annual: list[dict],
    audits: list[dict],
    daily: dict[str, pl.DataFrame],
    fingerprint: str,
) -> dict:
    root = protocol.output_root
    for name, values in (("summary", rows), ("annual", annual), ("reconciliation", audits)):
        write_table(root / name, pl.DataFrame(values, infer_schema_length=None))
    families, bias, deflated = family_diagnostics(protocol, rows, daily)
    for name, values in (
        ("families", families),
        ("selection_bias", bias),
        ("deflated_sharpe", deflated),
        ("block_bootstrap", baseline_bootstrap(protocol, daily)),
    ):
        write_table(root / name, pl.DataFrame(values, infer_schema_length=None))
    for name, frame in daily.items():
        (root / "daily").mkdir(exist_ok=True)
        frame.write_parquet(root / "daily" / f"{name}.parquet")
    failures = [r for r in rows if r["status"] != "completed"]
    result = {
        "status": "failed" if failures else "completed",
        "case_count": len(rows),
        "failed_count": len(failures),
        "fingerprint": fingerprint,
        "history_previously_inspected": protocol.history_previously_inspected,
        "robustness_verdict": "requires_review",
        "report_path": str(root / "report.html"),
    }
    write_json(root / "result.json", result)
    table = []
    for row in rows:
        values = [
            escape(str(row["run_id"])),
            escape(str(row["group"])),
            row["status"],
            number(row.get("total_return"), True),
            number(row.get("max_drawdown"), True),
            number(row.get("train_total_return"), True),
            number(row.get("check_total_return"), True),
            number(row.get("final_nav")),
            escape(str(row.get("base_currency", ""))),
            escape(str(row.get("error", ""))),
        ]
        table.append("<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>")
    assumptions = "".join(f"<li>{escape(value)}</li>" for value in protocol.assumptions)
    headers = [
        "Case",
        "Family",
        "Status",
        "Return",
        "Drawdown",
        "Train",
        "Check",
        "Final NAV",
        "Currency",
        "Error",
    ]
    links = " · ".join(
        f'<a href="{name}.csv">{name}.csv</a>'
        for name in (
            "summary",
            "annual",
            "families",
            "reconciliation",
            "block_bootstrap",
            "selection_bias",
            "deflated_sharpe",
        )
    )
    history = (
        "History was already inspected: the chronological check is not an untouched holdout."
        if protocol.history_previously_inspected
        else "Author declares history uninspected; preserve the fixed split and record every trial."
    )
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>{escape(protocol.name)}</title><style>
body{{font:15px system-ui;margin:24px;color:#dbe6ef;background:#102030}}
a{{color:#92cfff}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{padding:7px;border-bottom:1px solid #456;text-align:left}}th{{background:#20384b}}
code{{overflow-wrap:anywhere}}p{{max-width:1100px;line-height:1.5}}
</style><h1>{escape(protocol.name)}</h1>
<p>{len(rows)} cases; {len(failures)} failures. Split: {protocol.split_date}.
Returns and NAV come from the native portfolio ledger; drawdown uses bar valuations.</p>
<p>{history} A completed protocol means technical checks completed,
not that a strategy is profitable.
No automatic investment acceptance threshold is inferred.</p><p>{links}</p>
<table><thead><tr>{"".join(f"<th>{v}</th>" for v in headers)}</tr></thead>
<tbody>{"".join(table)}</tbody></table>
<h2>Assumptions</h2><ul>{assumptions}</ul>
<p>Cost variants execute full backtests. Daily moving-block bootstrap uses noncircular contiguous
blocks and empirical 5th/50th/95th percentiles. Loss share is a resampling diagnostic.
The reality check uses IID daily resampling; PBO/deflated Sharpe have method assumptions and do not
estimate future success. Failed/inactive trials stay in the tables. Different daily coverage or
incomplete families skip multiple-testing statistics rather than silently removing trials.</p>
<p>Best-trade subtraction is a cash attribution diagnostic, without compounding a new strategy.
An open ending position is valued by the declared native end policy. Fresh selected runs start
with the configured initial capital at the split; selection maximizes training return only.</p>
<p>Fingerprint: <code>{fingerprint}</code>. See manifest.json for frozen requests, input hashes,
runtime source hashes and original-run hashes when importing prior results.</p></html>"""
    (root / "report.html").write_text(html, encoding="utf-8")
    return result
