# Backtest Engine

Exchange-neutral historical execution and portfolio accounting for explicit market
datasets and user-provided Python/JavaScript source files. Requires Python 3.12+ and uv.

Includes daily/intraday event execution, order simulation, commissions, margin, native
ledger reconciliation, data-quality gates, reports, research scheduling and a local API.
Trading algorithms, tuned presets and private research results are supplied separately.

## Development

Clone the parent workspace recursively, then run from this directory:

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run python -m pytest -q
uv run backtest --help
uv run backtest-mcp --help
```

Normal runs require approved datasets, historical specifications, a calendar and explicit
execution assumptions. Future data access is rejected. The portfolio ledger is the
accounting source of truth. Broker integration and live order routing are not included.

The pinned SDK wheel contains contracts and a non-trading blank callback. Local source
files and research fixtures are ignored and excluded from public packages. The daily
buy-and-hold and moving-average controls are standard engine benchmarks, without a
personal strategy catalog or tuned parameters.
