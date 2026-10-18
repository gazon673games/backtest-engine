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


## Funding and research assumptions

Execution settings accept `funding_mode` in single-instrument, portfolio and Workbench
requests. Paper replay and candidate price preparation accept the same modes:

- `legacy` (default) preserves existing behavior: absent rates use the declared
  default, which remains zero; existing priced and unpriced inputs retain their valuation.
- `disabled` books no funding and does not require funding inputs.
- `historical` requires explicit event flags and observed event-priced funding.
  Missing event prices, estimated prices and default-rate substitutions are rejected.

These settings do not change the existing mark-price or margin requirements. Historical
mode validates supplied funding events; dataset approval must still establish coverage.
The selected policy is saved in the run configuration. Adding it changes new run
fingerprints, including when the financial assumptions are equivalent to a previous run.

Research selection excludes returns whose `next_ts` is after the training cutoff.
The configured minimum training history is enforced. Trials must use the same evaluation
timestamps; missing, duplicate or non-finite observations cannot be silently discarded.
Incomplete experiments and unavailable multiple-testing probabilities cannot pass the gate.

L2 maker orders share each price level's available volume in submission order. The current
decrease-based queue model assumes observed size decreases are trades when enabled;
aggregate book updates alone cannot distinguish executions from cancellations.
