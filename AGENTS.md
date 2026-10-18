# Backend development guide

The workspace `AGENTS.md` applies here. This independent repository owns historical
backtesting, exchange adapters, execution/accounting, research, Workbench and tests.
Current capabilities include daily/intraday execution, MOEX futures, Binance spot
and linear perpetuals, HTTP services and the local research MCP. The original
MVP scope is retained in [the historical guide](docs/archive/mvp-development-guide.md);
its daily-only restrictions are no longer active development rules.

## Boundaries and reproducibility

- Keep ingestion in adapters, series construction in its domain, backtesting in
  execution/accounting and presentation in reports/Workbench. Expose capabilities
  without exchange conditionals in the UI.
- Normal backtests require explicit approved datasets, catalog bindings, historical
  specifications and calendars. Reject missing, unknown, warning, split_required
  and rejected inputs. Diagnostic exceptions must be explicit and recorded.
- File existence or schema validity does not imply approval. Never silently repair
  gaps, timestamps, prices, currencies or rolls. Preserve raw responses and policy audits.
- Portfolio ledger output is accounting truth. Reports and MCP consume native outputs;
  they must not calculate a second balance or PnL.
- Preserve causal scheduling, bounded finalized history, independent strategy state,
  confirmed fills, quality gates and reconciliation. Save all execution assumptions.
- Historical Binance rules and funding event prices are explicit inputs. Never infer
  historical specifications from current exchange metadata. Record portfolio currency,
  capital, leverage and annualization; do not assume USDT equals USD/RUB.
- Live trading, broker integration and live order routing remain outside this system.
  Unsupported inverse/quanto perpetuals fail closed.

## Shared strategy SDK

- Author reusable algorithms in the configured external strategy-sdk catalog. Read its
  AGENTS.md and README.md; locate it via strategies.sdk_catalog_path or
  WORKBENCH_SDK_CATALOG_PATH, not assumed filesystem adjacency.
- Shipped strategies use the pinned vendor wheel. Keep SDK independent of this host.
  Package updates require a new wheel, lock update and packaged verification.
- Workbench stores SDK_CATALOG_STRATEGY references and presets. Freeze all sources and
  helpers before queueing runs/sweeps. Never reload code/change parameters during a run
  or silently fall back to the wheel. Preserve source fingerprints and state isolation.
- Use SDK catalog_sources and source_files for loading/editing; do not duplicate
  dependency resolution or import TradeRelay modules.
- Preserve revision conflicts, history backups and local-only write endpoints.
  Editing a draft does not publish it; only explicit "Save common SDK file" updates
  the catalog. Open presets as editable Python source, preserving helpers/defaults.
- Requests use script_language=python; saved records use language=python. Preserve
  legacy custom JS. Saved-record migration must match known source hashes with a backup.

## Research, artifacts and verification

Use [MCP or CLI](../docs/mcp_research.md) with the running Workbench: inspect approved
catalogs/contracts, freeze plans, submit to the common queue and poll progress.
Preserve failed/inactive trials, pinned inputs and fingerprints. Read large results
by columns/pages. Standalone protocol execution and archived analysis remain supported.

Keep reusable tools in tools/, small recorded fixtures in tests/fixtures/, permanent
notes in docs/, runtime data/results in ignored data/, and intermediate scripts,
audits and test files in ignored .scratch/. Logs use .scratch/logs/. Never move inputs
referenced by saved runs or commit credentials, private endpoints, caches or build artifacts.

Use Python 3.12+, typed boundaries and source modules of at most 300 physical lines.
Tests are offline and deterministic by default, using their own fixtures rather than
sibling runtime data. Live snapshot updates must be explicit and retain raw responses.

```powershell
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q -p no:cacheprovider --basetemp=.scratch/pytest/<unique-run-id>
uv run backtest --help
uv run backtest-mcp --help
```

Execution/accounting changes require golden fixtures and ledger reconciliation.
API/cross-module changes require integration and UI checks; packaging requires a
build and packaged launch. Inspect repository boundaries, preserve unrelated changes,
stage reviewed files and commit this repository separately. Report validation,
material limitations, output locations and commit hashes.
