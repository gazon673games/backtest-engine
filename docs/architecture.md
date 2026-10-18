# Architecture Notes

`backtest_engine` is organized as a historical research system with a small set of bounded
contexts. The goal is to keep market-domain rules separate from CLI adapters, file formats, and
research reports.

## Repository Layout

```text
ohlcv-replay/
|-- ui/                       Electron shell and browser Workbench
|-- settings/                 shared desktop/backend process settings
|-- build/                    generated intermediate build artifacts
|-- dist/                     generated installer and unpacked application
|-- backtest_engine/
|   |-- configs/              reproducible experiment and registry configs
|   |-- strategies/           user-authored strategy plugins
|   |-- src/backtest_engine/
|   |-- adapters/             generic contract, discovery, MOEX and Binance implementations
|   |-- accounting/           cash, futures, margin, and ledger truth
|   |-- core/                 shared models, errors, metrics, and timeframes
|   |-- daily/                daily-bar application runners
|   |-- data/                 datasets, catalogs, lineage, sessions, and quality
|   |-- execution/            order lifecycle and fill simulation
|   |-- intraday/             target and event-driven intraday applications
|   |-- portfolio/            aggregation and synchronized multi-instrument runs
|   |-- reporting/            reports, charts, attribution, and tearsheets
|   |-- research/             experiments, robustness, and acceptance gates
|   |-- strategies/           SDK, runtimes, and plugin boundaries
|   |   `-- workbench/        exchange-neutral UI backend and HTTP API
|   `-- tests/                unit, integration, golden, and packaged smoke tests
|-- continuous_futures_builder/
|-- data_store/
`-- moex_iss_adapter/
```

Generated executables belong to workspace `dist/`; mutable market data belongs to workspace
`data/` and is never mixed with the importable `backtest_engine/src/backtest_engine/data/` package.
Application settings are machine/process defaults, while strategy and experiment parameters remain
inside run manifests.

## Bounded Contexts

### Data Trust

Owns canonical data, dataset validation, gap/roll approvals, catalog binding, quality approvals, dataset
hashes, canonical storage layout helpers, parquet metadata indexing, and reproducible input
manifests. Dataset version records include file hashes, schema hashes, lineage, adapter snapshots,
roll config hashes, and schema migration records.

Adapters own provider transport, raw snapshots and normalization. The engines consume approved
canonical datasets and explicit specifications/calendars; they do not call exchange providers.
See [the market adapter contract](market_adapters.md).

Primary modules:

- `data/catalog.py`
- `data/lifecycle.py`
- `data/store.py`
- `adapters/moex/pipeline/runner.py`
- `adapters/moex/pipeline/stages.py`
- `data/session_gap_audit.py`
- `workbench/datasets.py`
- quality approval logic in event runners

### Market Session And Instrument Specs

Owns exchange/session vocabulary and instrument contract specs.

Primary modules:

- `data/session_model.py`
- `data/instrument_registry.py`
- `adapters/moex/registry.py`
- `adapters/moex/iss/snapshot.py`

Application runners should ask these modules whether a timestamp is tradable and which contract
spec applies to a row. They should not duplicate session windows, holiday parsing, wildcard symbol
matching, or valid date checks.

### Execution And Accounting

Owns bar-based execution timing, execution v3 order lifecycle, order-book state for non-IOC
intraday orders, fill realism controls, synchronized multi-leg fill policy, futures
notional/PnL/cost accounting, spot and perpetual settlement, and multi-currency cash ledgers.
Application runners should route equity cash through ledger runtime instead of maintaining separate
cash truth.

Primary modules:

- `execution/model.py`
- `execution/simulator.py`
- `intraday/event/execution_v3.py`
- `portfolio/multi/execution_v3.py`
- `accounting/futures.py`
- `accounting/futures_marks.py`
- `accounting/ledger.py`
- `accounting/ledger_frames.py`
- `accounting/ledger_runtime.py`

### Backtest Applications

Own orchestration: load data, call strategy plugins, apply domain services, produce events, and
save outputs.

Primary modules:

- `intraday/event/engine.py`
- `portfolio/multi/engine.py`
- `intraday/event/batch/runner.py`
- `portfolio/combine.py`

Application services should stay thin: they load data, call strategy/runtime contracts, apply
domain services, and write outputs. New market rules should be added to domain modules first.

### Reporting And Research

Owns summaries, reports, tearsheets, experiment registry, external validation, dataset benchmarks,
indicator caches, and out-of-sample/robustness analysis. Research validation also owns guardrail
gate artifacts that can reject an experiment formally.

Primary modules:

- `intraday/event/reporting/report.py`
- `research/validation/gate.py`
- `research/experiments/registry.py`
- `research/external_validation.py`
- `research/benchmark.py`
- `research/pack.py`
- `reporting/tearsheet/report.py`
- `strategies/runtime.py`

### Interface Adapters

Owns exchange adapter contracts, CLI parsing, Typer command wiring, and the local research UI.
The Workbench consumes `AdapterInfo.capabilities`; it must not branch on a concrete exchange id.
Market adapters own instrument discovery and dataset construction, while shared catalog, job,
research, and chart APIs remain exchange-neutral.

Primary modules:

- `cli/composition.py`
- `cli/parsing.py`
- `adapters/protocol.py`
- `adapters/registry.py`
- `adapters/moex/adapter.py`
- `workbench/server.py`
- `workbench/jobs.py`
- `workbench/research.py`

Workbench jobs execute in child processes. This makes cancellation real, isolates strategy
failures, and gives the HTTP server a fixed concurrency boundary. Packaged Electron builds use a
frozen Python backend and Electron's embedded Node runtime for JavaScript strategies.

## Decomposition Rules

- Domain modules should not import CLI, Typer, strategy plugins, or report rendering.
- Workbench must depend on `adapters/protocol.py`, never on a concrete exchange package.
- Exchange adapters may depend on shared data contracts but may not import Workbench or runners.
- Application runners may import domain services, but domain services should not call runners.
- Schema modules own table shapes; engines should not carry long Polars schema dictionaries inline.
- File/hash/manifest infrastructure should stay outside strategy and domain calculation code.
- New strategy-facing behavior should be represented in event/decision reports, not hidden only in
  equity curves.
- Strict runtime mode should require strategy metadata, typed params, warmup contracts, and
  no-lookahead strategy context where a runner supports it.

## Growth Watchlist

The source and test tree is kept below 300 lines per Python file. Files above 200 lines are normal
watchlist candidates; split them only when a responsibility boundary is real:

- execution/order lifecycle: keep models, fill rules, and runner integration separate.
- data trust: keep catalog, lifecycle audit, and data store versioning separate.
- strategy runtime: keep plugin loading separate from typed params and lookahead guards.
- reporting modules: separate metrics preparation from Markdown/HTML rendering.
