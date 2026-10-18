# Module Map

The Python package is grouped by responsibility rather than by execution command. Prefixes that
previously simulated directories in filenames are now represented by real packages.

| Domain | Location | Owns |
| --- | --- | --- |
| Core | `core/` | Shared models, errors, config parsing, metrics, timeframes |
| Adapters | `adapters/` | Adapter protocol, capabilities, discovery registry |
| MOEX | `adapters/moex/` | MOEX sessions, instrument registry, ISS and dataset pipeline |
| Data | `data/` | Canonical datasets, catalogs, lineage, quality and session model |
| Execution | `execution/` | Orders, lifecycle, fills, latency, queue and ambiguity policy |
| Accounting | `accounting/` | Contract economics, fees, cash, margin, FX and ledger runtime |
| Strategies | `strategies/` | Python/JS runtimes, SDK, typed params and lookahead boundary |
| Intraday | `intraday/` | Target-position and event-driven application services |
| Portfolio | `portfolio/` | Run aggregation and synchronized multi-instrument engine |
| Research | `research/` | Experiments, walk-forward, robustness, benchmarks and gates |
| Reporting | `reporting/` | Charts, tables, attribution, reports and tearsheets |
| Workbench | `workbench/` | Adapter-neutral API, UI, job isolation, history and reports |
| CLI | `cli/` | Thin Typer composition and argument translation |

## Adapter Extension

A future exchange module belongs under `adapters/<exchange>/`. It implements
`adapters.protocol.MarketDataAdapter`, declares `AdapterCapabilities`, and registers one instance in
`adapters.registry`. Instrument discovery and raw-to-canonical conversion remain inside that
adapter. Dataset cataloging, jobs, strategy execution, reports, and UI selection stay shared.

## Compatibility

The root `backtest_engine` package remains the stable public facade. Established imports such as
`backtest_engine.strategy_sdk` and `backtest_engine.intraday_event` are mapped to their new domain
modules. New code should use the organized paths, for example `backtest_engine.strategies.sdk` and
`backtest_engine.intraday.event.engine`.

## Commenting Policy

Package docstrings explain ownership and this map explains dependencies. Function docstrings and
inline comments are reserved for contracts, state transitions, non-obvious market rules, and
failure policy. Self-evident assignments are intentionally not narrated because such comments age
faster than the code they repeat.
