# Market adapters and engine independence

The engine consumes canonical, approved market data. An adapter owns transport,
raw response preservation, normalization and publication of historical inputs.
No engine calculation selects an exchange or imports its implementation.

## Adapter contract

Implement `backtest_engine.adapters.protocol.MarketDataAdapter`: `info`, `engines`,
`markets`, `boards`, `instruments`, and `build_dataset`. Discover adapters through
the `backtest_engine.market_adapters` package entry point or register a factory
explicitly. The desktop build collects installed entry points and their metadata.

`AdapterCapabilities` declares supported products, timeframes, instrument modes,
optional instrument catalogs, roll rules, delay, sessions and execution defaults.
UI choices and catalog endpoints consume those declarations. Product IDs such as
`cash`, `dated` or `swap` are adapter vocabulary; they need not match engine types.

`DatasetBuildRequest.adapter_id` is explicit. The selected adapter resolves omitted
engine, market and board values from its catalog. Ambiguous selections fail with
the available choices. Cache namespaces and delay use the selected adapter.
Unsupported modes/timeframes/roll rules fail instead of borrowing another provider's
defaults. Custom instrument catalogs expose the declared resource through a callable
whose name replaces resource hyphens with underscores.

Published inputs include a canonical dataset, immutable raw snapshot, manifest,
quality approval, historical instrument specifications and an independent calendar.
The Workbench registers a generic `workbench_dataset.json` declaration, so catalog
recovery does not require a provider-specific directory layout. Queued runs pin
catalog bindings, input hashes and strategy source snapshots.

## Calculation contract

Strategies use public runtime/order APIs and receive confirmed fills. The engine
selects financial behavior using `instrument_type`, not the exchange or asset class.
Current models cover spot assets, futures valued through contract point value,
and linear perpetuals with explicit margin, mark prices and funding inputs.
Inverse/quanto accounting remains unsupported and fails closed during specification resolution,
before orders or financial output. Direct perpetual and paper accounting also reject these
models. Their native payouts, fee/funding units, margin tiers, collateral FX and liquidation
policies require a separate financial model; changing a contract-type flag cannot enable them.

Quantity filters, fees, currencies, dated specifications and session calendars are
inputs. A non-crypto instrument uses the same spot/perpetual model and constraints
as a crypto instrument. A missing currency cannot silently become RUB or USDT;
mixed settlement currencies require an explicit portfolio currency and FX inputs.
The system never equates USD and USDT.

The portfolio ledger is cash/PnL truth for reports. Carried futures exposure is
valued from the preceding mark to the next fill, including close-to-open changes
when orders execute at the open. An unexplained contract change before the
execution interval fails and requires explicit roll fills. Existing declared
in-interval roll policies and their costs remain part of execution inputs.
The former independent `forts_accounting.parquet` calculation has been removed.

Annualization uses an explicit run parameter or calendar declaration. If neither
provides a factor, balances, total return and drawdown remain available; annualized
statistics are null and event/portfolio metrics explain the missing factor.
An explicit adapter reporting convention may supply a factor when producing its
calendar. It is not an engine constant.

## Installation and migration

The core wheel does not require `moex-iss-adapter` or `continuous-futures-builder`.
MOEX support is the `backtest-engine[moex]` extra. Workspace `uv sync` includes the
`market-adapters` development group, keeping the full desktop development setup.
Available-provider listing omits uninstalled optional adapters and requesting one
returns an actionable installation error. A broken installed provider is not hidden.

MOEX defaults live inside its adapter. Explicit `moex-*` CLI commands and lazy legacy
MOEX/Binance import aliases are compatibility boundaries; ordinary engine, CLI and
paper replay imports do not load those providers.
MOEX raw contract selection lives in `adapters/moex/iss/contract_selection.py`;
the common data layer does not interpret ISS contract fields.

Old programmatic callers must provide `adapter_id`, an account currency where it
cannot be resolved, and an annualization factor where needed. Time-based intraday
mode requires an explicit decision time. Generic event/Workbench requests default
to every bar, carry positions between decisions, and cover all dataset dates.
Saved strategies/experiments can continue to specify their previous schedule.

## Regression checks

`tests/test_exchange_independence.py` checks forbidden provider imports/constants,
public build-to-report flow, generic catalog recovery, missing annualization and
a subprocess that blocks MOEX/Binance imports. The subprocess also imports the
generic CLI and paper replay API.

`tests/test_neutral_multi_adapter.py` verifies the same recorded third-venue orders
through the portfolio engine, including long and short futures/perpetual positions.
`tests/test_market_input_resolution.py` checks currency, calendar, timestamp basis,
selected-adapter defaults, CLI currency propagation and non-crypto quantity rules.
Existing Binance/MOEX goldens and ledger reconciliation remain required checks.

## Historical MOEX margin inputs

MOEX builds download NCC market risk rates from
`/iss/rms/engines/futures/objects/limits/<asset>.json`. Requests, responses and SHA-256
digests are preserved under `reports/risk_history/` in each immutable dataset version.
Prefer, refresh, network and offline cache policies are supported. An offline cache
miss never starts a network request.

With `rules_enforcement=audit`, the explicit research model
`ncc_mr1_notional_estimate_v1` uses contract close * contract point value * historical
MR1, rounded upward to RUB 0.01. This estimates margin; it is not exchange or broker GO.
It excludes clearing scenarios, interest risk and client/broker add-ons. The risk
layer rejects positions above the first concentration limit.

Intervals record rate availability and exclusive validity ends. An evening update
cannot affect an earlier decision. Missing weekdays or opening regimes block
the run before strategy execution. Under the recorded research weekend policy,
weekends without a rate snapshot use the last known trading-day rate. Each carried
interval records its source trade date; future Monday rates cannot affect it. Current GO is never copied backward into history.

Estimates produce warning status and appear in the quality report. Require approved
data rejects estimates independently of an approval file. The checkbox and rule
mode are synchronized for adapters declaring rule-archive support.

Exact inputs use `DatasetBuildRequest.rules_path`: a JSON object with an `instruments`
list. Records require exact `symbol`, positive `initial_margin`, and inclusive dates
`valid_from` and `valid_to`. Explicit attributes override known instrument metadata.
Exact dated rules take precedence and require no risk download when they cover the
requested period. Independent-calendar checks still apply.

Rebuild existing datasets to obtain new immutable versions. Existing runs and their
inputs are preserved.
