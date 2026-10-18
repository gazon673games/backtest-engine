# Historical MVP guide

Archived on 2026-10-08. This records the original daily-only scope; active development rules are in the repository and workspace AGENTS.md files.

# AGENTS.md

## Project

`backtest_engine`

This repository is part of the wider `ohlcv-replay` workspace.

Expected workspace layout:

```text
ohlcv-replay/
  moex_iss_adapter/
  ohlcv-data-store/
  continuous-futures-builder/
  backtest_engine/
```

## Role of this project

`backtest_engine` is responsible for running historical backtests on already prepared, already validated market datasets.

It consumes derived continuous futures Parquet files produced by `continuous-futures-builder`.

It must not download, normalize, repair, or construct market data.

## Project boundary

### This project may do

```text
- read approved continuous futures Parquet datasets
- read quality gate manifests or quality gate result files
- reject datasets that are not approved for research/backtesting
- run simple historical daily-bar strategies
- calculate positions
- calculate close-to-close or next-bar returns
- calculate trades
- calculate equity curves
- calculate basic performance metrics
- save backtest outputs
- provide a Python API
- provide a CLI
- provide tests and fixtures
```

### This project must not do

```text
- call MOEX ISS
- download market data
- normalize raw market data
- create normalized candle datasets
- create metadata snapshots
- build continuous futures
- implement roll calendars
- perform back-adjustment
- repair data gaps
- modify source market data
- modify derived continuous futures data
- silently accept structurally bad datasets
- run on split_required or rejected datasets unless explicitly overridden for diagnostics
- implement live trading
- implement broker integration
- implement order routing
- implement intraday execution simulation in the MVP
- implement web APIs or dashboards in the MVP
```

## Upstream projects

### `moex_iss_adapter`

Responsible for:

```text
MOEX ISS API -> raw JSON -> normalized Parquet -> metadata snapshots -> quality reports
```

`backtest_engine` must not depend on MOEX ISS directly.

### `ohlcv-data-store`

Responsible for:

```text
local Parquet discovery
schema validation
read API for candles and metadata
CLI inspection/validation
```

`backtest_engine` may use `ohlcv-data-store` only as a read/validation helper if useful, but it must not change it without explicit instruction.

### `continuous-futures-builder`

Responsible for:

```text
concrete futures candles + metadata -> contract chain -> roll calendar -> continuous futures series
```

`backtest_engine` consumes its derived output.

Important known derived datasets:

```text
USDRUB_CONT:
  status = approved
  type = control dataset
  source family = Si*
  period ≈ 2015-01-05 -> 2026-07-03

CNYRUB_CONT:
  status = approved
  type = main CNY/RUB research candidate
  source family = CR*
  period ≈ 2022-05-12 -> 2026-07-03

CNYRUB_CONT_LONG:
  status = split_required
  type = long CY* + CR* candidate
  period ≈ 2015-03-17 -> 2026-07-03
  known issue = structural data gap at CYH0 -> CYM2
  reason = source_coverage_gap_days > threshold and roll_gap_pct > threshold
```

`CNYRUB_CONT_LONG` must not be used for normal backtests as one continuous series.

## Data-quality rule

The backtest engine must enforce a strict data-quality gate before running a normal backtest.

Allowed status:

```text
approved
```

Blocked statuses:

```text
split_required
rejected
warning unless explicitly allowed
unknown
missing
```

If a dataset has no quality gate result or manifest entry, the engine must fail closed by default.

Do not infer approval from file existence.

Do not infer approval from schema validity alone.

A Parquet file can be valid and still not acceptable for backtesting.

## Expected input data

The MVP should read daily continuous futures Parquet files from paths like:

```text
../continuous-futures-builder/data/derived/continuous_futures/exchange=MOEX/engine=futures/market=forts/continuous_symbol=CNYRUB_CONT/timeframe=1d/adjustment_mode=unadjusted/part-000.parquet
```

or:

```text
../continuous-futures-builder/data/derived/continuous_futures/exchange=MOEX/engine=futures/market=forts/continuous_symbol=CNYRUB_CONT/timeframe=1d/adjustment_mode=back_adjusted/part-000.parquet
```

The quality gate manifest may be located at:

```text
../continuous-futures-builder/data/reports/continuous_futures/quality_gate_manifest.json
```

Generated `data/` directories are usually ignored and must not be committed unless explicitly instructed.

## Expected continuous futures schema

The input continuous futures Parquet should contain at least:

```text
source
exchange
engine
market
continuous_symbol
timeframe
ts
open
high
low
close
volume
value
currency
active_contract
active_board
source_contract_symbol
roll_id
roll_flag
roll_date
adjustment_mode
adjustment_factor
```

The backtest engine may validate required columns before running.

For the MVP, assume:

```text
timeframe = 1d
one row per trading date
ts sorted ascending
OHLC prices are numeric
close is the primary signal and return price
```

## MVP assumptions

The first version should be deliberately simple.

Use daily bars only.

Preferred initial execution model:

```text
signal is computed at close[t]
position is applied from close[t] to close[t+1]
returns are close-to-close
no intraday fills
no partial fills
no order book
no bid/ask spread
no slippage
no commission unless explicitly configured
```

This avoids look-ahead bias.

The engine must make timing assumptions explicit in code, docs, and CLI output.

## MVP strategy scope

The MVP may include only simple deterministic strategies, for example:

```text
- buy and hold
- close-to-close momentum
- moving average crossover
```

Do not implement complex strategy frameworks prematurely.

Do not implement parameter optimization in the first MVP unless explicitly requested.

Do not implement portfolio-level multi-asset allocation in the first MVP unless explicitly requested.

## Look-ahead bias prevention

Never use future data to make a decision for the current bar.

Bad:

```text
use close[t+1] to decide position[t+1]
```

Acceptable MVP convention:

```text
compute signal[t] using data up to and including close[t]
apply position[t] to return from close[t] to close[t+1]
```

The output should include enough columns to audit this timing.

Suggested output columns:

```text
ts
close
signal
position
return_1d
strategy_return
equity
```

## Backtest output

Backtest runs should save outputs under:

```text
data/backtests/<run_id>/
```

Suggested files:

```text
config.json
equity.parquet
equity.csv
trades.parquet
trades.csv
metrics.json
dataset_quality.json
```

Do not commit generated backtest outputs.

## Basic metrics

The MVP should calculate at least:

```text
start_ts
end_ts
bar_count
total_return
annualized_return
annualized_volatility
sharpe
max_drawdown
trade_count
win_rate if trades are available
exposure
```

Use simple daily assumptions.

If annualization is used, document the trading-days-per-year assumption, for example:

```text
252
```

Metrics must be deterministic and tested.

## CLI expectations

The project should provide a CLI.

Possible command shape:

```bash
uv run backtest run \
  --dataset-path <PATH_TO_PARQUET> \
  --quality-manifest <PATH_TO_QUALITY_GATE_MANIFEST> \
  --strategy moving_average_crossover \
  --fast-window 20 \
  --slow-window 60 \
  --output-root data/backtests \
  --run-id cnyrub_ma_20_60
```

Also useful:

```bash
uv run backtest inspect-dataset \
  --dataset-path <PATH_TO_PARQUET> \
  --quality-manifest <PATH_TO_QUALITY_GATE_MANIFEST>
```

The exact CLI can differ, but it must be documented.

## Config

Prefer explicit config files for reproducible runs once the CLI works.

Possible config format:

```toml
[dataset]
path = "../continuous-futures-builder/data/derived/continuous_futures/exchange=MOEX/engine=futures/market=forts/continuous_symbol=CNYRUB_CONT/timeframe=1d/adjustment_mode=back_adjusted/part-000.parquet"
quality_manifest = "../continuous-futures-builder/data/reports/continuous_futures/quality_gate_manifest.json"

[strategy]
name = "moving_average_crossover"
fast_window = 20
slow_window = 60

[execution]
timing = "signal_at_close_apply_next_close"
commission_bps = 0.0
slippage_bps = 0.0

[output]
root = "data/backtests"
run_id = "cnyrub_ma_20_60"
```

## Testing requirements

Tests must not depend on real sibling-project data.

Use small fixture Parquet files generated inside tests or committed under `tests/fixtures`.

Test at least:

```text
- approved manifest allows a dataset
- rejected manifest blocks a dataset
- split_required manifest blocks a dataset
- missing manifest entry blocks by default
- required column validation
- sorted timestamp validation
- no look-ahead timing convention
- buy-and-hold returns
- moving-average crossover signals
- equity calculation
- metrics calculation
- CLI smoke test
```

## Style and implementation

Use Python 3.12+.

Prefer:

```text
polars
pyarrow
pydantic
typer
pytest
ruff
```

Keep modules small.

Avoid hidden global state.

Avoid network calls in tests.

Avoid nondeterministic timestamps in test outputs unless injected or fixed.

Use typed models for:

```text
DatasetQualityStatus
BacktestConfig
StrategyConfig
ExecutionConfig
BacktestResult
PerformanceMetrics
```

## Documentation

README should explain:

```text
- what this project does
- what this project does not do
- required input data
- quality-gate requirement
- supported strategies
- execution timing assumptions
- example commands
- output files
- known limitations
```

## Git rules

Commit source code, tests, docs, and small fixtures.

Do not commit:

```text
data/
large generated Parquet outputs
large generated CSV outputs
IDE files
local cache files
.venv/
```

Before committing, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Final report expectation for agents

When completing a task, report:

```text
1. What was implemented
2. What was intentionally not implemented
3. Input datasets used
4. Whether quality gate enforcement is active
5. Strategy/execution assumptions
6. Output paths
7. Metrics produced
8. Test/lint results
9. Git commit hash
10. Recommended next step
```

## Strategy SDK integration

- Author new reusable algorithms in the configured external `strategy-sdk` catalog.
  Read its AGENTS.md and README.md; use only the standard Python SDK contract.
  Find the folder via strategies.sdk_catalog_path or WORKBENCH_SDK_CATALOG_PATH.
- Workbench exposes SDK_CATALOG_STRATEGY references. Preserve the snapshot of the
  strategy and helper modules for each queued run and parameter sweep. Never reload
  code or change parameters during an active run; never silently fall back to the wheel.

- Shipped strategies use strategy-sdk from the pinned vendor wheel. Keep the SDK independent of this host.
- The SDK adapter receives bounded finalized bars, signed actual position quantity and confirmed fills.
- Preserve quality gates, causal scheduling, execution, accounting and per-run state isolation.
- Workbench Python source uses script_language=python; saved records use language=python.
- Preserve legacy custom JS plugins and migrate saved records only by matching source hash with a backup.
- SDK package changes require a new wheel, uv.lock update and the PyInstaller hook for catalog source fingerprints.

- Shared SDK source editing must preserve revision conflict checks, history backups,
  local-only write endpoints and pinned snapshots for queued runs. Editing a Python
  draft does not publish it; only explicit "Save common SDK file" updates the catalog.
- Use SDK catalog_sources and source_files for shared loading and safe writes. Do not
  duplicate their dependency resolution in this host or import TradeRelay modules.

- Open saved SDK presets as actual Python source in Workbench. Selector references
  remain a compatibility/storage format, never the sole editable representation.
  Preserve all helper files and variant defaults when saving an edited preset.
