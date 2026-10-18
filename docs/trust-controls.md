# Trust-sensitive crypto and portfolio controls

These features are fail-closed where missing history would make a result misleading. They do not
turn a backtest into a guarantee of future execution.

## Historical Binance rules

`DatasetBuildRequest.rules_path` accepts a JSON archive with effective dates. Use
`rules_enforcement="strict"` for money-sensitive research. Strict mode rejects gaps and snapshots
without filters, maker/taker fees, and (for perpetuals) margin tiers.

```json
{
  "schema_version": 1,
  "venue": "binance",
  "product": "perpetual",
  "symbol": "BTCUSDT",
  "snapshots": [{
    "valid_from": "2024-01-01",
    "valid_to": "2024-06-30",
    "tick_size": 0.1,
    "quantity_step": 0.001,
    "minimum_notional": 5.0,
    "maker_fee_bps": 2.0,
    "taker_fee_bps": 5.0,
    "liquidation_fee_bps": 125.0,
    "margin_tiers": [{
      "notional_floor": 0.0,
      "notional_cap": null,
      "maintenance_margin_rate": 0.004,
      "max_leverage": 125.0
    }]
  }]
}
```

The build writes the versioned `instrument_registry.json`, a separate
`instrument_registry_audit.json`, and hashes both in the dataset manifest. Current exchange rules
are only a fallback and produce a warning under the default audit policy.

## Historical L2 execution

Import book snapshots/deltas with `data_kind="book"` and a local CSV or Parquet `source_path`.
Required fields are:

```text
ts, sequence, previous_sequence, update_type, side, price, size
```

The first update must be a `snapshot`; later `delta` sequence numbers must be contiguous. Configure
an intraday event run with `execution.engine="v3"`, `order_book_path`, `latency_ms`, and
`l2_queue_ahead_fraction`. The engine supports market/limit depth sweeps, maker queue estimates,
post-only, reduce-only, FOK/IOC/GTC, and self-trade prevention. Stop and synthetic bracket orders
are deliberately rejected in L2 mode until an exchange-event model exists for them.

## Shared perpetual cross-margin

Multi-instrument runs with more than one perpetual leg require:

```json
{
  "portfolio": {
    "margin_mode": "cross",
    "collateral_currency": "USDT",
    "initial_collateral": 10000.0,
    "leverage": 5.0,
    "require_margin_tiers": true
  }
}
```

All legs must share one settlement currency. A single wallet covers combined initial and
maintenance margin; funding, trading fees, realized PnL, and liquidation fees are posted to the
portfolio ledger. Liquidation reduces the worst positions until the configured maintenance buffer
is restored.

## Strategy bias audit

Run a standalone gate with:

```bash
uv run backtest strategy-audit \
  --strategy-file strategies/my_strategy.py \
  --dataset-path data/market.parquet \
  --output-path data/audits/my_strategy.json \
  --gate
```

The audit compares full-history exposure with true prefix-only exposure and compares multiple
startup windows for recursive instability. The research pipeline runs the gate automatically in
strict mode for every dataset and every unique set of plugin parameters. Set
`strategy_audit_policy` to `audit` for reports without rejection or `off` only for diagnostics.

## Paper execution

`paper-replay` consumes Parquet events in batches and uses the production strategy contract,
`ExecutionSimulatorV3`, spot/perpetual accounting, funding, fees, margin, liquidation, and ledger.

```bash
uv run backtest paper-replay \
  --config-path configs/paper.json \
  --events-path data/live_capture.parquet \
  --output-path data/paper/result.json
```

`BinanceClosedKlineFeed` is available to applications that need polling of finalized public spot or
perpetual candles. Paper mode never submits an exchange order. A configured checkpoint is replaced
atomically and records positions, equity, fills, open orders, and the last availability timestamp.

## Point-in-time dynamic universe

Set `universe.path` in a multi-instrument config. Parquet and CSV are accepted. Required columns
are `effective_at`, `available_at`, `asset_id`, `status`, and the configured ranking column
(`quote_volume` by default). `available_at` must not precede `effective_at`.

```json
{
  "universe": {
    "path": "data/universe.parquet",
    "ranking_column": "quote_volume",
    "top_n": 20,
    "min_score": 1000000.0,
    "min_price": 0.01,
    "allowlist": [],
    "blocklist": [],
    "removal_policy": "liquidate",
    "require_point_in_time": true
  }
}
```

At each event, selection uses only rows whose effective and publication timestamps are already
known. Strategies receive `universe_assets`, `universe_added`, and `universe_removed`; orders for
non-members are ignored. `liquidate` creates an audited flatten decision at the next executable
event, while `hold` keeps valuing an existing position without allowing new strategy orders.
