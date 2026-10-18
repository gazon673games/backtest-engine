"""Host adapter: independent SDK decisions share native execution and one ledger."""

from decimal import Decimal
from pathlib import Path

from strategy_sdk import entry_quantity
from strategy_sdk.catalog_sources import catalog_snapshot, load_catalog_snapshot
from strategy_sdk.time import timeframe_minutes

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.data.bar_availability import row_decision_boundary
from backtest_engine.portfolio.multi.capital import SleeveCapital
from backtest_engine.portfolio.multi.data import instrument_spec_for_row
from backtest_engine.portfolio.multi.execution_window import apply_increase_window
from backtest_engine.portfolio.multi.sdk_history import SDKHistory
from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.sdk_adapter import SDKReplayStrategy
from backtest_engine.strategies.sdk_numbers import native_quantity


class SDKPortfolioStrategy:
    def __init__(self, config, loaded) -> None:
        self.config = config
        self.loaded = loaded
        sleeves = config.strategy.sdk_sleeves
        if {s.asset_id for s in sleeves} != set(loaded):
            raise StrategyPluginError("SDK sleeves must match portfolio legs exactly")
        currencies = {
            p["config"].settlement_currency or p["config"].pnl_currency for p in loaded.values()
        }
        if currencies != {config.portfolio.base_currency}:
            raise StrategyPluginError("SDK sleeves require a shared native settlement currency")
        self.capital = SleeveCapital(
            {s.asset_id: s.initial_capital for s in sleeves},
            config.portfolio.initial_capital,
            config.strategy.capital_policy,
        )
        self.adapters, self.histories, self.sleeves, self.snapshots = {}, {}, {}, {}
        for sleeve in sleeves:
            path = Path(sleeve.strategy_file)
            snapshot = (
                sleeve.source_snapshot
                if sleeve.source_snapshot is not None
                else catalog_snapshot(path.parent, path.stem)
            )
            module = load_catalog_snapshot(snapshot)
            adapter = SDKReplayStrategy(module, path, params=sleeve.params)
            self.adapters[sleeve.asset_id] = adapter
            self.histories[sleeve.asset_id] = SDKHistory(
                loaded[sleeve.asset_id]["frame"],
                adapter.strategy.requirements,
                sleeve.signal_offset_minutes,
            )
            self.sleeves[sleeve.asset_id] = sleeve
            self.snapshots[sleeve.asset_id] = snapshot

    def __call__(self, ctx, params):
        self.capital.redistribute(ctx.current_ts, ctx.previous_contracts, ctx.equity_cash)
        decisions = {}
        for asset in ctx.tradable_assets:
            adapter = self.adapters[asset]
            history = self.histories[asset]
            index = history.advance(ctx.current_ts)
            if index is None:
                continue
            quantity = ctx.previous_contracts[asset]
            budget = self.capital.budgets[asset].equity
            row = ctx.current_bar(asset)
            instrument = instrument_spec_for_row(row, ctx.current_ts, self.loaded[asset]["config"])
            if budget <= 0:
                decisions[asset] = {
                    "target": 0,
                    "contracts": 0,
                    "reason": "sleeve_capital_exhausted",
                }
                continue
            context = IntradayStrategyContext(
                history.frame,
                end_index=index,
                timeframe=adapter.strategy.requirements.timeframe,
                bar_minutes=timeframe_minutes(adapter.strategy.requirements.timeframe),
                previous_position=(quantity > 0) - (quantity < 0),
                position_quantity=quantity,
                instrument_spec=instrument,
                equity_cash=budget,
                available_funds=ctx.equity_cash,
                point_value=instrument.contract_multiplier * instrument.lot_size,
                base_currency=self.config.portfolio.base_currency,
            )
            decision = adapter(context, self.sleeves[asset].params)
            if decision["contracts"] is None:
                raise StrategyPluginError(f"SDK sleeve {asset} must specify contracts")
            # Host risk budget also applies to HOLD and reductions after a loss.
            capped = float(
                entry_quantity(
                    budget,
                    row["close"],
                    adapter.context.instrument,
                    percent=Decimal(str(self.sleeves[asset].max_exposure)) * 100,
                    rounding="down",
                )
            )
            if native_quantity(
                decision["contracts"], adapter.context.instrument.quantity_step
            ) > Decimal(str(capped)):
                decision.update(contracts=capped, reason="sleeve_exposure_cap")
                if capped == 0:
                    decision["target"] = 0
            boundary = row_decision_boundary(
                history.frame.row(index, named=True),
                timeframe_minutes(adapter.strategy.requirements.timeframe),
            )
            apply_increase_window(decision, quantity, boundary, self.sleeves[asset])
            decisions[asset] = decision
        return decisions

    def notify_fill(self, event):
        asset = event["asset_id"]
        adapter = self.adapters[asset]
        # Liquidation may be partial; notify its actual resulting position separately.
        filled = float(event.get("filled_contract_delta", event["contract_delta"]))
        before = float(event["previous_contracts"])
        if filled:
            fill = {
                **event,
                "contract_delta": filled,
                "target_contracts": before + filled,
                "liquidated": False,
            }
            adapter.notify_fill(fill)
        ending = float(event["target_contracts"])
        if ending != before + filled:
            liquidation = {
                **event,
                "contract_delta": ending - before - filled,
                "target_contracts": ending,
                "fill_ts": event["next_ts"],
                "fill_price": event.get("liquidation_price") or event["exit_price"],
                "liquidated": False,
            }
            adapter.notify_fill(liquidation)

    def notify_settlement(self, rows, equity):
        self.capital.settle(rows, equity)

    def finish(self, summary):
        self.capital.reconcile(summary["final_equity_cash"])
