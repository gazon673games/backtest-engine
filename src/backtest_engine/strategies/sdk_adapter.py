"""Per-run adapter for catalog entries and trusted Python SDK strategy files."""

import hashlib
import json
from decimal import Decimal

from strategy_sdk import Fill, ParameterError, Strategy, __version__
from strategy_sdk import StrategyMetadata as SDKMetadata
from strategy_sdk.catalog import load_catalog_strategy
from strategy_sdk.time import normalize_timeframe, timeframe_minutes

from backtest_engine.core.exceptions import DatasetValidationError, StrategyPluginError
from backtest_engine.strategies.intraday_api import IntradayStrategyContext
from backtest_engine.strategies.runtime import StrategyMetadata, StrategyParamSpec
from backtest_engine.strategies.sdk_context import localized, sdk_context
from backtest_engine.strategies.sdk_numbers import native_quantity
from backtest_engine.strategies.sdk_plots import plot_features


class SDKReplayStrategy:
    def __init__(self, source, path, *, daily=False, params=None):
        self.source = source
        self.path = path
        self.daily = daily
        self.strategy = None
        self.state = None
        self.context = None
        self.params_key = None
        source_bytes = path.read_bytes() if isinstance(source, str) else source.__strategy_source__
        self.source_revision = hashlib.sha256(source_bytes + __version__.encode()).hexdigest()
        self.fill_ids = set()
        self.__strategy_call_mode__ = "positional"
        try:
            defaults = self.bind(params or {})
        except ParameterError as error:
            raise DatasetValidationError(str(error)) from error
        self.__strategy_metadata__ = replay_metadata(defaults)
        self.__strategy_fingerprint__ = defaults.revision
        self.__strategy_warmup_bars__ = 0  # SDK owns warmup and its error/skip/flat policy.
        self.__strategy_param_specs__ = self.__strategy_metadata__.params
        self.__strategy_sdk__ = True

    def bind(self, params):
        if not isinstance(self.source, str):
            params = {**getattr(self.source, "__strategy_defaults__", {}), **params}
        key = json.dumps(params, sort_keys=True, default=str)
        if self.params_key == key:
            return self.strategy
        if self.state is not None:
            raise StrategyPluginError("SDK parameters cannot change during an active run")
        if isinstance(self.source, str):
            strategy = load_catalog_strategy(self.source, params)
        else:
            module = self.source
            strategy = Strategy(
                module.STRATEGY_METADATA,
                params,
                module.data_requirements,
                source_revision=self.source_revision,
                generate_signal=module.generate_signal,
                initialize=getattr(module, "initialize", None),
                on_fill=getattr(module, "on_fill", None),
            )
        self.strategy = strategy
        self.params_key = key
        self.__strategy_fingerprint__ = strategy.revision
        return strategy

    def __call__(self, ctx, params):
        try:
            if any(spec.name == "timeframe" for spec in self.strategy.metadata.params):
                params = {
                    "timeframe": normalize_timeframe(
                        getattr(ctx, "timeframe", self.strategy.requirements.timeframe)
                    ),
                    **params,
                }
            strategy = self.bind(params)
            if self.daily:
                return self.daily_signals(ctx)
            return self.evaluate(ctx, strategy)
        except StrategyPluginError:
            raise
        except Exception as error:
            raise StrategyPluginError(f"SDK strategy failed: {error}") from error

    def evaluate(self, ctx, strategy):
        context = sdk_context(ctx, strategy, self.state)
        if (
            strategy.requirements.decision_times
            and context.current_time not in strategy.requirements.decision_times
        ):
            if ctx.position_quantity is None and ctx.previous_position:
                raise StrategyPluginError("SDK HOLD requires the actual position quantity")
            return {
                "target": context.previous_position,
                "contracts": float(abs(context.position_quantity)),
                "reason": "outside_decision_time",
                "features": {},
            }
        result = strategy.evaluate(context)
        if result.decision.kind in {"HOLD", "SKIP"}:
            if ctx.position_quantity is None and ctx.previous_position:
                raise StrategyPluginError("SDK HOLD/SKIP requires the actual position quantity")
            target = context.previous_position
            # Validated SDK numbers remove float noise, but HOLD must retain the
            # exact native inventory so it cannot create a microscopic order.
            actual = ctx.position_quantity
            contracts = (
                abs(Decimal(str(actual))) if actual is not None else abs(context.position_quantity)
            )
        else:
            target = result.decision.target
            contracts = result.decision.contracts
        self.state = result.state
        self.context = context
        return {
            "target": target,
            "contracts": float(contracts) if contracts is not None else None,
            "reason": result.decision.reason,
            "features": plot_features(strategy.metadata.name, result.decision.features),
            "skip_unaffordable_add": result.decision.skip_unaffordable_add,
        }

    def daily_signals(self, ctx):
        frame = ctx.frame()
        if frame.height == 0:
            return []
        signals = []
        for index in range(frame.height):
            visible = IntradayStrategyContext(
                frame,
                end_index=index,
                timeframe=self.strategy.requirements.timeframe,
                bar_minutes=timeframe_minutes(self.strategy.requirements.timeframe),
                previous_position=signals[-1] if signals else 0,
                position_quantity=signals[-1] if signals else 0,
            )
            result = self.evaluate(visible, self.strategy)
            signals.append(float(result["target"]))
        return signals

    def notify_fill(self, event):
        if self.context is None or self.state is None:
            raise StrategyPluginError("SDK fill arrived before a strategy decision")
        delta = Decimal(str(event.get("contract_delta", 0)))
        quantity = abs(delta) if "contract_delta" in event else Decimal(str(event["quantity"]))
        side = "BUY" if delta > 0 else "SELL"
        if "contract_delta" not in event:
            side = str(event["side"]).upper()
        position = event.get("target_contracts", event.get("position"))
        if event.get("liquidated"):
            quantity = Decimal(str(event["liquidation_quantity"]))
            position = 0
            side = "SELL" if event["previous_contracts"] > 0 else "BUY"
        zone = self.context.current_ts.tzinfo
        at = localized(event.get("fill_ts") or event["ts"], zone)
        symbol = str(event.get("active_contract") or self.context.active_contract)
        identifier = str(
            event.get("fill_id")
            or event.get("id")
            or (event.get("order_id"), symbol, at, side, quantity, position)
        )
        if identifier in self.fill_ids:
            raise StrategyPluginError("Duplicate SDK fill event")
        fill = Fill(
            identifier,
            at,
            symbol,
            side,
            native_quantity(quantity, self.context.instrument.quantity_step),
            event.get("liquidation_price")
            if event.get("liquidated")
            else event.get("fill_price", event.get("price")),
            position_after=native_quantity(position, self.context.instrument.quantity_step)
            if position is not None
            else None,
            liquidated=event.get("liquidated", False),
        )
        self.state = self.strategy.apply_fill(self.state, fill)
        self.fill_ids.add(identifier)


def replay_metadata(strategy):
    requirements = strategy.requirements
    specs = tuple(
        StrategyParamSpec(
            spec.name,
            spec.param_type,
            default=str(strategy.parameters[spec.name])
            if spec.name == "decision_time"
            else str(spec.default)
            if isinstance(spec.default, Decimal)
            else spec.default,
            required=spec.required,
            min_value=spec.min_value,
            max_value=spec.max_value,
            choices=spec.choices,
        )
        for spec in strategy.metadata.params
    )
    frame = requirements.timeframe
    timeframe = frame[1:] + {"M": "m", "H": "h", "D": "d"}[frame[0]]
    times = tuple(clock.isoformat(timespec="minutes") for clock in requirements.decision_times)
    return StrategyMetadata(
        strategy.metadata.name,
        strategy.metadata.version,
        params=specs,
        timeframe=timeframe,
        schedule="times" if times else "every-bar",
        decision_times=times,
        decision_tolerance_minutes=0,
        missing_decision_policy=requirements.missing_decision_policy,
        history_bars=requirements.history_bars,
        decision_time_param="decision_time"
        if "decision_time" in strategy.parameters and len(times) == 1
        else None,
    )


def sdk_plugin(module, path, *, daily=False, params=None):
    snapshot = getattr(module, "SDK_CATALOG_SNAPSHOT", None)
    reference = getattr(module, "SDK_CATALOG_STRATEGY", None)
    if snapshot is not None or reference is not None:
        from backtest_engine.core.process_settings import configured_sdk_catalog
        from backtest_engine.strategies.sdk_catalog_sources import (
            catalog_reference,
            catalog_snapshot,
            load_catalog_snapshot,
        )

        if snapshot is None:
            if catalog_reference(module.__strategy_source__.decode("utf-8-sig")) != reference:
                raise StrategyPluginError("SDK_CATALOG_STRATEGY must be the entire source")
            snapshot = catalog_snapshot(configured_sdk_catalog(), reference)
        return SDKReplayStrategy(load_catalog_snapshot(snapshot), path, daily=daily, params=params)
    catalog = getattr(module, "SDK_STRATEGY", None)
    if catalog is not None:
        if not isinstance(catalog, str):
            raise StrategyPluginError("SDK_STRATEGY must name a catalog entry")
        return SDKReplayStrategy(catalog, path, daily=daily, params=params)
    if isinstance(getattr(module, "STRATEGY_METADATA", None), SDKMetadata):
        return SDKReplayStrategy(module, path, daily=daily, params=params)
    return None
