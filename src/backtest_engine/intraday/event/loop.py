from __future__ import annotations

from typing import Any

import polars as pl

from backtest_engine.data.bar_availability import row_available_at, row_open_ts
from backtest_engine.execution.model import execution_interval
from backtest_engine.intraday.event.calculations import (
    build_instrument_state,
    build_pnl_state,
    build_position_state,
)
from backtest_engine.intraday.event.config import IntradayEventBacktestConfig
from backtest_engine.intraday.event.costs import calculate_event_costs
from backtest_engine.intraday.event.data import resolve_event_config
from backtest_engine.intraday.event.decisions import (
    apply_strategy_decision,
    apply_volume_state,
    initial_decision_state,
)
from backtest_engine.intraday.event.lifecycle import finalize_event_run, notify_event_fill
from backtest_engine.intraday.event.loop_support import (
    EventProgressCallback,
    cached_instrument_spec,
    emit_progress,
    load_event_runtime,
)
from backtest_engine.intraday.event.orders import (
    event_loop_stop,
    resolved_volume_result,
)
from backtest_engine.intraday.event.output_buffer import EventFrameBuffer
from backtest_engine.intraday.event.risk_safety import (
    validate_event_after_settlement,
    validate_event_post_fill,
)
from backtest_engine.intraday.event.row_source import FrameRows
from backtest_engine.intraday.event.rows import build_event_row
from backtest_engine.intraday.event.schedule import (
    build_call_schedule,
    row_session_eligible_for_event,
)
from backtest_engine.intraday.event.schemas import events_schema
from backtest_engine.intraday.event.settlement import EventAccountingRuntime
from backtest_engine.intraday.event.settlement_state import apply_settlement_state
from backtest_engine.intraday.event.valuation import EventValuation


def build_intraday_event_events(
    config: IntradayEventBacktestConfig,
    df: pl.DataFrame,
    dataset_validation: dict[str, Any],
    progress_callback: EventProgressCallback | None = None,
    ledger_snapshots: list[dict[str, Any]] | None = None,
) -> pl.DataFrame:
    config = resolve_event_config(config, dataset_validation)
    rows = FrameRows(df)
    call_schedule = build_call_schedule(
        rows,
        config.execution,
        config.session,
        int(dataset_validation["bar_minutes"]),
    )
    generate_signal, execution_book = load_event_runtime(config)
    valuation = EventValuation.create(config, row_available_at(rows[0]) if rows else None)
    ledger_runtime = valuation.ledger
    accounting_runtime = EventAccountingRuntime(ledger_runtime)
    current_signal = 0.0
    current_contracts = 0.0
    current_equity_cash = ledger_runtime.equity_cash
    event_buffer = EventFrameBuffer(events_schema(df.schema["ts"]))
    instrument_spec_cache: dict[tuple[Any, ...], Any] = {}
    indicator_cache = {}
    total_rows = event_loop_stop(df.height, config.execution)
    progress_stride = max(1, total_rows // 200)
    for index in range(total_rows):
        row = rows[index]
        next_row = rows[index + 1]
        current_ts = row_available_at(row)
        if current_ts.date() < config.execution.from_date:
            emit_progress(progress_callback, index, total_rows, progress_stride, None)
            continue

        interval = execution_interval(
            rows, index, config.execution.timing, config.execution.data_mode
        )
        fill_ts = interval.fill_ts
        fill_price = interval.fill_price
        exit_price = interval.exit_price
        fill_row = rows[interval.fill_index]
        exit_row = rows[interval.exit_index]
        config = valuation.activate(config, current_ts)
        decision_equity_cash = ledger_runtime.decision_equity_at(current_ts)
        contract_roll = str(fill_row["active_contract"]) != str(exit_row["active_contract"])
        decision_instrument_spec = cached_instrument_spec(
            instrument_spec_cache, row, current_ts, config.instrument
        )
        instrument_spec = cached_instrument_spec(
            instrument_spec_cache, fill_row, fill_ts, config.instrument
        )
        if execution_book is not None:
            execution_book.simulator.quantity_step = instrument_spec.quantity_step
        previous_signal = current_signal
        previous_contracts = current_contracts
        decision_state = initial_decision_state(
            index,
            call_schedule,
            current_signal,
            current_contracts,
            config,
            strategy_eligible=row_session_eligible_for_event(row_open_ts(row), config.session)
            and current_ts.date() not in call_schedule.skipped_dates,
        )
        if decision_state["missing_decision"]:
            current_signal = 0.0
            current_contracts = 0.0
        elif decision_state["strategy_called"]:
            fill_price, current_signal, current_contracts = apply_strategy_decision(
                generate_signal=generate_signal,
                df=df,
                rows=rows,
                index=index,
                dataset_validation=dataset_validation,
                config=config,
                call_time=call_schedule.call_time(index),
                previous_signal=previous_signal,
                previous_contracts=previous_contracts,
                fill_price=fill_price,
                row=row,
                next_row=next_row,
                instrument_spec=decision_instrument_spec,
                fill_instrument_spec=instrument_spec,
                equity_cash=current_equity_cash,
                decision_equity_cash=decision_equity_cash,
                decision_cash=ledger_runtime.decision_cash_at(current_ts),
                decision_state=decision_state,
                execution_book=execution_book,
                indicator_cache=indicator_cache,
            )
        elif execution_book is not None and execution_book.has_active_orders:
            pending_result = execution_book.process_pending(
                rows=rows,
                index=index,
                base_fill_price=fill_price,
            )
            if pending_result["filled"]:
                decision_state["order_status"] = pending_result["status"]
                decision_state["order_filled"] = True
                decision_state["requested_signal"] = float(pending_result["target_signal"])
                decision_state["requested_contracts"] = float(pending_result["requested_contracts"])
                fill_price = float(pending_result["fill_price"])
                current_signal = float(pending_result["target_signal"])
                current_contracts = float(pending_result["target_contracts"])
                decision_state["signal_reason"] = "pending_order_filled"

        config = valuation.activate(config, fill_ts)
        volume_result = resolved_volume_result(
            execution=config.execution,
            allow_fractional_contracts=config.portfolio.allow_fractional_contracts,
            confirmed=execution_book is not None,
            previous_signal=previous_signal,
            previous_contracts=previous_contracts,
            target_signal=current_signal,
            target_contracts=current_contracts,
            requested_contracts=decision_state["requested_contracts"],
            row=row,
            next_row=next_row,
        )
        current_signal, current_contracts = apply_volume_state(
            decision_state,
            volume_result,
            previous_contracts,
        )
        position_state = build_position_state(
            fill_row,
            previous_signal,
            previous_contracts,
            current_signal,
            current_contracts,
        )
        instrument_state = build_instrument_state(
            instrument_spec,
            fill_price,
            position_state,
            config,
        )
        validate_event_post_fill(
            config,
            instrument_spec,
            instrument_state,
            current_contracts,
            current_equity_cash,
            previous_contracts=previous_contracts,
            fill_price=fill_price,
        )
        previous_equity_cash = current_equity_cash
        cost_parts, crypto_settlement = calculate_event_costs(
            fill_row=fill_row,
            exit_row=exit_row,
            fill_price=fill_price,
            exit_price=exit_price,
            current_contracts=current_contracts,
            position_state=position_state,
            instrument_state=instrument_state,
            instrument=instrument_spec,
            config=config,
            previous_equity_cash=previous_equity_cash,
        )
        if crypto_settlement is not None:
            exit_price = crypto_settlement.effective_exit_price
        pnl_state = build_pnl_state(
            config=config,
            current_signal=current_signal,
            current_contracts=current_contracts,
            fill_price=fill_price,
            exit_price=exit_price,
            instrument_state=instrument_state,
            cost_parts=cost_parts,
            previous_equity_cash=previous_equity_cash,
            contract_roll=contract_roll,
        )
        settlement_result = accounting_runtime.settle_interval(
            config=config,
            instrument=instrument_spec,
            fill_row=fill_row,
            exit_row=exit_row,
            fill_ts=fill_ts,
            valuation_ts=interval.exit_ts,
            fill_price=fill_price,
            mark_price=exit_price,
            quantity_delta=position_state["contract_delta"],
            quantity=current_contracts,
            previous_equity_cash=previous_equity_cash,
            pnl_state=pnl_state,
            cost_parts=cost_parts,
            crypto_settlement=crypto_settlement,
        )
        current_equity_cash = settlement_result.equity_cash
        crypto_settlement = settlement_result.crypto_settlement
        exit_price = settlement_result.effective_exit_price
        current_signal, current_contracts = apply_settlement_state(
            settlement_result,
            current_signal=current_signal,
            current_contracts=current_contracts,
            previous_contracts=previous_contracts,
            position_state=position_state,
            instrument_state=instrument_state,
        )
        validate_event_after_settlement(
            config,
            instrument_spec,
            instrument_state,
            current_contracts,
            current_equity_cash,
        )
        pnl_state["current_equity_cash"] = current_equity_cash
        pnl_state["current_equity"] = current_equity_cash / config.portfolio.initial_capital
        event_row = build_event_row(
            config=config,
            row=fill_row,
            exit_row=exit_row,
            current_ts=current_ts,
            fill_ts=fill_ts,
            fill_price=fill_price,
            exit_price=exit_price,
            call_time=call_schedule.call_time(index),
            missing_decision_times=call_schedule.missing_closes.get(index, []),
            decision_state=decision_state,
            position_state=position_state,
            instrument_state=instrument_state,
            pnl_state=pnl_state,
            cost_parts=cost_parts,
            contract_roll=contract_roll,
            crypto_settlement=crypto_settlement,
        )
        event_buffer.append(event_row)
        notify_event_fill(generate_signal, event_row)
        if crypto_settlement is not None and crypto_settlement.liquidation_status == "full":
            current_signal = 0.0
            current_contracts = 0.0
        emit_progress(
            progress_callback,
            index,
            total_rows,
            progress_stride,
            event_row,
        )
    return finalize_event_run(
        generate_signal,
        event_buffer,
        ledger_runtime,
        ledger_snapshots,
        current_signal,
        current_contracts,
        config=config,
    )
