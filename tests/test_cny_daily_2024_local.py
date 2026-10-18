"""Opt-in regression for an approved local 2024 CNY dataset, never a network test."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.workbench.models import WorkbenchRunRequest
from backtest_engine.workbench.runner import run_workbench_backtest

REQUEST_PATH = os.environ.get("CNY_2024_REQUEST_PATH")
EXPECTED = json.loads(
    (Path(__file__).parent / "fixtures/cny_daily_2024_margin_call.json").read_text()
)


@pytest.mark.skipif(
    not REQUEST_PATH, reason="CNY_2024_REQUEST_PATH enables the local market control"
)
def test_2024_daily_cny_accumulation_stops_at_the_same_margin_call(tmp_path: Path) -> None:
    request = WorkbenchRunRequest.model_validate_json(
        Path(REQUEST_PATH).read_text(encoding="utf-8")
    )
    assert (
        hashlib.sha256(request.dataset_path.read_bytes()).hexdigest() == EXPECTED["dataset_sha256"]
    )
    for field, expected in EXPECTED["inputs"].items():
        assert hashlib.sha256(getattr(request, field).read_bytes()).hexdigest() == expected
    assert request.initial_capital == EXPECTED["assumptions"]["initial_capital"]
    assert request.commission_bps == EXPECTED["assumptions"]["commission_bps"]
    request.output_root, request.run_id = tmp_path / "runs", "yearly_cny_control"
    with pytest.raises(DatasetValidationError, match="Margin call after settlement") as failure:
        run_workbench_backtest(request)
    assert str(failure.value) == EXPECTED["error"]
    traceback = failure.value.__traceback__
    # The public failure contains balances; its engine frame also identifies the exact rejected bar.
    while (
        traceback is not None and traceback.tb_frame.f_code.co_name != "build_intraday_event_events"
    ):
        traceback = traceback.tb_next
    assert traceback is not None
    state = traceback.tb_frame.f_locals
    assert str(state["current_ts"]) == EXPECTED["stop"]["ts"]
    assert str(state["fill_ts"]) == EXPECTED["stop"]["fill_ts"]
    assert state["current_contracts"] == EXPECTED["stop"]["contracts"]
    assert state["ledger_runtime"].equity_cash == EXPECTED["stop"]["equity_cash"]
