from __future__ import annotations

import json

import pytest
from test_binance_workbench import SeriesTransport, build_request, run_request

from backtest_engine.adapters.binance.adapter import BinanceAdapter
from backtest_engine.core.exceptions import DatasetValidationError
from backtest_engine.workbench.runner import run_workbench_backtest


@pytest.mark.parametrize(
    "field", ["minimum_quantity", "liquidation_fee_bps", "margin_tiers.max_leverage"]
)
def test_strict_history_cannot_inherit_missing_rules_from_current_metadata(tmp_path, field):
    request = build_request(tmp_path, "perpetual")
    archive = json.loads(request.rules_path.read_text(encoding="utf-8"))
    snapshot = archive["snapshots"][0]
    if "." in field:
        snapshot["margin_tiers"][0].pop("max_leverage")
    else:
        snapshot.pop(field)
    request.rules_path.write_text(json.dumps(archive), encoding="utf-8")
    transport = SeriesTransport()
    with pytest.raises(DatasetValidationError, match=field):
        BinanceAdapter(transport).build_dataset(request)
    assert not any(path.endswith("klines") for _, path, _ in transport.calls)


def test_disabled_rule_audit_does_not_approve_incomplete_research_inputs(tmp_path):
    request = build_request(tmp_path, rules_path=None, rules_enforcement="off")
    result = BinanceAdapter(SeriesTransport()).build_dataset(request)
    assert result.approval_status == "warning"
    with pytest.raises(DatasetValidationError, match="not approved"):
        run_workbench_backtest(run_request(result, tmp_path))
