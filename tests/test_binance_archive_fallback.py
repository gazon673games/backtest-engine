"""Invalid public archives are replaced only under an explicit, audited policy."""

from types import SimpleNamespace

import pytest

from backtest_engine.adapters.binance.stream import history_rows
from backtest_engine.adapters.models import DatasetBuildRequest
from backtest_engine.core.exceptions import DatasetValidationError


def history(monkeypatch, tmp_path, *, fallback="missing_only", source="auto", invalid=True):
    archive_rows = [[1000, "1", "1", "1", "1", "0", 999 if invalid else 1999]]
    rest_rows = [[2000, "1", "1", "1", "1", "1", 2999]]
    calls, sources = [], []
    monkeypatch.setattr(
        "backtest_engine.adapters.binance.stream.archive_klines",
        lambda *args, **kwargs: (archive_rows, {"path": "immutable.zip", "sha256": "original"}),
    )

    def download(*args, **kwargs):
        calls.append(kwargs)
        return rest_rows

    monkeypatch.setattr("backtest_engine.adapters.binance.stream.download_klines", download)
    request = DatasetBuildRequest(
        adapter_id="binance",
        engine="spot",
        security="EURUSDT",
        from_date="2020-12-01",
        till_date="2020-12-31",
        history_source=source,
        archive_fallback_policy=fallback,
    )
    transport = SimpleNamespace(transport=object(), root=tmp_path)
    rows = history_rows(request, transport, sources, "2020-12", 0, 3000, 3000, 4000)
    return rows, archive_rows, sources, calls


def test_default_does_not_change_invalid_archive(monkeypatch, tmp_path):
    rows, original, sources, calls = history(monkeypatch, tmp_path)
    assert rows == original
    assert not calls
    assert sources == [{"path": "immutable.zip", "sha256": "original"}]


def test_explicit_policy_replaces_entire_month_and_records_invalid_raw_rows(monkeypatch, tmp_path):
    rows, original, sources, calls = history(monkeypatch, tmp_path, fallback="invalid_timestamps")
    assert rows[0][0] == 2000
    assert len(calls) == 1
    assert (calls[0]["start_ms"], calls[0]["end_ms"]) == (0, 3000)
    assert sources[0]["sha256"] == "original"
    assert sources[0]["status"] == "excluded_invalid_timestamps"
    assert sources[0]["invalid_rows"] == original
    assert sources[0]["replacement_policy"] == "download_entire_month_from_rest"


def test_archive_only_cannot_silently_switch_to_rest(monkeypatch, tmp_path):
    with pytest.raises(DatasetValidationError, match="require auto REST fallback"):
        history(monkeypatch, tmp_path, fallback="invalid_timestamps", source="archive")


def test_valid_archive_does_not_need_rest(monkeypatch, tmp_path):
    rows, original, sources, calls = history(
        monkeypatch,
        tmp_path,
        fallback="invalid_timestamps",
        invalid=False,
    )
    assert rows == original
    assert not calls
    assert "status" not in sources[0]
