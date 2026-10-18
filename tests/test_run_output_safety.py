from pathlib import Path

import pytest

from backtest_engine.core.run_output import publish_run_output


def test_run_output_preserves_existing_evidence(tmp_path: Path) -> None:
    published = tmp_path / "run"
    published.mkdir()
    (published / "stale.txt").write_text("stale", encoding="utf-8")

    def writer(staging: Path, final: Path) -> None:
        assert final == published
        (staging / "complete.txt").write_text("complete", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already published"):
        publish_run_output(tmp_path, "run", writer)
    assert not (published / "complete.txt").exists()
    assert (published / "stale.txt").read_text(encoding="utf-8") == "stale"
    assert not list(tmp_path.glob(".run.*"))


def test_failed_run_output_never_publishes_partial_run(tmp_path: Path) -> None:

    def failing_writer(staging: Path, final: Path) -> None:
        del final
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        publish_run_output(tmp_path, "run", failing_writer)

    assert not (tmp_path / "run").exists()
    assert not list(tmp_path.glob(".run.*"))
