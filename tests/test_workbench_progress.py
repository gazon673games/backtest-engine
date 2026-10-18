from pathlib import Path

import pytest

from backtest_engine.workbench.progress import read_progress, write_progress


def test_progress_write_retries_transient_windows_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "fixture contention")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)

    write_progress(
        tmp_path / "runs",
        "retry",
        status="running",
        stage="fixture",
        message="fixture",
        pct=0.5,
    )

    assert attempts == 3
    assert read_progress(tmp_path / "runs", "retry")["stage"] == "fixture"


def test_progress_read_recovers_from_transient_windows_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "runs"
    expected = write_progress(
        output_root, "retry", status="running", stage="fixture", message="fixture", pct=0.5
    )
    original_read = Path.read_text
    attempts = 0

    def flaky_read(path: Path, *args: object, **kwargs: object) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(13, "fixture contention")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read)
    monkeypatch.setattr("backtest_engine.workbench.progress.time.sleep", lambda _: None)

    assert read_progress(output_root, "retry") == expected


def test_progress_read_does_not_hide_persistent_permission_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied_read(path: Path, *args: object, **kwargs: object) -> str:
        raise PermissionError(13, "persistent denial")

    monkeypatch.setattr(Path, "read_text", denied_read)
    monkeypatch.setattr("backtest_engine.workbench.progress.time.sleep", lambda _: None)

    with pytest.raises(PermissionError, match="persistent denial"):
        read_progress(tmp_path / "runs", "denied")


def test_missing_progress_retains_unknown_status(tmp_path: Path) -> None:
    progress = read_progress(tmp_path / "runs", "missing")

    assert progress["run_id"] == "missing"
    assert progress["status"] == "unknown"
