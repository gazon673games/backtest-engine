from __future__ import annotations

import importlib
from pathlib import Path

import backtest_engine
from backtest_engine.workbench.settings import load_workbench_settings, runs_root


def test_source_root_contains_only_the_public_package_facade() -> None:
    package_root = Path(backtest_engine.__file__).parent

    assert [path.name for path in package_root.glob("*.py")] == ["__init__.py"]


def test_legacy_strategy_modules_resolve_to_domain_packages() -> None:
    legacy = importlib.import_module("backtest_engine.strategy_sdk")
    organized = importlib.import_module("backtest_engine.strategies.sdk")

    assert legacy is organized


def test_repository_workbench_settings_are_loadable(tmp_path: Path) -> None:
    settings_path = tmp_path / "workbench.json"
    settings_path.write_text(
        '{"server":{"port":8765},"jobs":{"max_concurrency":2}}',
        encoding="utf-8",
    )
    settings = load_workbench_settings(settings_path)

    assert settings.server.port == 8765
    assert settings.jobs.max_concurrency == 2
    assert settings.proxy.url == "localhost:10808"
    assert settings.proxy.enabled is False
    assert runs_root(settings) == Path("data/workbench/runs")
