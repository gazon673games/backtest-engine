"""Exposes configured SDK strategies as live references, without copying saved records."""

import ast

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.core.process_settings import configured_sdk_catalog as configured_catalog
from backtest_engine.core.process_settings import load_workbench_settings
from backtest_engine.strategies.sdk_catalog_sources import (
    catalog_entries,
    catalog_files,
    catalog_snapshot,
    snapshot_from_files,
)
from backtest_engine.strategies.sdk_catalog_sources import (
    catalog_reference as catalog_reference,
)

CATALOG_PREFIX = "sdk:"


def catalog_record(name, defaults):
    return {
        "id": CATALOG_PREFIX + name,
        "name": f"SDK: {name}",
        "tags": ["sdk-catalog"],
        "source": f'SDK_CATALOG_STRATEGY = "{name}"\n',
        "language": "python",
        "params": defaults,
        "origin": "sdk-catalog",
    }


def list_catalog_records():
    root = load_workbench_settings().strategies.sdk_catalog_path
    if root is None:
        return []
    entries = catalog_entries(catalog_files(root))
    return [catalog_record(name, entries[name][1]) for name in sorted(entries)]


def get_catalog_record(strategy_id):
    name = strategy_id.removeprefix(CATALOG_PREFIX)
    snapshot = catalog_snapshot(configured_catalog(), name)
    return catalog_record(name, snapshot["defaults"])


def freeze_catalog_source(source, language):
    if language != "python":
        return source
    if source.strip().startswith("SDK_CATALOG_DRAFT"):
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
            raise StrategyPluginError("SDK_CATALOG_DRAFT must be the entire source")
        assignment = tree.body[0]
        if (
            len(assignment.targets) != 1
            or not isinstance(assignment.targets[0], ast.Name)
            or assignment.targets[0].id != "SDK_CATALOG_DRAFT"
        ):
            raise StrategyPluginError("Invalid SDK draft declaration")
        draft = ast.literal_eval(assignment.value)
        if (
            not isinstance(draft, dict)
            or set(draft) != {"name", "source"}
            or not isinstance(draft["name"], str)
            or not isinstance(draft["source"], str)
        ):
            raise StrategyPluginError("Invalid SDK draft")
        files = catalog_files(configured_catalog())
        entries = catalog_entries(files)
        if draft["name"] not in entries:
            raise StrategyPluginError("Unknown SDK catalog strategy")
        module, _defaults = entries[draft["name"]]
        # Resolve draft imports from this one catalog read; never update shared files.
        files[module] = draft["source"]
        snapshot = snapshot_from_files(files, draft["name"])
        return "SDK_CATALOG_SNAPSHOT = " + repr(snapshot) + "\n"
    name = catalog_reference(source)
    if name is None:
        return source
    snapshot = catalog_snapshot(configured_catalog(), name)
    return "SDK_CATALOG_SNAPSHOT = " + repr(snapshot) + "\n"


def freeze_run_request(request):
    path = request.script_path
    language = (
        ("python" if path.suffix.lower() == ".py" else "javascript")
        if path
        else request.script_language
    )
    original = path.read_text(encoding="utf-8-sig") if path else request.script_source or ""
    frozen = freeze_catalog_source(original, language)
    return request.model_copy(
        update={"script_path": None, "script_source": frozen, "script_language": language}
    )
