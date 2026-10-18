"""Show actual Python sources for saved SDK presets and configured catalog entries."""

import ast
from pathlib import Path

import strategy_sdk.catalog
from strategy_sdk.source_files import SourceConflict, read_source, save_source

from backtest_engine.core.exceptions import StrategyPluginError
from backtest_engine.strategies.sdk_catalog_sources import catalog_snapshot
from backtest_engine.workbench.sdk_catalog import (
    CATALOG_PREFIX,
    configured_catalog,
    freeze_catalog_source,
)
from backtest_engine.workbench.strategies import metadata_from_source, strategies_root

MARKERS = {"SDK_STRATEGY", "SDK_CATALOG_STRATEGY", "SDK_CATALOG_SNAPSHOT", "SDK_CATALOG_DRAFT"}


def source_snapshot(source):
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None, None
    declaration = tree.body[0]
    if len(declaration.targets) != 1 or not isinstance(declaration.targets[0], ast.Name):
        return None, None
    marker = declaration.targets[0].id
    if marker not in MARKERS:
        return None, None
    value = ast.literal_eval(declaration.value)
    if marker == "SDK_CATALOG_SNAPSHOT":
        return value, None
    if marker == "SDK_CATALOG_DRAFT":
        frozen = freeze_catalog_source(source, "python")
        return ast.literal_eval(ast.parse(frozen).body[0].value), None
    root = (
        configured_catalog()
        if marker == "SDK_CATALOG_STRATEGY"
        else Path(strategy_sdk.catalog.__file__).parent
    )
    return catalog_snapshot(root, value), root if marker == "SDK_CATALOG_STRATEGY" else None


def editor_record(record):
    if record.get("language", "javascript") != "python":
        return record
    snapshot, root = source_snapshot(record["source"])
    if snapshot is None:
        return record
    module = snapshot["module"]
    # A variant may only re-export callbacks; open the file containing their bodies.
    visited = set()
    while module not in visited:
        visited.add(module)
        tree = ast.parse(snapshot["files"][module])
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "generate_signal"
            for node in tree.body
        ):
            break
        imported = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ImportFrom)
                and any((alias.asname or alias.name) == "generate_signal" for alias in node.names)
            ),
            None,
        )
        dependency = imported.module.rsplit(".", 1)[-1] if imported and imported.module else None
        if dependency not in snapshot["files"]:
            break
        module = dependency
    revisions = {}
    if root:
        for name, code in snapshot["files"].items():
            opened = read_source(root, name + ".py")
            if opened["source"] != code:
                raise SourceConflict("SDK files changed while opening; reload the strategy")
            revisions[name] = opened["revision"]
    binding = {
        "snapshot": snapshot,
        "module": module,
        "revisions": revisions,
        "id": CATALOG_PREFIX + snapshot["name"] if root else None,
        "location": str(root) if root else "installed SDK or saved snapshot",
    }
    return {**record, "editor_source": snapshot["files"][module], "sdk_editor": binding}


def source_open_payload(output_root, payload):
    name = payload["id"].removeprefix(CATALOG_PREFIX)
    root = configured_catalog()
    snapshot = catalog_snapshot(root, name)
    return {**read_source(root, snapshot["module"] + ".py"), "id": CATALOG_PREFIX + name}


def source_save_payload(output_root, payload):
    opened = source_open_payload(output_root, payload)
    name = payload["id"].removeprefix(CATALOG_PREFIX)
    snapshot = catalog_snapshot(configured_catalog(), name)
    filename = payload.get("filename", opened["filename"])
    module = filename.removesuffix(".py")
    if module not in snapshot["files"] or filename != module + ".py":
        raise StrategyPluginError("Only files belonging to the selected strategy can be updated")
    snapshot["files"][module] = payload["source"]
    draft = "SDK_CATALOG_SNAPSHOT = " + repr(snapshot) + "\n"
    metadata_from_source(
        draft, strategies_root(output_root), language="python", params=payload.get("params")
    )
    saved = save_source(configured_catalog(), filename, payload["source"], payload.get("revision"))
    return {"status": "completed", **saved, "id": payload["id"]}
