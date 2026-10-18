"""Translate portable catalog loading errors at the backtest boundary."""

import ast

from strategy_sdk import StrategyError
from strategy_sdk import catalog_sources as sdk

from backtest_engine.core.exceptions import StrategyPluginError


def _load(function, *args):
    try:
        return function(*args)
    except StrategyError as error:
        raise StrategyPluginError(str(error)) from error


def catalog_files(root):
    return _load(sdk.catalog_files, root)


def catalog_entries(files):
    return _load(sdk.catalog_entries, files)


def catalog_snapshot(root, name):
    return _load(sdk.catalog_snapshot, root, name)


def load_catalog_snapshot(snapshot):
    return _load(sdk.load_catalog_snapshot, snapshot)


def snapshot_from_files(files, name):
    return _load(sdk.snapshot_from_files, files, name)


def catalog_reference(source):
    if "SDK_CATALOG_STRATEGY" not in source:
        return None
    tree = ast.parse(source)
    declarations = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SDK_CATALOG_STRATEGY"
            for target in node.targets
        )
    ]
    if not declarations:
        return None
    declaration = declarations[0]
    if (
        len(tree.body) != 1
        or len(declaration.targets) != 1
        or not isinstance(declaration.value, ast.Constant)
        or not isinstance(declaration.value.value, str)
    ):
        raise StrategyPluginError(
            "SDK_CATALOG_STRATEGY must be the entire source and use a literal name"
        )
    return declaration.value.value
