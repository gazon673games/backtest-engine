"""Reject local catalog modules and research material in public distributions."""

import sys
import tarfile
import zipfile
from pathlib import Path

ALLOWED_CATALOG = {"__init__.py", "empty.py"}


def validate(names):
    for name in names:
        normalized = name.replace("\\", "/")
        if "strategy_sdk/catalog/" in normalized:
            relative = normalized.split("strategy_sdk/catalog/", 1)[1]
            if relative and relative not in ALLOWED_CATALOG:
                raise SystemExit(f"Local catalog material in distribution: {relative}")
        if any(
            part in normalized.split("/")
            for part in (".private-originals", ".history", "sdk_source_patches")
        ):
            raise SystemExit("Local-only material in distribution")


def main():
    paths = [Path(argument) for argument in sys.argv[1:]]
    if not paths:
        paths = list(Path("dist").glob("*.whl")) + list(Path("dist").glob("*.tar.gz"))
    if not paths:
        raise SystemExit("No distributions to inspect")
    for path in paths:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                validate(archive.namelist())
        else:
            with tarfile.open(path) as archive:
                validate(archive.getnames())
        print(f"Public distribution verified: {path.name}")


if __name__ == "__main__":
    main()
