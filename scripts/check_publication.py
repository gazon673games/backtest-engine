"""Check public Git objects against local-only publication boundaries."""

import subprocess
import sys
from pathlib import PurePosixPath


def check(paths):
    paths = [path for path in paths if path]
    for path in paths:
        parts = PurePosixPath(path).parts
        if any(part in {".private-originals", ".history", "sdk_source_patches"} for part in parts):
            raise SystemExit("Local-only material cannot be published")
        if path.startswith("src/strategy_sdk/catalog/"):
            module = path.removeprefix("src/strategy_sdk/catalog/")
            if module not in {"__init__.py", "empty.py"}:
                raise SystemExit("Local catalog source cannot be published")
        if path.startswith(("strategies/", "docs/research/", "docs/research_baselines/")):
            raise SystemExit("Local algorithms and research cannot be published")
    if paths:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--stdin", "-z"],
            input="\0".join(paths) + "\0",
            text=True,
            capture_output=True,
        )
        if result.returncode not in {0, 1}:
            raise SystemExit(result.stderr)
        if result.stdout:
            raise SystemExit("Public Git objects contain files excluded by .gitignore")


def main():
    if "--pre-push" in sys.argv:
        for line in sys.stdin:
            _, revision, _, _ = line.split()
            if set(revision) == {"0"}:
                continue
            objects = subprocess.check_output(
                ["git", "rev-list", "--objects", revision],
                text=True,
            )
            check([line.split(" ", 1)[1] for line in objects.splitlines() if " " in line])
    else:
        files = subprocess.check_output(["git", "ls-files", "-z"], text=True)
        check([path for path in files.split("\0") if path])
    print("Public source boundaries verified")


if __name__ == "__main__":
    main()
