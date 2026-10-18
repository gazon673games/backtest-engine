from pathlib import Path


def test_source_files_stay_within_workspace_line_limit() -> None:
    source_root = Path(__file__).parents[1] / "src"
    oversized = {}
    for path in source_root.rglob("*"):
        if path.suffix not in {".py", ".js", ".mjs"}:
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > 300:
            oversized[str(path.relative_to(source_root))] = line_count
    assert oversized == {}
