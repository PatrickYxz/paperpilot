from pathlib import Path

import pytest


DOCS_KEEP_ALLOWLIST = {
    "docs/codex-only-plans/README.md",
    "docs/codex-only-plans/2026-08-07-paperpilot-conversation-langgraph-vertical-slice-design.md",
    "docs/codex-only-plans/2026-08-07-paperpilot-conversation-langgraph-vertical-slice-plan.md",
    "docs/codex-only-plans/2026-08-11-new-architecture-cleanup-design.md",
    "docs/codex-only-plans/2026-08-11-new-architecture-cleanup-implementation-plan.md",
}


def _assert_no_historical_day_scripts(scripts_root: Path) -> None:
    historical_scripts = [
        path
        for path in scripts_root.rglob("day*")
        if path.is_file() and path.suffix in {".py", ".pyc", ".pyo"}
    ]
    assert not historical_scripts


@pytest.mark.parametrize(
    "relative",
    (
        "day4_smoke.py",
        "nested/day4_smoke.py",
        "__pycache__/day4_smoke.cpython-312.pyc",
        "__pycache__/day4_smoke.cpython-312.pyo",
    ),
)
def test_day_script_gate_rejects_source_and_bytecode(
    tmp_path: Path, relative: str
) -> None:
    scripts_root = tmp_path / "scripts"
    historical_script = scripts_root / relative
    historical_script.parent.mkdir(parents=True, exist_ok=True)
    historical_script.touch()

    with pytest.raises(AssertionError):
        _assert_no_historical_day_scripts(scripts_root)


def test_day_script_gate_allows_unrelated_bytecode(tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    unrelated_bytecode = scripts_root / "__pycache__" / "helper.cpython-312.pyc"
    unrelated_bytecode.parent.mkdir(parents=True)
    unrelated_bytecode.touch()

    _assert_no_historical_day_scripts(scripts_root)


def test_repository_contains_no_historical_runtime_or_eval_groups() -> None:
    root = Path(__file__).parents[2]
    for relative in (
        "paperpilot/eval",
        "paperpilot/skills",
        "tests/eval",
        "data/eval",
        "data/traces",
        "docs/superpowers",
    ):
        assert not (root / relative).exists()
    _assert_no_historical_day_scripts(root / "scripts")


def test_repository_docs_match_keep_allowlist() -> None:
    root = Path(__file__).parents[2]
    actual_docs = {
        path.relative_to(root).as_posix()
        for path in (root / "docs").rglob("*")
        if path.is_file()
    }

    assert actual_docs == DOCS_KEEP_ALLOWLIST
