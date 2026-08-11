from pathlib import Path


DOCS_KEEP_ALLOWLIST = {
    "docs/codex-only-plans/README.md",
    "docs/codex-only-plans/2026-08-07-paperpilot-conversation-langgraph-vertical-slice-design.md",
    "docs/codex-only-plans/2026-08-07-paperpilot-conversation-langgraph-vertical-slice-plan.md",
    "docs/codex-only-plans/2026-08-11-new-architecture-cleanup-design.md",
    "docs/codex-only-plans/2026-08-11-new-architecture-cleanup-implementation-plan.md",
}


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
    assert not list((root / "scripts").glob("day*.py"))


def test_repository_docs_match_keep_allowlist() -> None:
    root = Path(__file__).parents[2]
    actual_docs = {
        path.relative_to(root).as_posix()
        for path in (root / "docs").rglob("*")
        if path.is_file()
    }

    assert actual_docs == DOCS_KEEP_ALLOWLIST
