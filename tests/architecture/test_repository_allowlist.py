from pathlib import Path

import pytest


DOCS_KEEP_ALLOWLIST = {
    "docs/codex-only-plans/README.md",
    "docs/codex-only-plans/2026-08-07-paperpilot-conversation-langgraph-vertical-slice-design.md",
    "docs/codex-only-plans/2026-08-07-paperpilot-conversation-langgraph-vertical-slice-plan.md",
    "docs/codex-only-plans/2026-08-11-new-architecture-cleanup-design.md",
    "docs/codex-only-plans/2026-08-11-new-architecture-cleanup-implementation-plan.md",
    "docs/codex-only-plans/2026-08-13-web-business-module-reorganization-design.md",
    "docs/codex-only-plans/2026-08-13-web-business-module-reorganization-plan.md",
    "docs/codex-only-plans/2026-08-14-deep-reading-nodes-split-design.md",
    "docs/codex-only-plans/2026-08-14-deep-reading-nodes-split-plan.md",
    "docs/codex-only-plans/2026-08-17-context-engineering-design.md",
    "docs/codex-only-plans/2026-08-17-context-engineering-implementation-plan.md",
    "docs/codex-only-plans/2026-08-23-agent-status-bar-design.md",
    "docs/codex-only-plans/2026-08-23-agent-status-bar-implementation-plan.md",
    "docs/codex-only-plans/2026-09-01-context-compression-design.md",
    "docs/codex-only-plans/2026-09-01-context-compression-implementation-plan.md",
    "docs/codex-only-plans/2026-09-02-context-compression-review-fixes-plan.md",
    "docs/codex-only-plans/2026-09-17-real-smoke-fixes-plan.md",
    "docs/codex-only-plans/2026-10-03-real-business-test-system-plan.md",
    "docs/codex-only-plans/2026-10-03-test-system-phase2-plan.md",
    "docs/codex-only-plans/2026-10-04-test-system-phase3-plan.md",
    "docs/codex-only-plans/2026-10-04-citation-anchoring-check-plan.md",
    "docs/codex-only-plans/2026-10-04-test-set-dimensions-plan.md",
}

ACTIVE_ENV_VARIABLES = {
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "MCP_TOOL_TIMEOUT",
    "MCP_INITIALIZE_TIMEOUT",
    "PAPERPILOT_GRAPH_PATH",
    "PAPERPILOT_TASK_DB_PATH",
    "PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH",
    "LANGGRAPH_STRICT_MSGPACK",
    "PAPERPILOT_TASK_EXECUTOR",
    "PAPERPILOT_THREAD_WORKERS",
    "PAPERPILOT_THREAD_QUEUE_CAPACITY",
    "PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS",
    "PAPERPILOT_TASK_MAX_RETRIES",
    "PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS",
    "PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS",
    "PAPERPILOT_LOG_LEVEL",
    "PAPERPILOT_LOG_FORMAT",
    "PAPERPILOT_SLOW_REQUEST_MS",
    "PAPERPILOT_ENV",
    "PAPERPILOT_CELERY_BROKER_URL",
    "PAPERPILOT_TASK_SOFT_TIME_LIMIT_SECONDS",
    "PAPERPILOT_TASK_TIME_LIMIT_SECONDS",
    "PAPERPILOT_REDIS_VISIBILITY_TIMEOUT_SECONDS",
    "PAPERPILOT_SUMMARY_TOKEN_THRESHOLD",
    "PAPERPILOT_SUMMARY_RECENT_TURNS",
    "PAPERPILOT_RESEARCH_RECURSION_LIMIT",
    "PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT",
    "PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT",
    "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS",
    "PAPERPILOT_RESEARCH_MODEL_RETRIES",
    "PAPERPILOT_RESEARCH_MODEL_NAME",
    "PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED",
    "PAPERPILOT_FULL_COMPACTION_ENABLED",
    "PAPERPILOT_CONTEXT_MODEL_WINDOW_TOKENS",
    "PAPERPILOT_CONTEXT_ARTIFACT_ROOT",
    "PAPERPILOT_CONTEXT_TOOL_INLINE_MAX_TOKENS",
    "PAPERPILOT_CONTEXT_ARTIFACT_READ_MAX_TOKENS",
    "PAPERPILOT_CONTEXT_MICRO_COMPACTION_TRIGGER_RATIO",
    "PAPERPILOT_CONTEXT_MICRO_COMPACTION_MIN_RECLAIM_TOKENS",
    "PAPERPILOT_CONTEXT_MICRO_COMPACTION_MIN_RECLAIM_RATIO",
    "PAPERPILOT_CONTEXT_MICRO_COMPACTION_KEEP_RECENT_TOOL_RESULTS",
    "PAPERPILOT_CONTEXT_ARCHIVE_BUDGET_TOKENS",
    "PAPERPILOT_CONTEXT_ARCHIVE_MAX_RECORDS",
    "PAPERPILOT_CONTEXT_ARCHIVE_RECENT_RECORDS",
    "PAPERPILOT_CONTEXT_FULL_COMPACTION_TRIGGER_RATIO",
    "PAPERPILOT_CONTEXT_SESSION_MEMORY_TARGET_RATIO",
    "PAPERPILOT_CONTEXT_FULL_COMPACTION_TARGET_RATIO",
    "PAPERPILOT_CONTEXT_FULL_COMPACTION_RECENT_TURNS",
    "PAPERPILOT_CONTEXT_COMPRESSION_FAILURE_THRESHOLD",
    "PAPERPILOT_CONTEXT_COMPRESSION_TRANSIENT_RETRY_COUNT",
    "PAPERPILOT_CONTEXT_COMPRESSION_BREAKER_COOLDOWN_SECONDS",
    "PAPERPILOT_CONTEXT_SAFETY_MARGIN_RATIO",
}

REMOVED_ENV_VARIABLES = {
    "DEFAULT_MODEL",
    "MAX_ITERATIONS",
    "BUDGET_TOKENS",
    "S2_API_KEY",
}


def _env_variable_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            line = line.removeprefix("#").strip()
        if not line or "=" not in line:
            continue
        name = line.split("=", 1)[0].strip()
        if name:
            names.add(name)
    return names


def _assert_exact_active_env_variables(path: Path) -> None:
    assert _env_variable_names(path) == ACTIVE_ENV_VARIABLES


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


def test_env_example_declares_exactly_the_active_variables() -> None:
    root = Path(__file__).parents[2]
    _assert_exact_active_env_variables(root / ".env.example")


def test_env_gate_rejects_unlisted_variable_mutation(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    mutated = tmp_path / ".env.example"
    mutated.write_text(
        (root / ".env.example").read_text(encoding="utf-8")
        + "\nPAPERPILOT_MISSPELLED_PRODUCTION_KEY=1\n",
        encoding="utf-8",
    )
    names = _env_variable_names(mutated)

    assert ACTIVE_ENV_VARIABLES <= names
    assert names.isdisjoint(REMOVED_ENV_VARIABLES)
    with pytest.raises(AssertionError):
        _assert_exact_active_env_variables(mutated)
