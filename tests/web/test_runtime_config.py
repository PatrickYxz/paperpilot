"""Runtime configuration artifact tests."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]


def test_compose_enables_persistent_redis_queue():
    text = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "redis:7-alpine" in text
    assert '"--appendonly"' in text
    assert '"yes"' in text
    assert '"--appendfsync"' in text
    assert '"everysec"' in text
    assert "redis-cli" in text
    assert "paperpilot-redis-data" in text
    assert '"127.0.0.1:6379:6379"' in text


def test_env_example_declares_both_sqlite_paths_and_executor_examples():
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    for value in (
        "PAPERPILOT_TASK_DB_PATH=data/web/tasks.sqlite3",
        "PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH=data/langgraph/checkpoints.sqlite3",
        "LANGGRAPH_STRICT_MSGPACK=true",
        "PAPERPILOT_TASK_EXECUTOR=thread",
        "PAPERPILOT_TASK_EXECUTOR=celery",
        "PAPERPILOT_CELERY_BROKER_URL=redis://127.0.0.1:6379/0",
        "PAPERPILOT_TASK_MAX_RETRIES=3",
        "PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS=1",
        "PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS=30",
        "PAPERPILOT_RESEARCH_RECURSION_LIMIT=33",
        "PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT=12",
        "PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT=12",
        "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS=4096",
        "PAPERPILOT_RESEARCH_MODEL_RETRIES=1",
        "PAPERPILOT_RESEARCH_MODEL_NAME=deepseek-v4-flash",
        "PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED=false",
        "PAPERPILOT_FULL_COMPACTION_ENABLED=false",
        "PAPERPILOT_CONTEXT_MODEL_WINDOW_TOKENS=1048576",
        "PAPERPILOT_CONTEXT_ARTIFACT_ROOT=data/context-artifacts",
        "PAPERPILOT_CONTEXT_TOOL_INLINE_MAX_TOKENS=2000",
        "PAPERPILOT_CONTEXT_ARTIFACT_READ_MAX_TOKENS=2000",
        "PAPERPILOT_CONTEXT_MICRO_COMPACTION_TRIGGER_RATIO=0.70",
        "PAPERPILOT_CONTEXT_MICRO_COMPACTION_MIN_RECLAIM_TOKENS=8000",
        "PAPERPILOT_CONTEXT_MICRO_COMPACTION_MIN_RECLAIM_RATIO=0.10",
        "PAPERPILOT_CONTEXT_MICRO_COMPACTION_KEEP_RECENT_TOOL_RESULTS=3",
        "PAPERPILOT_CONTEXT_ARCHIVE_BUDGET_TOKENS=4000",
        "PAPERPILOT_CONTEXT_ARCHIVE_MAX_RECORDS=5",
        "PAPERPILOT_CONTEXT_ARCHIVE_RECENT_RECORDS=2",
        "PAPERPILOT_CONTEXT_FULL_COMPACTION_TRIGGER_RATIO=0.80",
        "PAPERPILOT_CONTEXT_SESSION_MEMORY_TARGET_RATIO=0.65",
        "PAPERPILOT_CONTEXT_FULL_COMPACTION_TARGET_RATIO=0.50",
        "PAPERPILOT_CONTEXT_FULL_COMPACTION_RECENT_TURNS=2",
        "PAPERPILOT_CONTEXT_COMPRESSION_FAILURE_THRESHOLD=3",
        "PAPERPILOT_CONTEXT_COMPRESSION_TRANSIENT_RETRY_COUNT=1",
        "PAPERPILOT_CONTEXT_COMPRESSION_BREAKER_COOLDOWN_SECONDS=300",
        "PAPERPILOT_CONTEXT_SAFETY_MARGIN_RATIO=0.05",
    ):
        assert value in text


def test_readme_documents_context_rollout_safety_and_verification_boundary():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for value in (
        "PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED=false",
        "PAPERPILOT_FULL_COMPACTION_ENABLED=false",
        "PAPERPILOT_CONTEXT_MODEL_WINDOW_TOKENS=1048576",
        "20260901_0003",
        "data/context-artifacts",
        "chmod 750",
        "artifact_externalized",
        "关闭两个 flag",
        "本地测试不等于生产收益",
    ):
        assert value in text
