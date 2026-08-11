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
        "PAPERPILOT_RESEARCH_RECURSION_LIMIT=24",
        "PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT=8",
        "PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT=12",
        "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS=4096",
        "PAPERPILOT_RESEARCH_MODEL_RETRIES=1",
    ):
        assert value in text
