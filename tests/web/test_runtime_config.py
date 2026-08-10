"""Runtime configuration artifact tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
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


def test_readme_documents_celery_runtime_commands():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "PAPERPILOT_TASK_EXECUTOR=celery" in text
    assert "PAPERPILOT_TASK_DB_PATH" in text
    assert "PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH" in text
    assert "PAPERPILOT_REDIS_VISIBILITY_TIMEOUT_SECONDS" in text
    assert "at-least-once" in text
    assert "transactional outbox" in text
    assert "docker compose up -d redis" in text
    assert "paperpilot.web.celery_app:celery_app worker" in text
    assert "20260807_0002" in text
    assert "database`, `checkpoint`, and `executor" in text
    assert "`running`, `completed`, or `failed`" in text

    startup_commands = (
        "uv pip sync requirements-lock.txt --python .venv/bin/python",
        "./.venv/bin/python -m alembic -c alembic.ini upgrade head",
        "LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python "
        "-m paperpilot.web.checkpoint --setup",
        "./.venv/bin/celery -A paperpilot.web.celery_app:celery_app "
        "worker --loglevel=INFO",
        "./.venv/bin/uvicorn paperpilot.web.app:app "
        "--host 127.0.0.1 --port 8000",
    )
    positions = [text.index(command) for command in startup_commands]
    assert positions == sorted(positions)


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


def test_readme_exports_dotenv_before_starting_celery_runtime():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "does not load `.env` automatically" in text
    dotenv_commands = (
        "cp .env.example .env",
        "set -a",
        "source .env",
        "set +a",
        "uv pip sync requirements-lock.txt --python .venv/bin/python",
    )
    positions = [text.index(command) for command in dotenv_commands]
    assert positions == sorted(positions)
    assert "set `PAPERPILOT_TASK_EXECUTOR=celery` in `.env`" in normalized
    assert text.index("docker compose up -d redis") < text.index(
        "./.venv/bin/celery -A paperpilot.web.celery_app:celery_app "
        "worker --loglevel=INFO"
    )


def test_readme_starts_worker_and_api_in_separate_loaded_shells():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "two bash/zsh terminals" in normalized
    assert "load the same `.env` in each" in normalized
    assert "Start the Worker in terminal 1" in normalized
    assert "start the API in terminal 2" in normalized
    assert "POSIX shell" not in text


def test_readme_backs_up_and_restores_sqlite_files_as_timestamped_pair():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert 'BACKUP_DIR="data/backups/$(date +%Y%m%d-%H%M%S)"' in text
    assert 'mkdir -p "$BACKUP_DIR"' in text
    assert '"$BACKUP_DIR/business.sqlite3"' in text
    assert '"$BACKUP_DIR/checkpoints.sqlite3"' in text
    assert 'RESTORE_DIR="data/backups/<selected-timestamp>"' in text
    assert '"$RESTORE_DIR/business.sqlite3"' in text
    assert '"$RESTORE_DIR/checkpoints.sqlite3"' in text
    assert "data/backups/business.sqlite3" not in text
    assert "data/backups/checkpoints.sqlite3" not in text


def test_admission_benchmark_proves_bound_and_recovery():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_web_admission.py",
            "--workers",
            "1",
            "--queue-capacity",
            "0",
            "--requests",
            "2",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["configured_capacity"] == 1
    assert payload["accepted"] == 1
    assert payload["rejected"] == 1
    assert payload["task_rows_before_recovery"] == 1
    assert payload["event_rows_before_recovery"] == 1
    assert payload["capacity_recovered"] is True
    assert payload["unexpected_statuses"] == []


def test_admission_benchmark_does_not_touch_configured_database(tmp_path):
    sentinel_path = tmp_path / "external-tasks.sqlite3"
    environment = os.environ.copy()
    environment["PAPERPILOT_TASK_DB_PATH"] = str(sentinel_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_web_admission.py",
            "--workers",
            "1",
            "--queue-capacity",
            "0",
            "--requests",
            "2",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["capacity_recovered"] is True
    assert not sentinel_path.exists()


def test_readme_documents_runtime_protection_contract():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "6 unfinished tasks per Uvicorn process" in normalized
    assert "theoretical aggregate capacity across both processes is 12" in normalized
    assert "process-local total is 12" not in normalized

    for value in (
        "PAPERPILOT_THREAD_WORKERS",
        "PAPERPILOT_THREAD_QUEUE_CAPACITY",
        "PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS",
        "PAPERPILOT_LOG_FORMAT",
        "/health/live",
        "/health/ready",
        "Retry-After",
        "X-Request-ID",
        "--no-access-log",
        "benchmark_web_admission.py",
    ):
        assert value in text


def test_readme_documents_research_budget_and_failure_taxonomy() -> None:
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for value in (
        "PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT",
        "PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT",
        "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS",
        "PAPERPILOT_RESEARCH_MODEL_RETRIES",
        "32,768",
        "131,072",
        "generated tokens",
        "usage metadata",
        "terminal",
        "transient",
    ):
        assert value in text
