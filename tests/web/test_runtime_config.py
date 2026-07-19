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
    assert "PAPERPILOT_REDIS_VISIBILITY_TIMEOUT_SECONDS" in text
    assert "at-least-once" in text
    assert "transactional outbox" in text
    assert "docker compose up -d redis" in text
    assert "paperpilot.web.celery_app:celery_app worker" in text


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
