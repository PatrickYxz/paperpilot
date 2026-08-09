"""Web runtime configuration tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from paperpilot.web.config import WebRuntimeConfig


def test_runtime_config_defaults_are_local_and_bounded():
    config = WebRuntimeConfig.from_env({})

    assert config.task_executor == "thread"
    assert config.thread_workers == 2
    assert config.thread_queue_capacity == 4
    assert config.overload_retry_after_seconds == 1
    assert config.log_level == "INFO"
    assert config.log_format == "json"
    assert config.slow_request_ms == 1000
    assert config.environment == "development"
    assert config.checkpoint_db_path == Path("data/langgraph/checkpoints.sqlite3")
    assert config.summary_token_threshold == 32_000
    assert config.summary_recent_turns == 6
    assert config.research_recursion_limit == 12


def test_runtime_config_accepts_explicit_overrides():
    config = WebRuntimeConfig.from_env(
        {
            "PAPERPILOT_TASK_EXECUTOR": "CELERY",
            "PAPERPILOT_THREAD_WORKERS": "3",
            "PAPERPILOT_THREAD_QUEUE_CAPACITY": "0",
            "PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS": "5",
            "PAPERPILOT_LOG_LEVEL": "debug",
            "PAPERPILOT_LOG_FORMAT": "text",
            "PAPERPILOT_SLOW_REQUEST_MS": "250",
            "PAPERPILOT_ENV": "test",
            "PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH": "/tmp/paperpilot-checkpoints.sqlite3",
            "PAPERPILOT_SUMMARY_TOKEN_THRESHOLD": "64000",
            "PAPERPILOT_SUMMARY_RECENT_TURNS": "8",
            "PAPERPILOT_RESEARCH_RECURSION_LIMIT": "20",
        }
    )

    assert config.task_executor == "celery"
    assert config.thread_workers == 3
    assert config.thread_queue_capacity == 0
    assert config.overload_retry_after_seconds == 5
    assert config.log_level == "DEBUG"
    assert config.log_format == "text"
    assert config.slow_request_ms == 250
    assert config.environment == "test"
    assert config.checkpoint_db_path == Path("/tmp/paperpilot-checkpoints.sqlite3")
    assert config.summary_token_threshold == 64_000
    assert config.summary_recent_turns == 8
    assert config.research_recursion_limit == 20


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("PAPERPILOT_TASK_EXECUTOR", "other", "PAPERPILOT_TASK_EXECUTOR"),
        ("PAPERPILOT_THREAD_WORKERS", "0", "PAPERPILOT_THREAD_WORKERS"),
        ("PAPERPILOT_THREAD_WORKERS", "many", "PAPERPILOT_THREAD_WORKERS"),
        ("PAPERPILOT_THREAD_QUEUE_CAPACITY", "-1", "PAPERPILOT_THREAD_QUEUE_CAPACITY"),
        ("PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS", "0", "PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS"),
        ("PAPERPILOT_LOG_LEVEL", "verbose", "PAPERPILOT_LOG_LEVEL"),
        ("PAPERPILOT_LOG_FORMAT", "yaml", "PAPERPILOT_LOG_FORMAT"),
        ("PAPERPILOT_SLOW_REQUEST_MS", "0", "PAPERPILOT_SLOW_REQUEST_MS"),
        ("PAPERPILOT_ENV", "   ", "PAPERPILOT_ENV"),
        (
            "PAPERPILOT_SUMMARY_TOKEN_THRESHOLD",
            "0",
            "PAPERPILOT_SUMMARY_TOKEN_THRESHOLD",
        ),
        (
            "PAPERPILOT_SUMMARY_TOKEN_THRESHOLD",
            "many",
            "PAPERPILOT_SUMMARY_TOKEN_THRESHOLD",
        ),
        ("PAPERPILOT_SUMMARY_RECENT_TURNS", "0", "PAPERPILOT_SUMMARY_RECENT_TURNS"),
        (
            "PAPERPILOT_RESEARCH_RECURSION_LIMIT",
            "0",
            "PAPERPILOT_RESEARCH_RECURSION_LIMIT",
        ),
    ],
)
def test_runtime_config_rejects_invalid_values(name, value, message):
    with pytest.raises(ValueError, match=message):
        WebRuntimeConfig.from_env({name: value})
