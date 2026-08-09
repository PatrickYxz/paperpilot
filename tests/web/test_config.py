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
    assert config.task_max_retries == 3
    assert config.task_retry_backoff_seconds == 1
    assert config.task_retry_backoff_max_seconds == 30
    assert config.log_level == "INFO"
    assert config.log_format == "json"
    assert config.slow_request_ms == 1000
    assert config.environment == "development"
    assert config.checkpoint_db_path == Path("data/langgraph/checkpoints.sqlite3")
    assert config.summary_token_threshold == 32_000
    assert config.summary_recent_turns == 6
    assert config.research_recursion_limit == 12
    assert config.research_model_call_limit == 6
    assert config.research_tool_call_limit == 12
    assert config.research_max_output_tokens == 4096
    assert config.research_model_retries == 1


def test_runtime_config_accepts_explicit_overrides():
    config = WebRuntimeConfig.from_env(
        {
            "PAPERPILOT_TASK_EXECUTOR": "CELERY",
            "PAPERPILOT_THREAD_WORKERS": "3",
            "PAPERPILOT_THREAD_QUEUE_CAPACITY": "0",
            "PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS": "5",
            "PAPERPILOT_TASK_MAX_RETRIES": "5",
            "PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS": "3",
            "PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS": "12",
            "PAPERPILOT_LOG_LEVEL": "debug",
            "PAPERPILOT_LOG_FORMAT": "text",
            "PAPERPILOT_SLOW_REQUEST_MS": "250",
            "PAPERPILOT_ENV": "test",
            "PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH": "/tmp/paperpilot-checkpoints.sqlite3",
            "PAPERPILOT_SUMMARY_TOKEN_THRESHOLD": "64000",
            "PAPERPILOT_SUMMARY_RECENT_TURNS": "8",
            "PAPERPILOT_RESEARCH_RECURSION_LIMIT": "20",
            "PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT": "8",
            "PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT": "16",
            "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS": "2048",
            "PAPERPILOT_RESEARCH_MODEL_RETRIES": "0",
        }
    )

    assert config.task_executor == "celery"
    assert config.thread_workers == 3
    assert config.thread_queue_capacity == 0
    assert config.overload_retry_after_seconds == 5
    assert config.task_max_retries == 5
    assert config.task_retry_backoff_seconds == 3
    assert config.task_retry_backoff_max_seconds == 12
    assert config.log_level == "DEBUG"
    assert config.log_format == "text"
    assert config.slow_request_ms == 250
    assert config.environment == "test"
    assert config.checkpoint_db_path == Path("/tmp/paperpilot-checkpoints.sqlite3")
    assert config.summary_token_threshold == 64_000
    assert config.summary_recent_turns == 8
    assert config.research_recursion_limit == 20
    assert config.research_model_call_limit == 8
    assert config.research_tool_call_limit == 16
    assert config.research_max_output_tokens == 2048
    assert config.research_model_retries == 0


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("PAPERPILOT_TASK_EXECUTOR", "other", "PAPERPILOT_TASK_EXECUTOR"),
        ("PAPERPILOT_THREAD_WORKERS", "0", "PAPERPILOT_THREAD_WORKERS"),
        ("PAPERPILOT_THREAD_WORKERS", "many", "PAPERPILOT_THREAD_WORKERS"),
        ("PAPERPILOT_THREAD_QUEUE_CAPACITY", "-1", "PAPERPILOT_THREAD_QUEUE_CAPACITY"),
        ("PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS", "0", "PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS"),
        ("PAPERPILOT_TASK_MAX_RETRIES", "-1", "PAPERPILOT_TASK_MAX_RETRIES"),
        ("PAPERPILOT_TASK_MAX_RETRIES", "many", "PAPERPILOT_TASK_MAX_RETRIES"),
        (
            "PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS",
            "0",
            "PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS",
        ),
        (
            "PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS",
            "0",
            "PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS",
        ),
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
        (
            "PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT",
            "0",
            "PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT",
        ),
        (
            "PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT",
            "many",
            "PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT",
        ),
        (
            "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS",
            "0",
            "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS",
        ),
        (
            "PAPERPILOT_RESEARCH_MODEL_RETRIES",
            "-1",
            "PAPERPILOT_RESEARCH_MODEL_RETRIES",
        ),
    ],
)
def test_runtime_config_rejects_invalid_values(name, value, message):
    with pytest.raises(ValueError, match=message):
        WebRuntimeConfig.from_env({name: value})


def test_runtime_config_rejects_retry_backoff_above_cap():
    with pytest.raises(
        ValueError,
        match="PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS",
    ):
        WebRuntimeConfig.from_env(
            {
                "PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS": "31",
                "PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS": "30",
            }
        )


@pytest.mark.parametrize(
    "name",
    [
        "PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT",
        "PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT",
    ],
)
def test_runtime_config_requires_budget_for_both_structured_attempts(name):
    with pytest.raises(ValueError, match=name):
        WebRuntimeConfig.from_env({name: "1"})
