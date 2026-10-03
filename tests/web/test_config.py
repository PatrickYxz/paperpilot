"""Web runtime configuration tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from paperpilot.web.config import ContextManagementConfig, WebRuntimeConfig


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
    assert config.research_recursion_limit == 33
    assert config.research_model_call_limit == 12
    assert config.research_tool_call_limit == 12
    assert config.research_max_output_tokens == 4096
    assert config.research_model_retries == 1
    assert config.research_model_name == "deepseek-v4-flash"
    assert config.context_management == ContextManagementConfig()
    assert config.context_management.enabled is False
    assert config.context_management.full_compaction_enabled is False
    assert config.context_management.model_context_window_tokens == 1_048_576
    assert config.context_management.artifact_root == Path("data/context-artifacts")
    assert config.context_management.tool_inline_max_tokens == 2_000
    assert config.context_management.artifact_read_max_tokens == 2_000
    assert config.context_management.micro_compaction_trigger_ratio == 0.70
    assert config.context_management.micro_compaction_min_reclaim_tokens == 8_000
    assert config.context_management.micro_compaction_min_reclaim_ratio == 0.10
    assert config.context_management.micro_compaction_keep_recent_tool_results == 3
    assert config.context_management.archive_context_budget_tokens == 4_000
    assert config.context_management.archive_context_max_records == 5
    assert config.context_management.archive_context_recent_records == 2
    assert config.context_management.full_compaction_trigger_ratio == 0.80
    assert config.context_management.session_memory_target_ratio == 0.65
    assert config.context_management.full_compaction_target_ratio == 0.50
    assert config.context_management.full_compaction_recent_turns == 2
    assert config.context_management.compression_failure_threshold == 3
    assert config.context_management.compression_transient_retry_count == 1
    assert config.context_management.compression_breaker_cooldown_seconds == 300
    assert config.context_management.context_safety_margin_ratio == 0.05


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
            "PAPERPILOT_RESEARCH_RECURSION_LIMIT": "30",
            "PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT": "10",
            "PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT": "16",
            "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS": "2048",
            "PAPERPILOT_RESEARCH_MODEL_RETRIES": "0",
            "PAPERPILOT_RESEARCH_MODEL_NAME": "custom-model",
            "PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED": "true",
            "PAPERPILOT_FULL_COMPACTION_ENABLED": "true",
            "PAPERPILOT_CONTEXT_MODEL_WINDOW_TOKENS": "200000",
            "PAPERPILOT_CONTEXT_ARTIFACT_ROOT": "/tmp/context-artifacts",
            "PAPERPILOT_CONTEXT_TOOL_INLINE_MAX_TOKENS": "1500",
            "PAPERPILOT_CONTEXT_ARTIFACT_READ_MAX_TOKENS": "1700",
            "PAPERPILOT_CONTEXT_MICRO_COMPACTION_TRIGGER_RATIO": "0.9",
            "PAPERPILOT_CONTEXT_MICRO_COMPACTION_MIN_RECLAIM_TOKENS": "9000",
            "PAPERPILOT_CONTEXT_MICRO_COMPACTION_MIN_RECLAIM_RATIO": "0.2",
            "PAPERPILOT_CONTEXT_MICRO_COMPACTION_KEEP_RECENT_TOOL_RESULTS": "4",
            "PAPERPILOT_CONTEXT_ARCHIVE_BUDGET_TOKENS": "5000",
            "PAPERPILOT_CONTEXT_ARCHIVE_MAX_RECORDS": "6",
            "PAPERPILOT_CONTEXT_ARCHIVE_RECENT_RECORDS": "3",
            "PAPERPILOT_CONTEXT_FULL_COMPACTION_TRIGGER_RATIO": "0.95",
            "PAPERPILOT_CONTEXT_SESSION_MEMORY_TARGET_RATIO": "0.7",
            "PAPERPILOT_CONTEXT_FULL_COMPACTION_TARGET_RATIO": "0.6",
            "PAPERPILOT_CONTEXT_FULL_COMPACTION_RECENT_TURNS": "3",
            "PAPERPILOT_CONTEXT_COMPRESSION_FAILURE_THRESHOLD": "4",
            "PAPERPILOT_CONTEXT_COMPRESSION_TRANSIENT_RETRY_COUNT": "2",
            "PAPERPILOT_CONTEXT_COMPRESSION_BREAKER_COOLDOWN_SECONDS": "600",
            "PAPERPILOT_CONTEXT_SAFETY_MARGIN_RATIO": "0.08",
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
    assert config.research_recursion_limit == 30
    assert config.research_model_call_limit == 10
    assert config.research_tool_call_limit == 16
    assert config.research_max_output_tokens == 2048
    assert config.research_model_retries == 0
    assert config.research_model_name == "custom-model"
    context = config.context_management
    assert context.enabled is True
    assert context.full_compaction_enabled is True
    assert context.model_context_window_tokens == 200_000
    assert context.artifact_root == Path("/tmp/context-artifacts")
    assert context.tool_inline_max_tokens == 1500
    assert context.artifact_read_max_tokens == 1700
    assert context.micro_compaction_trigger_ratio == 0.9
    assert context.micro_compaction_min_reclaim_tokens == 9000
    assert context.micro_compaction_min_reclaim_ratio == 0.2
    assert context.micro_compaction_keep_recent_tool_results == 4
    assert context.archive_context_budget_tokens == 5000
    assert context.archive_context_max_records == 6
    assert context.archive_context_recent_records == 3
    assert context.full_compaction_trigger_ratio == 0.95
    assert context.session_memory_target_ratio == 0.7
    assert context.full_compaction_target_ratio == 0.6
    assert context.full_compaction_recent_turns == 3
    assert context.compression_failure_threshold == 4
    assert context.compression_transient_retry_count == 2
    assert context.compression_breaker_cooldown_seconds == 600
    assert context.context_safety_margin_ratio == 0.08


def test_context_config_rejects_invalid_relationships():
    with pytest.raises(ValueError, match="FULL_COMPACTION_ENABLED"):
        WebRuntimeConfig.from_env({"PAPERPILOT_FULL_COMPACTION_ENABLED": "true"})

    with pytest.raises(ValueError, match="full_compaction_target_ratio"):
        WebRuntimeConfig.from_env(
            {
                "PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED": "true",
                "PAPERPILOT_CONTEXT_FULL_COMPACTION_TARGET_RATIO": "0.8",
                "PAPERPILOT_CONTEXT_SESSION_MEMORY_TARGET_RATIO": "0.7",
                "PAPERPILOT_CONTEXT_FULL_COMPACTION_TRIGGER_RATIO": "0.9",
            }
        )

    with pytest.raises(ValueError, match="positive input budget"):
        WebRuntimeConfig.from_env(
            {
                "PAPERPILOT_CONTEXT_MODEL_WINDOW_TOKENS": "100",
                "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS": "96",
                "PAPERPILOT_CONTEXT_SAFETY_MARGIN_RATIO": "0.05",
            }
        )


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
        (
            "PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED",
            "maybe",
            "PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED",
        ),
        (
            "PAPERPILOT_CONTEXT_MODEL_WINDOW_TOKENS",
            "0",
            "PAPERPILOT_CONTEXT_MODEL_WINDOW_TOKENS",
        ),
        (
            "PAPERPILOT_CONTEXT_ARTIFACT_ROOT",
            "   ",
            "PAPERPILOT_CONTEXT_ARTIFACT_ROOT",
        ),
        (
            "PAPERPILOT_CONTEXT_MICRO_COMPACTION_TRIGGER_RATIO",
            "1",
            "PAPERPILOT_CONTEXT_MICRO_COMPACTION_TRIGGER_RATIO",
        ),
        (
            "PAPERPILOT_CONTEXT_SAFETY_MARGIN_RATIO",
            "0",
            "PAPERPILOT_CONTEXT_SAFETY_MARGIN_RATIO",
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
