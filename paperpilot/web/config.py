"""Validated runtime configuration for the PaperPilot Web backend."""
from __future__ import annotations

import os
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Literal, Mapping, cast

TaskExecutorBackend = Literal["thread", "celery"]
LogFormat = Literal["json", "text"]
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


@dataclass(frozen=True)
class ContextManagementConfig:
    """Immutable bounds for the provider-neutral context manager."""

    enabled: bool = False
    full_compaction_enabled: bool = False
    model_context_window_tokens: int = 1_048_576
    artifact_root: Path = Path("data/context-artifacts")
    tool_inline_max_tokens: int = 2_000
    artifact_read_max_tokens: int = 2_000
    micro_compaction_trigger_ratio: float = 0.70
    micro_compaction_min_reclaim_tokens: int = 8_000
    micro_compaction_min_reclaim_ratio: float = 0.10
    micro_compaction_keep_recent_tool_results: int = 3
    archive_context_budget_tokens: int = 4_000
    archive_context_max_records: int = 5
    archive_context_recent_records: int = 2
    full_compaction_trigger_ratio: float = 0.80
    session_memory_target_ratio: float = 0.65
    full_compaction_target_ratio: float = 0.50
    full_compaction_recent_turns: int = 2
    compression_failure_threshold: int = 3
    compression_transient_retry_count: int = 1
    compression_breaker_cooldown_seconds: int = 300
    context_safety_margin_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.full_compaction_enabled and not self.enabled:
            raise ValueError(
                "PAPERPILOT_FULL_COMPACTION_ENABLED requires "
                "PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED"
            )
        if not str(self.artifact_root).strip():
            raise ValueError("PAPERPILOT_CONTEXT_ARTIFACT_ROOT must not be empty")
        positive_ints = {
            "model_context_window_tokens": self.model_context_window_tokens,
            "tool_inline_max_tokens": self.tool_inline_max_tokens,
            "artifact_read_max_tokens": self.artifact_read_max_tokens,
            "micro_compaction_min_reclaim_tokens": self.micro_compaction_min_reclaim_tokens,
            "micro_compaction_keep_recent_tool_results": self.micro_compaction_keep_recent_tool_results,
            "archive_context_budget_tokens": self.archive_context_budget_tokens,
            "archive_context_max_records": self.archive_context_max_records,
            "archive_context_recent_records": self.archive_context_recent_records,
            "full_compaction_recent_turns": self.full_compaction_recent_turns,
            "compression_failure_threshold": self.compression_failure_threshold,
            "compression_transient_retry_count": self.compression_transient_retry_count,
            "compression_breaker_cooldown_seconds": self.compression_breaker_cooldown_seconds,
        }
        for name, value in positive_ints.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.archive_context_recent_records > self.archive_context_max_records:
            raise ValueError(
                "archive_context_recent_records must not exceed "
                "archive_context_max_records"
            )
        ratios = {
            "micro_compaction_trigger_ratio": self.micro_compaction_trigger_ratio,
            "micro_compaction_min_reclaim_ratio": self.micro_compaction_min_reclaim_ratio,
            "full_compaction_trigger_ratio": self.full_compaction_trigger_ratio,
            "session_memory_target_ratio": self.session_memory_target_ratio,
            "full_compaction_target_ratio": self.full_compaction_target_ratio,
            "context_safety_margin_ratio": self.context_safety_margin_ratio,
        }
        for name, value in ratios.items():
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not (
            0
            < self.full_compaction_target_ratio
            < self.session_memory_target_ratio
            < self.full_compaction_trigger_ratio
            < 1
        ):
            raise ValueError(
                "full_compaction_target_ratio, session_memory_target_ratio, "
                "and full_compaction_trigger_ratio must satisfy "
                "0 < full_target < session_target < trigger < 1"
            )


@dataclass(frozen=True)
class WebRuntimeConfig:
    task_executor: TaskExecutorBackend = "thread"
    thread_workers: int = 2
    thread_queue_capacity: int = 4
    overload_retry_after_seconds: int = 1
    task_max_retries: int = 3
    task_retry_backoff_seconds: int = 1
    task_retry_backoff_max_seconds: int = 30
    log_level: str = "INFO"
    log_format: LogFormat = "json"
    slow_request_ms: int = 1000
    environment: str = "development"
    checkpoint_db_path: Path = Path("data/langgraph/checkpoints.sqlite3")
    summary_token_threshold: int = 32_000
    summary_recent_turns: int = 6
    research_recursion_limit: int = 24
    research_model_call_limit: int = 12
    research_tool_call_limit: int = 12
    research_max_output_tokens: int = 4096
    research_model_retries: int = 1
    research_model_name: str = "deepseek-v4-flash"
    context_management: ContextManagementConfig = ContextManagementConfig()

    def __post_init__(self) -> None:
        if not self.research_model_name.strip():
            raise ValueError("PAPERPILOT_RESEARCH_MODEL_NAME must not be empty")
        usable = (
            self.context_management.model_context_window_tokens
            - self.research_max_output_tokens
            - ceil(
                self.context_management.model_context_window_tokens
                * self.context_management.context_safety_margin_ratio
            )
        )
        if usable <= 0:
            raise ValueError(
                "context configuration must leave a positive input budget"
            )

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "WebRuntimeConfig":
        values = os.environ if environ is None else environ
        task_executor = values.get("PAPERPILOT_TASK_EXECUTOR", "thread").strip().lower()
        if task_executor not in {"thread", "celery"}:
            raise ValueError(
                "PAPERPILOT_TASK_EXECUTOR must be 'thread' or 'celery'"
            )

        log_level = values.get("PAPERPILOT_LOG_LEVEL", "INFO").strip().upper()
        if log_level not in VALID_LOG_LEVELS:
            raise ValueError(
                "PAPERPILOT_LOG_LEVEL must be one of "
                + ", ".join(sorted(VALID_LOG_LEVELS))
            )

        log_format = values.get("PAPERPILOT_LOG_FORMAT", "json").strip().lower()
        if log_format not in {"json", "text"}:
            raise ValueError("PAPERPILOT_LOG_FORMAT must be 'json' or 'text'")

        environment = values.get("PAPERPILOT_ENV", "development").strip()
        if not environment:
            raise ValueError("PAPERPILOT_ENV must not be empty")

        research_model_name = values.get(
            "PAPERPILOT_RESEARCH_MODEL_NAME", "deepseek-v4-flash"
        ).strip()
        if not research_model_name:
            raise ValueError("PAPERPILOT_RESEARCH_MODEL_NAME must not be empty")

        task_retry_backoff_seconds = _read_int(
            values,
            "PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS",
            default=1,
            minimum=1,
        )
        task_retry_backoff_max_seconds = _read_int(
            values,
            "PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS",
            default=30,
            minimum=1,
        )
        if task_retry_backoff_seconds > task_retry_backoff_max_seconds:
            raise ValueError(
                "PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS must not exceed "
                "PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS"
            )

        return cls(
            task_executor=cast(TaskExecutorBackend, task_executor),
            thread_workers=_read_int(
                values, "PAPERPILOT_THREAD_WORKERS", default=2, minimum=1
            ),
            thread_queue_capacity=_read_int(
                values,
                "PAPERPILOT_THREAD_QUEUE_CAPACITY",
                default=4,
                minimum=0,
            ),
            overload_retry_after_seconds=_read_int(
                values,
                "PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS",
                default=1,
                minimum=1,
            ),
            task_max_retries=_read_int(
                values,
                "PAPERPILOT_TASK_MAX_RETRIES",
                default=3,
                minimum=0,
            ),
            task_retry_backoff_seconds=task_retry_backoff_seconds,
            task_retry_backoff_max_seconds=task_retry_backoff_max_seconds,
            log_level=log_level,
            log_format=cast(LogFormat, log_format),
            slow_request_ms=_read_int(
                values, "PAPERPILOT_SLOW_REQUEST_MS", default=1000, minimum=1
            ),
            environment=environment,
            checkpoint_db_path=Path(
                values.get(
                    "PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH",
                    "data/langgraph/checkpoints.sqlite3",
                ).strip()
            ),
            summary_token_threshold=_read_int(
                values,
                "PAPERPILOT_SUMMARY_TOKEN_THRESHOLD",
                default=32_000,
                minimum=1,
            ),
            summary_recent_turns=_read_int(
                values,
                "PAPERPILOT_SUMMARY_RECENT_TURNS",
                default=6,
                minimum=1,
            ),
            research_recursion_limit=_read_int(
                values,
                "PAPERPILOT_RESEARCH_RECURSION_LIMIT",
                default=24,
                minimum=1,
            ),
            research_model_call_limit=_read_int(
                values,
                "PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT",
                default=12,
                minimum=2,
            ),
            research_tool_call_limit=_read_int(
                values,
                "PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT",
                default=12,
                minimum=2,
            ),
            research_max_output_tokens=_read_int(
                values,
                "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS",
                default=4096,
                minimum=1,
            ),
            research_model_retries=_read_int(
                values,
                "PAPERPILOT_RESEARCH_MODEL_RETRIES",
                default=1,
                minimum=0,
            ),
            research_model_name=research_model_name,
            context_management=ContextManagementConfig(
                enabled=_read_bool(
                    values, "PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED", default=False
                ),
                full_compaction_enabled=_read_bool(
                    values, "PAPERPILOT_FULL_COMPACTION_ENABLED", default=False
                ),
                model_context_window_tokens=_read_int(
                    values,
                    "PAPERPILOT_CONTEXT_MODEL_WINDOW_TOKENS",
                    default=1_048_576,
                    minimum=1,
                ),
                artifact_root=_read_path(
                    values,
                    "PAPERPILOT_CONTEXT_ARTIFACT_ROOT",
                    default="data/context-artifacts",
                ),
                tool_inline_max_tokens=_read_int(
                    values,
                    "PAPERPILOT_CONTEXT_TOOL_INLINE_MAX_TOKENS",
                    default=2_000,
                    minimum=1,
                ),
                artifact_read_max_tokens=_read_int(
                    values,
                    "PAPERPILOT_CONTEXT_ARTIFACT_READ_MAX_TOKENS",
                    default=2_000,
                    minimum=1,
                ),
                micro_compaction_trigger_ratio=_read_ratio(
                    values,
                    "PAPERPILOT_CONTEXT_MICRO_COMPACTION_TRIGGER_RATIO",
                    default=0.70,
                ),
                micro_compaction_min_reclaim_tokens=_read_int(
                    values,
                    "PAPERPILOT_CONTEXT_MICRO_COMPACTION_MIN_RECLAIM_TOKENS",
                    default=8_000,
                    minimum=1,
                ),
                micro_compaction_min_reclaim_ratio=_read_ratio(
                    values,
                    "PAPERPILOT_CONTEXT_MICRO_COMPACTION_MIN_RECLAIM_RATIO",
                    default=0.10,
                ),
                micro_compaction_keep_recent_tool_results=_read_int(
                    values,
                    "PAPERPILOT_CONTEXT_MICRO_COMPACTION_KEEP_RECENT_TOOL_RESULTS",
                    default=3,
                    minimum=1,
                ),
                archive_context_budget_tokens=_read_int(
                    values,
                    "PAPERPILOT_CONTEXT_ARCHIVE_BUDGET_TOKENS",
                    default=4_000,
                    minimum=1,
                ),
                archive_context_max_records=_read_int(
                    values,
                    "PAPERPILOT_CONTEXT_ARCHIVE_MAX_RECORDS",
                    default=5,
                    minimum=1,
                ),
                archive_context_recent_records=_read_int(
                    values,
                    "PAPERPILOT_CONTEXT_ARCHIVE_RECENT_RECORDS",
                    default=2,
                    minimum=1,
                ),
                full_compaction_trigger_ratio=_read_ratio(
                    values,
                    "PAPERPILOT_CONTEXT_FULL_COMPACTION_TRIGGER_RATIO",
                    default=0.80,
                ),
                session_memory_target_ratio=_read_ratio(
                    values,
                    "PAPERPILOT_CONTEXT_SESSION_MEMORY_TARGET_RATIO",
                    default=0.65,
                ),
                full_compaction_target_ratio=_read_ratio(
                    values,
                    "PAPERPILOT_CONTEXT_FULL_COMPACTION_TARGET_RATIO",
                    default=0.50,
                ),
                full_compaction_recent_turns=_read_int(
                    values,
                    "PAPERPILOT_CONTEXT_FULL_COMPACTION_RECENT_TURNS",
                    default=2,
                    minimum=1,
                ),
                compression_failure_threshold=_read_int(
                    values,
                    "PAPERPILOT_CONTEXT_COMPRESSION_FAILURE_THRESHOLD",
                    default=3,
                    minimum=1,
                ),
                compression_transient_retry_count=_read_int(
                    values,
                    "PAPERPILOT_CONTEXT_COMPRESSION_TRANSIENT_RETRY_COUNT",
                    default=1,
                    minimum=0,
                ),
                compression_breaker_cooldown_seconds=_read_int(
                    values,
                    "PAPERPILOT_CONTEXT_COMPRESSION_BREAKER_COOLDOWN_SECONDS",
                    default=300,
                    minimum=1,
                ),
                context_safety_margin_ratio=_read_ratio(
                    values,
                    "PAPERPILOT_CONTEXT_SAFETY_MARGIN_RATIO",
                    default=0.05,
                ),
            ),
        )


def _read_int(
    environ: Mapping[str, str],
    name: str,
    *,
    default: int,
    minimum: int,
) -> int:
    raw = environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _read_bool(
    environ: Mapping[str, str],
    name: str,
    *,
    default: bool,
) -> bool:
    raw = environ.get(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _read_ratio(
    environ: Mapping[str, str],
    name: str,
    *,
    default: float,
) -> float:
    raw = environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _read_path(
    environ: Mapping[str, str],
    name: str,
    *,
    default: str,
) -> Path:
    raw = environ.get(name, default).strip()
    if not raw:
        raise ValueError(f"{name} must not be empty")
    return Path(raw)
