"""Validated runtime configuration for the PaperPilot Web backend."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, cast

TaskExecutorBackend = Literal["thread", "celery"]
LogFormat = Literal["json", "text"]
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


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
    research_recursion_limit: int = 12

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
                default=12,
                minimum=1,
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
