"""Celery application for durable PaperPilot research task dispatch."""
from __future__ import annotations

import os

from celery import Celery

DEFAULT_BROKER_URL = "redis://127.0.0.1:6379/0"
EXECUTE_RESEARCH_TASK_NAME = "paperpilot.web.execute_research_task"
DEFAULT_SOFT_TIME_LIMIT_SECONDS = 10_800
DEFAULT_TIME_LIMIT_SECONDS = 11_100
DEFAULT_VISIBILITY_TIMEOUT_SECONDS = 14_400


def create_celery_app() -> Celery:
    soft_time_limit = _positive_int_env(
        "PAPERPILOT_TASK_SOFT_TIME_LIMIT_SECONDS",
        DEFAULT_SOFT_TIME_LIMIT_SECONDS,
    )
    time_limit = _positive_int_env(
        "PAPERPILOT_TASK_TIME_LIMIT_SECONDS",
        DEFAULT_TIME_LIMIT_SECONDS,
    )
    visibility_timeout = _positive_int_env(
        "PAPERPILOT_REDIS_VISIBILITY_TIMEOUT_SECONDS",
        DEFAULT_VISIBILITY_TIMEOUT_SECONDS,
    )
    if soft_time_limit >= time_limit:
        raise ValueError("task soft time limit must be less than hard time limit")
    if time_limit >= visibility_timeout:
        raise ValueError("task time limit must be less than visibility timeout")

    app = Celery(
        "paperpilot",
        broker=os.environ.get(
            "PAPERPILOT_CELERY_BROKER_URL",
            DEFAULT_BROKER_URL,
        ),
        include=["paperpilot.web.worker_tasks"],
    )
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_cancel_long_running_tasks_on_connection_loss=True,
        broker_connection_retry_on_startup=True,
        task_publish_retry=True,
        task_publish_retry_policy={
            "max_retries": 5,
            "interval_start": 0,
            "interval_step": 0.5,
            "interval_max": 5,
        },
        task_soft_time_limit=soft_time_limit,
        task_time_limit=time_limit,
        broker_transport_options={
            "visibility_timeout": visibility_timeout,
        },
    )
    return app


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


celery_app = create_celery_app()
