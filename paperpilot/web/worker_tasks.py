"""Celery worker tasks with one lazy MCP runtime per worker process."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from celery.signals import worker_process_init, worker_process_shutdown

from paperpilot.deep_reading.runner import DeepReadingRunner
from paperpilot.tools.mcp_runtime import MCPRuntime
from paperpilot.web.celery_app import (
    EXECUTE_RESEARCH_TASK_NAME,
    celery_app,
)
from paperpilot.web.checkpoint import SqliteCheckpointRuntime
from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.task_executor import task_retry_countdown_seconds
from paperpilot.web.task_store import TaskStore

_runtime: MCPRuntime | None = None
_checkpoint_runtime: SqliteCheckpointRuntime | None = None
_runtime_lock = threading.Lock()
_runtime_factory: Callable[[], MCPRuntime] = MCPRuntime
_checkpoint_runtime_factory: Callable[[Path], SqliteCheckpointRuntime] = (
    SqliteCheckpointRuntime.open
)
_store_factory: Callable[[], TaskStore] = TaskStore
_LOGGER = logging.getLogger("paperpilot.web.runtime")


class _RetryableConversationExecution(Exception):
    """Typed signal carrying a retryable Conversation infrastructure error."""

    def __init__(self, original: Exception) -> None:
        super().__init__("retryable Conversation execution failure")
        self.original = original


def _get_runtime() -> MCPRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = _runtime_factory()
        return _runtime


def _get_checkpoint_runtime(path: Path) -> SqliteCheckpointRuntime:
    global _checkpoint_runtime
    with _runtime_lock:
        if _checkpoint_runtime is None:
            _checkpoint_runtime = _checkpoint_runtime_factory(path)
        return _checkpoint_runtime


def _build_deep_reading_runner(
    store: TaskStore,
    mcp_runtime: MCPRuntime,
    checkpoint_runtime: SqliteCheckpointRuntime | None = None,
    config: WebRuntimeConfig | None = None,
) -> DeepReadingRunner:
    runtime_config = config or WebRuntimeConfig.from_env()
    checkpoint = (
        checkpoint_runtime
        if checkpoint_runtime is not None
        else _get_checkpoint_runtime(runtime_config.checkpoint_db_path)
    )
    return DeepReadingRunner(
        task_store=store,
        checkpointer=checkpoint.saver,
        mcp_runtime=mcp_runtime,
        summary_token_threshold=runtime_config.summary_token_threshold,
        summary_recent_turns=runtime_config.summary_recent_turns,
        research_recursion_limit=runtime_config.research_recursion_limit,
        research_model_call_limit=runtime_config.research_model_call_limit,
        research_tool_call_limit=runtime_config.research_tool_call_limit,
        research_max_output_tokens=runtime_config.research_max_output_tokens,
        research_model_retries=runtime_config.research_model_retries,
    )


def _execute_research_task(
    task_id: str,
    *,
    redelivered: bool = False,
    config: WebRuntimeConfig | None = None,
) -> None:
    store = _store_factory()
    retryable_error: _RetryableConversationExecution | None = None
    try:
        try:
            task = store.get_task(task_id)
        except Exception as exc:
            raise _RetryableConversationExecution(exc) from exc
        if task is None:
            raise ValueError(f"task not found: {task_id}")
        try:
            runtime = _get_runtime()
            deep_reading_runner = _build_deep_reading_runner(
                store,
                runtime,
                config=config,
            )
            deep_reading_runner.run(
                task_id,
                allow_running=redelivered,
            )
        except Exception as exc:
            raise _RetryableConversationExecution(exc) from exc
    except _RetryableConversationExecution as exc:
        retryable_error = exc
        raise
    finally:
        try:
            store.close()
        except Exception:
            if retryable_error is None:
                raise
            _LOGGER.exception(
                "Task store cleanup failed after Conversation execution",
                extra={
                    "event": "task.execution_cleanup_failed",
                    "executor": "celery",
                    "reason": "store_exception",
                },
            )


@celery_app.task(
    bind=True,
    name=EXECUTE_RESEARCH_TASK_NAME,
    ignore_result=True,
)
def execute_research_task(
    self,
    task_id: str,
) -> None:
    config = WebRuntimeConfig.from_env()
    delivery_info = self.request.delivery_info or {}
    retries = int(self.request.retries or 0)
    try:
        _execute_research_task(
            task_id,
            redelivered=(
                bool(delivery_info.get("redelivered")) or retries > 0
            ),
            config=config,
        )
    except _RetryableConversationExecution as retryable:
        exc = retryable.original
        if retries < config.task_max_retries:
            countdown = task_retry_countdown_seconds(
                retries,
                initial_seconds=config.task_retry_backoff_seconds,
                max_seconds=config.task_retry_backoff_max_seconds,
            )
            _LOGGER.warning(
                "Conversation execution retry %d/%d in %d seconds",
                retries + 1,
                config.task_max_retries,
                countdown,
                exc_info=(type(exc), exc, exc.__traceback__),
                extra={
                    "event": "task.execution_retry",
                    "executor": "celery",
                    "reason": "workflow_exception",
                    "exception_type": type(exc).__name__,
                },
            )
            raise self.retry(
                exc=exc,
                countdown=countdown,
                max_retries=config.task_max_retries,
            )

        try:
            _fail_retry_exhausted(
                task_id,
                attempts=retries + 1,
                max_retries=config.task_max_retries,
                exc=exc,
            )
        except Exception:
            _LOGGER.exception(
                "Conversation retry exhaustion finalization failed",
                extra={
                    "event": "task.execution_failure_finalize_failed",
                    "executor": "celery",
                    "reason": "store_exception",
                },
            )
        raise exc


def _fail_retry_exhausted(
    task_id: str,
    *,
    attempts: int,
    max_retries: int,
    exc: Exception,
) -> None:
    store = _store_factory()
    try:
        DeepReadingRunner(
            task_store=store,
            checkpointer=object(),
            mcp_runtime=object(),
        ).fail_retry_exhausted(
            task_id,
            backend="celery",
            attempts=attempts,
            max_retries=max_retries,
            exc=exc,
        )
    finally:
        store.close()


@worker_process_init.connect
def _reset_worker_resources(**kwargs) -> None:
    """Drop parent-process handles before the child can perform any work."""
    del kwargs
    global _runtime, _checkpoint_runtime, _runtime_lock
    # This signal runs synchronously during child initialization. Do not acquire
    # a lock copied from the parent: it could have been held by a vanished thread
    # at fork time.
    _runtime = None
    _checkpoint_runtime = None
    _runtime_lock = threading.Lock()


@worker_process_shutdown.connect
def _close_worker_runtime(**kwargs) -> None:
    del kwargs
    global _runtime, _checkpoint_runtime
    with _runtime_lock:
        runtime, _runtime = _runtime, None
        checkpoint_runtime, _checkpoint_runtime = _checkpoint_runtime, None
    try:
        if checkpoint_runtime is not None:
            checkpoint_runtime.close()
    finally:
        if runtime is not None:
            runtime.close()
