"""Celery worker tasks with one lazy MCP runtime per worker process."""
from __future__ import annotations

import threading
from collections.abc import Callable

from celery.signals import worker_process_shutdown

from paperpilot.conversation import run as run_conversation
from paperpilot.tools.mcp_runtime import MCPRuntime
from paperpilot.web.celery_app import (
    EXECUTE_RESEARCH_TASK_NAME,
    celery_app,
)
from paperpilot.web.task_executor import ExecutionMode
from paperpilot.web.task_store import TaskStore
from paperpilot.web.workflow import WorkflowRunner

_runtime: MCPRuntime | None = None
_runtime_lock = threading.Lock()
_runtime_factory: Callable[[], MCPRuntime] = MCPRuntime
_store_factory: Callable[[], TaskStore] = TaskStore


def _get_runtime() -> MCPRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = _runtime_factory()
        return _runtime


def _execute_research_task(
    task_id: str,
    execution_mode: ExecutionMode,
    *,
    redelivered: bool = False,
) -> None:
    if execution_mode not in {"simulated", "real"}:
        raise ValueError(f"invalid execution mode: {execution_mode!r}")
    store = _store_factory()
    try:
        claimed = store.claim_task(task_id, allow_running=redelivered)
        if claimed is None and store.get_task(task_id) is None:
            raise ValueError(f"task not found: {task_id}")
        if claimed is None:
            return
        if execution_mode == "simulated":
            WorkflowRunner(store).run_simulated(task_id)
            return

        runtime = _get_runtime()

        def real_runner(query: str, *, on_event=None) -> list[dict]:
            return run_conversation(
                query,
                on_event=on_event,
                mcp_runtime=runtime,
            )

        WorkflowRunner(store, real_runner=real_runner).run_real(task_id)
    finally:
        store.close()


@celery_app.task(
    bind=True,
    name=EXECUTE_RESEARCH_TASK_NAME,
    ignore_result=True,
)
def execute_research_task(
    self,
    task_id: str,
    execution_mode: ExecutionMode,
) -> None:
    delivery_info = self.request.delivery_info or {}
    _execute_research_task(
        task_id,
        execution_mode,
        redelivered=bool(delivery_info.get("redelivered")),
    )


@worker_process_shutdown.connect
def _close_worker_runtime(**kwargs) -> None:
    global _runtime
    with _runtime_lock:
        runtime, _runtime = _runtime, None
    if runtime is not None:
        runtime.close()
