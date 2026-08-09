"""Celery configuration and process-local worker runtime tests."""
from __future__ import annotations

import pytest

from paperpilot.tools.mcp_runtime import MCPRuntime
from paperpilot.web.celery_app import create_celery_app
from paperpilot.web.task_store import TaskStore
from paperpilot.web import worker_tasks


class RuntimeFactory:
    def __init__(self) -> None:
        self.instances: list[MCPRuntime] = []

    def __call__(self) -> MCPRuntime:
        runtime = MCPRuntime(lambda: None)  # The fake runner never leases tools.
        self.instances.append(runtime)
        return runtime


class ClosableRuntime:
    def __init__(self, name: str = "runtime", close_order=None) -> None:
        self.name = name
        self.close_order = close_order
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        if self.close_order is not None:
            self.close_order.append(self.name)


class TrackingTaskStore(TaskStore):
    def __init__(self, db_path) -> None:
        super().__init__(db_path)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


class RaisingWorkflowRunner:
    def __init__(self, store: TaskStore) -> None:
        self.store = store

    def run_simulated(self, task_id: str) -> None:
        raise RuntimeError("workflow failed before cleanup")


def test_celery_app_uses_long_task_safety_settings(monkeypatch):
    monkeypatch.setenv(
        "PAPERPILOT_CELERY_BROKER_URL",
        "redis://example.invalid:6379/4",
    )

    app = create_celery_app()

    assert app.conf.broker_url == "redis://example.invalid:6379/4"
    assert app.conf.task_serializer == "json"
    assert app.conf.accept_content == ["json"]
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.task_acks_late is True
    assert app.conf.task_reject_on_worker_lost is True
    assert app.conf.worker_cancel_long_running_tasks_on_connection_loss is True
    assert app.conf.broker_connection_retry_on_startup is True
    assert app.conf.task_publish_retry is True
    assert app.conf.task_soft_time_limit == 10_800
    assert app.conf.task_time_limit == 11_100
    assert app.conf.broker_transport_options["visibility_timeout"] == 14_400


def test_celery_app_rejects_time_limits_beyond_visibility_timeout(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_TASK_SOFT_TIME_LIMIT_SECONDS", "100")
    monkeypatch.setenv("PAPERPILOT_TASK_TIME_LIMIT_SECONDS", "200")
    monkeypatch.setenv("PAPERPILOT_REDIS_VISIBILITY_TIMEOUT_SECONDS", "150")

    with pytest.raises(ValueError, match="visibility timeout"):
        create_celery_app()


def test_worker_closes_task_store_after_success(tmp_path, monkeypatch):
    store = TrackingTaskStore(tmp_path / "success.sqlite3")
    task = store.create_task(question="close after success")
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: store)

    worker_tasks._execute_research_task(task.id, "simulated")

    assert store.close_calls == 1


def test_worker_closes_task_store_after_failure(tmp_path, monkeypatch):
    store = TrackingTaskStore(tmp_path / "failure.sqlite3")
    task = store.create_task(question="close after failure")
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: store)
    monkeypatch.setattr(worker_tasks, "WorkflowRunner", RaisingWorkflowRunner)

    with pytest.raises(RuntimeError, match="workflow failed before cleanup"):
        worker_tasks._execute_research_task(task.id, "simulated")

    assert store.close_calls == 1


def test_worker_reuses_one_runtime_for_two_real_tasks(tmp_path, monkeypatch):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    first = store.create_task(question="first")
    second = store.create_task(question="second")
    runtime_factory = RuntimeFactory()
    seen_runtimes: list[MCPRuntime] = []

    def fake_run(query, *, on_event=None, mcp_runtime=None):
        seen_runtimes.append(mcp_runtime)
        return [{"role": "assistant", "content": f"answer: {query}"}]

    monkeypatch.setattr(worker_tasks, "_runtime", None)
    monkeypatch.setattr(worker_tasks, "_runtime_factory", runtime_factory)
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: store)
    monkeypatch.setattr(worker_tasks, "run_conversation", fake_run)

    worker_tasks._execute_research_task(first.id, "real")
    worker_tasks._execute_research_task(second.id, "real")

    assert len(runtime_factory.instances) == 1
    assert seen_runtimes == [runtime_factory.instances[0], runtime_factory.instances[0]]
    assert store.get_task(first.id).status == "completed"
    assert store.get_task(second.id).status == "completed"


def test_simulated_worker_task_does_not_create_runtime(tmp_path, monkeypatch):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="simulation")
    runtime_factory = RuntimeFactory()

    monkeypatch.setattr(worker_tasks, "_runtime", None)
    monkeypatch.setattr(worker_tasks, "_runtime_factory", runtime_factory)
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: store)

    worker_tasks._execute_research_task(task.id, "simulated")

    assert runtime_factory.instances == []
    assert store.get_task(task.id).status == "completed"


def test_worker_skips_redelivered_completed_task(tmp_path, monkeypatch):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="already complete")
    store.update_status(task.id, "completed")
    runtime_factory = RuntimeFactory()

    monkeypatch.setattr(worker_tasks, "_runtime", None)
    monkeypatch.setattr(worker_tasks, "_runtime_factory", runtime_factory)
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: store)

    worker_tasks._execute_research_task(task.id, "real")

    assert runtime_factory.instances == []
    event_page = store.list_events_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    )
    assert event_page is not None
    assert event_page.items == []


def test_only_redelivered_message_can_recover_running_task(tmp_path, monkeypatch):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="recover me")
    store.update_status(task.id, "running")
    runtime_factory = RuntimeFactory()
    seen: list[str] = []

    def fake_run(query, *, on_event=None, mcp_runtime=None):
        seen.append(query)
        return [{"role": "assistant", "content": "recovered"}]

    monkeypatch.setattr(worker_tasks, "_runtime", None)
    monkeypatch.setattr(worker_tasks, "_runtime_factory", runtime_factory)
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: store)
    monkeypatch.setattr(worker_tasks, "run_conversation", fake_run)

    worker_tasks._execute_research_task(task.id, "real")
    assert seen == []

    worker_tasks._execute_research_task(task.id, "real", redelivered=True)

    assert seen == ["recover me"]
    assert store.get_task(task.id).status == "completed"


def test_worker_shutdown_closes_and_forgets_runtime(monkeypatch):
    close_order: list[str] = []
    runtime = ClosableRuntime("mcp", close_order)
    checkpoint_runtime = ClosableRuntime("checkpoint", close_order)
    monkeypatch.setattr(worker_tasks, "_runtime", runtime)
    monkeypatch.setattr(
        worker_tasks,
        "_checkpoint_runtime",
        checkpoint_runtime,
        raising=False,
    )

    worker_tasks._close_worker_runtime()

    assert runtime.close_count == 1
    assert checkpoint_runtime.close_count == 1
    assert close_order == ["checkpoint", "mcp"]
    assert worker_tasks._runtime is None
    assert worker_tasks._checkpoint_runtime is None

    worker_tasks._close_worker_runtime()
    assert runtime.close_count == 1
    assert checkpoint_runtime.close_count == 1


def test_worker_process_init_forgets_parent_process_resources(monkeypatch):
    parent_mcp = ClosableRuntime("mcp")
    parent_checkpoint = ClosableRuntime("checkpoint")
    monkeypatch.setattr(worker_tasks, "_runtime", parent_mcp)
    monkeypatch.setattr(
        worker_tasks,
        "_checkpoint_runtime",
        parent_checkpoint,
        raising=False,
    )

    worker_tasks._reset_worker_resources()

    assert worker_tasks._runtime is None
    assert worker_tasks._checkpoint_runtime is None
    assert parent_mcp.close_count == 0
    assert parent_checkpoint.close_count == 0
