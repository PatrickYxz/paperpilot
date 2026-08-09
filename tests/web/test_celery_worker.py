"""Celery configuration and process-local worker runtime tests."""
from __future__ import annotations

import threading

import pytest

from paperpilot.papers import PaperCandidate
from paperpilot.tools.mcp_runtime import MCPRuntime
from paperpilot.web.config import WebRuntimeConfig
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


class RetryScheduled(RuntimeError):
    pass


def _new_conversation_task(store: TaskStore):
    user = store.create_user(
        username="celery-retry-user",
        password_hash="hash",
        password_salt="salt",
    )
    conversation = store.create_conversation(
        user_id=user.id,
        paper=PaperCandidate(
            external_id="2401.99992v1",
            title="Celery retry paper",
            authors=["Grace Hopper"],
            abstract="Celery retry abstract.",
            source_url="https://arxiv.org/abs/2401.99992v1",
        ),
    )
    return store.create_conversation_turn(
        user_id=user.id,
        conversation_id=conversation.id,
        content="Retry this Celery conversation.",
        depth="standard",
        expected_head_message_id=None,
    ).task


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


def test_celery_task_retries_ordinary_execution_exception(monkeypatch):
    execution_error = ConnectionError("temporary checkpoint outage")
    retry_calls: list[dict[str, object]] = []

    def fail_execution(*_args, **_kwargs):
        raise execution_error

    def schedule_retry(**kwargs):
        retry_calls.append(kwargs)
        raise RetryScheduled("scheduled")

    monkeypatch.setattr(worker_tasks, "_execute_research_task", fail_execution)
    monkeypatch.setattr(
        worker_tasks.WebRuntimeConfig,
        "from_env",
        lambda: WebRuntimeConfig(),
    )
    monkeypatch.setattr(worker_tasks.execute_research_task, "retry", schedule_retry)

    result = worker_tasks.execute_research_task.apply(
        args=["task_retry", "real"],
        throw=False,
    )

    assert isinstance(result.result, RetryScheduled)
    assert retry_calls == [
        {
            "exc": execution_error,
            "countdown": 1,
            "max_retries": 3,
        }
    ]


def test_celery_retry_request_can_reclaim_running_task(monkeypatch):
    calls: list[dict[str, object]] = []

    def record_execution(task_id, execution_mode, *, redelivered=False, config=None):
        calls.append(
            {
                "task_id": task_id,
                "execution_mode": execution_mode,
                "redelivered": redelivered,
                "config": config,
            }
        )

    config = WebRuntimeConfig()
    monkeypatch.setattr(worker_tasks, "_execute_research_task", record_execution)
    monkeypatch.setattr(worker_tasks.WebRuntimeConfig, "from_env", lambda: config)

    result = worker_tasks.execute_research_task.apply(
        args=["task_running", "real"],
        retries=1,
        throw=False,
    )

    assert result.successful()
    assert calls == [
        {
            "task_id": "task_running",
            "execution_mode": "real",
            "redelivered": True,
            "config": config,
        }
    ]


def test_celery_retry_exhaustion_fails_real_sqlite_conversation(monkeypatch, tmp_path):
    db_path = tmp_path / "celery-retry.sqlite3"
    seed = TaskStore(db_path)
    task = _new_conversation_task(seed)
    seed.update_status(task.id, "running")
    seed.close()
    execution_error = OSError("database locked secret-paper-text")

    def fail_execution(*_args, **_kwargs):
        raise execution_error

    monkeypatch.setattr(worker_tasks, "_execute_research_task", fail_execution)
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: TaskStore(db_path))
    monkeypatch.setattr(
        worker_tasks.WebRuntimeConfig,
        "from_env",
        lambda: WebRuntimeConfig(),
    )

    result = worker_tasks.execute_research_task.apply(
        args=[task.id, "real"],
        retries=3,
        throw=False,
    )

    assert result.failed()
    assert result.result is execution_error
    check = TaskStore(db_path)
    assert check.get_task(task.id).status == "failed"
    events = check.list_events_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    )
    assert events is not None
    failures = [event for event in events.items if event.type == "failed"]
    assert len(failures) == 1
    failure = failures[0]
    assert failure.stage == "execution_retry_exhausted"
    assert failure.message == "Conversation execution failed after retry limit."
    assert failure.payload == {
        "backend": "celery",
        "attempts": 4,
        "max_retries": 3,
        "error_type": "OSError",
    }
    assert "database locked" not in str(failure.to_dict())
    assert "secret-paper-text" not in str(failure.to_dict())
    check.close()


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
    parent_lock = threading.Lock()
    parent_lock.acquire()
    monkeypatch.setattr(worker_tasks, "_runtime", parent_mcp)
    monkeypatch.setattr(
        worker_tasks,
        "_checkpoint_runtime",
        parent_checkpoint,
        raising=False,
    )
    monkeypatch.setattr(worker_tasks, "_runtime_lock", parent_lock)
    replacement_runtime = ClosableRuntime("replacement")
    monkeypatch.setattr(worker_tasks, "_runtime_factory", lambda: replacement_runtime)

    worker_tasks._reset_worker_resources()

    assert worker_tasks._runtime is None
    assert worker_tasks._checkpoint_runtime is None
    assert worker_tasks._runtime_lock is not parent_lock
    assert worker_tasks._runtime_lock.locked() is False
    assert worker_tasks._get_runtime() is replacement_runtime
    assert parent_mcp.close_count == 0
    assert parent_checkpoint.close_count == 0
    parent_lock.release()
