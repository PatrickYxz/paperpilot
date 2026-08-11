"""Celery configuration and process-local worker runtime tests."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from paperpilot.papers import PaperCandidate
from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.celery_app import create_celery_app
from paperpilot.web.task_store import TaskStore
from paperpilot.web import worker_tasks


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


class RetryScheduled(RuntimeError):
    pass


class NonRetryableWorkerError(RuntimeError):
    pass


class FailingDeepReadingRunner:
    def __init__(self, exc: Exception, store=None) -> None:
        self.exc = exc
        self.store = store

    def run(self, task_id: str, *, allow_running: bool = False) -> bool:
        if self.store is not None:
            assert self.store.claim_task(
                task_id,
                allow_running=allow_running,
            ) is not None
        raise self.exc


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


def _capture_retry(monkeypatch):
    retry_calls: list[dict[str, object]] = []

    def schedule_retry(**kwargs):
        retry_calls.append(kwargs)
        raise RetryScheduled("scheduled")

    monkeypatch.setattr(worker_tasks.execute_research_task, "retry", schedule_retry)
    return retry_calls


def _configure_failing_conversation(
    monkeypatch,
    *,
    db_path,
    execution_error: Exception,
    store_factory=None,
) -> None:
    monkeypatch.setattr(
        worker_tasks,
        "_store_factory",
        store_factory or (lambda: TaskStore(db_path)),
    )
    monkeypatch.setattr(worker_tasks, "_get_runtime", lambda: object())
    monkeypatch.setattr(
        worker_tasks,
        "_build_deep_reading_runner",
        lambda store, *_args, **_kwargs: FailingDeepReadingRunner(
            execution_error,
            store,
        ),
    )
    monkeypatch.setattr(
        worker_tasks.WebRuntimeConfig,
        "from_env",
        lambda: WebRuntimeConfig(),
    )


def test_mode_free_celery_worker_passes_retry_recovery_flag(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    config = WebRuntimeConfig()

    def record_execution(task_id, *, redelivered=False, config=None):
        calls.append(
            {
                "task_id": task_id,
                "redelivered": redelivered,
                "config": config,
            }
        )

    monkeypatch.setattr(worker_tasks, "_execute_research_task", record_execution)
    monkeypatch.setattr(worker_tasks.WebRuntimeConfig, "from_env", lambda: config)

    result = worker_tasks.execute_research_task.apply(
        args=["task_running"],
        retries=1,
        throw=False,
    )

    assert result.successful()
    assert calls == [
        {
            "task_id": "task_running",
            "redelivered": True,
            "config": config,
        }
    ]


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


def test_celery_task_retries_real_conversation_execution_exception(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "conversation-retry.sqlite3"
    seed = TaskStore(db_path)
    task = _new_conversation_task(seed)
    seed.close()
    execution_error = ConnectionError("temporary checkpoint outage")
    _configure_failing_conversation(
        monkeypatch,
        db_path=db_path,
        execution_error=execution_error,
    )
    retry_calls = _capture_retry(monkeypatch)

    result = worker_tasks.execute_research_task.apply(
        args=[task.id],
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
    check = TaskStore(db_path)
    assert check.get_task(task.id).status == "running"
    check.close()


@pytest.mark.parametrize("failure_point", ["mcp", "checkpoint"])
def test_celery_task_retries_real_conversation_resource_failure(
    monkeypatch,
    tmp_path,
    failure_point,
):
    db_path = tmp_path / f"conversation-{failure_point}.sqlite3"
    seed = TaskStore(db_path)
    task = _new_conversation_task(seed)
    seed.close()
    execution_error = ConnectionError(f"{failure_point} unavailable")
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: TaskStore(db_path))
    if failure_point == "mcp":
        monkeypatch.setattr(
            worker_tasks,
            "_get_runtime",
            lambda: (_ for _ in ()).throw(execution_error),
        )
    else:
        monkeypatch.setattr(worker_tasks, "_get_runtime", lambda: object())
        monkeypatch.setattr(
            worker_tasks,
            "_build_deep_reading_runner",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(execution_error),
        )
    monkeypatch.setattr(
        worker_tasks.WebRuntimeConfig,
        "from_env",
        lambda: WebRuntimeConfig(),
    )
    retry_calls = _capture_retry(monkeypatch)

    result = worker_tasks.execute_research_task.apply(
        args=[task.id],
        throw=False,
    )

    assert isinstance(result.result, RetryScheduled)
    assert retry_calls[0]["exc"] is execution_error


def test_celery_task_does_not_retry_missing_task(monkeypatch, tmp_path):
    db_path = tmp_path / "missing.sqlite3"
    seed = TaskStore(db_path)
    seed.close()
    retry_calls = _capture_retry(monkeypatch)
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: TaskStore(db_path))
    monkeypatch.setattr(
        worker_tasks.WebRuntimeConfig,
        "from_env",
        lambda: WebRuntimeConfig(),
    )

    result = worker_tasks.execute_research_task.apply(
        args=["task_missing"],
        throw=False,
    )

    assert result.failed()
    assert isinstance(result.result, ValueError)
    assert retry_calls == []


def test_celery_task_does_not_retry_store_construction_failure(monkeypatch):
    store_error = NonRetryableWorkerError("store unavailable")
    retry_calls = _capture_retry(monkeypatch)
    monkeypatch.setattr(
        worker_tasks,
        "_store_factory",
        lambda: (_ for _ in ()).throw(store_error),
    )
    monkeypatch.setattr(
        worker_tasks.WebRuntimeConfig,
        "from_env",
        lambda: WebRuntimeConfig(),
    )

    result = worker_tasks.execute_research_task.apply(
        args=["task_store"],
        throw=False,
    )

    assert result.failed()
    assert result.result is store_error
    assert retry_calls == []


def test_celery_task_retries_claim_infrastructure_failure(monkeypatch):
    claim_error = OSError("claim unavailable")
    retry_calls = _capture_retry(monkeypatch)

    class ClaimFailingStore:
        close_calls = 0

        def get_task(self, task_id):
            return SimpleNamespace(id=task_id)

        def claim_task(self, task_id, *, allow_running=False):
            raise claim_error

        def close(self):
            self.close_calls += 1

    store = ClaimFailingStore()
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: store)
    monkeypatch.setattr(worker_tasks, "_get_runtime", lambda: object())

    class ClaimFailingRunner:
        def run(self, task_id, *, allow_running=False):
            return store.claim_task(task_id, allow_running=allow_running)

    monkeypatch.setattr(
        worker_tasks,
        "_build_deep_reading_runner",
        lambda *_args, **_kwargs: ClaimFailingRunner(),
    )
    monkeypatch.setattr(
        worker_tasks.WebRuntimeConfig,
        "from_env",
        lambda: WebRuntimeConfig(),
    )

    result = worker_tasks.execute_research_task.apply(
        args=["task_claim"],
        throw=False,
    )

    assert isinstance(result.result, RetryScheduled)
    assert retry_calls[0]["exc"] is claim_error
    assert store.close_calls == 1


def test_celery_task_does_not_retry_standalone_cleanup_failure(monkeypatch):
    cleanup_error = NonRetryableWorkerError("cleanup failed")
    retry_calls = _capture_retry(monkeypatch)

    class CleanupFailingStore:
        def claim_task(self, task_id, *, allow_running=False):
            return None

        def get_task(self, task_id):
            return SimpleNamespace(id=task_id)

        def close(self):
            raise cleanup_error

    monkeypatch.setattr(worker_tasks, "_store_factory", CleanupFailingStore)
    monkeypatch.setattr(worker_tasks, "_get_runtime", lambda: object())

    class NoopRunner:
        def run(self, task_id, *, allow_running=False):
            del task_id, allow_running
            return False

    monkeypatch.setattr(
        worker_tasks,
        "_build_deep_reading_runner",
        lambda *_args, **_kwargs: NoopRunner(),
    )
    monkeypatch.setattr(
        worker_tasks.WebRuntimeConfig,
        "from_env",
        lambda: WebRuntimeConfig(),
    )

    result = worker_tasks.execute_research_task.apply(
        args=["task_cleanup"],
        throw=False,
    )

    assert result.failed()
    assert result.result is cleanup_error
    assert retry_calls == []


def test_conversation_execution_error_wins_over_cleanup_failure(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "conversation-cleanup.sqlite3"
    seed = TaskStore(db_path)
    task = _new_conversation_task(seed)
    seed.close()
    execution_error = ConnectionError("checkpoint unavailable")
    cleanup_error = NonRetryableWorkerError("cleanup failed")

    class CloseFailingStore:
        def __init__(self) -> None:
            self.inner = TaskStore(db_path)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def close(self):
            self.inner.close()
            raise cleanup_error

    _configure_failing_conversation(
        monkeypatch,
        db_path=db_path,
        execution_error=execution_error,
        store_factory=CloseFailingStore,
    )
    retry_calls = _capture_retry(monkeypatch)

    result = worker_tasks.execute_research_task.apply(
        args=[task.id],
        throw=False,
    )

    assert isinstance(result.result, RetryScheduled)
    assert retry_calls[0]["exc"] is execution_error


def test_celery_retry_exhaustion_fails_real_sqlite_conversation(monkeypatch, tmp_path):
    db_path = tmp_path / "celery-retry.sqlite3"
    seed = TaskStore(db_path)
    task = _new_conversation_task(seed)
    seed.update_status(task.id, "running")
    seed.close()
    execution_error = OSError("database locked secret-paper-text")
    _configure_failing_conversation(
        monkeypatch,
        db_path=db_path,
        execution_error=execution_error,
    )

    result = worker_tasks.execute_research_task.apply(
        args=[task.id],
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
    task = _new_conversation_task(store)
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: store)
    monkeypatch.setattr(worker_tasks, "_get_runtime", lambda: object())

    class SuccessfulRunner:
        def run(self, task_id, *, allow_running=False):
            del task_id, allow_running
            return True

    monkeypatch.setattr(
        worker_tasks,
        "_build_deep_reading_runner",
        lambda *_args, **_kwargs: SuccessfulRunner(),
    )

    worker_tasks._execute_research_task(task.id)

    assert store.close_calls == 1


def test_worker_closes_task_store_after_failure(tmp_path, monkeypatch):
    store = TrackingTaskStore(tmp_path / "failure.sqlite3")
    task = _new_conversation_task(store)
    failure = RuntimeError("runner failed before cleanup")
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: store)
    monkeypatch.setattr(worker_tasks, "_get_runtime", lambda: object())
    monkeypatch.setattr(
        worker_tasks,
        "_build_deep_reading_runner",
        lambda *_args, **_kwargs: FailingDeepReadingRunner(failure),
    )

    with pytest.raises(worker_tasks._RetryableConversationExecution) as exc_info:
        worker_tasks._execute_research_task(task.id)

    assert exc_info.value.original is failure
    assert store.close_calls == 1


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
