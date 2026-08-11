"""FastAPI composition, lifecycle, health, and static-hosting tests."""
from __future__ import annotations

import logging
import sqlite3

import pytest
from fastapi.testclient import TestClient

import paperpilot.web.app as web_app_module
from paperpilot.web.app import create_app
from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.observability import ACCESS_LOGGER_NAME
from paperpilot.web.task_executor import SynchronousTaskExecutor
from paperpilot.web.task_store import TaskStore


class _UnusedRunner:
    def run(self, task_id: str, *, allow_running: bool = False) -> bool:
        raise AssertionError(f"unexpected task execution: {task_id}, {allow_running}")

    def fail_retry_exhausted(self, *_args, **_kwargs) -> None:
        raise AssertionError("unexpected retry exhaustion")


class FailingExecutor:
    is_shutdown = False

    def shutdown(self) -> None:
        self.is_shutdown = True


class FailingHealthStore(TaskStore):
    def check_health(self) -> None:
        raise sqlite3.OperationalError("database unavailable")


class FailingHealthCheckpoint:
    saver = object()

    def __init__(self) -> None:
        self.health_calls = 0

    def check_health(self) -> None:
        self.health_calls += 1
        raise sqlite3.OperationalError("checkpoint unavailable")

    def close(self) -> None:
        pass


class TrackingTaskStore(TaskStore):
    def __init__(self, db_path) -> None:
        super().__init__(db_path)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


class RecordHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _client(tmp_path) -> TestClient:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    executor = SynchronousTaskExecutor(_UnusedRunner())
    return TestClient(create_app(store, task_executor=executor))


def _register(client: TestClient, username: str = "alice") -> dict:
    response = client.post(
        "/api/auth/register",
        json={"username": username, "password": "secret123"},
    )
    assert response.status_code == 201
    return response.json()


@pytest.fixture
def client(tmp_path) -> TestClient:
    return _client(tmp_path)


def test_openapi_exposes_only_new_product_routes(client) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert not any(
        path == "/api/tasks" or path.startswith("/api/tasks/")
        for path in paths
    )
    assert not any(
        path == "/api/eval" or path.startswith("/api/eval/")
        for path in paths
    )
    assert (
        "/api/conversations/{conversation_id}/tasks/{task_id}/updates"
        in paths
    )


def test_openapi_route_allowlist(client) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])

    assert paths == {
        "/api/auth/register",
        "/api/auth/login",
        "/api/auth/logout",
        "/api/auth/me",
        "/api/papers/search",
        "/api/conversations",
        "/api/conversations/{conversation_id}",
        "/api/conversations/{conversation_id}/messages",
        "/api/conversations/{conversation_id}/messages/{message_id}/alternatives",
        "/api/conversations/{conversation_id}/rollback",
        "/api/conversations/{conversation_id}/tasks/{task_id}/updates",
    }


def test_app_shutdown_closes_owned_task_store(tmp_path, monkeypatch):
    store = TrackingTaskStore(tmp_path / "owned.sqlite3")
    executor = FailingExecutor()
    monkeypatch.setattr(web_app_module, "TaskStore", lambda: store)

    with TestClient(
        create_app(
            task_executor=executor,
            runtime_config=WebRuntimeConfig(),
        )
    ):
        pass

    assert executor.is_shutdown is True
    assert store.close_calls == 1


def test_app_shutdown_does_not_close_injected_task_store(tmp_path):
    store = TrackingTaskStore(tmp_path / "injected.sqlite3")
    executor = FailingExecutor()

    with TestClient(
        create_app(
            store,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(),
        )
    ):
        pass

    assert executor.is_shutdown is True
    assert store.close_calls == 0
    store.close()


def test_authentication_error_has_request_id(tmp_path):
    client = _client(tmp_path)

    response = client.get("/api/conversations")

    assert response.status_code == 401
    assert response.json() == {"detail": "authentication required"}
    assert response.headers["x-request-id"]


def test_authenticated_user_is_added_to_request_context(tmp_path):
    logger = logging.getLogger(ACCESS_LOGGER_NAME)
    handler = RecordHandler()
    logger.addHandler(handler)
    try:
        client = _client(tmp_path)
        user = _register(client)

        response = client.get("/api/conversations")

        assert response.status_code == 200
        record = next(
            record
            for record in reversed(handler.records)
            if getattr(record, "route", None) == "/api/conversations"
        )
        assert record.user_id == user["id"]
    finally:
        logger.removeHandler(handler)


def test_create_app_fails_fast_for_invalid_runtime_environment(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("PAPERPILOT_THREAD_WORKERS", "0")
    store = TaskStore(tmp_path / "tasks.sqlite3")

    with pytest.raises(ValueError, match="PAPERPILOT_THREAD_WORKERS"):
        create_app(store)


def test_health_routes_are_public_hidden_and_split_dependencies(tmp_path):
    client = _client(tmp_path)

    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json() == {"status": "ready"}
    paths = client.get("/openapi.json").json()["paths"]
    assert "/health/live" not in paths
    assert "/health/ready" not in paths


def test_ready_fails_for_database_but_live_remains_ok(tmp_path):
    store = FailingHealthStore(tmp_path / "tasks.sqlite3")
    executor = SynchronousTaskExecutor(_UnusedRunner())
    client = TestClient(create_app(store, task_executor=executor))

    assert client.get("/health/live").status_code == 200
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {
            "database": "failed",
            "checkpoint": "ok",
            "executor": "ok",
        },
    }


def test_ready_fails_for_checkpoint_but_live_does_not_probe_it(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    executor = SynchronousTaskExecutor(_UnusedRunner())
    checkpoint = FailingHealthCheckpoint()
    client = TestClient(
        create_app(
            store,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(),
            checkpoint_runtime=checkpoint,
            deep_reading_runner=object(),
        )
    )

    assert client.get("/health/live").json() == {"status": "ok"}
    assert checkpoint.health_calls == 0
    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {
            "database": "ok",
            "checkpoint": "failed",
            "executor": "ok",
        },
    }
    assert checkpoint.health_calls == 1


def test_ready_fails_after_executor_shutdown(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    executor = SynchronousTaskExecutor(_UnusedRunner())
    executor.shutdown()
    client = TestClient(create_app(store, task_executor=executor))

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "database": "ok",
        "checkpoint": "ok",
        "executor": "failed",
    }


def test_index_and_static_assets_are_served(tmp_path):
    client = _client(tmp_path)

    html = client.get("/")
    css = client.get("/static/styles.css")
    javascript = client.get("/static/app.js")

    assert html.status_code == 200
    assert "PaperPilot" in html.text
    assert 'href="/static/styles.css"' in html.text
    assert 'src="/static/app.js" defer' in html.text
    assert css.status_code == 200
    assert javascript.status_code == 200
