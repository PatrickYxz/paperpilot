"""FastAPI Web workbench tests."""
from __future__ import annotations

import logging
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

import paperpilot.web.app as web_app_module
from paperpilot.web.app import create_app
from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.observability import ACCESS_LOGGER_NAME, RUNTIME_LOGGER_NAME
from paperpilot.web.task_executor import (
    SynchronousTaskExecutor,
    TaskExecutor,
    TaskExecutorAtCapacityError,
    TaskExecutorShuttingDownError,
    TaskSubmissionReservation,
)
from paperpilot.web.task_store import TaskStore
from paperpilot.web.workflow import WorkflowRunner


class FailingExecutor:
    is_shutdown = False

    def reserve(self):
        def fail(task_id: str, execution_mode: str):
            raise ConnectionError("broker unavailable")

        return TaskSubmissionReservation(fail)

    def shutdown(self) -> None:
        self.is_shutdown = True


class ClaimingFailingExecutor:
    def __init__(self, store: TaskStore) -> None:
        self.store = store
        self.is_shutdown = False

    def reserve(self):
        def claim_then_fail(task_id: str, execution_mode: str):
            assert self.store.claim_task(task_id) is not None
            raise ConnectionError("publish result was ambiguous")

        return TaskSubmissionReservation(claim_then_fail)

    def shutdown(self) -> None:
        self.is_shutdown = True


class RejectingExecutor:
    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.is_shutdown = reason == "shutdown"

    def reserve(self):
        if self.reason == "capacity":
            raise TaskExecutorAtCapacityError("full")
        raise TaskExecutorShuttingDownError("stopping")

    def shutdown(self) -> None:
        self.is_shutdown = True


class BlockingRunner:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def run_simulated(self, task_id: str) -> None:
        self.started.set()
        assert self.release.wait(timeout=2)
        self.finished.set()

    def run_real(self, task_id: str) -> None:
        self.run_simulated(task_id)


class FailingQueuedTaskStore(TaskStore):
    def create_queued_task(self, **kwargs):
        raise sqlite3.OperationalError("write unavailable")


class CleanupFailingTaskStore(TaskStore):
    def __init__(self, db_path) -> None:
        super().__init__(db_path)
        self.cleanup_error = sqlite3.OperationalError(
            "cleanup write unavailable"
        )

    def fail_pending_task(self, task_id: str):
        raise self.cleanup_error


class ReleaseSpyExecutor:
    def __init__(self) -> None:
        self.is_shutdown = False
        self.released = 0
        self.submitted = 0

    def reserve(self):
        def release():
            self.released += 1

        def submit(task_id, execution_mode):
            self.submitted += 1

        return TaskSubmissionReservation(submit, release)

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
    runner = WorkflowRunner(store, delay_seconds=0)
    executor = SynchronousTaskExecutor(runner)
    return TestClient(
        create_app(store, workflow_runner=runner, task_executor=executor)
    )


def _register(client: TestClient, username: str = "alice") -> dict:
    response = client.post(
        "/api/auth/register",
        json={"username": username, "password": "secret123"},
    )
    assert response.status_code == 201
    return response.json()


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

    response = client.get("/api/tasks")

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

        response = client.get("/api/tasks")

        assert response.status_code == 200
        task_list_record = next(
            record
            for record in reversed(handler.records)
            if getattr(record, "route", None) == "/api/tasks"
        )
        assert task_list_record.user_id == user["id"]
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


@pytest.mark.parametrize(
    ("reason", "detail", "retry_after"),
    [
        ("capacity", "task executor is at capacity", "1"),
        ("shutdown", "task executor is shutting down", None),
    ],
)
def test_task_admission_rejection_writes_nothing(
    tmp_path,
    reason,
    detail,
    retry_after,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    executor = RejectingExecutor(reason)
    config = WebRuntimeConfig()
    app = create_app(
        store,
        workflow_runner=runner,
        task_executor=executor,
        runtime_config=config,
    )
    client = TestClient(app)
    user = _register(client)

    response = client.post(
        "/api/tasks",
        json={"question": "Do not persist me", "depth": "quick"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": detail}
    assert response.headers.get("retry-after") == retry_after
    assert response.headers["x-request-id"]
    assert app.state.runtime_config is config
    assert app.state.task_store is store
    assert app.state.task_executor is executor
    assert store.list_tasks_page(user_id=user["id"], limit=100).items == []
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_artifacts").fetchone()[0] == 0


def test_thread_capacity_recovers_after_background_work_finishes(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = BlockingRunner()
    executor = TaskExecutor(runner, max_workers=1, queue_capacity=0)
    app = create_app(
        store,
        workflow_runner=runner,
        task_executor=executor,
        runtime_config=WebRuntimeConfig(
            thread_workers=1,
            thread_queue_capacity=0,
        ),
    )
    with TestClient(app) as client:
        _register(client)
        first = client.post(
            "/api/tasks",
            json={"question": "first", "depth": "quick"},
        )
        assert first.status_code == 201
        assert runner.started.wait(timeout=1)

        overloaded = client.post(
            "/api/tasks",
            json={"question": "second", "depth": "quick"},
        )
        assert overloaded.status_code == 503

        runner.release.set()
        assert runner.finished.wait(timeout=1)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            recovered = client.post(
                "/api/tasks",
                json={"question": "third", "depth": "quick"},
            )
            if recovered.status_code == 201:
                break
            assert recovered.status_code == 503
            time.sleep(0.01)
        else:
            raise AssertionError("task capacity was not restored")
        assert recovered.status_code == 201


def test_database_failure_releases_reservation_without_submit(tmp_path):
    store = FailingQueuedTaskStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    executor = ReleaseSpyExecutor()
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(),
        )
    )
    _register(client)

    response = client.post(
        "/api/tasks",
        json={"question": "database failure", "depth": "quick"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "internal server error"
    assert executor.released == 1
    assert executor.submitted == 0


def test_health_routes_are_public_hidden_and_split_dependencies(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    executor = SynchronousTaskExecutor(runner)
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(),
        )
    )

    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json() == {"status": "ready"}
    paths = client.get("/openapi.json").json()["paths"]
    assert "/health/live" not in paths
    assert "/health/ready" not in paths


def test_ready_fails_for_database_but_live_remains_ok(tmp_path):
    store = FailingHealthStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    executor = SynchronousTaskExecutor(runner)
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(),
        )
    )

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
    runner = WorkflowRunner(store, delay_seconds=0)
    executor = SynchronousTaskExecutor(runner)
    checkpoint = FailingHealthCheckpoint()
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
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
    runner = WorkflowRunner(store, delay_seconds=0)
    executor = SynchronousTaskExecutor(runner)
    executor.shutdown()
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(),
        )
    )

    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"] == {
        "database": "ok",
        "checkpoint": "ok",
        "executor": "failed",
    }


def test_ready_stays_ok_while_thread_executor_is_saturated(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = BlockingRunner()
    executor = TaskExecutor(runner, max_workers=1, queue_capacity=0)
    reservation = executor.reserve()
    future = reservation.submit("health_probe_task", "simulated")
    assert runner.started.wait(timeout=1)
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(
                thread_workers=1,
                thread_queue_capacity=0,
            ),
        )
    )

    try:
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}
    finally:
        runner.release.set()
        future.result(timeout=2)
        executor.shutdown()


def test_create_task_api_returns_pending_task(tmp_path):
    client = _client(tmp_path)
    _register(client)

    response = client.post(
        "/api/tasks",
        json={"question": "What is retrieval augmented generation?", "depth": "quick"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["id"].startswith("task_")
    assert payload["question"] == "What is retrieval augmented generation?"
    assert payload["depth"] == "quick"
    assert payload["status"] == "pending"


def test_create_task_marks_failed_when_queue_submission_fails(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=FailingExecutor(),
        )
    )
    registered_user = _register(client)

    response = client.post(
        "/api/tasks",
        json={"question": "Queue this task", "depth": "quick"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "task queue unavailable"
    task = store.list_tasks_page(
        user_id=registered_user["id"],
        limit=100,
    ).items[0]
    event_page = store.list_events_page(
        task.id,
        user_id=registered_user["id"],
        after_id=0,
        limit=100,
    )
    assert event_page is not None
    events = event_page.items
    assert task.status == "failed"
    assert [event.type for event in events] == ["queued", "failed"]
    assert events[-1].stage == "queue"
    assert events[-1].payload["error_type"] == "ConnectionError"


def test_create_task_logs_cleanup_exception_when_submission_cleanup_fails(tmp_path):
    store = CleanupFailingTaskStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=FailingExecutor(),
        )
    )
    logger = logging.getLogger(RUNTIME_LOGGER_NAME)
    handler = RecordHandler()
    logger.addHandler(handler)
    try:
        _register(client)

        response = client.post(
            "/api/tasks",
            json={"question": "Cleanup failure", "depth": "quick"},
        )
    finally:
        logger.removeHandler(handler)

    assert response.status_code == 503
    assert response.json() == {"detail": "task queue unavailable"}
    assert response.headers["x-request-id"]
    assert "broker unavailable" not in response.text
    assert "cleanup write unavailable" not in response.text
    cleanup_record = next(
        record
        for record in handler.records
        if getattr(record, "event", None) == "task.submission_cleanup_failed"
    )
    assert cleanup_record.exception_type == "OperationalError"
    assert cleanup_record.exc_info is not None
    exception_type, exception, traceback = cleanup_record.exc_info
    assert exception_type is sqlite3.OperationalError
    assert exception is store.cleanup_error
    assert traceback is not None


def test_create_task_does_not_fail_work_claimed_during_publish_error(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=ClaimingFailingExecutor(store),
        )
    )
    registered_user = _register(client)

    response = client.post(
        "/api/tasks",
        json={"question": "Ambiguous publish", "depth": "quick"},
    )

    assert response.status_code == 201
    task = store.list_tasks_page(
        user_id=registered_user["id"],
        limit=100,
    ).items[0]
    assert task.status == "running"
    event_page = store.list_events_page(
        task.id,
        user_id=registered_user["id"],
        after_id=0,
        limit=100,
    )
    assert event_page is not None
    assert [event.type for event in event_page.items] == ["queued"]


def test_list_tasks_api_returns_created_tasks(tmp_path):
    client = _client(tmp_path)
    _register(client)
    created = client.post(
        "/api/tasks",
        json={"question": "Compare ColBERT and BM25", "depth": "standard"},
    ).json()

    response = client.get("/api/tasks")

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["id"] == created["id"]
    assert payload["next_cursor"] is None
    assert payload["has_more"] is False


def test_list_tasks_api_traverses_cursor_pages_without_duplicates(tmp_path):
    client = _client(tmp_path)
    _register(client)
    created_ids = {
        client.post(
            "/api/tasks",
            json={"question": f"Task {index}", "depth": "standard"},
        ).json()["id"]
        for index in range(3)
    }

    first = client.get("/api/tasks?limit=2")
    first_payload = first.json()
    second = client.get(
        "/api/tasks",
        params={"limit": 2, "cursor": first_payload["next_cursor"]},
    )
    second_payload = second.json()

    assert first.status_code == 200
    assert first_payload["has_more"] is True
    assert first_payload["next_cursor"]
    assert second.status_code == 200
    assert second_payload["has_more"] is False
    assert second_payload["next_cursor"] is None
    actual_ids = [
        item["id"]
        for item in first_payload["items"] + second_payload["items"]
    ]
    assert set(actual_ids) == created_ids
    assert len(actual_ids) == len(set(actual_ids)) == 3


def test_list_tasks_api_rejects_malformed_and_other_user_cursors(tmp_path):
    alice = _client(tmp_path)
    bob = _client(tmp_path)
    _register(alice, "alice")
    _register(bob, "bob")
    for index in range(2):
        alice.post(
            "/api/tasks",
            json={"question": f"Alice task {index}", "depth": "standard"},
        )
    alice_cursor = alice.get("/api/tasks?limit=1").json()["next_cursor"]

    malformed = alice.get("/api/tasks?cursor=not-a-valid-cursor")
    other_user = bob.get(
        "/api/tasks",
        params={"cursor": alice_cursor},
    )

    assert malformed.status_code == 422
    assert malformed.json()["detail"] == "invalid task cursor"
    assert other_user.status_code == 422
    assert other_user.json()["detail"] == "invalid task cursor"


def test_get_task_api_returns_404_for_missing_task(tmp_path):
    client = _client(tmp_path)
    _register(client)

    response = client.get("/api/tasks/task_missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "task not found"


def test_create_task_api_rejects_invalid_payload(tmp_path):
    client = _client(tmp_path)
    _register(client)

    response = client.post("/api/tasks", json={"question": "", "depth": "standard"})

    assert response.status_code == 422


def test_task_filter_rejects_invalid_status(tmp_path):
    client = _client(tmp_path)
    _register(client)

    response = client.get("/api/tasks?status=unknown")

    assert response.status_code == 422


def test_index_page_is_served(tmp_path):
    client = _client(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert "PaperPilot" in response.text


def test_event_ui_static_assets_are_served(tmp_path):
    client = _client(tmp_path)

    css = client.get("/static/styles.css")
    js = client.get("/static/app.js")
    html = client.get("/")

    assert "Evaluation Snapshot" in html.text
    assert "Calibration Candidates" in html.text
    assert 'id="authForm"' in html.text
    assert 'id="workbench"' in html.text
    assert css.status_code == 200
    assert ".auth-panel" in css.text
    assert ".event-category" in css.text
    assert ".event-payload" in css.text
    assert ".metric-card" in css.text
    assert ".candidate-card" in css.text
    assert js.status_code == 200
    assert "async function loadCurrentUser" in js.text
    assert "submitAuth" in js.text
    assert "function eventCategory" in js.text
    assert "renderEventRow" in js.text
    assert "loadEvalSnapshot" in js.text
    assert "loadCalibrationCandidates" in js.text
    assert "payload.items" in js.text
    assert "/updates" in js.text
    assert "after_event_id" in js.text
    assert "after_artifact_id" in js.text
    assert "let taskListVersion = 0;" in js.text
    assert "const requestVersion = ++taskListVersion;" in js.text
    assert "requestVersion !== taskListVersion" in js.text
    assert "requestedStatus !== statusFilter.value" in js.text
    assert "invalidateTaskListRequests();" in js.text
    assert 'requestJson(`/api/tasks/${encodedTaskId}/events`)' not in js.text
    assert 'requestJson(`/api/tasks/${encodedTaskId}/artifacts`)' not in js.text


def test_eval_summary_api_returns_snapshot(tmp_path, monkeypatch):
    client = _client(tmp_path)

    monkeypatch.setattr(
        "paperpilot.web.app.build_eval_snapshot",
        lambda: {
            "available": True,
            "total_cases": 2,
            "strict": {"pass_count": 1, "rate": 0.5},
            "semantic": {
                "correct_count": 1,
                "correct_rate": 0.5,
                "weighted_count": 1.5,
                "weighted_rate": 0.75,
                "label_counts": {"correct": 1, "partial": 1},
            },
            "calibrated": {
                "available": True,
                "candidate_count": 1,
                "correct_count": 2,
                "correct_rate": 1.0,
                "weighted_count": 2,
                "weighted_rate": 1.0,
                "label_counts": {"correct": 2},
                "decision_counts": {"accept_as_correct": 1},
            },
        },
    )

    response = client.get("/api/eval/summary")

    assert response.status_code == 200
    assert response.json()["total_cases"] == 2
    assert response.json()["calibrated"]["correct_rate"] == 1.0


def test_calibration_candidates_api_returns_filtered_payload(tmp_path, monkeypatch):
    client = _client(tmp_path)
    seen: dict[str, str | None] = {}

    def fake_candidates(category=None, review_decision=None):
        seen["category"] = category
        seen["review_decision"] = review_decision
        return {
            "available": True,
            "total_candidates": 1,
            "filtered_count": 1,
            "candidates": [
                {
                    "case_id": "case-1",
                    "category": category,
                    "review_decision": review_decision,
                }
            ],
        }

    monkeypatch.setattr(
        "paperpilot.web.app.list_calibration_candidates",
        fake_candidates,
    )

    response = client.get(
        "/api/eval/calibration-candidates"
        "?category=partial_high&review_decision=accept_as_correct"
    )

    assert response.status_code == 200
    assert seen == {
        "category": "partial_high",
        "review_decision": "accept_as_correct",
    }
    assert response.json()["candidates"][0]["case_id"] == "case-1"


def test_create_task_records_simulated_workflow_events(tmp_path):
    client = _client(tmp_path)
    _register(client)
    created = client.post(
        "/api/tasks",
        json={"question": "Trace a simulated workflow", "depth": "standard"},
    ).json()

    detail = client.get(f"/api/tasks/{created['id']}").json()
    events = client.get(f"/api/tasks/{created['id']}/events").json()["items"]

    assert detail["status"] == "completed"
    assert [event["type"] for event in events] == [
        "queued",
        "started",
        "progress",
        "progress",
        "completed",
    ]
    assert [event["stage"] for event in events] == [
        "queue",
        "start",
        "prepare",
        "deep_read_placeholder",
        "complete",
    ]
    assert events[0]["payload"]["depth"] == "standard"
    assert events[0]["payload"]["simulated"] is True

    artifacts = client.get(f"/api/tasks/{created['id']}/artifacts").json()["items"]
    assert len(artifacts) == 1
    assert artifacts[0]["kind"] == "result"
    assert artifacts[0]["title"] == "Simulated research result"
    assert "Real PaperPilot deep-read" in artifacts[0]["content"]
    assert artifacts[0]["payload"]["depth"] == "standard"


def test_create_task_real_mode_uses_injected_runner(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    seen_queries: list[str] = []

    def fake_real_runner(query: str) -> list[dict]:
        seen_queries.append(query)
        return [{"role": "assistant", "content": "real answer"}]

    from paperpilot.web.workflow import WorkflowRunner

    runner = WorkflowRunner(store, delay_seconds=0, real_runner=fake_real_runner)
    executor = SynchronousTaskExecutor(runner)
    client = TestClient(
        create_app(store, workflow_runner=runner, task_executor=executor)
    )
    _register(client)

    created = client.post(
        "/api/tasks",
        json={
            "question": "Run real PaperPilot",
            "depth": "standard",
            "execution_mode": "real",
        },
    ).json()

    detail = client.get(f"/api/tasks/{created['id']}").json()
    events = client.get(f"/api/tasks/{created['id']}/events").json()["items"]
    artifacts = client.get(f"/api/tasks/{created['id']}/artifacts").json()["items"]

    assert seen_queries == ["Run real PaperPilot"]
    assert detail["status"] == "completed"
    assert [event["stage"] for event in events] == [
        "queue",
        "real_start",
        "real_complete",
    ]
    assert artifacts[0]["title"] == "PaperPilot result"
    assert artifacts[0]["content"] == "real answer"
    assert artifacts[0]["payload"]["execution_mode"] == "real"


def test_task_events_api_returns_404_for_missing_task(tmp_path):
    client = _client(tmp_path)
    _register(client)

    response = client.get("/api/tasks/task_missing/events")

    assert response.status_code == 404
    assert response.json()["detail"] == "task not found"


def test_task_artifacts_api_returns_404_for_missing_task(tmp_path):
    client = _client(tmp_path)
    _register(client)

    response = client.get("/api/tasks/task_missing/artifacts")

    assert response.status_code == 404
    assert response.json()["detail"] == "task not found"


@pytest.mark.parametrize(
    "path",
    [
        "/api/tasks?limit=0",
        "/api/tasks?limit=101",
        "/api/tasks/task_missing/events?after_id=-1",
        "/api/tasks/task_missing/artifacts?after_id=-1",
        "/api/tasks/task_missing/updates?after_event_id=-1",
        "/api/tasks/task_missing/updates?after_artifact_id=-1",
    ],
)
def test_task_page_apis_reject_invalid_limits_and_watermarks(tmp_path, path):
    client = _client(tmp_path)
    _register(client)

    response = client.get(path)

    assert response.status_code == 422


def test_task_updates_api_advances_event_and_artifact_watermarks(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=SynchronousTaskExecutor(runner),
        )
    )
    _register(client)
    created = client.post(
        "/api/tasks",
        json={"question": "Poll task updates", "depth": "standard"},
    ).json()
    for index in range(2):
        store.add_artifact(
            task_id=created["id"],
            kind="note",
            title=f"Extra artifact {index}",
            content=f"Content {index}",
        )

    first = client.get(f"/api/tasks/{created['id']}/updates?limit=2")
    first_payload = first.json()
    second = client.get(
        f"/api/tasks/{created['id']}/updates",
        params={
            "limit": 2,
            "after_event_id": first_payload["events"]["next_after_id"],
            "after_artifact_id": first_payload["artifacts"]["next_after_id"],
        },
    )
    second_payload = second.json()

    assert first.status_code == 200
    assert second.status_code == 200
    assert first_payload["task"]["id"] == created["id"]
    first_event_ids = {item["id"] for item in first_payload["events"]["items"]}
    second_event_ids = {item["id"] for item in second_payload["events"]["items"]}
    first_artifact_ids = {
        item["id"] for item in first_payload["artifacts"]["items"]
    }
    second_artifact_ids = {
        item["id"] for item in second_payload["artifacts"]["items"]
    }
    assert first_event_ids.isdisjoint(second_event_ids)
    assert first_artifact_ids.isdisjoint(second_artifact_ids)
    assert len(first_event_ids | second_event_ids) == 4
    assert len(first_artifact_ids | second_artifact_ids) == 3


def test_task_updates_api_preserves_missing_and_other_user_404(tmp_path):
    alice = _client(tmp_path)
    bob = _client(tmp_path)
    _register(alice, "alice")
    _register(bob, "bob")
    created = alice.post(
        "/api/tasks",
        json={"question": "Alice private updates", "depth": "standard"},
    ).json()

    missing = alice.get("/api/tasks/task_missing/updates")
    other_user = bob.get(f"/api/tasks/{created['id']}/updates")

    assert missing.status_code == 404
    assert missing.json()["detail"] == "task not found"
    assert other_user.status_code == 404
    assert other_user.json()["detail"] == "task not found"


def test_task_api_requires_login(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/api/tasks",
        json={"question": "Private task", "depth": "quick"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "authentication required"


def test_task_api_isolates_tasks_by_user(tmp_path):
    alice = _client(tmp_path)
    bob = _client(tmp_path)
    _register(alice, "alice")
    _register(bob, "bob")
    created = alice.post(
        "/api/tasks",
        json={"question": "Alice private task", "depth": "standard"},
    ).json()

    bob_tasks = bob.get("/api/tasks")
    bob_detail = bob.get(f"/api/tasks/{created['id']}")
    bob_events = bob.get(f"/api/tasks/{created['id']}/events")
    bob_artifacts = bob.get(f"/api/tasks/{created['id']}/artifacts")

    assert bob_tasks.status_code == 200
    assert bob_tasks.json() == {
        "items": [],
        "next_cursor": None,
        "has_more": False,
    }
    assert bob_detail.status_code == 404
    assert bob_events.status_code == 404
    assert bob_artifacts.status_code == 404

    alice_detail = alice.get(f"/api/tasks/{created['id']}")
    assert alice_detail.status_code == 200
    assert alice_detail.json()["question"] == "Alice private task"
