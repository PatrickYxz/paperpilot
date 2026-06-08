"""FastAPI Web workbench tests."""
from __future__ import annotations

from fastapi.testclient import TestClient

from paperpilot.web.app import create_app
from paperpilot.web.task_store import TaskStore


def _client(tmp_path) -> TestClient:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    return TestClient(create_app(store, simulation_delay_seconds=0.0))


def test_create_task_api_returns_pending_task(tmp_path):
    client = _client(tmp_path)

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


def test_list_tasks_api_returns_created_tasks(tmp_path):
    client = _client(tmp_path)
    created = client.post(
        "/api/tasks",
        json={"question": "Compare ColBERT and BM25", "depth": "standard"},
    ).json()

    response = client.get("/api/tasks")

    assert response.status_code == 200
    assert response.json()[0]["id"] == created["id"]


def test_get_task_api_returns_404_for_missing_task(tmp_path):
    client = _client(tmp_path)

    response = client.get("/api/tasks/task_missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "task not found"


def test_create_task_api_rejects_invalid_payload(tmp_path):
    client = _client(tmp_path)

    response = client.post("/api/tasks", json={"question": "", "depth": "standard"})

    assert response.status_code == 422


def test_task_filter_rejects_invalid_status(tmp_path):
    client = _client(tmp_path)

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

    assert css.status_code == 200
    assert ".event-category" in css.text
    assert ".event-payload" in css.text
    assert js.status_code == 200
    assert "function eventCategory" in js.text
    assert "renderEventRow" in js.text


def test_create_task_records_simulated_workflow_events(tmp_path):
    client = _client(tmp_path)
    created = client.post(
        "/api/tasks",
        json={"question": "Trace a simulated workflow", "depth": "standard"},
    ).json()

    detail = client.get(f"/api/tasks/{created['id']}").json()
    events = client.get(f"/api/tasks/{created['id']}/events").json()

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

    artifacts = client.get(f"/api/tasks/{created['id']}/artifacts").json()
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
    client = TestClient(create_app(store, workflow_runner=runner))

    created = client.post(
        "/api/tasks",
        json={
            "question": "Run real PaperPilot",
            "depth": "standard",
            "execution_mode": "real",
        },
    ).json()

    detail = client.get(f"/api/tasks/{created['id']}").json()
    events = client.get(f"/api/tasks/{created['id']}/events").json()
    artifacts = client.get(f"/api/tasks/{created['id']}/artifacts").json()

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

    response = client.get("/api/tasks/task_missing/events")

    assert response.status_code == 404
    assert response.json()["detail"] == "task not found"


def test_task_artifacts_api_returns_404_for_missing_task(tmp_path):
    client = _client(tmp_path)

    response = client.get("/api/tasks/task_missing/artifacts")

    assert response.status_code == 404
    assert response.json()["detail"] == "task not found"
