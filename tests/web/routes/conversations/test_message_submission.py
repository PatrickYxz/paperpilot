"""Conversation route tests grouped by HTTP responsibility."""
from __future__ import annotations

from tests.web.routes.conversations.conftest import *

def test_message_submission_reserves_then_persists_then_submits_real(tmp_path):
    harness = _harness(tmp_path)
    _register(harness.client)
    conversation = _create_conversation(harness.client)

    response = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Compare its method.",
            "depth": "deep",
            "expected_head_message_id": None,
        },
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["user_message"]["content"] == "Compare its method."
    assert payload["task"]["status"] == "pending"
    assert payload["stable_head_message_id"] is None
    assert harness.executor.submissions == [payload["task"]["id"]]
    stored = harness.store.get_task(payload["task"]["id"])
    assert stored is not None and stored.conversation_id == conversation["id"]
    assert harness.runner.model_calls == 0

def test_message_submission_maps_busy_and_stale_head_to_409(tmp_path):
    harness = _harness(tmp_path)
    _register(harness.client)
    conversation = _create_conversation(harness.client)
    body = {
        "content": "First",
        "depth": "standard",
        "expected_head_message_id": None,
    }

    first = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages", json=body
    )
    busy = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages", json=body
    )
    task_id = first.json()["task"]["id"]
    assert harness.store.fail_pending_task(task_id) is not None
    stale = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={**body, "content": "Stale", "expected_head_message_id": "msg_old"},
    )

    assert first.status_code == 202
    assert busy.status_code == 409
    assert busy.json() == {"detail": "conversation already has an active task"}
    assert stale.status_code == 409
    assert stale.json() == {"detail": "conversation head has changed"}

def test_message_capacity_rejection_writes_nothing(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    executor = RejectingExecutor()
    runner = FakeDeepReadingRunner()
    checkpoint = FakeCheckpointRuntime()
    calls: list[tuple[str, int]] = []
    client = TestClient(
        create_app(
            store,
            task_executor=executor,
            checkpoint_runtime=checkpoint,
            deep_reading_runner=runner,
            paper_search=_paper_search(calls),
        )
    )
    user = _register(client)
    conversation = _create_conversation(client)

    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Do not persist",
            "depth": "quick",
            "expected_head_message_id": None,
        },
    )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    detail = store.get_conversation_detail(conversation["id"], user_id=user["id"])
    assert detail is not None
    assert detail.active_task is None
    assert store.list_active_messages(conversation["id"], user_id=user["id"]) == []
    assert store.get_unstable_turn(conversation["id"], user_id=user["id"]) is None

def test_message_capacity_rejection_uses_configured_retry_after(tmp_path):
    harness = _harness(
        tmp_path,
        executor=RejectingExecutor(),
        runtime_config=WebRuntimeConfig(overload_retry_after_seconds=5),
    )
    _register(harness.client)
    conversation = _create_conversation(harness.client)

    response = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Retry after configured delay",
            "depth": "quick",
            "expected_head_message_id": None,
        },
    )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"

def test_submit_failure_marks_pending_task_failed_and_preserves_stable_head(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = FakeDeepReadingRunner()
    checkpoint = FakeCheckpointRuntime()
    executor = FailingExecutor()
    calls: list[tuple[str, int]] = []
    client = TestClient(
        create_app(
            store,
            task_executor=executor,
            checkpoint_runtime=checkpoint,
            deep_reading_runner=runner,
            paper_search=_paper_search(calls),
        )
    )
    user = _register(client)
    conversation = _create_conversation(client)

    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Queue failure",
            "depth": "quick",
            "expected_head_message_id": None,
        },
    )

    assert response.status_code == 503
    detail = store.get_conversation_detail(conversation["id"], user_id=user["id"])
    assert detail is not None and detail.conversation.head_message_id is None
    unstable = store.get_unstable_turn(conversation["id"], user_id=user["id"])
    assert unstable is not None and unstable.task.status == "failed"
    events = store.list_events_page(
        unstable.task.id, user_id=user["id"], after_id=0, limit=100
    )
    assert events is not None
    assert [event.type for event in events.items] == ["queued", "failed"]
    assert events.items[-1].stage == "queue"
    visible = client.get(f"/api/conversations/{conversation['id']}/messages")
    assert visible.status_code == 200
    assert visible.json()["unstable_turn"]["task"]["status"] == "failed"

def test_ambiguous_submit_failure_does_not_fail_worker_claimed_task(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    executor = ClaimingFailingExecutor(store)
    runner = FakeDeepReadingRunner()
    checkpoint = FakeCheckpointRuntime()
    calls: list[tuple[str, int]] = []
    client = TestClient(
        create_app(
            store,
            task_executor=executor,
            checkpoint_runtime=checkpoint,
            deep_reading_runner=runner,
            paper_search=_paper_search(calls),
        )
    )
    user = _register(client)
    conversation = _create_conversation(client)

    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Ambiguous publish",
            "depth": "standard",
            "expected_head_message_id": None,
        },
    )

    assert response.status_code == 202
    unstable = store.get_unstable_turn(conversation["id"], user_id=user["id"])
    assert unstable is not None
    task = unstable.task
    assert task.status == "running"
    events = store.list_events_page(task.id, user_id=user["id"], after_id=0, limit=100)
    assert events is not None
    assert [event.type for event in events.items] == ["queued"]

def test_ambiguous_submit_after_worker_failure_still_returns_accepted(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    executor = WorkerFailingThenAmbiguousExecutor(store)
    runner = FakeDeepReadingRunner()
    checkpoint = FakeCheckpointRuntime()
    calls: list[tuple[str, int]] = []
    client = TestClient(
        create_app(
            store,
            task_executor=executor,
            checkpoint_runtime=checkpoint,
            deep_reading_runner=runner,
            paper_search=_paper_search(calls),
        )
    )
    user = _register(client)
    conversation = _create_conversation(client)

    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Worker fails quickly",
            "depth": "standard",
            "expected_head_message_id": None,
        },
    )

    assert response.status_code == 202
    task_id = response.json()["task"]["id"]
    task = store.get_task(task_id, user_id=user["id"])
    assert task is not None and task.status == "failed"
    events = store.list_events_page(
        task_id,
        user_id=user["id"],
        after_id=0,
        limit=100,
    )
    assert events is not None
    assert [event.type for event in events.items] == ["queued", "failed"]
    assert events.items[-1].stage == "deep_reading"
    assert events.items[-1].message == "Worker failed after claiming the task."
