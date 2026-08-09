"""HTTP contracts for the PaperPilot conversation workspace."""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from paperpilot.deep_reading.runner import DeepReadingCheckpoint
from paperpilot.papers import PaperCandidate
from paperpilot.web.app import create_app
from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.task_executor import (
    TaskExecutorAtCapacityError,
    TaskSubmissionReservation,
)
from paperpilot.web.task_store import TaskStore


PRIMARY = PaperCandidate(
    external_id="2401.12345v2",
    title="A Test Paper",
    authors=["Ada Lovelace"],
    abstract="abstract",
    source_url="https://arxiv.org/abs/2401.12345v2",
)


class FakeCheckpointRuntime:
    def __init__(self) -> None:
        self.saver = object()
        self.health_calls = 0
        self.close_calls = 0
        self.health_error: Exception | None = None

    def check_health(self) -> None:
        self.health_calls += 1
        if self.health_error is not None:
            raise self.health_error

    def close(self) -> None:
        self.close_calls += 1


class FakeDeepReadingRunner:
    def __init__(self) -> None:
        self.checkpoints: dict[tuple[str, str], DeepReadingCheckpoint | None] = {}
        self.read_calls: list[tuple[str, str]] = []
        self.model_calls = 0

    def read_checkpoint(
        self,
        conversation_id: str,
        checkpoint_id: str,
    ) -> DeepReadingCheckpoint | None:
        self.read_calls.append((conversation_id, checkpoint_id))
        return self.checkpoints.get((conversation_id, checkpoint_id))

    def run(self, task_id: str) -> None:
        self.model_calls += 1
        raise AssertionError(f"Web API must not run the model directly: {task_id}")


class RecordingExecutor:
    def __init__(self) -> None:
        self.is_shutdown = False
        self.submissions: list[tuple[str, str]] = []
        self.releases = 0

    def reserve(self) -> TaskSubmissionReservation:
        def release() -> None:
            self.releases += 1

        return TaskSubmissionReservation(self._submit, release)

    def _submit(self, task_id: str, execution_mode: str) -> object:
        self.submissions.append((task_id, execution_mode))
        future: Future[None] = Future()
        future.set_result(None)
        return future

    def submit(self, task_id: str, execution_mode: str) -> object:
        return self.reserve().submit(task_id, execution_mode)

    def shutdown(self) -> None:
        self.is_shutdown = True


class RejectingExecutor(RecordingExecutor):
    def reserve(self) -> TaskSubmissionReservation:
        raise TaskExecutorAtCapacityError("full")


class FailingExecutor(RecordingExecutor):
    def _submit(self, task_id: str, execution_mode: str) -> object:
        raise ConnectionError("broker unavailable")


class ClaimingFailingExecutor(RecordingExecutor):
    def __init__(self, store: TaskStore) -> None:
        super().__init__()
        self.store = store

    def _submit(self, task_id: str, execution_mode: str) -> object:
        assert self.store.claim_task(task_id) is not None
        raise ConnectionError("publish result was ambiguous")


@dataclass
class AppHarness:
    store: TaskStore
    runner: FakeDeepReadingRunner
    checkpoint: FakeCheckpointRuntime
    executor: RecordingExecutor
    client: TestClient
    paper_search_calls: list[tuple[str, int]]


def _paper_search(calls: list[tuple[str, int]]):
    def search(query: str, limit: int) -> list[PaperCandidate]:
        calls.append((query, limit))
        return [PRIMARY]

    return search


def _harness(tmp_path, *, executor: RecordingExecutor | None = None) -> AppHarness:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = FakeDeepReadingRunner()
    checkpoint = FakeCheckpointRuntime()
    actual_executor = executor or RecordingExecutor()
    calls: list[tuple[str, int]] = []
    app = create_app(
        store,
        task_executor=actual_executor,
        runtime_config=WebRuntimeConfig(),
        checkpoint_runtime=checkpoint,
        deep_reading_runner=runner,
        paper_search=_paper_search(calls),
    )
    return AppHarness(
        store=store,
        runner=runner,
        checkpoint=checkpoint,
        executor=actual_executor,
        client=TestClient(app),
        paper_search_calls=calls,
    )


def _register(client: TestClient, username: str = "alice") -> dict[str, str]:
    response = client.post(
        "/api/auth/register",
        json={"username": username, "password": "secret123"},
    )
    assert response.status_code == 201
    return response.json()


def _create_conversation(client: TestClient, *, title: str | None = None) -> dict:
    body: dict[str, Any] = {
        "paper": {"source": "arxiv", "external_id": PRIMARY.external_id},
        "title": title,
    }
    response = client.post("/api/conversations", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _complete_turn(
    store: TaskStore,
    *,
    user_id: str,
    conversation_id: str,
    expected_head_message_id: str | None,
    checkpoint_id: str,
    content: str,
):
    turn = store.create_conversation_turn(
        user_id=user_id,
        conversation_id=conversation_id,
        content=f"Question for {content}",
        depth="standard",
        expected_head_message_id=expected_head_message_id,
    )
    published = store.publish_conversation_result(
        task_id=turn.task.id,
        content=content,
        metadata={"answer_draft": {"content": content}},
        used_papers=[],
    )
    detail = store.get_conversation_detail(conversation_id, user_id=user_id)
    assert detail is not None
    finalized = store.finalize_conversation_task(
        task_id=turn.task.id,
        assistant_message_id=published.message.id,
        final_checkpoint_id=checkpoint_id,
        result_quality="complete",
        active_paper_ids=[detail.conversation.primary_paper_id],
    )
    return turn, published, finalized


def _checkpoint(
    *,
    checkpoint_id: str,
    task_id: str,
    message_id: str,
    active_paper_ids: list[str],
    schema_version: int = 1,
    graph_version: str = "conversation-v1",
    is_complete: bool = True,
) -> DeepReadingCheckpoint:
    return DeepReadingCheckpoint(
        checkpoint_id=checkpoint_id,
        state={
            "schema_version": schema_version,
            "graph_version": graph_version,
            "current_task_id": task_id,
            "published_message_id": message_id,
            "active_paper_ids": active_paper_ids,
        },
        is_complete=is_complete,
    )


def test_paper_search_supports_natural_language_and_arxiv_url_without_llm(tmp_path):
    harness = _harness(tmp_path)
    _register(harness.client)

    natural = harness.client.get(
        "/api/papers/search", params={"q": "retrieval augmented generation", "limit": 7}
    )
    exact = harness.client.get(
        "/api/papers/search",
        params={"q": "https://arxiv.org/abs/2401.12345v2", "limit": 10},
    )

    expected = {
        "items": [
            {
                "source": "arxiv",
                "external_id": "2401.12345v2",
                "title": "A Test Paper",
                "authors": ["Ada Lovelace"],
                "abstract": "abstract",
                "source_url": "https://arxiv.org/abs/2401.12345v2",
            }
        ]
    }
    assert natural.status_code == 200
    assert natural.json() == expected
    assert exact.status_code == 200
    assert exact.json() == expected
    assert harness.paper_search_calls == [
        ("retrieval augmented generation", 7),
        ("https://arxiv.org/abs/2401.12345v2", 10),
    ]
    assert harness.runner.model_calls == 0


@pytest.mark.parametrize(
    "path",
    [
        "/api/papers/search?q=test",
        "/api/conversations",
        "/api/conversations/conv_missing",
        "/api/conversations/conv_missing/messages",
    ],
)
def test_conversation_read_apis_require_authentication(tmp_path, path):
    harness = _harness(tmp_path)

    response = harness.client.get(path)

    assert response.status_code == 401
    assert response.json() == {"detail": "authentication required"}


def test_conversation_create_list_detail_title_and_soft_archive(tmp_path):
    harness = _harness(tmp_path)
    _register(harness.client)

    created = _create_conversation(harness.client, title="Initial title")
    listed = harness.client.get("/api/conversations")
    detail = harness.client.get(f"/api/conversations/{created['id']}")
    renamed = harness.client.patch(
        f"/api/conversations/{created['id']}", json={"title": "Renamed"}
    )
    archived = harness.client.patch(
        f"/api/conversations/{created['id']}", json={"archived": True}
    )

    assert created["title"] == "Initial title"
    assert created["head_message_id"] is None
    assert created["head_checkpoint_id"] is None
    assert listed.status_code == 200
    assert listed.json()["items"] == [created]
    assert detail.status_code == 200
    assert detail.json()["conversation"] == created
    assert detail.json()["primary_paper"]["external_id"] == PRIMARY.external_id
    assert detail.json()["active_papers"] == [detail.json()["primary_paper"]]
    assert detail.json()["active_task"] is None
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed"
    assert archived.status_code == 200
    assert archived.json()["archived_at"] is not None
    assert harness.client.get("/api/conversations").json() == {"items": []}
    archived_list = harness.client.get(
        "/api/conversations", params={"include_archived": True}
    )
    assert [item["id"] for item in archived_list.json()["items"]] == [created["id"]]
    assert harness.runner.model_calls == 0


def test_conversation_create_resolves_server_side_paper_metadata(tmp_path):
    harness = _harness(tmp_path)
    _register(harness.client)

    created = _create_conversation(harness.client)

    assert created["title"] == PRIMARY.title
    assert harness.paper_search_calls == [(PRIMARY.external_id, 1)]


def test_conversation_owner_isolation_uses_uniform_404(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = FakeDeepReadingRunner()
    checkpoint = FakeCheckpointRuntime()
    executor = RecordingExecutor()
    calls: list[tuple[str, int]] = []
    app = create_app(
        store,
        task_executor=executor,
        checkpoint_runtime=checkpoint,
        deep_reading_runner=runner,
        paper_search=_paper_search(calls),
    )
    alice = TestClient(app)
    bob = TestClient(app)
    _register(alice, "alice")
    _register(bob, "bob")
    created = _create_conversation(alice)

    responses = [
        bob.get(f"/api/conversations/{created['id']}"),
        bob.patch(f"/api/conversations/{created['id']}", json={"title": "stolen"}),
        bob.get(f"/api/conversations/{created['id']}/messages"),
        bob.get(
            f"/api/conversations/{created['id']}/messages/msg_missing/alternatives"
        ),
        bob.post(
            f"/api/conversations/{created['id']}/messages",
            json={
                "content": "stolen",
                "depth": "standard",
                "expected_head_message_id": None,
            },
        ),
        bob.post(
            f"/api/conversations/{created['id']}/rollback",
            json={"message_id": "msg_missing", "expected_head_message_id": None},
        ),
    ]

    assert [response.status_code for response in responses] == [404] * 6


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
    assert harness.executor.submissions == [(payload["task"]["id"], "real")]
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
    harness.store.update_status(task_id, "failed")
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
    assert store.list_tasks_page(user_id=user["id"], limit=100).items == []
    assert store.list_active_messages(conversation["id"], user_id=user["id"]) == []
    assert store.get_unstable_turn(conversation["id"], user_id=user["id"]) is None


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
    task = store.list_tasks_page(user_id=user["id"], limit=100).items[0]
    assert task.status == "running"
    events = store.list_events_page(task.id, user_id=user["id"], after_id=0, limit=100)
    assert events is not None
    assert [event.type for event in events.items] == ["queued"]


def test_messages_expose_active_path_and_unstable_turn(tmp_path):
    harness = _harness(tmp_path)
    user = _register(harness.client)
    conversation = _create_conversation(harness.client)
    _, published, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=None,
        checkpoint_id="cp-1",
        content="Stable answer",
    )
    pending = harness.store.create_conversation_turn(
        user_id=user["id"],
        conversation_id=conversation["id"],
        content="Pending follow-up",
        depth="quick",
        expected_head_message_id=published.message.id,
    )

    response = harness.client.get(f"/api/conversations/{conversation['id']}/messages")

    assert response.status_code == 200
    payload = response.json()
    assert [item["role"] for item in payload["items"]] == ["user", "assistant"]
    assert payload["items"][-1]["id"] == published.message.id
    assert payload["unstable_turn"]["user_message"]["id"] == pending.user_message.id
    assert payload["unstable_turn"]["task"]["id"] == pending.task.id
    assert payload["unstable_turn"]["task"]["status"] == "pending"


def test_message_alternatives_return_direct_completed_user_assistant_pairs(tmp_path):
    harness = _harness(tmp_path)
    user = _register(harness.client)
    conversation = _create_conversation(harness.client)
    detail = harness.store.get_conversation_detail(conversation["id"], user_id=user["id"])
    assert detail is not None
    _, branch_point, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=None,
        checkpoint_id="cp-1",
        content="Branch point",
    )
    _, first_branch, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=branch_point.message.id,
        checkpoint_id="cp-2",
        content="First branch",
    )
    harness.store.switch_conversation_head(
        conversation["id"],
        user_id=user["id"],
        expected_head_message_id=first_branch.message.id,
        target_message_id=branch_point.message.id,
        target_checkpoint_id="cp-1",
        active_paper_ids=[detail.conversation.primary_paper_id],
    )
    _, second_branch, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=branch_point.message.id,
        checkpoint_id="cp-3",
        content="Second branch",
    )

    response = harness.client.get(
        f"/api/conversations/{conversation['id']}/messages/"
        f"{branch_point.message.id}/alternatives"
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert {item["assistant_message"]["id"] for item in items} == {
        first_branch.message.id,
        second_branch.message.id,
    }
    assert all(item["user_message"]["role"] == "user" for item in items)


def test_rollback_switches_head_from_valid_checkpoint_without_model_call(tmp_path):
    harness = _harness(tmp_path)
    user = _register(harness.client)
    conversation = _create_conversation(harness.client)
    detail = harness.store.get_conversation_detail(conversation["id"], user_id=user["id"])
    assert detail is not None
    first_turn, first, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=None,
        checkpoint_id="cp-1",
        content="First answer",
    )
    _, second, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=first.message.id,
        checkpoint_id="cp-2",
        content="Second answer",
    )
    harness.runner.checkpoints[(conversation["id"], "cp-1")] = _checkpoint(
        checkpoint_id="cp-1",
        task_id=first_turn.task.id,
        message_id=first.message.id,
        active_paper_ids=[detail.conversation.primary_paper_id],
    )

    response = harness.client.post(
        f"/api/conversations/{conversation['id']}/rollback",
        json={
            "message_id": first.message.id,
            "expected_head_message_id": second.message.id,
        },
    )

    assert response.status_code == 200
    assert response.json()["head_message_id"] == first.message.id
    assert response.json()["head_checkpoint_id"] == "cp-1"
    assert harness.runner.read_calls == [(conversation["id"], "cp-1")]
    assert harness.runner.model_calls == 0


@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    [
        ("missing", "checkpoint is unavailable"),
        ("schema", "checkpoint schema is unsupported"),
        ("graph", "checkpoint graph is unsupported"),
        ("incomplete", "checkpoint is not complete"),
        ("published", "checkpoint does not match target message"),
        ("task", "checkpoint does not match target task"),
        ("active", "checkpoint active paper ids are invalid"),
    ],
)
def test_rollback_rejects_missing_or_invalid_checkpoint_without_moving_head(
    tmp_path,
    mutation,
    expected_detail,
):
    harness = _harness(tmp_path)
    user = _register(harness.client)
    conversation = _create_conversation(harness.client)
    detail = harness.store.get_conversation_detail(conversation["id"], user_id=user["id"])
    assert detail is not None
    first_turn, first, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=None,
        checkpoint_id="cp-1",
        content="First answer",
    )
    _, second, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=first.message.id,
        checkpoint_id="cp-2",
        content="Second answer",
    )
    checkpoint: DeepReadingCheckpoint | None = _checkpoint(
        checkpoint_id="cp-1",
        task_id=first_turn.task.id,
        message_id=first.message.id,
        active_paper_ids=[detail.conversation.primary_paper_id],
    )
    if mutation == "missing":
        checkpoint = None
    elif mutation == "schema":
        checkpoint.state["schema_version"] = 999
    elif mutation == "graph":
        checkpoint.state["graph_version"] = "conversation-v999"
    elif mutation == "incomplete":
        checkpoint = _checkpoint(
            checkpoint_id="cp-1",
            task_id=first_turn.task.id,
            message_id=first.message.id,
            active_paper_ids=[detail.conversation.primary_paper_id],
            is_complete=False,
        )
    elif mutation == "published":
        checkpoint.state["published_message_id"] = "msg_other"
    elif mutation == "task":
        checkpoint.state["current_task_id"] = "task_other"
    elif mutation == "active":
        checkpoint.state["active_paper_ids"] = [detail.conversation.primary_paper_id] * 2
    harness.runner.checkpoints[(conversation["id"], "cp-1")] = checkpoint

    response = harness.client.post(
        f"/api/conversations/{conversation['id']}/rollback",
        json={
            "message_id": first.message.id,
            "expected_head_message_id": second.message.id,
        },
    )

    assert response.status_code == 409
    assert response.json() == {"detail": expected_detail}
    after = harness.store.get_conversation_detail(conversation["id"], user_id=user["id"])
    assert after is not None
    assert after.conversation.head_message_id == second.message.id
    assert after.conversation.head_checkpoint_id == "cp-2"
    assert harness.runner.model_calls == 0


def test_rollback_rejects_active_task_and_stale_head_before_checkpoint_read(tmp_path):
    harness = _harness(tmp_path)
    user = _register(harness.client)
    conversation = _create_conversation(harness.client)
    _, target, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=None,
        checkpoint_id="cp-1",
        content="Target answer",
    )
    active = harness.store.create_conversation_turn(
        user_id=user["id"],
        conversation_id=conversation["id"],
        content="Active",
        depth="standard",
        expected_head_message_id=target.message.id,
    )

    busy = harness.client.post(
        f"/api/conversations/{conversation['id']}/rollback",
        json={
            "message_id": target.message.id,
            "expected_head_message_id": target.message.id,
        },
    )
    harness.store.update_status(active.task.id, "failed")
    stale = harness.client.post(
        f"/api/conversations/{conversation['id']}/rollback",
        json={
            "message_id": target.message.id,
            "expected_head_message_id": "msg_stale",
        },
    )

    assert busy.status_code == 409
    assert busy.json() == {"detail": "conversation already has an active task"}
    assert stale.status_code == 409
    assert stale.json() == {"detail": "conversation head has changed"}
    assert harness.runner.read_calls == []


def test_expected_head_fields_are_required_but_nullable(tmp_path):
    harness = _harness(tmp_path)
    _register(harness.client)
    conversation = _create_conversation(harness.client)

    missing_message_head = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={"content": "Question", "depth": "standard"},
    )
    missing_rollback_head = harness.client.post(
        f"/api/conversations/{conversation['id']}/rollback",
        json={"message_id": "msg_missing"},
    )

    assert missing_message_head.status_code == 422
    assert missing_rollback_head.status_code == 422
