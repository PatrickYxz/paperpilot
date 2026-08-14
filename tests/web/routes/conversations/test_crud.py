"""Conversation route tests grouped by HTTP responsibility."""
from __future__ import annotations

from tests.web.routes.conversations.conftest import *

def test_conversation_router_keeps_route_method_set(tmp_path) -> None:
    harness = _harness(tmp_path)
    schema = harness.client.get("/openapi.json").json()
    actual = {
        (path, method)
        for path, operations in schema["paths"].items()
        if path.startswith("/api/conversations") and "/tasks/" not in path
        for method in operations
    }
    assert actual == {
        ("/api/conversations", "get"),
        ("/api/conversations", "post"),
        ("/api/conversations/{conversation_id}", "get"),
        ("/api/conversations/{conversation_id}", "patch"),
        ("/api/conversations/{conversation_id}/messages", "get"),
        ("/api/conversations/{conversation_id}/messages", "post"),
        (
            "/api/conversations/{conversation_id}/messages/"
            "{message_id}/alternatives",
            "get",
        ),
        ("/api/conversations/{conversation_id}/rollback", "post"),
    }

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
