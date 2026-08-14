"""Conversation route tests grouped by HTTP responsibility."""
from __future__ import annotations

from tests.web.routes.conversations.conftest import *

def test_conversation_task_updates_require_matching_owner_conversation_and_task(
    tmp_path,
):
    harness = _harness(tmp_path)
    alice = _register(harness.client, "alice-task-updates-api")
    bob = TestClient(harness.client.app)
    _register(bob, "bob-task-updates-api")
    conversation = _create_conversation(harness.client)
    other_conversation = _create_conversation(harness.client)
    submitted = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Scope this progress update.",
            "depth": "standard",
            "expected_head_message_id": None,
        },
    )
    assert submitted.status_code == 202
    task_id = submitted.json()["task"]["id"]

    owner_response = harness.client.get(
        f"/api/conversations/{conversation['id']}/tasks/{task_id}/updates"
    )
    forbidden_responses = [
        bob.get(f"/api/conversations/{conversation['id']}/tasks/{task_id}/updates"),
        harness.client.get(f"/api/conversations/conv_missing/tasks/{task_id}/updates"),
        harness.client.get(
            f"/api/conversations/{conversation['id']}/tasks/task_missing/updates"
        ),
        harness.client.get(
            f"/api/conversations/{other_conversation['id']}/tasks/{task_id}/updates"
        ),
    ]

    assert owner_response.status_code == 200
    assert owner_response.json()["task"]["id"] == task_id
    assert [response.status_code for response in forbidden_responses] == [404] * 4
    schema = harness.client.get("/openapi.json").json()
    assert (
        "/api/conversations/{conversation_id}/tasks/{task_id}/updates"
        in schema["paths"]
    )
