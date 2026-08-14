"""Conversation route tests grouped by HTTP responsibility."""
from __future__ import annotations

from tests.web.routes.conversations.conftest import *

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
