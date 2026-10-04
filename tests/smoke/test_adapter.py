"""HTTP adapter: entry-point call order, payloads, and failure surfacing."""
from __future__ import annotations

import pytest

from paperpilot.smoke.adapter import ConversationApi


def test_adapter_calls_public_entrypoints_in_order(fake_client, make_scenario) -> None:
    client = fake_client(
        assistant={"id": "m-a", "role": "assistant", "content": "answer", "metadata": {}}
    )
    api = ConversationApi(client)
    scenario = make_scenario()

    api.open_conversation(scenario)
    assert client.calls[0][0:2] == ("POST", "/api/auth/register")
    assert client.calls[0][2]["password"] == "smoke-password-1"
    assert client.calls[0][2]["username"].startswith("smoke-")
    assert client.calls[1] == (
        "POST",
        "/api/conversations",
        {"paper": {"source": "arxiv", "external_id": "2001.09899v1"}},
    )

    task_id = api.ask(scenario.question, scenario.depth)
    assert task_id == "task-1"
    assert client.calls[2][2] == {
        "content": scenario.question,
        "depth": "quick",
        "expected_head_message_id": None,
    }

    task, events, artifacts = api.poll_until_terminal(timeout_s=5)
    assert task["status"] == "completed"
    assert events == [] and artifacts == []

    assistant = api.assistant_message()
    assert assistant is not None and assistant["role"] == "assistant"


def test_adapter_raises_runtime_error_on_http_failure(fake_client, make_scenario) -> None:
    client = fake_client(register_status=500)
    api = ConversationApi(client)
    with pytest.raises(RuntimeError, match="register failed: HTTP 500"):
        api.open_conversation(make_scenario())


def test_adapter_returns_task_events_artifacts_batch(fake_client) -> None:
    client = fake_client(
        final_status="failed",
        events=[{"type": "task_started"}, {"type": "task_failed"}],
        artifacts=[{"id": 1}],
    )
    api = ConversationApi(client)
    api.conversation_id, api.task_id = "conv-1", "task-1"
    task, events, artifacts = api.poll_until_terminal(timeout_s=5)
    assert task["status"] == "failed"
    assert [e["type"] for e in events] == ["task_started", "task_failed"]
    assert artifacts == [{"id": 1}]


def test_adapter_chains_head_id_for_follow_ups(fake_client, make_scenario) -> None:
    client = fake_client(
        turns=[
            {"final_status": "completed", "assistant": {"role": "assistant", "content": "first answer"}},
            {"final_status": "completed", "assistant": {"role": "assistant", "content": "second answer"}},
        ]
    )
    api = ConversationApi(client)
    api.open_conversation(make_scenario())

    api.ask("first question", "quick", None)
    api.poll_until_terminal(timeout_s=5)
    first_assistant = api.assistant_message()
    assert first_assistant is not None and first_assistant["content"] == "first answer"

    api.ask("follow up?", "quick", first_assistant["id"])
    api.poll_until_terminal(timeout_s=5)
    assert api.assistant_message()["content"] == "second answer"

    message_posts = [c for c in client.calls if c[1].endswith("/messages") and c[0] == "POST"]
    assert message_posts[1][2]["expected_head_message_id"] == "m-a-1"
    assert message_posts[1][2]["content"] == "follow up?"
