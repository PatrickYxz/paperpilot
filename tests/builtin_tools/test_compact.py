"""compact_context built-in tool unit tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from paperpilot.builtin_tools.compact import (
    COMPACT_CONTEXT_NUDGE,
    compact_context_tool,
)
from paperpilot.core.adapter import ParsedResponse


def _fake_client(text: str) -> MagicMock:
    client = MagicMock()
    client.call.return_value = ParsedResponse(
        text=text,
        tool_calls=[],
        usage={"total_tokens": 50},
        raw=None,
    )
    return client


def _make_long_messages(n: int) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": "original query"}]
    for i in range(1, n):
        role = "assistant" if i % 2 == 1 else "user"
        messages.append({"role": role, "content": f"turn-{i}"})
    return messages


def test_handler_too_short_returns_noop():
    messages: list[dict] = [{"role": "user", "content": "hi"}]
    client = _fake_client("should not be called")
    tool = compact_context_tool(
        messages_ref=messages,
        client_factory=lambda: client,
        on_event=lambda k, v: None,
    )

    result = tool.handler({})

    assert "already compact" in result
    assert messages == [{"role": "user", "content": "hi"}]
    client.call.assert_not_called()


def test_handler_normal_path_replaces_middle():
    messages = _make_long_messages(10)
    head_obj = messages[0]
    tail_objs = list(messages[-6:])

    client = _fake_client("FAKE_SUMMARY_BODY")
    tool = compact_context_tool(
        messages_ref=messages,
        client_factory=lambda: client,
        on_event=lambda k, v: None,
    )

    result = tool.handler({})

    assert result.startswith("compacted ")
    assert len(messages) == 1 + 1 + 6
    assert messages[0] is head_obj
    summary_msg = messages[1]
    assert summary_msg["role"] == "user"
    assert "<context_summary>" in summary_msg["content"]
    assert "FAKE_SUMMARY_BODY" in summary_msg["content"]
    for index, expected_obj in enumerate(tail_objs):
        assert messages[2 + index] is expected_obj
    client.call.assert_called_once()


def test_handler_summarize_failure_reraises_and_keeps_messages():
    messages = _make_long_messages(10)
    snapshot = list(messages)

    client = MagicMock()
    client.call.side_effect = RuntimeError("LLM explodes")
    tool = compact_context_tool(
        messages_ref=messages,
        client_factory=lambda: client,
        on_event=lambda k, v: None,
    )

    with pytest.raises(RuntimeError, match="LLM explodes"):
        tool.handler({})
    assert messages == snapshot


def test_lifecycle_events_emitted():
    messages = _make_long_messages(10)
    events: list[tuple[str, dict]] = []

    tool = compact_context_tool(
        messages_ref=messages,
        client_factory=lambda: _fake_client("S"),
        on_event=lambda k, v: events.append((k, dict(v))),
    )
    tool.handler({})

    kinds = [k for k, _ in events]
    assert kinds == ["compact_start", "compact_done"]
    assert events[0][1] == {"middle_count": 3}
    assert events[1][1] == {"kept_recent": 6}


def test_tool_metadata_and_nudge():
    tool = compact_context_tool(
        messages_ref=[],
        client_factory=lambda: _fake_client("x"),
        on_event=lambda k, v: None,
    )

    assert tool.name == "compact_context"
    assert tool.input_schema["required"] == []
    assert tool.input_schema["additionalProperties"] is False
    assert tool.input_schema["properties"] == {}
    assert "compact_context" in COMPACT_CONTEXT_NUDGE
    assert "Long conversation" in COMPACT_CONTEXT_NUDGE


def test_handler_preserves_head_and_tail_identity():
    messages = _make_long_messages(10)
    same_list_id = id(messages)
    head_id = id(messages[0])
    tail_ids = [id(message) for message in messages[-6:]]

    tool = compact_context_tool(
        messages_ref=messages,
        client_factory=lambda: _fake_client("S"),
        on_event=lambda k, v: None,
    )
    tool.handler({})

    assert id(messages) == same_list_id
    assert id(messages[0]) == head_id
    assert [id(message) for message in messages[2:]] == tail_ids


def test_handler_does_not_orphan_leading_tool_result():
    messages: list[dict] = [
        {"role": "user", "content": "original query"},
        {"role": "assistant", "content": "turn-1"},
        {"role": "user", "content": "turn-2"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call-1", "name": "x"}],
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "call-1",
                "content": "result",
            }],
        },
        {"role": "assistant", "content": "turn-5"},
        {"role": "user", "content": "turn-6"},
        {"role": "assistant", "content": "turn-7"},
        {"role": "user", "content": "turn-8"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call-2", "name": "compact"}],
        },
    ]
    boundary_assistant = messages[3]
    boundary_result = messages[4]

    tool = compact_context_tool(
        messages_ref=messages,
        client_factory=lambda: _fake_client("S"),
        on_event=lambda k, v: None,
    )
    tool.handler({})

    assert messages[2] is boundary_assistant
    assert messages[3] is boundary_result
