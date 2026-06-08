"""PaperPilot agent event to Web task event mapper tests."""
from __future__ import annotations

from paperpilot.web.event_mapper import (
    MAX_EVENT_TEXT_CHARS,
    map_paperpilot_event,
)


def test_maps_tool_call_to_progress_event():
    event = map_paperpilot_event(
        "tool_call",
        {
            "name": "mcp__arxiv__search_papers",
            "arguments": {"query": "retrieval augmented generation"},
        },
    )

    assert event["type"] == "progress"
    assert event["stage"] == "tool_call"
    assert event["message"] == "Calling tool: mcp__arxiv__search_papers"
    assert event["payload"]["tool_name"] == "mcp__arxiv__search_papers"
    assert "retrieval augmented generation" in event["payload"]["arguments_preview"]


def test_maps_tool_result_with_truncated_content():
    event = map_paperpilot_event(
        "tool_result",
        {
            "name": "mcp__colbert__search",
            "content": "x" * (MAX_EVENT_TEXT_CHARS + 100),
        },
    )

    assert event["type"] == "progress"
    assert event["stage"] == "tool_result"
    assert event["payload"]["tool_name"] == "mcp__colbert__search"
    assert event["payload"]["truncated"] is True
    assert len(event["payload"]["content_preview"]) == MAX_EVENT_TEXT_CHARS


def test_maps_turn_event():
    event = map_paperpilot_event(
        "turn",
        {"iteration": 2, "tool_calls": ["load_skill"], "text": None},
    )

    assert event["type"] == "progress"
    assert event["stage"] == "agent_turn"
    assert event["message"] == "Agent turn 2"
    assert event["payload"]["tool_calls"] == ["load_skill"]
    assert event["payload"]["has_text"] is False


def test_maps_unknown_non_json_payload_to_safe_preview():
    class Block:
        pass

    event = map_paperpilot_event("custom_event", {"block": Block()})

    assert event["type"] == "progress"
    assert event["stage"] == "agent_event"
    assert event["payload"]["source_kind"] == "custom_event"
    assert "Block" in event["payload"]["payload_preview"]

