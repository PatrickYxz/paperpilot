"""TodoStore / research_todo_tool / render unit tests."""
from __future__ import annotations

import pytest

from paperpilot.builtin_tools.research_todo import (
    TodoStore,
    render,
    research_todo_tool,
)


def test_store_starts_empty():
    assert TodoStore().items() == []


def test_store_replace_overwrites():
    s = TodoStore()
    s.replace([{"content": "a", "status": "pending"}])
    s.replace([
        {"content": "b", "status": "in_progress"},
        {"content": "c", "status": "pending"},
    ])
    items = s.items()
    assert len(items) == 2
    assert items[0]["content"] == "b"
    assert items[1]["content"] == "c"


def test_store_items_returns_copy():
    s = TodoStore()
    s.replace([{"content": "a", "status": "pending"}])
    items = s.items()
    items.append({"content": "x", "status": "pending"})
    assert len(s.items()) == 1


def test_handler_replaces_state():
    s = TodoStore()
    tool = research_todo_tool(s)
    tool.handler({"todos": [
        {"content": "step 1", "status": "in_progress"},
        {"content": "step 2", "status": "pending"},
    ]})
    items = s.items()
    assert len(items) == 2
    assert items[0]["status"] == "in_progress"


def test_handler_returns_render_string():
    s = TodoStore()
    tool = research_todo_tool(s)
    out = tool.handler({"todos": [
        {"content": "step 1", "status": "pending"},
    ]})
    assert isinstance(out, str)
    assert "## Research Todos" in out
    assert "step 1" in out


def test_handler_marks_in_progress_with_arrow():
    s = TodoStore()
    tool = research_todo_tool(s)
    out = tool.handler({"todos": [
        {"content": "doing", "status": "in_progress"},
    ]})
    assert "[->]" in out
    assert "(in progress)" in out


def test_handler_marks_completed_with_x():
    s = TodoStore()
    tool = research_todo_tool(s)
    out = tool.handler({"todos": [
        {"content": "done", "status": "completed"},
        {"content": "todo", "status": "pending"},
    ]})
    assert "[x] done" in out
    assert "[ ] todo" in out


def test_handler_empty_list_returns_empty_render():
    s = TodoStore()
    tool = research_todo_tool(s)
    out = tool.handler({"todos": []})
    assert out == "## Research Todos (empty)"
    assert s.items() == []


def test_handler_rejects_multiple_in_progress():
    s = TodoStore()
    tool = research_todo_tool(s)
    with pytest.raises(ValueError) as exc:
        tool.handler({"todos": [
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "in_progress"},
        ]})
    msg = str(exc.value)
    assert "only one in_progress" in msg
    assert "got 2" in msg


def test_handler_allows_zero_in_progress():
    s = TodoStore()
    tool = research_todo_tool(s)
    out = tool.handler({"todos": [
        {"content": "a", "status": "pending"},
        {"content": "b", "status": "completed"},
    ]})
    assert "## Research Todos (2 items)" in out


def test_handler_allows_one_in_progress():
    s = TodoStore()
    tool = research_todo_tool(s)
    out = tool.handler({"todos": [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "in_progress"},
        {"content": "c", "status": "pending"},
    ]})
    assert "[x] a" in out
    assert "[->] b" in out
    assert "[ ] c" in out


def test_tool_metadata():
    s = TodoStore()
    tool = research_todo_tool(s)
    assert tool.name == "research_todo"
    assert "todos" in tool.input_schema["properties"]
    item_schema = tool.input_schema["properties"]["todos"]["items"]
    assert item_schema["additionalProperties"] is False
    assert item_schema["properties"]["status"]["enum"] == [
        "pending", "in_progress", "completed",
    ]


def test_handler_raises_keyerror_when_todos_missing():
    s = TodoStore()
    tool = research_todo_tool(s)
    with pytest.raises(KeyError):
        tool.handler({})
