"""ask_user built-in tool tests."""
from __future__ import annotations

import pytest

from paperpilot.builtin_tools.ask_user import ASK_USER_NUDGE, ask_user_tool


def test_ask_user_tool_returns_provider_answer_and_emits_events():
    prompts: list[str] = []
    events: list[tuple[str, dict]] = []
    tool = ask_user_tool(
        input_provider=lambda question: prompts.append(question) or "用户回答",
        on_event=lambda kind, payload: events.append((kind, dict(payload))),
    )

    result = tool.handler({
        "question": "需要比较哪两篇论文?",
        "reason": "缺少比较对象",
    })

    assert result == "用户回答"
    assert prompts == ["需要比较哪两篇论文?"]
    assert events == [
        (
            "ask_user_prompt",
            {"question": "需要比较哪两篇论文?", "reason": "缺少比较对象"},
        ),
        (
            "ask_user_answer",
            {"question": "需要比较哪两篇论文?", "answer": "用户回答"},
        ),
    ]


def test_ask_user_tool_requires_question():
    tool = ask_user_tool(
        input_provider=lambda question: "unused",
        on_event=lambda kind, payload: None,
    )

    with pytest.raises(ValueError, match="question is required"):
        tool.handler({})


def test_ask_user_tool_metadata_and_nudge():
    tool = ask_user_tool(
        input_provider=lambda question: "answer",
        on_event=lambda kind, payload: None,
    )

    assert tool.name == "ask_user"
    assert tool.input_schema["required"] == ["question"]
    assert tool.input_schema["additionalProperties"] is False
    assert "question" in tool.input_schema["properties"]
    assert "reason" in tool.input_schema["properties"]
    assert "ask_user" in ASK_USER_NUDGE

