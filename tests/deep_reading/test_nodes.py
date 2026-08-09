"""State and node behavior for the deep-reading graph."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from langchain.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.runtime import Runtime

from paperpilot.deep_reading.nodes import (
    DeepReadingContext,
    initialize_turn,
    needs_summary,
    summarize_history,
)
from paperpilot.deep_reading.schemas import ConversationSummary
from paperpilot.deep_reading.state import GRAPH_VERSION, SCHEMA_VERSION


SUMMARY = ConversationSummary(
    confirmed_facts=["Fact"],
    paper_findings=["Finding"],
    comparison_context=["Comparison"],
    open_questions=["Question"],
)


class _StructuredModel:
    def __init__(self, result: ConversationSummary | dict[str, object]) -> None:
        self.result = result
        self.invocations: list[object] = []

    def invoke(self, model_input: object) -> object:
        self.invocations.append(model_input)
        return self.result


class _FakeModel:
    def __init__(
        self,
        result: ConversationSummary | dict[str, object] = SUMMARY,
    ) -> None:
        self.structured = _StructuredModel(result)
        self.schemas: list[type[ConversationSummary]] = []

    def with_structured_output(
        self, schema: type[ConversationSummary]
    ) -> _StructuredModel:
        self.schemas.append(schema)
        return self.structured


def _context(
    model: object,
    *,
    threshold: int = 32_000,
    recent_turns: int = 6,
    recursion_limit: int = 12,
) -> DeepReadingContext:
    return DeepReadingContext(
        user_id="user-1",
        conversation_id="conversation-1",
        task_id="task-current",
        current_user_message_id="message-current",
        base_checkpoint_id="checkpoint-base",
        task_store=object(),  # type: ignore[arg-type]
        model=model,
        mcp_tools={},
        paper_search=lambda _query, _limit: [],
        event_sink=lambda _event, _payload: None,
        summary_token_threshold=threshold,
        summary_recent_turns=recent_turns,
        research_recursion_limit=recursion_limit,
    )


def test_context_is_frozen_and_rejects_unbounded_configuration() -> None:
    context = _context(_FakeModel())

    with pytest.raises(FrozenInstanceError):
        context.task_id = "changed"  # type: ignore[misc]

    for overrides in (
        {"threshold": 0},
        {"recent_turns": 0},
        {"recursion_limit": 0},
    ):
        with pytest.raises(ValueError):
            _context(_FakeModel(), **overrides)


def test_initialize_turn_clears_only_per_turn_fields() -> None:
    messages = [HumanMessage(content="Earlier question", id="human-old")]
    old_summary = SUMMARY.model_dump(mode="json")
    state = {
        "schema_version": SCHEMA_VERSION,
        "graph_version": GRAPH_VERSION,
        "messages": messages,
        "conversation_summary": old_summary,
        "current_task_id": "task-old",
        "current_user_message_id": "message-old",
        "primary_paper_id": "paper-primary",
        "active_paper_ids": ["paper-primary", "paper-related"],
        "evidence_items": [{"id": "old-evidence"}],
        "answer_draft": {"content": "old draft"},
        "published_message_id": "assistant-old",
        "error": {"message": "old error"},
    }

    update = initialize_turn(state, Runtime(context=_context(_FakeModel())))

    assert update == {
        "schema_version": 1,
        "graph_version": "conversation-v1",
        "current_task_id": "task-current",
        "current_user_message_id": "message-current",
        "evidence_items": [],
        "answer_draft": None,
        "published_message_id": None,
        "error": None,
    }
    merged = state | update
    assert merged["messages"] is messages
    assert merged["conversation_summary"] == old_summary
    assert merged["primary_paper_id"] == "paper-primary"
    assert merged["active_paper_ids"] == ["paper-primary", "paper-related"]


def test_needs_summary_uses_exact_character_estimate_boundary() -> None:
    context = _context(_FakeModel(), threshold=2)

    assert not needs_summary(
        {"messages": [HumanMessage(content="12345678901", id="short")]},
        Runtime(context=context),
    )
    assert needs_summary(
        {"messages": [HumanMessage(content="123456789012", id="long")]},
        Runtime(context=context),
    )


def test_summarize_history_below_threshold_does_not_call_model() -> None:
    model = _FakeModel()
    state = {"messages": [HumanMessage(content="short", id="human-1")]}

    update = summarize_history(
        state,
        Runtime(context=_context(model, threshold=10)),
    )

    assert update == {}
    assert model.schemas == []
    assert model.structured.invocations == []


def test_summarize_history_writes_json_and_retains_recent_six_turns() -> None:
    model = _FakeModel(SUMMARY.model_dump(mode="json"))
    messages: list[Any] = []
    for index in range(8):
        messages.extend(
            [
                HumanMessage(content=f"question-{index}-long", id=f"h-{index}"),
                AIMessage(content=f"answer-{index}-long", id=f"a-{index}"),
            ]
        )
    state = {
        "messages": messages,
        "conversation_summary": {
            "confirmed_facts": ["Older fact"],
            "paper_findings": [],
            "comparison_context": [],
            "open_questions": [],
        },
    }

    update = summarize_history(
        state,
        Runtime(context=_context(model, threshold=1, recent_turns=6)),
    )

    assert model.schemas == [ConversationSummary]
    assert update["conversation_summary"] == SUMMARY.model_dump(mode="json")
    message_update = update["messages"]
    assert isinstance(message_update[0], RemoveMessage)
    assert message_update[0].id == REMOVE_ALL_MESSAGES
    assert [message.id for message in message_update[1:]] == [
        "h-2",
        "a-2",
        "h-3",
        "a-3",
        "h-4",
        "a-4",
        "h-5",
        "a-5",
        "h-6",
        "a-6",
        "h-7",
        "a-7",
    ]
    assert [message.id for message in add_messages(messages, message_update)] == [
        "h-2",
        "a-2",
        "h-3",
        "a-3",
        "h-4",
        "a-4",
        "h-5",
        "a-5",
        "h-6",
        "a-6",
        "h-7",
        "a-7",
    ]
