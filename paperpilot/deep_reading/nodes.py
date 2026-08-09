"""Initial state-management nodes for the deep-reading graph."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langchain.messages import AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from paperpilot.core.adapter import Tool
from paperpilot.papers import PaperCandidate
from paperpilot.web.task_store import TaskStore

from .schemas import ConversationSummary
from .state import GRAPH_VERSION, SCHEMA_VERSION, DeepReadingState

PaperSearch = Callable[[str, int], list[PaperCandidate]]
EventSink = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class DeepReadingContext:
    """Trusted, per-run dependencies and identifiers excluded from checkpoints."""

    user_id: str
    conversation_id: str
    task_id: str
    current_user_message_id: str
    base_checkpoint_id: str | None
    task_store: TaskStore
    model: Any
    mcp_tools: Mapping[str, Tool]
    paper_search: PaperSearch
    event_sink: EventSink
    summary_token_threshold: int = 32_000
    summary_recent_turns: int = 6
    research_recursion_limit: int = 12

    def __post_init__(self) -> None:
        for name in (
            "summary_token_threshold",
            "summary_recent_turns",
            "research_recursion_limit",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


def initialize_turn(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:
    """Reset per-turn outputs while retaining checkpointed conversation context."""
    del state
    context = runtime.context
    return {
        "schema_version": SCHEMA_VERSION,
        "graph_version": GRAPH_VERSION,
        "current_task_id": context.task_id,
        "current_user_message_id": context.current_user_message_id,
        "research_result": None,
        "answer_draft": None,
        "published_message_id": None,
        "error": None,
    }


def needs_summary(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> bool:
    """Return whether estimated checkpoint context exceeds the configured budget."""
    return _estimated_tokens(state) > runtime.context.summary_token_threshold


def summarize_history(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:
    """Summarize long history and replace current State messages with recent turns."""
    if not needs_summary(state, runtime):
        return {}

    context = runtime.context
    summary_model = context.model.with_structured_output(ConversationSummary)
    summary_input: list[AnyMessage] = [
        SystemMessage(
            content=(
                "Summarize the conversation for continued scholarly paper reading. "
                "Preserve only confirmed facts, paper findings, comparison context, "
                "and open questions."
            )
        )
    ]
    previous_summary = state.get("conversation_summary")
    if previous_summary is not None:
        summary_input.append(
            HumanMessage(
                content=(
                    "Previous structured summary:\n"
                    + json.dumps(previous_summary, ensure_ascii=False, sort_keys=True)
                )
            )
        )
    summary_input.extend(state.get("messages", []))

    summary = ConversationSummary.model_validate(summary_model.invoke(summary_input))
    recent_messages = _recent_turns(
        state.get("messages", []), context.summary_recent_turns
    )
    return {
        "conversation_summary": summary.model_dump(mode="json"),
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent_messages],
    }


def _estimated_tokens(state: DeepReadingState) -> int:
    total_chars = sum(
        _message_character_count(message) for message in state.get("messages", [])
    )
    summary = state.get("conversation_summary")
    if summary is not None:
        total_chars += len(
            json.dumps(
                summary,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return max(1, total_chars // 4)


def _message_character_count(message: AnyMessage) -> int:
    content = message.content
    if isinstance(content, str):
        return len(content)
    return len(json.dumps(content, ensure_ascii=False, separators=(",", ":")))


def _recent_turns(messages: list[AnyMessage], turn_count: int) -> list[AnyMessage]:
    turns_seen = 0
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].type == "human":
            turns_seen += 1
            if turns_seen == turn_count:
                return messages[index:]
    return list(messages)
