"""Serializable state for the deep-reading LangGraph."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages

SCHEMA_VERSION = 2
GRAPH_VERSION = "conversation-v2"
LEGACY_SCHEMA_VERSION = 1
LEGACY_GRAPH_VERSION = "conversation-v1"


class DeepReadingState(TypedDict, total=False):
    """Checkpointed state; runtime clients and database handles never belong here."""

    schema_version: int
    graph_version: str
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_summary: dict[str, object] | None
    current_task_id: str
    current_user_message_id: str
    primary_paper_id: str
    active_paper_ids: list[str]
    context_view: dict[str, object] | None
    continuation_capsule: dict[str, object] | None
    retrieved_archive_ids: list[str]
    context_input_tokens: int | None
    research_result: dict[str, object] | None
    user_memory_context: str | None
    research_trace: dict[str, object] | None
    research_context_delta: dict[str, object] | None
    answer_draft: dict[str, object] | None
    published_message_id: str | None
    error: dict[str, object] | None
