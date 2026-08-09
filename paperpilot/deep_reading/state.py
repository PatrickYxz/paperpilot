"""Serializable state for the deep-reading LangGraph."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages

SCHEMA_VERSION = 1
GRAPH_VERSION = "conversation-v1"


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
    evidence_items: list[dict[str, object]]
    answer_draft: dict[str, object] | None
    published_message_id: str | None
    error: dict[str, object] | None
