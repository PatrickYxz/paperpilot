"""Turn initialization and optional-history routing node."""
from __future__ import annotations

from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command

from ..state import GRAPH_VERSION, SCHEMA_VERSION, DeepReadingState
from .context import DeepReadingContext
from .summarize_history import needs_summary


def initialize_turn(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> Command[Literal["summarize_history", "prepare_context", "prepare_primary_paper"]]:
    """Reset per-turn outputs and route through the optional summary stage."""
    context = runtime.context
    if context.context_management.enabled:
        goto = "prepare_context"
    else:
        goto = "summarize_history" if needs_summary(state, runtime) else "prepare_primary_paper"
    update: DeepReadingState = {
            "schema_version": SCHEMA_VERSION,
            "graph_version": GRAPH_VERSION,
            "current_task_id": context.task_id,
            "current_user_message_id": context.current_user_message_id,
            "context_view": None,
            "retrieved_archive_ids": [],
            "context_input_tokens": None,
            "research_trace": None,
            "research_context_delta": None,
            "research_result": None,
            "answer_draft": None,
            "published_message_id": None,
            "error": None,
        }
    return Command(
        update=update,
        goto=goto,
    )
