"""LangGraph orchestration for the fixed deep-reading workflow."""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import (
    DeepReadingContext,
    initialize_turn,
    prepare_primary_paper,
    publish_result,
    prepare_context,
    research_evidence,
    summarize_history,
    write_answer,
)
from .state import DeepReadingState


def build_deep_reading_graph(checkpointer: Any):
    """Compile the fixed macro workflow with the provided checkpointer."""
    builder = StateGraph(DeepReadingState, context_schema=DeepReadingContext)
    builder.add_node("initialize_turn", initialize_turn)
    builder.add_node("summarize_history", summarize_history)
    builder.add_node("prepare_context", prepare_context)
    builder.add_node("prepare_primary_paper", prepare_primary_paper)
    builder.add_node("research_evidence", research_evidence)
    builder.add_node("write_answer", write_answer)
    builder.add_node("publish_result", publish_result)
    builder.add_edge(START, "initialize_turn")
    builder.add_edge("summarize_history", "prepare_primary_paper")
    builder.add_edge("prepare_context", "prepare_primary_paper")
    builder.add_edge("prepare_primary_paper", "research_evidence")
    builder.add_edge("research_evidence", "write_answer")
    builder.add_edge("write_answer", "publish_result")
    builder.add_edge("publish_result", END)
    return builder.compile(checkpointer=checkpointer)
