"""LangGraph-based deep-reading workflow contracts."""

from .graph import build_deep_reading_graph
from .nodes import (
    DeepReadingContext,
    initialize_turn,
    needs_summary,
    prepare_primary_paper,
    publish_result,
    research_evidence,
    route_after_initialize,
    summarize_history,
    write_answer,
)
from .schemas import (
    AnswerCitation,
    AnswerDraft,
    ConversationSummary,
    EvidenceItem,
    PaperUse,
    ResearchResult,
)
from .runner import (
    DeepReadingCheckpoint,
    DeepReadingRunner,
    DeepReadingTaskError,
    build_deep_reading_model,
)
from .state import GRAPH_VERSION, SCHEMA_VERSION, DeepReadingState

__all__ = [
    "AnswerCitation",
    "AnswerDraft",
    "ConversationSummary",
    "DeepReadingContext",
    "DeepReadingCheckpoint",
    "DeepReadingRunner",
    "DeepReadingState",
    "DeepReadingTaskError",
    "EvidenceItem",
    "GRAPH_VERSION",
    "PaperUse",
    "ResearchResult",
    "SCHEMA_VERSION",
    "build_deep_reading_graph",
    "build_deep_reading_model",
    "initialize_turn",
    "needs_summary",
    "prepare_primary_paper",
    "publish_result",
    "research_evidence",
    "route_after_initialize",
    "summarize_history",
    "write_answer",
]
