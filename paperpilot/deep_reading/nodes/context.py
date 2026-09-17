"""Trusted context and dependency contracts for deep-reading nodes."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from paperpilot.papers import PaperCandidate
from paperpilot.tools.types import Tool
from paperpilot.web.config import ContextManagementConfig
from paperpilot.web.task_store import TaskStore

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
    context_management: ContextManagementConfig = ContextManagementConfig()
    context_management_runtime: Any | None = None
    summary_token_threshold: int = 32_000
    summary_recent_turns: int = 6
    research_recursion_limit: int = 24
    research_model_call_limit: int = 12
    research_tool_call_limit: int = 12
    research_max_output_tokens: int = 4096
    research_model_retries: int = 1

    def __post_init__(self) -> None:
        for name in (
            "summary_token_threshold",
            "summary_recent_turns",
            "research_recursion_limit",
            "research_max_output_tokens",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "research_model_call_limit",
            "research_tool_call_limit",
        ):
            if getattr(self, name) < 2:
                raise ValueError(
                    f"{name} must cover both structured-response attempts"
                )
        if self.research_model_retries < 0:
            raise ValueError("research_model_retries must be nonnegative")
