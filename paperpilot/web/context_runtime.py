"""Web-owned assembly of context-management ports and per-task adapters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from paperpilot.deep_reading.context_management.artifacts import ToolResultIngestor
from paperpilot.deep_reading.context_management.budget import (
    ModelAwareTokenCounter,
    usable_input_budget,
)
from paperpilot.deep_reading.context_management.breaker import CompressionCircuitBreaker
from paperpilot.deep_reading.context_management.compaction import CompressionCoordinator
from paperpilot.deep_reading.context_management.editing import LocalContextEditingAdapter
from paperpilot.deep_reading.context_management.ports import TokenCounter
from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.context_artifacts import LocalContextArtifactStore
from paperpilot.web.task_store import TaskStore


@dataclass(frozen=True)
class ContextManagementRuntime:
    """Immutable dependency bundle excluded from checkpointed graph state."""

    task_store: TaskStore
    config: WebRuntimeConfig
    model: Any
    token_counter: TokenCounter
    artifact_store: LocalContextArtifactStore
    archive_store: TaskStore
    compression_state_port: TaskStore
    tool_result_ingestor: ToolResultIngestor
    editing_adapter: LocalContextEditingAdapter
    compression_coordinator: CompressionCoordinator
    breaker: CompressionCircuitBreaker | None


def build_context_management_runtime(
    task_store: TaskStore,
    config: WebRuntimeConfig,
    model: Any,
    *,
    conversation_id: str | None = None,
    task_id: str | None = None,
    event_sink: Callable[[str, dict[str, object]], None] | None = None,
) -> ContextManagementRuntime | None:
    """Build no resources when the master flag is disabled."""
    if not config.context_management.enabled:
        return None
    settings = config.context_management
    token_counter = ModelAwareTokenCounter(model)
    artifact_store = LocalContextArtifactStore(
        root=Path(settings.artifact_root),
        task_store=task_store,
        token_counter=token_counter,
        read_max_tokens=settings.artifact_read_max_tokens,
    )
    input_budget = usable_input_budget(
        settings.model_context_window_tokens,
        config.research_max_output_tokens,
        settings.context_safety_margin_ratio,
    )
    sink = event_sink or (lambda _type, _payload: None)
    editing_adapter = LocalContextEditingAdapter(
        token_counter=token_counter,
        usable_input_budget=input_budget,
        trigger_ratio=settings.micro_compaction_trigger_ratio,
        min_reclaim_tokens=settings.micro_compaction_min_reclaim_tokens,
        min_reclaim_ratio=settings.micro_compaction_min_reclaim_ratio,
        keep_recent_tool_results=settings.micro_compaction_keep_recent_tool_results,
        event_sink=sink,
    )
    ingestor = ToolResultIngestor(
        artifact_store=artifact_store,
        token_counter=token_counter,
        inline_max_tokens=settings.tool_inline_max_tokens,
        event_sink=sink,
    )
    compressor_version = (
        f"{config.research_model_name}:session-memory-compressor-v1:"
        "continuation-compressor-v1:context-view-v1"
    )
    breaker = None
    if conversation_id is not None:
        breaker = CompressionCircuitBreaker(
            state_port=task_store,
            conversation_id=conversation_id,
            compressor_version=compressor_version,
            cooldown_seconds=settings.compression_breaker_cooldown_seconds,
            failure_threshold=settings.compression_failure_threshold,
        )
    coordinator = CompressionCoordinator(
        token_counter=token_counter,
        usable_input_budget=input_budget,
        full_compaction_enabled=settings.full_compaction_enabled,
        model=model,
        event_sink=sink,
        trigger_ratio=settings.full_compaction_trigger_ratio,
        session_target_ratio=settings.session_memory_target_ratio,
        full_target_ratio=settings.full_compaction_target_ratio,
        recent_turns=settings.full_compaction_recent_turns,
        transient_retry_count=settings.compression_transient_retry_count,
        breaker=breaker,
    )
    return ContextManagementRuntime(
        task_store=task_store,
        config=config,
        model=model,
        token_counter=token_counter,
        artifact_store=artifact_store,
        archive_store=task_store,
        compression_state_port=task_store,
        tool_result_ingestor=ingestor,
        editing_adapter=editing_adapter,
        compression_coordinator=coordinator,
        breaker=breaker,
    )
