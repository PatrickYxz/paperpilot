"""Prepare the single authoritative dynamic model ContextView."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
from typing import Any

from langgraph.runtime import Runtime

from ..context_management.budget import ModelAwareTokenCounter
from ..context_management.compaction import ContextCapacityExhaustedError
from ..context_management.models import ArtifactRef, ContextView
from ..context_management.views import ArchiveRetriever, ContextViewBuilder
from ..research_agent import _research_request_budget_inputs
from ..state import DeepReadingState
from ...web.store.records import TurnArchiveSeedRecord
from ...web.context_artifacts import ArtifactPutRequest
from .binding import _validate_runtime_binding
from .context import DeepReadingContext


def prepare_context(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:
    """Rebuild context from stable messages and persisted archives for this turn."""
    context = runtime.context
    if not context.context_management.enabled:
        return {}
    _task, user_message, _detail = _validate_runtime_binding(state, context)
    _migrate_legacy_summary_once(state, context, user_message)
    token_counter = ModelAwareTokenCounter(context.model)
    archives = context.task_store.list_turn_archives(context.conversation_id)
    stable_messages = context.task_store.list_active_messages(
        context.conversation_id,
        user_id=context.user_id,
    )
    source_messages: list[object] = (
        list(stable_messages) if stable_messages is not None else list(state.get("messages", []))
    )
    rendered_messages = [_message_view(item) for item in source_messages]
    if not any(item.get("id") == user_message.id for item in rendered_messages):
        rendered_messages.append(_message_view(user_message))
    retriever = ArchiveRetriever(
        token_counter=token_counter,
        budget_tokens=context.context_management.archive_context_budget_tokens,
        max_records=context.context_management.archive_context_max_records,
        recent_records=context.context_management.archive_context_recent_records,
    )
    fixed_messages, trailing_messages, tool_schemas = _research_request_budget_inputs(
        state,
        context,
    )
    view_builder = ContextViewBuilder(
        token_counter=token_counter,
        archive_retriever=retriever,
        fixed_messages=fixed_messages,
        trailing_messages=trailing_messages,
        tool_schemas=tool_schemas,
        recent_turns=context.context_management.full_compaction_recent_turns,
    )
    view = view_builder.build(
        current_goal=user_message.content,
        active_paper_ids=state.get("active_paper_ids", []),
        messages=rendered_messages,
        archives=archives,
        archive_query=user_message.content,
        continuation_capsule=state.get("continuation_capsule"),
    )
    compression_coordinator = getattr(
        context.context_management_runtime,
        "compression_coordinator",
        None,
    )
    if compression_coordinator is not None:
        decision = compression_coordinator.prepare(
            view,
            task_id=context.task_id,
            conversation_id=context.conversation_id,
            view_token_counter=view_builder.count_view,
        )
        view = decision.view
        if not decision.accepted and view.input_tokens > compression_coordinator.usable_input_budget:
            try:
                view = view_builder.build_minimal_safe_view_with_budget(
                    view,
                    usable_input_budget=compression_coordinator.usable_input_budget,
                )
                context.event_sink(
                    "compression_degraded",
                    _compression_event_payload(
                        view=decision.view,
                        stage="safe_minimum",
                        reason=decision.reason,
                        before_tokens=decision.view.input_tokens,
                        after_tokens=view.input_tokens,
                    ),
                )
            except ContextCapacityExhaustedError:
                capacity_artifact = None
                try:
                    capacity_artifact = _persist_capacity_capsule(context, decision.view)
                except Exception:
                    context.event_sink(
                        "context_capacity_artifact_failed",
                        {
                            "stage": "safe_minimum",
                            "reason": "artifact_persistence_failed",
                        },
                    )
                payload = _compression_event_payload(
                    view=decision.view,
                    stage="safe_minimum",
                    reason="protected_context_oversized",
                    before_tokens=decision.view.input_tokens,
                    after_tokens=decision.view.input_tokens,
                )
                if capacity_artifact is not None:
                    payload.update(
                        {
                            "artifact_id": capacity_artifact.artifact_id,
                            "artifact_sha256": capacity_artifact.sha256,
                            "artifact_token_estimate": capacity_artifact.token_estimate,
                        }
                    )
                context.event_sink(
                    "context_capacity_exhausted",
                    payload,
                )
                raise
    return {
        "context_view": view.model_dump(mode="json"),
        "continuation_capsule": view.continuation_capsule,
        "retrieved_archive_ids": [
            item.archive_id for item in view.retrieved_archives
        ],
        "context_input_tokens": view.input_tokens,
    }


def _persist_capacity_capsule(
    context: DeepReadingContext,
    view: ContextView,
) -> ArtifactRef | None:
    runtime = context.context_management_runtime
    artifact_store = getattr(runtime, "artifact_store", None)
    if artifact_store is None:
        return None
    projection = view.active_projection
    current_goal = projection.current_goal if projection is not None else ""
    current_message = next(
        (
            {
                key: value
                for key, value in message.items()
                if key in {"role", "id", "content"}
            }
            for message in reversed(view.messages)
            if message.get("role") in {"user", "human"}
            and message.get("content") == current_goal
        ),
        None,
    )
    payload = {
        "schema_version": "paperpilot-capacity-continuation-v1",
        "current_message": current_message,
        "active_projection": (
            projection.model_dump(mode="json") if projection is not None else None
        ),
        "continuation_capsule": view.continuation_capsule,
        "authority_archive_ids": list(view.authority_archive_ids),
        "authority_evidence_ids": list(view.authority_evidence_ids),
        "authority_artifact_refs": [
            item.model_dump(mode="json") for item in view.authority_artifact_refs
        ],
        "verification": list(view.authority_verification),
        "failed_paths": list(view.authority_failed_paths),
        "rollback_notes": list(view.authority_rollback_notes),
    }
    preview = (
        "capacity continuation: "
        f"archives={len(view.authority_archive_ids)}, "
        f"evidence={len(view.authority_evidence_ids)}, "
        f"artifacts={len(view.authority_artifact_refs)}"
    )
    record = artifact_store.put(
        ArtifactPutRequest(
            conversation_id=context.conversation_id,
            task_id=context.task_id,
            tool_call_id=f"context-capacity:{context.task_id}",
            tool_name="context_capacity",
            kind="continuation_capsule",
            payload=payload,
            preview=preview,
            initial_action="EXTERNALIZE_NOW",
            future_retention="PROTECTED",
        )
    )
    return ArtifactRef(
        artifact_id=record.artifact_id,
        sha256=record.sha256,
        token_estimate=record.token_estimate,
        preview=record.preview,
    )


def _migrate_legacy_summary_once(
    state: DeepReadingState,
    context: DeepReadingContext,
    user_message: object,
) -> None:
    summary = state.get("conversation_summary")
    if not isinstance(summary, Mapping):
        return
    archives = context.task_store.list_turn_archives(context.conversation_id)
    if any(
        getattr(item, "archive_version", None) == "legacy-summary-v1"
        for item in archives
    ):
        return
    archive_id = "legacy_" + hashlib.sha256(
        context.conversation_id.encode("utf-8")
    ).hexdigest()[:24]
    message_id = getattr(user_message, "id", context.current_user_message_id)
    created_at = datetime.now(timezone.utc).isoformat()
    context.task_store.seed_turn_archive(
        seed=TurnArchiveSeedRecord(
            archive_id=archive_id,
            conversation_id=context.conversation_id,
            task_id=context.task_id,
            user_message_id=message_id,
            terminal_status="success",
            archive_version="legacy-summary-v1",
            seed_json={
                "archive_type": "legacy-summary-v1",
                "untrusted": True,
                "legacy_summary": dict(summary),
            },
            supersedes_json=[],
            created_at=created_at,
        )
    )


def _compression_event_payload(
    *,
    view: object,
    stage: str,
    reason: str,
    before_tokens: int,
    after_tokens: int,
) -> dict[str, object]:
    projection = getattr(view, "active_projection", None)
    return {
        "stage": stage,
        "compressor_version": "context-compression-v1",
        "reason": reason,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "reclaimed_tokens": max(0, before_tokens - after_tokens),
        "protected_item_count": len(getattr(projection, "protected_items", ())),
        "archive_ref_count": len(getattr(view, "authority_archive_ids", ())),
        "artifact_ref_count": len(getattr(view, "authority_artifact_ids", ())),
        "validation_failure_type": None,
        "breaker_state": "CLOSED",
    }


def _message_view(message: object) -> dict[str, Any]:
    message_id = getattr(message, "id", None)
    role = getattr(message, "role", None)
    if role is None:
        message_type = getattr(message, "type", "message")
        role = {
            "human": "user",
            "ai": "assistant",
            "tool": "tool",
            "system": "system",
        }.get(str(message_type), str(message_type))
    content = getattr(message, "content", "")
    if role == "tool":
        content = "<tool_result_externalized>"
    result: dict[str, Any] = {"role": str(role), "content": content}
    if isinstance(message_id, str) and message_id.strip():
        result["id"] = message_id
    return result
