"""Conversation rollback HTTP handler and checkpoint admission checks.

This module validates a target Assistant Message, its completed Task, and the
checkpoint state before one Store transaction switches both Conversation heads
and active papers.  It does not delete history, query the database directly,
or own Message submission behavior.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from paperpilot.deep_reading.runner import DeepReadingRunner
from paperpilot.deep_reading.state import GRAPH_VERSION, SCHEMA_VERSION
from paperpilot.web.routes.auth import RequireUser
from paperpilot.web.routes.conversations.presenters import conversation_dict
from paperpilot.web.schemas import ConversationResponse, RollbackRequest
from paperpilot.web.task_store import (
    ConversationBusyError,
    ConversationDetail,
    StaleConversationHeadError,
    TaskStore,
    WebUser,
)


def build_rollback_router(
    *,
    store: TaskStore,
    require_user: RequireUser,
    deep_reading_runner: DeepReadingRunner,
) -> APIRouter:
    router = APIRouter()

    @router.post("/{conversation_id}/rollback", response_model=ConversationResponse)
    def rollback_conversation(
        conversation_id: str,
        payload: RollbackRequest,
        user: WebUser = Depends(require_user),
    ) -> dict:
        detail = _owned_detail(store, conversation_id, user.id)
        _validate_rollback_admission(detail, payload.expected_head_message_id)
        target = store.get_message(
            conversation_id,
            payload.message_id,
            user_id=user.id,
        )
        if target is None:
            raise HTTPException(status_code=404, detail="message not found")
        if target.role != "assistant" or target.status != "complete":
            raise HTTPException(
                status_code=409,
                detail="target must be a complete assistant message",
            )
        if target.task_id is None:
            raise HTTPException(status_code=409, detail="target task is unavailable")
        task = store.get_task(target.task_id, user_id=user.id)
        if task is None or task.conversation_id != conversation_id:
            raise HTTPException(status_code=409, detail="target task is unavailable")
        if task.status != "completed" or not task.final_checkpoint_id:
            raise HTTPException(status_code=409, detail="checkpoint is unavailable")

        checkpoint = deep_reading_runner.read_checkpoint(
            conversation_id,
            task.final_checkpoint_id,
        )
        active_paper_ids = _validate_rollback_checkpoint(
            checkpoint,
            checkpoint_id=task.final_checkpoint_id,
            task_id=task.id,
            message_id=target.id,
            primary_paper_id=detail.conversation.primary_paper_id,
        )
        try:
            switched = store.switch_conversation_head(
                conversation_id,
                user_id=user.id,
                expected_head_message_id=payload.expected_head_message_id,
                target_message_id=target.id,
                target_checkpoint_id=task.final_checkpoint_id,
                active_paper_ids=active_paper_ids,
            )
        except (ConversationBusyError, StaleConversationHeadError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if switched is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return conversation_dict(switched)

    return router


def _owned_detail(
    store: TaskStore,
    conversation_id: str,
    user_id: str,
) -> ConversationDetail:
    detail = store.get_conversation_detail(conversation_id, user_id=user_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return detail


def _validate_turn_admission(
    detail: ConversationDetail,
    expected_head_message_id: str | None,
) -> None:
    if detail.conversation.archived_at is not None:
        raise HTTPException(status_code=409, detail="conversation is archived")
    if detail.active_task is not None:
        raise HTTPException(
            status_code=409,
            detail="conversation already has an active task",
        )
    if detail.conversation.head_message_id != expected_head_message_id:
        raise HTTPException(status_code=409, detail="conversation head has changed")


def _validate_rollback_admission(
    detail: ConversationDetail,
    expected_head_message_id: str | None,
) -> None:
    _validate_turn_admission(detail, expected_head_message_id)


def _validate_rollback_checkpoint(
    checkpoint: object,
    *,
    checkpoint_id: str,
    task_id: str,
    message_id: str,
    primary_paper_id: str,
) -> list[str]:
    if checkpoint is None:
        raise HTTPException(status_code=409, detail="checkpoint is unavailable")
    if getattr(checkpoint, "checkpoint_id", None) != checkpoint_id:
        raise HTTPException(status_code=409, detail="checkpoint is unavailable")
    state = getattr(checkpoint, "state", None)
    if not isinstance(state, dict):
        raise HTTPException(status_code=409, detail="checkpoint schema is unsupported")
    if state.get("schema_version") != SCHEMA_VERSION:
        raise HTTPException(status_code=409, detail="checkpoint schema is unsupported")
    if state.get("graph_version") != GRAPH_VERSION:
        raise HTTPException(status_code=409, detail="checkpoint graph is unsupported")
    if getattr(checkpoint, "is_complete", None) is not True:
        raise HTTPException(status_code=409, detail="checkpoint is not complete")
    if state.get("published_message_id") != message_id:
        raise HTTPException(
            status_code=409,
            detail="checkpoint does not match target message",
        )
    if state.get("current_task_id") != task_id:
        raise HTTPException(
            status_code=409,
            detail="checkpoint does not match target task",
        )
    active_paper_ids = state.get("active_paper_ids")
    if (
        not isinstance(active_paper_ids, list)
        or not active_paper_ids
        or any(
            not isinstance(paper_id, str) or not paper_id
            for paper_id in active_paper_ids
        )
        or len(set(active_paper_ids)) != len(active_paper_ids)
        or primary_paper_id not in active_paper_ids
    ):
        raise HTTPException(
            status_code=409,
            detail="checkpoint active paper ids are invalid",
        )
    return list(active_paper_ids)
