"""Conversation Message read and submit HTTP handlers.

This module owns active-path and alternatives reads plus executor admission and
user-turn submission.  It does not own Store transactions or rollback
checkpoint validation.  Capacity is reserved before creating the turn; a
failed broker submission is cleaned up only while the Task is still pending,
because a Worker may already have claimed it.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from paperpilot.web.observability import RUNTIME_LOGGER_NAME
from paperpilot.web.routes.auth import RequireUser
from paperpilot.web.routes.conversations.presenters import message_dict, task_dict
from paperpilot.web.schemas import (
    ConversationAlternativesResponse,
    ConversationMessagesResponse,
    CreateMessageRequest,
    CreateMessageResponse,
)
from paperpilot.web.task_executor import (
    TaskExecutorAtCapacityError,
    TaskExecutorLike,
    TaskExecutorShuttingDownError,
)
from paperpilot.web.task_store import (
    ConversationBusyError,
    ConversationDetail,
    StaleConversationHeadError,
    TaskStore,
    WebUser,
)


_LOGGER = logging.getLogger(RUNTIME_LOGGER_NAME)


def build_message_router(
    *,
    store: TaskStore,
    executor: TaskExecutorLike,
    require_user: RequireUser,
    overload_retry_after_seconds: int,
) -> APIRouter:
    router = APIRouter()

    @router.get(
        "/{conversation_id}/messages",
        response_model=ConversationMessagesResponse,
    )
    def list_messages(
        conversation_id: str,
        user: WebUser = Depends(require_user),
    ) -> dict:
        _owned_detail(store, conversation_id, user.id)
        messages = store.list_active_messages(conversation_id, user_id=user.id)
        if messages is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        unstable = store.get_unstable_turn(conversation_id, user_id=user.id)
        return {
            "items": [message_dict(message) for message in messages],
            "unstable_turn": (
                {
                    "user_message": message_dict(unstable.user_message),
                    "task": task_dict(unstable.task),
                }
                if unstable is not None
                else None
            ),
        }

    @router.get(
        "/{conversation_id}/messages/{message_id}/alternatives",
        response_model=ConversationAlternativesResponse,
    )
    def list_alternatives(
        conversation_id: str,
        message_id: str,
        user: WebUser = Depends(require_user),
    ) -> dict:
        _owned_detail(store, conversation_id, user.id)
        try:
            alternatives = store.list_message_alternatives(
                conversation_id,
                message_id,
                user_id=user.id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if alternatives is None:
            raise HTTPException(status_code=404, detail="message not found")
        return {
            "items": [
                {
                    "user_message": message_dict(item.user_message),
                    "assistant_message": message_dict(item.assistant_message),
                }
                for item in alternatives
            ]
        }

    @router.post(
        "/{conversation_id}/messages",
        response_model=CreateMessageResponse,
        status_code=202,
    )
    def create_message(
        conversation_id: str,
        payload: CreateMessageRequest,
        user: WebUser = Depends(require_user),
    ) -> dict:
        detail = _owned_detail(store, conversation_id, user.id)
        _validate_turn_admission(detail, payload.expected_head_message_id)
        try:
            reservation = executor.reserve()
        except TaskExecutorAtCapacityError as exc:
            raise HTTPException(
                status_code=503,
                detail="task executor is at capacity",
                headers={"Retry-After": str(overload_retry_after_seconds)},
            ) from exc
        except TaskExecutorShuttingDownError as exc:
            raise HTTPException(
                status_code=503,
                detail="task executor is shutting down",
            ) from exc

        try:
            turn = store.create_conversation_turn(
                user_id=user.id,
                conversation_id=conversation_id,
                content=payload.content,
                depth=payload.depth,
                expected_head_message_id=payload.expected_head_message_id,
            )
        except ConversationBusyError as exc:
            reservation.release()
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except StaleConversationHeadError as exc:
            reservation.release()
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            reservation.release()
            if store.get_conversation_detail(
                conversation_id,
                user_id=user.id,
            ) is None:
                raise HTTPException(
                    status_code=404,
                    detail="conversation not found",
                ) from exc
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception:
            reservation.release()
            raise

        response = {
            "user_message": message_dict(turn.user_message),
            "task": task_dict(turn.task),
            "stable_head_message_id": detail.conversation.head_message_id,
        }
        try:
            reservation.submit(turn.task.id)
        except Exception as exc:
            try:
                failed = store.fail_pending_task(turn.task.id)
                if failed is None:
                    current = store.get_task(turn.task.id, user_id=user.id)
                    if current is not None and current.status in {
                        "running",
                        "completed",
                        "failed",
                    }:
                        return response
                else:
                    store.add_event(
                        task_id=turn.task.id,
                        type="failed",
                        stage="queue",
                        message="Task queue submission failed.",
                        payload={"error_type": type(exc).__name__},
                    )
            except Exception:
                _LOGGER.exception(
                    "Conversation task submission cleanup failed",
                    extra={
                        "event": "conversation.submission_cleanup_failed",
                        "user_id": user.id,
                        "task_id": turn.task.id,
                    },
                )
            raise HTTPException(
                status_code=503,
                detail="task queue unavailable",
            ) from exc
        return response

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
