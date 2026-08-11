"""Conversation-scoped task progress updates."""
from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Query

from paperpilot.web.schemas import TaskUpdatesResponse
from paperpilot.web.task_store import TaskStore, TaskUpdates, WebUser


RequireUser = Callable[..., WebUser]


def create_task_updates_router(
    *,
    store: TaskStore,
    require_user: RequireUser,
) -> APIRouter:
    router = APIRouter()

    @router.get(
        "/api/conversations/{conversation_id}/tasks/{task_id}/updates",
        response_model=TaskUpdatesResponse,
    )
    def get_conversation_task_updates(
        conversation_id: str,
        task_id: str,
        after_event_id: int = Query(default=0, ge=0),
        after_artifact_id: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
        user: WebUser = Depends(require_user),
    ) -> TaskUpdatesResponse:
        updates = store.get_conversation_task_updates(
            conversation_id,
            task_id,
            user_id=user.id,
            after_event_id=after_event_id,
            after_artifact_id=after_artifact_id,
            limit=limit,
        )
        if updates is None:
            raise HTTPException(status_code=404, detail="task not found")
        return TaskUpdatesResponse.model_validate(task_updates_dict(updates))

    return router


def task_updates_dict(updates: TaskUpdates) -> dict:
    return {
        "task": updates.task.to_dict(),
        "events": {
            "items": [event.to_dict() for event in updates.events.items],
            "next_after_id": updates.events.next_after_id,
            "has_more": updates.events.has_more,
        },
        "artifacts": {
            "items": [artifact.to_dict() for artifact in updates.artifacts.items],
            "next_after_id": updates.artifacts.next_after_id,
            "has_more": updates.artifacts.has_more,
        },
    }
