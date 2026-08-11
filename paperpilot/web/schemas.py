"""Shared strict HTTP response schemas for the Web workbench."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskResponse(StrictApiModel):
    id: str
    question: str
    depth: str
    status: str
    created_at: str
    updated_at: str


class TaskEventResponse(StrictApiModel):
    id: int
    task_id: str
    type: str
    stage: str | None
    message: str
    payload: dict
    created_at: str


class TaskArtifactResponse(StrictApiModel):
    id: int
    task_id: str
    kind: str
    title: str
    content: str
    payload: dict
    created_at: str


class TaskPageResponse(StrictApiModel):
    items: list[TaskResponse]
    next_cursor: str | None
    has_more: bool


class TaskEventPageResponse(StrictApiModel):
    items: list[TaskEventResponse]
    next_after_id: int
    has_more: bool


class TaskArtifactPageResponse(StrictApiModel):
    items: list[TaskArtifactResponse]
    next_after_id: int
    has_more: bool


class TaskUpdatesResponse(StrictApiModel):
    task: TaskResponse
    events: TaskEventPageResponse
    artifacts: TaskArtifactPageResponse
