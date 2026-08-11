"""Shared strict HTTP schemas for the Web workbench."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AuthRequest(StrictApiModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=6)


class UserResponse(StrictApiModel):
    id: str
    username: str
    created_at: str


class PaperReference(StrictApiModel):
    source: Literal["arxiv"]
    external_id: str = Field(..., min_length=1)


class CreateConversationRequest(StrictApiModel):
    paper: PaperReference
    title: str | None = None


class UpdateConversationRequest(StrictApiModel):
    title: str | None = None
    archived: bool | None = None

    @model_validator(mode="after")
    def require_change(self) -> "UpdateConversationRequest":
        if self.title is None and self.archived is None:
            raise ValueError("title or archived is required")
        return self


class CreateMessageRequest(StrictApiModel):
    content: str = Field(..., min_length=1)
    depth: Literal["quick", "standard", "deep"] = "standard"
    expected_head_message_id: str | None


class RollbackRequest(StrictApiModel):
    message_id: str = Field(..., min_length=1)
    expected_head_message_id: str | None


class PaperResponse(StrictApiModel):
    source: str
    external_id: str
    title: str
    authors: list[str]
    abstract: str | None
    source_url: str


class PaperSearchResponse(StrictApiModel):
    items: list[PaperResponse]


class ConversationResponse(StrictApiModel):
    id: str
    primary_paper_id: str
    title: str
    head_message_id: str | None
    head_checkpoint_id: str | None
    created_at: str
    updated_at: str
    archived_at: str | None


class ConversationListResponse(StrictApiModel):
    items: list[ConversationResponse]


class TaskResponse(StrictApiModel):
    id: str
    question: str
    depth: str
    status: str
    created_at: str
    updated_at: str


class ConversationDetailResponse(StrictApiModel):
    conversation: ConversationResponse
    primary_paper: PaperResponse
    active_papers: list[PaperResponse]
    active_task: TaskResponse | None


class MessageResponse(StrictApiModel):
    id: str
    conversation_id: str
    task_id: str | None
    parent_message_id: str | None
    role: str
    content: str
    status: str
    metadata: dict
    created_at: str


class UnstableTurnResponse(StrictApiModel):
    user_message: MessageResponse
    task: TaskResponse


class ConversationMessagesResponse(StrictApiModel):
    items: list[MessageResponse]
    unstable_turn: UnstableTurnResponse | None


class ConversationAlternativeResponse(StrictApiModel):
    user_message: MessageResponse
    assistant_message: MessageResponse


class ConversationAlternativesResponse(StrictApiModel):
    items: list[ConversationAlternativeResponse]


class CreateMessageResponse(StrictApiModel):
    user_message: MessageResponse
    task: TaskResponse
    stable_head_message_id: str | None


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
