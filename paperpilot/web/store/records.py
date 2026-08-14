"""Immutable business records returned by the Web persistence layer.

These dataclasses describe the Web domain and are intentionally separate from
SQLAlchemy rows.  They carry no queries or transaction behavior; the business
modules own persistence and keep their transaction, ownership, and idempotency
rules around these values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from paperpilot.papers import PaperCandidate


VALID_DEPTHS: set[str] = {"quick", "standard", "deep"}


class DuplicateUsernameError(ValueError):
    """Raised when the users table rejects a duplicate username."""


class ConversationBusyError(ValueError):
    """Raised when a conversation already has pending or running work."""


class StaleConversationHeadError(ValueError):
    """Raised when a caller submits against an obsolete stable head."""


@dataclass(frozen=True)
class ResearchTask:
    id: str
    question: str
    depth: str
    status: str
    created_at: str
    updated_at: str
    user_id: str | None = None
    conversation_id: str | None = None
    base_checkpoint_id: str | None = None
    final_checkpoint_id: str | None = None
    result_quality: str | None = None

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "question": self.question,
            "depth": self.depth,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class WebUser:
    id: str
    username: str
    password_hash: str
    password_salt: str
    created_at: str

    def to_public_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "username": self.username,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TaskEvent:
    id: int
    task_id: str
    type: str
    stage: str | None
    message: str
    payload: dict
    created_at: str

    def to_dict(self) -> dict[str, int | str | dict | None]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "type": self.type,
            "stage": self.stage,
            "message": self.message,
            "payload": self.payload,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TaskArtifact:
    id: int
    task_id: str
    kind: str
    title: str
    content: str
    payload: dict
    created_at: str

    def to_dict(self) -> dict[str, int | str | dict]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "kind": self.kind,
            "title": self.title,
            "content": self.content,
            "payload": self.payload,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TaskEventBatch:
    items: list[TaskEvent]
    next_after_id: int
    has_more: bool


@dataclass(frozen=True)
class TaskArtifactBatch:
    items: list[TaskArtifact]
    next_after_id: int
    has_more: bool


@dataclass(frozen=True)
class TaskUpdates:
    task: ResearchTask
    events: TaskEventBatch
    artifacts: TaskArtifactBatch


@dataclass(frozen=True)
class PaperRecord:
    id: str
    source: str
    external_id: str
    title: str
    authors: list[str]
    abstract: str | None
    source_url: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ConversationRecord:
    id: str
    user_id: str
    primary_paper_id: str
    title: str
    head_message_id: str | None
    head_checkpoint_id: str | None
    created_at: str
    updated_at: str
    archived_at: str | None


@dataclass(frozen=True)
class ConversationPaperRecord:
    conversation_id: str
    paper_id: str
    role: str
    added_by: str
    source_task_id: str | None
    source_message_id: str | None
    is_active: bool
    created_at: str


@dataclass(frozen=True)
class ConversationDetail:
    conversation: ConversationRecord
    primary_paper: PaperRecord
    active_papers: list[PaperRecord]
    paper_associations: list[ConversationPaperRecord]
    active_task: ResearchTask | None


@dataclass(frozen=True)
class MessageRecord:
    id: str
    conversation_id: str
    task_id: str | None
    parent_message_id: str | None
    role: str
    content: str
    status: str
    metadata: dict
    created_at: str


@dataclass(frozen=True)
class ConversationTurn:
    user_message: MessageRecord
    task: ResearchTask


@dataclass(frozen=True)
class ConversationAlternative:
    user_message: MessageRecord
    assistant_message: MessageRecord


@dataclass(frozen=True)
class UsedPaperInput:
    paper: PaperCandidate
    role: Literal["comparison", "citation", "background", "follow_up"]


@dataclass(frozen=True)
class PublishedConversationResult:
    message: MessageRecord
    artifact: TaskArtifact
    active_paper_ids: list[str]


@dataclass(frozen=True)
class FinalizedConversationTask:
    task: ResearchTask
    conversation: ConversationRecord
