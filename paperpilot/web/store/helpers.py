"""Shared row mapping and ownership helpers for Web business stores.

This module owns only helpers used by at least two business modules.  It does
not own business transactions, HTTP behavior, or the TaskStore facade.  The
helpers map SQLAlchemy rows to immutable records, enforce shared ownership and
active-paper rules, and provide the common SQLite time/JSON primitives.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from paperpilot.web.db_models import (
    ConversationPaperRow,
    ConversationRow,
    CompressionStateRow,
    ContextArtifactRow,
    MessageRow,
    PaperRow,
    ResearchTaskRow,
    TaskArtifactRow,
    TaskEventRow,
    TurnArchiveRow,
    UserRow,
)
from paperpilot.web.store.records import (
    ConversationPaperRecord,
    ConversationRecord,
    MessageRecord,
    PaperRecord,
    ResearchTask,
    TaskArtifact,
    TaskArtifactBatch,
    TaskEvent,
    TaskEventBatch,
    WebUser,
    CompressionStateRecord,
    ContextArtifactRecord,
    TurnArchiveRecord,
)


SessionFactory = sessionmaker[Session]


class ContextDataCorruptionError(ValueError):
    """Raised when an internal context JSON column is malformed."""


def task_from_row(row: ResearchTaskRow) -> ResearchTask:
    return ResearchTask(
        id=row.id,
        question=row.question,
        depth=row.depth,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
        user_id=row.user_id,
        conversation_id=row.conversation_id,
        base_checkpoint_id=row.base_checkpoint_id,
        final_checkpoint_id=row.final_checkpoint_id,
        result_quality=row.result_quality,
    )


def user_from_row(row: UserRow) -> WebUser:
    return WebUser(
        id=row.id,
        username=row.username,
        password_hash=row.password_hash,
        password_salt=row.password_salt,
        created_at=row.created_at,
    )


def paper_from_row(row: PaperRow) -> PaperRecord:
    return PaperRecord(
        id=row.id,
        source=row.source,
        external_id=row.external_id,
        title=row.title,
        authors=decode_json_list(row.authors_json),
        abstract=row.abstract,
        source_url=row.source_url,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def conversation_from_row(row: ConversationRow) -> ConversationRecord:
    return ConversationRecord(
        id=row.id,
        user_id=row.user_id,
        primary_paper_id=row.primary_paper_id,
        title=row.title,
        head_message_id=row.head_message_id,
        head_checkpoint_id=row.head_checkpoint_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        archived_at=row.archived_at,
    )


def conversation_paper_from_row(
    row: ConversationPaperRow,
) -> ConversationPaperRecord:
    return ConversationPaperRecord(
        conversation_id=row.conversation_id,
        paper_id=row.paper_id,
        role=row.role,
        added_by=row.added_by,
        source_task_id=row.source_task_id,
        source_message_id=row.source_message_id,
        is_active=row.is_active,
        created_at=row.created_at,
    )


def message_from_row(row: MessageRow) -> MessageRecord:
    return MessageRecord(
        id=row.id,
        conversation_id=row.conversation_id,
        task_id=row.task_id,
        parent_message_id=row.parent_message_id,
        role=row.role,
        content=row.content,
        status=row.status,
        metadata=decode_payload(row.metadata_json),
        created_at=row.created_at,
    )


def event_from_row(row: TaskEventRow) -> TaskEvent:
    return TaskEvent(
        id=row.id,
        task_id=row.task_id,
        type=row.type,
        stage=row.stage,
        message=row.message,
        payload=decode_payload(row.payload_json),
        created_at=row.created_at,
    )


def artifact_from_row(row: TaskArtifactRow) -> TaskArtifact:
    return TaskArtifact(
        id=row.id,
        task_id=row.task_id,
        kind=row.kind,
        title=row.title,
        content=row.content,
        payload=decode_payload(row.payload_json),
        created_at=row.created_at,
    )


def context_artifact_from_row(row: ContextArtifactRow) -> ContextArtifactRecord:
    return ContextArtifactRecord(
        artifact_id=row.artifact_id,
        conversation_id=row.conversation_id,
        task_id=row.task_id,
        tool_call_id=row.tool_call_id,
        tool_name=row.tool_name,
        kind=row.kind,
        storage_key=row.storage_key,
        sha256=row.sha256,
        byte_size=row.byte_size,
        token_estimate=row.token_estimate,
        preview=row.preview,
        initial_action=row.initial_action,
        future_retention=row.future_retention,
        created_at=row.created_at,
    )


def turn_archive_from_row(row: TurnArchiveRow) -> TurnArchiveRecord:
    return TurnArchiveRecord(
        archive_id=row.archive_id,
        conversation_id=row.conversation_id,
        task_id=row.task_id,
        user_message_id=row.user_message_id,
        terminal_status=row.terminal_status,
        archive_version=row.archive_version,
        seed_json=decode_context_json(row.seed_json, "seed_json"),
        supersedes_json=decode_context_json(
            row.supersedes_json, "supersedes_json"
        ),
        created_at=row.created_at,
        narrative_summary=row.narrative_summary,
        narrative_status=row.narrative_status,
        updated_at=row.updated_at,
    )


def compression_state_from_row(row: CompressionStateRow) -> CompressionStateRecord:
    return CompressionStateRecord(
        conversation_id=row.conversation_id,
        compressor_version=row.compressor_version,
        state=row.state,
        consecutive_failures=row.consecutive_failures,
        last_failure_type=row.last_failure_type,
        last_input_digest=row.last_input_digest,
        opened_at=row.opened_at,
        updated_at=row.updated_at,
    )


def decode_context_json(value: object, field_name: str) -> Any:
    try:
        decoded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ContextDataCorruptionError(
            f"invalid context JSON in {field_name}"
        ) from exc
    if not isinstance(decoded, (dict, list)):
        raise ContextDataCorruptionError(
            f"context JSON in {field_name} must be an object or array"
        )
    return decoded


def select_owned_task(
    session: Session,
    task_id: str,
    user_id: str | None,
) -> ResearchTaskRow | None:
    filters = [ResearchTaskRow.id == task_id]
    if user_id is not None:
        filters.append(ResearchTaskRow.user_id == user_id)
    return session.scalar(select(ResearchTaskRow).where(*filters))


def select_owned_conversation(
    session: Session,
    conversation_id: str,
    user_id: str,
) -> ConversationRow | None:
    return session.scalar(
        select(ConversationRow).where(
            ConversationRow.id == conversation_id,
            ConversationRow.user_id == user_id,
        )
    )


def validate_active_paper_ids(
    session: Session,
    *,
    conversation_row: ConversationRow,
    active_paper_ids: list[str],
) -> list[str]:
    requested_ids = set(active_paper_ids)
    requested_ids.add(conversation_row.primary_paper_id)
    association_rows = list(
        session.scalars(
            select(ConversationPaperRow)
            .where(ConversationPaperRow.conversation_id == conversation_row.id)
            .order_by(
                ConversationPaperRow.created_at.asc(),
                ConversationPaperRow.paper_id.asc(),
            )
        )
    )
    associated_ids = {row.paper_id for row in association_rows}
    unknown_ids = requested_ids - associated_ids
    if unknown_ids:
        raise ValueError(
            "active papers are not associated with conversation: "
            + ", ".join(sorted(unknown_ids))
        )
    return [conversation_row.primary_paper_id] + [
        row.paper_id
        for row in association_rows
        if row.paper_id != conversation_row.primary_paper_id
        and row.paper_id in requested_ids
    ]


def read_active_paper_ids(
    session: Session,
    conversation_row: ConversationRow,
) -> list[str]:
    rows = list(
        session.scalars(
            select(ConversationPaperRow)
            .where(
                ConversationPaperRow.conversation_id == conversation_row.id,
                ConversationPaperRow.is_active.is_(True),
            )
            .order_by(
                ConversationPaperRow.created_at.asc(),
                ConversationPaperRow.paper_id.asc(),
            )
        )
    )
    active_ids = {row.paper_id for row in rows}
    return (
        [conversation_row.primary_paper_id]
        if conversation_row.primary_paper_id in active_ids
        else []
    ) + [
        row.paper_id
        for row in rows
        if row.paper_id != conversation_row.primary_paper_id
    ]


def set_active_paper_ids(
    session: Session,
    *,
    conversation_row: ConversationRow,
    active_paper_ids: list[str],
) -> None:
    session.execute(
        update(ConversationPaperRow)
        .where(ConversationPaperRow.conversation_id == conversation_row.id)
        .values(is_active=False)
    )
    session.execute(
        update(ConversationPaperRow)
        .where(
            ConversationPaperRow.conversation_id == conversation_row.id,
            ConversationPaperRow.paper_id.in_(active_paper_ids),
        )
        .values(is_active=True)
    )


def decode_payload(value: object) -> dict:
    if value is None:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def decode_json_list(value: object) -> list[str]:
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_in(*, days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(
        timespec="seconds"
    )
