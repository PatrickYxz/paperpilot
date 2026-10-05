"""Short transactions for append-only user long-term memories."""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import func, select

from paperpilot.web.db_models import (
    ConversationRow,
    MessageRow,
    TurnArchiveRow,
    UserMemoryProfileRow,
    UserMemoryRow,
)
from paperpilot.web.store.helpers import (
    SessionFactory,
    utc_now,
    user_memory_from_row,
)
from paperpilot.web.store.records import (
    NewUserMemory,
    UserProfileRecord,
    UserMemoryRecord,
    UserTurnSummary,
)

MEMORY_STATUS_ACTIVE = "active"


def append_user_memory(
    session_factory: SessionFactory,
    *,
    record: NewUserMemory,
) -> UserMemoryRecord:
    """Append one memory fact. Inserting an existing memory_id is idempotent."""
    with session_factory.begin() as session:
        existing = session.get(UserMemoryRow, record.memory_id)
        if existing is not None:
            return user_memory_from_row(existing)
        row = UserMemoryRow(
            memory_id=record.memory_id,
            user_id=record.user_id,
            kind=record.kind,
            content=record.content,
            context_json=_encode_context(record.context),
            source_conversation_id=record.source_conversation_id,
            source_task_id=record.source_task_id,
            source_message_id=record.source_message_id,
            support_span=record.support_span,
            status=MEMORY_STATUS_ACTIVE,
            created_at=record.created_at or utc_now(),
        )
        session.add(row)
        session.flush()
        return user_memory_from_row(row)


def list_user_memories(
    session_factory: SessionFactory,
    user_id: str,
    *,
    active_only: bool = True,
) -> list[UserMemoryRecord]:
    """List one user's memories, newest first. Isolation is enforced here."""
    with session_factory.begin() as session:
        query = select(UserMemoryRow).where(UserMemoryRow.user_id == user_id)
        if active_only:
            query = query.where(UserMemoryRow.status == MEMORY_STATUS_ACTIVE)
        rows = session.scalars(
            query.order_by(
                UserMemoryRow.created_at.desc(), UserMemoryRow.memory_id.desc()
            )
        ).all()
        return [user_memory_from_row(row) for row in rows]


def count_task_memories(session_factory: SessionFactory, task_id: str) -> int:
    """How many memories were already extracted for one source task."""
    with session_factory.begin() as session:
        return int(
            session.scalar(
                select(func.count()).where(
                    UserMemoryRow.source_task_id == task_id
                )
            )
            or 0
        )


def _encode_context(context: dict[str, Any]) -> str:
    return json.dumps(context, ensure_ascii=False, sort_keys=True)


def get_user_profile(
    session_factory: SessionFactory,
    user_id: str,
) -> UserProfileRecord | None:
    with session_factory.begin() as session:
        row = session.get(UserMemoryProfileRow, user_id)
        if row is None:
            return None
        return UserProfileRecord(
            user_id=row.user_id,
            profile_text=row.profile_text,
            source_memory_count=row.source_memory_count,
            updated_at=row.updated_at,
        )


def upsert_user_profile(
    session_factory: SessionFactory,
    *,
    user_id: str,
    profile_text: str,
    source_memory_count: int,
) -> UserProfileRecord:
    with session_factory.begin() as session:
        row = session.get(UserMemoryProfileRow, user_id)
        if row is None:
            row = UserMemoryProfileRow(
                user_id=user_id,
                profile_text=profile_text,
                source_memory_count=source_memory_count,
                updated_at=utc_now(),
            )
            session.add(row)
        else:
            row.profile_text = profile_text
            row.source_memory_count = source_memory_count
            row.updated_at = utc_now()
        session.flush()
        return UserProfileRecord(
            user_id=row.user_id,
            profile_text=row.profile_text,
            source_memory_count=row.source_memory_count,
            updated_at=row.updated_at,
        )


def list_user_turn_summaries(
    session_factory: SessionFactory,
    user_id: str,
    *,
    limit: int = 50,
) -> list[UserTurnSummary]:
    """Cross-conversation turn digests, newest first, user-scoped."""
    with session_factory.begin() as session:
        rows = session.execute(
            select(TurnArchiveRow, MessageRow)
            .join(
                ConversationRow,
                TurnArchiveRow.conversation_id == ConversationRow.id,
            )
            .join(MessageRow, TurnArchiveRow.user_message_id == MessageRow.id)
            .where(ConversationRow.user_id == user_id)
            .order_by(TurnArchiveRow.created_at.desc())
            .limit(limit)
        ).all()
        return [
            UserTurnSummary(
                conversation_id=archive.conversation_id,
                task_id=archive.task_id,
                user_message_id=archive.user_message_id,
                question=question.content[:200],
                narrative=archive.narrative_summary,
                created_at=archive.created_at,
            )
            for archive, question in rows
        ]
