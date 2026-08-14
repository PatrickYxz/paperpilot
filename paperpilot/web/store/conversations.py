"""Conversation CRUD and paper associations.

This module owns ConversationRow, PaperRow, and ConversationPaperRow operations, including ownership, title validation, and the primary-paper association. It does not own the message tree, Task publication, or HTTP routing. Conversation creation keeps Paper upsert, Conversation creation, and primary association in one transaction."""

from __future__ import annotations

import json
import secrets
import uuid
from typing import Literal

from sqlalchemy import (
    Engine,
    delete,
    exists,
    func,
    insert,
    literal,
    select,
    text,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from paperpilot.papers import PaperCandidate
from paperpilot.web.db_models import (
    ConversationPaperRow,
    ConversationRow,
    LoginSessionRow,
    MessageRow,
    PaperRow,
    ResearchTaskRow,
    TaskArtifactRow,
    TaskEventRow,
    UserRow,
)
from paperpilot.web.store.helpers import (
    artifact_from_row,
    conversation_from_row,
    conversation_paper_from_row,
    decode_json_list,
    decode_payload,
    event_from_row,
    message_from_row,
    paper_from_row,
    read_active_paper_ids,
    select_owned_conversation,
    select_owned_task,
    set_active_paper_ids,
    task_from_row,
    user_from_row,
    utc_in,
    utc_now,
    validate_active_paper_ids,
)
from paperpilot.web.store.records import (
    ConversationAlternative,
    ConversationBusyError,
    ConversationDetail,
    ConversationPaperRecord,
    ConversationRecord,
    ConversationTurn,
    DuplicateUsernameError,
    FinalizedConversationTask,
    MessageRecord,
    PaperRecord,
    PublishedConversationResult,
    ResearchTask,
    StaleConversationHeadError,
    TaskArtifact,
    TaskArtifactBatch,
    TaskEvent,
    TaskEventBatch,
    TaskUpdates,
    UsedPaperInput,
    VALID_DEPTHS,
    WebUser,
)
from paperpilot.web.store.helpers import SessionFactory

def create_conversation(session_factory: SessionFactory,
    *,
    user_id: str,
    paper: PaperCandidate,
    title: str | None = None,
) -> ConversationRecord:
    paper_title = paper.title.strip()
    conversation_title = _validate_conversation_title(
        paper_title if title is None else title
    )
    source = paper.source.strip()
    external_id = paper.external_id.strip()
    now = utc_now()
    authors_json = json.dumps(paper.authors, ensure_ascii=False)
    source_url = paper.source_url.strip()

    with session_factory.begin() as session:
        paper_row = session.scalar(
            sqlite_insert(PaperRow)
            .values(
                id=f"paper_{uuid.uuid4().hex}",
                source=source,
                external_id=external_id,
                title=paper_title,
                authors_json=authors_json,
                abstract=paper.abstract,
                source_url=source_url,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=[PaperRow.source, PaperRow.external_id],
                set_={
                    "title": paper_title,
                    "authors_json": authors_json,
                    "abstract": paper.abstract,
                    "source_url": source_url,
                    "updated_at": now,
                },
            )
            .returning(PaperRow)
        )
        if paper_row is None:
            raise RuntimeError("paper upsert did not return a row")

        conversation_row = ConversationRow(
            id=f"conv_{uuid.uuid4().hex}",
            user_id=user_id,
            primary_paper_id=paper_row.id,
            title=conversation_title,
            head_message_id=None,
            head_checkpoint_id=None,
            created_at=now,
            updated_at=now,
            archived_at=None,
        )
        session.add(conversation_row)
        session.flush()
        session.add(
            ConversationPaperRow(
                conversation_id=conversation_row.id,
                paper_id=paper_row.id,
                role="primary",
                added_by="user",
                source_task_id=None,
                source_message_id=None,
                is_active=True,
                created_at=now,
            )
        )
        session.flush()
        return conversation_from_row(conversation_row)

def list_conversations(session_factory: SessionFactory,
    *,
    user_id: str,
    include_archived: bool = False,
    limit: int = 100,
) -> list[ConversationRecord]:
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    filters = [ConversationRow.user_id == user_id]
    if not include_archived:
        filters.append(ConversationRow.archived_at.is_(None))
    statement = (
        select(ConversationRow)
        .where(*filters)
        .order_by(
            ConversationRow.updated_at.desc(),
            ConversationRow.id.desc(),
        )
        .limit(limit)
    )
    with session_factory() as session:
        rows = list(session.scalars(statement))
    return [conversation_from_row(row) for row in rows]

def get_conversation_detail(session_factory: SessionFactory,
    conversation_id: str,
    *,
    user_id: str,
) -> ConversationDetail | None:
    with session_factory() as session:
        conversation_row = select_owned_conversation(
            session,
            conversation_id,
            user_id,
        )
        if conversation_row is None:
            return None

        primary_paper_row = session.get(
            PaperRow,
            conversation_row.primary_paper_id,
        )
        if primary_paper_row is None:
            raise RuntimeError("conversation primary paper is missing")

        paper_rows = session.execute(
            select(ConversationPaperRow, PaperRow)
            .join(PaperRow, PaperRow.id == ConversationPaperRow.paper_id)
            .where(
                ConversationPaperRow.conversation_id == conversation_id,
                ConversationPaperRow.is_active.is_(True),
            )
            .order_by(
                ConversationPaperRow.created_at.asc(),
                ConversationPaperRow.paper_id.asc(),
            )
        ).all()
        active_task_row = session.scalar(
            select(ResearchTaskRow).where(
                ResearchTaskRow.conversation_id == conversation_id,
                ResearchTaskRow.user_id == user_id,
                ResearchTaskRow.status.in_(("pending", "running")),
            )
        )

        return ConversationDetail(
            conversation=conversation_from_row(conversation_row),
            primary_paper=paper_from_row(primary_paper_row),
            active_papers=[paper_from_row(row[1]) for row in paper_rows],
            paper_associations=[
                conversation_paper_from_row(row[0]) for row in paper_rows
            ],
            active_task=(
                task_from_row(active_task_row)
                if active_task_row is not None
                else None
            ),
        )

def update_conversation(session_factory: SessionFactory,
    conversation_id: str,
    *,
    user_id: str,
    title: str | None = None,
    archived: bool | None = None,
) -> ConversationRecord | None:
    with session_factory.begin() as session:
        row = select_owned_conversation(
            session,
            conversation_id,
            user_id,
        )
        if row is None:
            return None

        cleaned_title = (
            _validate_conversation_title(title) if title is not None else None
        )
        if archived:
            active_task_id = session.scalar(
                select(ResearchTaskRow.id).where(
                    ResearchTaskRow.conversation_id == conversation_id,
                    ResearchTaskRow.user_id == user_id,
                    ResearchTaskRow.status.in_(("pending", "running")),
                )
            )
            if active_task_id is not None:
                raise ValueError("cannot archive conversation with an active task")

        changed = False
        if cleaned_title is not None:
            row.title = cleaned_title
            changed = True
        if archived is not None:
            row.archived_at = utc_now() if archived else None
            changed = True
        if changed:
            row.updated_at = utc_now()
        session.flush()
        return conversation_from_row(row)

def _validate_conversation_title(title: str) -> str:
    cleaned = title.strip()
    if not 1 <= len(cleaned) <= 200:
        raise ValueError("conversation title must be between 1 and 200 characters")
    return cleaned

