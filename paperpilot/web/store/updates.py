"""Task Event, Artifact, and incremental update persistence.

This module owns TaskEventRow and TaskArtifactRow writes and cursor-based reads. It does not own Task state transitions or HTTP response formatting. The conversation-scoped updates operation keeps Task, Event, and Artifact reads in one session transaction so all three watermarks observe one database snapshot."""

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

def add_event(session_factory: SessionFactory,
    *,
    task_id: str,
    type: str,
    message: str,
    stage: str | None = None,
    payload: dict | None = None,
) -> TaskEvent:
    if not message.strip():
        raise ValueError("event message is required")
    row = TaskEventRow(
        task_id=task_id,
        type=type,
        stage=stage,
        message=message.strip(),
        payload_json=json.dumps(payload or {}, ensure_ascii=False),
        created_at=utc_now(),
    )
    with session_factory.begin() as session:
        if session.get(ResearchTaskRow, task_id) is None:
            raise ValueError(f"task not found: {task_id}")
        session.add(row)
        session.flush()
    return event_from_row(row)

def list_events_page(session_factory: SessionFactory,
    task_id: str,
    *,
    user_id: str | None,
    after_id: int,
    limit: int,
) -> TaskEventBatch | None:
    _validate_incremental_page(after_id, limit)
    with session_factory() as session:
        if select_owned_task(session, task_id, user_id) is None:
            return None
        return _read_event_batch(
            session,
            task_id,
            after_id=after_id,
            limit=limit,
        )

def add_artifact(session_factory: SessionFactory,
    *,
    task_id: str,
    kind: str,
    title: str,
    content: str,
    payload: dict | None = None,
) -> TaskArtifact:
    if not kind.strip():
        raise ValueError("artifact kind is required")
    if not title.strip():
        raise ValueError("artifact title is required")
    if not content.strip():
        raise ValueError("artifact content is required")
    row = TaskArtifactRow(
        task_id=task_id,
        kind=kind.strip(),
        title=title.strip(),
        content=content.strip(),
        payload_json=json.dumps(payload or {}, ensure_ascii=False),
        created_at=utc_now(),
    )
    with session_factory.begin() as session:
        if session.get(ResearchTaskRow, task_id) is None:
            raise ValueError(f"task not found: {task_id}")
        session.add(row)
        session.flush()
    return artifact_from_row(row)

def list_artifacts_page(session_factory: SessionFactory,
    task_id: str,
    *,
    user_id: str | None,
    after_id: int,
    limit: int,
) -> TaskArtifactBatch | None:
    _validate_incremental_page(after_id, limit)
    with session_factory() as session:
        if select_owned_task(session, task_id, user_id) is None:
            return None
        return _read_artifact_batch(
            session,
            task_id,
            after_id=after_id,
            limit=limit,
        )

def get_conversation_task_updates(session_factory: SessionFactory,
    conversation_id: str,
    task_id: str,
    *,
    user_id: str,
    after_event_id: int = 0,
    after_artifact_id: int = 0,
    limit: int = 50,
) -> TaskUpdates | None:
    _validate_incremental_page(after_event_id, limit)
    _validate_incremental_page(after_artifact_id, limit)
    with session_factory.begin() as session:
        row = select_owned_task(session, task_id, user_id)
        if row is None or row.conversation_id != conversation_id:
            return None
        return TaskUpdates(
            task=task_from_row(row),
            events=_read_event_batch(
                session,
                task_id,
                after_id=after_event_id,
                limit=limit,
            ),
            artifacts=_read_artifact_batch(
                session,
                task_id,
                after_id=after_artifact_id,
                limit=limit,
            ),
        )

def _read_event_batch(
    session: Session,
    task_id: str,
    *,
    after_id: int,
    limit: int,
) -> TaskEventBatch:
    rows = list(
        session.scalars(
            select(TaskEventRow)
            .where(
                TaskEventRow.task_id == task_id,
                TaskEventRow.id > after_id,
            )
            .order_by(TaskEventRow.id.asc())
            .limit(limit + 1)
        )
    )
    items = [event_from_row(row) for row in rows[:limit]]
    return TaskEventBatch(
        items=items,
        next_after_id=items[-1].id if items else after_id,
        has_more=len(rows) > limit,
    )

def _read_artifact_batch(
    session: Session,
    task_id: str,
    *,
    after_id: int,
    limit: int,
) -> TaskArtifactBatch:
    rows = list(
        session.scalars(
            select(TaskArtifactRow)
            .where(
                TaskArtifactRow.task_id == task_id,
                TaskArtifactRow.id > after_id,
            )
            .order_by(TaskArtifactRow.id.asc())
            .limit(limit + 1)
        )
    )
    items = [artifact_from_row(row) for row in rows[:limit]]
    return TaskArtifactBatch(
        items=items,
        next_after_id=items[-1].id if items else after_id,
        has_more=len(rows) > limit,
    )

def _validate_incremental_page(after_id: int, limit: int) -> None:
    if after_id < 0:
        raise ValueError("after_id must be non-negative")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")

