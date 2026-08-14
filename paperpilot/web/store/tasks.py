"""Task lookup, claiming, submission failure, and database health.

This module owns ResearchTaskRow lifecycle operations that do not advance a Conversation head. It does not own result publication or HTTP behavior. Claim and failure paths preserve the existing compare-and-update and redelivery semantics; health checks use the Engine directly."""

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

def check_health(engine: Engine) -> None:
    """Run the lightweight database health probe on the owned Engine."""
    with engine.connect() as connection:
        row = connection.execute(text("SELECT 1")).one_or_none()
    if row is None or int(row[0]) != 1:
        raise RuntimeError("SQLite health probe returned an invalid result")

def get_task(session_factory: SessionFactory,
    task_id: str,
    *,
    user_id: str | None = None,
) -> ResearchTask | None:
    """Read one Task, optionally constrained to its owner."""
    filters = [ResearchTaskRow.id == task_id]
    if user_id is not None:
        filters.append(ResearchTaskRow.user_id == user_id)
    with session_factory() as session:
        row = session.scalar(select(ResearchTaskRow).where(*filters))
    return task_from_row(row) if row is not None else None

def claim_task(session_factory: SessionFactory,
    task_id: str,
    *,
    allow_running: bool = False,
) -> ResearchTask | None:
    """Atomically claim pending work or recover a redelivered running task."""
    claimable_statuses = ("pending", "running") if allow_running else ("pending",)
    updated_at = utc_now()
    with session_factory.begin() as session:
        result = session.execute(
            update(ResearchTaskRow)
            .where(
                ResearchTaskRow.id == task_id,
                ResearchTaskRow.status.in_(claimable_statuses),
            )
            .values(status="running", updated_at=updated_at)
        )
        if result.rowcount != 1:
            return None
        row = session.get(ResearchTaskRow, task_id)
        if row is None:
            raise RuntimeError("claimed task disappeared within transaction")
        return task_from_row(row)

def fail_pending_task(session_factory: SessionFactory, task_id: str) -> ResearchTask | None:
    """Mark an unclaimed task failed without overwriting active work."""
    with session_factory.begin() as session:
        result = session.execute(
            update(ResearchTaskRow)
            .where(
                ResearchTaskRow.id == task_id,
                ResearchTaskRow.status == "pending",
            )
            .values(status="failed", updated_at=utc_now())
        )
        if result.rowcount != 1:
            return None
        row = session.get(ResearchTaskRow, task_id)
        if row is None:
            raise RuntimeError("failed task disappeared within transaction")
        return task_from_row(row)
