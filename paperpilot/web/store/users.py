"""User and login-session persistence.

This module owns UserRow and LoginSessionRow reads/writes, including username uniqueness and session expiry. It does not own Conversation, Task, or HTTP behavior. Each public operation uses a short session transaction; duplicate usernames are translated at this boundary."""

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

def create_user(session_factory: SessionFactory,
    *,
    username: str,
    password_hash: str,
    password_salt: str,
) -> WebUser:
    """Create a user atomically and translate the unique-username race."""
    username = username.strip()
    if not username:
        raise ValueError("username is required")
    row = UserRow(
        id=f"user_{uuid.uuid4().hex}",
        username=username,
        password_hash=password_hash,
        password_salt=password_salt,
        created_at=utc_now(),
    )
    try:
        with session_factory.begin() as session:
            session.add(row)
    except IntegrityError as exc:
        raise DuplicateUsernameError("username already exists") from exc
    return user_from_row(row)

def get_user_by_username(session_factory: SessionFactory, username: str) -> WebUser | None:
    """Read a user by username without exposing SQLAlchemy rows."""
    with session_factory() as session:
        row = session.scalar(
            select(UserRow).where(UserRow.username == username)
        )
    return user_from_row(row) if row is not None else None

def get_user_by_id(session_factory: SessionFactory, user_id: str) -> WebUser | None:
    """Read a user by stable identifier."""
    with session_factory() as session:
        row = session.get(UserRow, user_id)
    return user_from_row(row) if row is not None else None

def create_session(session_factory: SessionFactory, user_id: str) -> str:
    """Create a seven-day session only for an existing user."""
    if get_user_by_id(session_factory, user_id) is None:
        raise ValueError(f"user not found: {user_id}")
    token = f"session_{secrets.token_urlsafe(32)}"
    created_at = utc_now()
    expires_at = utc_in(days=7)
    with session_factory.begin() as session:
        session.add(
            LoginSessionRow(
                token=token,
                user_id=user_id,
                created_at=created_at,
                expires_at=expires_at,
            )
        )
    return token

def get_user_for_session(session_factory: SessionFactory, token: str) -> WebUser | None:
    """Resolve a non-expired login token to its owner."""
    now = utc_now()
    statement = (
        select(UserRow)
        .join(LoginSessionRow, LoginSessionRow.user_id == UserRow.id)
        .where(
            LoginSessionRow.token == token,
            LoginSessionRow.expires_at > now,
        )
    )
    with session_factory() as session:
        row = session.scalar(statement)
    return user_from_row(row) if row is not None else None

def delete_session(session_factory: SessionFactory, token: str) -> None:
    """Delete a login session idempotently."""
    with session_factory.begin() as session:
        session.execute(
            delete(LoginSessionRow).where(LoginSessionRow.token == token)
        )
