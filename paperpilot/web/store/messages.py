"""Conversation message-tree persistence and user-turn creation.

This module owns MessageRow reads, active-path traversal, alternatives, unstable turns, and the atomic user-message plus pending-task plus queued-event transaction. It does not reserve executor capacity or publish/finalize results. Stable heads remain unchanged until finalization."""

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

def create_conversation_turn(session_factory: SessionFactory,
    *,
    user_id: str,
    conversation_id: str,
    content: str,
    depth: str,
    expected_head_message_id: str | None,
) -> ConversationTurn:
    task = _new_task(
        question=content,
        depth=depth,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    now = task.created_at
    user_message_row = MessageRow(
        id=f"msg_{uuid.uuid4().hex}",
        conversation_id=conversation_id,
        task_id=task.id,
        parent_message_id=expected_head_message_id,
        role="user",
        content=task.question,
        status="complete",
        metadata_json="{}",
        created_at=now,
    )
    head_matches = (
        ConversationRow.head_message_id.is_(None)
        if expected_head_message_id is None
        else ConversationRow.head_message_id == expected_head_message_id
    )
    active_task_exists = exists(
        select(ResearchTaskRow.id).where(
            ResearchTaskRow.conversation_id == conversation_id,
            ResearchTaskRow.status.in_(("pending", "running")),
        )
    )
    eligible_conversation = (
        select(
            literal(task.id),
            literal(task.question),
            literal(task.depth),
            literal(task.status),
            literal(task.created_at),
            literal(task.updated_at),
            literal(task.user_id),
            ConversationRow.id,
            ConversationRow.head_checkpoint_id,
            literal(None),
            literal(None),
        )
        .select_from(ConversationRow)
        .where(
            ConversationRow.id == conversation_id,
            ConversationRow.user_id == user_id,
            ConversationRow.archived_at.is_(None),
            head_matches,
            ~active_task_exists,
        )
    )
    task_insert = insert(ResearchTaskRow).from_select(
        [
            ResearchTaskRow.id,
            ResearchTaskRow.question,
            ResearchTaskRow.depth,
            ResearchTaskRow.status,
            ResearchTaskRow.created_at,
            ResearchTaskRow.updated_at,
            ResearchTaskRow.user_id,
            ResearchTaskRow.conversation_id,
            ResearchTaskRow.base_checkpoint_id,
            ResearchTaskRow.final_checkpoint_id,
            ResearchTaskRow.result_quality,
        ],
        eligible_conversation,
    )

    try:
        with session_factory.begin() as session:
            result = session.execute(task_insert)
            if result.rowcount != 1:
                _raise_conversation_turn_conflict(
                    session,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    expected_head_message_id=expected_head_message_id,
                )
            task_row = session.get(ResearchTaskRow, task.id)
            if task_row is None:
                raise RuntimeError("conversation task disappeared within transaction")
            task = task_from_row(task_row)
            user_message_row.parent_message_id = expected_head_message_id
            session.add(user_message_row)
            session.flush()
            session.add(
                TaskEventRow(
                    task_id=task.id,
                    type="queued",
                    stage="queue",
                    message="Task queued for deep reading.",
                    payload_json=json.dumps(
                        {"depth": task.depth},
                        ensure_ascii=False,
                    ),
                    created_at=utc_now(),
                )
            )
            session.flush()
            return ConversationTurn(
                user_message=message_from_row(user_message_row),
                task=task,
            )
    except IntegrityError as exc:
        if _is_active_task_unique_conflict(exc):
            raise ConversationBusyError(
                "conversation already has an active task"
            ) from exc
        raise

def get_message(session_factory: SessionFactory,
    conversation_id: str,
    message_id: str,
    *,
    user_id: str,
) -> MessageRecord | None:
    with session_factory() as session:
        row = session.scalar(
            select(MessageRow)
            .join(
                ConversationRow,
                ConversationRow.id == MessageRow.conversation_id,
            )
            .where(
                MessageRow.id == message_id,
                MessageRow.conversation_id == conversation_id,
                ConversationRow.user_id == user_id,
            )
        )
    return message_from_row(row) if row is not None else None

def get_task_message(session_factory: SessionFactory,
    task_id: str,
    role: str,
) -> MessageRecord | None:
    if role not in {"user", "assistant", "system"}:
        raise ValueError(f"invalid message role: {role!r}")
    with session_factory() as session:
        row = session.scalar(
            select(MessageRow).where(
                MessageRow.task_id == task_id,
                MessageRow.role == role,
            )
        )
    return message_from_row(row) if row is not None else None

def list_active_messages(session_factory: SessionFactory,
    conversation_id: str,
    *,
    user_id: str,
) -> list[MessageRecord] | None:
    with session_factory() as session:
        conversation_row = select_owned_conversation(
            session,
            conversation_id,
            user_id,
        )
        if conversation_row is None:
            return None
        if conversation_row.head_message_id is None:
            return []

        rows = list(
            session.scalars(
                select(MessageRow).where(
                    MessageRow.conversation_id == conversation_id
                )
            )
        )
        rows_by_id = {row.id: row for row in rows}
        head_row = rows_by_id.get(conversation_row.head_message_id)
        if head_row is None:
            _raise_broken_message_reference(
                session,
                conversation_id=conversation_id,
                message_id=conversation_row.head_message_id,
                relation="head",
            )
        assert head_row is not None
        if head_row.role != "assistant" or head_row.status != "complete":
            raise RuntimeError(
                "conversation head is not a complete assistant message: "
                f"{head_row.id}"
            )

        path: list[MessageRow] = []
        seen: set[str] = set()
        current: MessageRow | None = head_row
        while current is not None:
            if current.id in seen:
                raise RuntimeError(
                    f"message tree cycle detected at message: {current.id}"
                )
            seen.add(current.id)
            path.append(current)
            parent_id = current.parent_message_id
            if parent_id is None:
                break
            current = rows_by_id.get(parent_id)
            if current is None:
                _raise_broken_message_reference(
                    session,
                    conversation_id=conversation_id,
                    message_id=parent_id,
                    relation="parent",
                )

        path.reverse()
        return [message_from_row(row) for row in path]

def list_message_alternatives(session_factory: SessionFactory,
    conversation_id: str,
    message_id: str,
    *,
    user_id: str,
) -> list[ConversationAlternative] | None:
    with session_factory() as session:
        if (
            select_owned_conversation(
                session,
                conversation_id,
                user_id,
            )
            is None
        ):
            return None
        branch_point = session.scalar(
            select(MessageRow).where(
                MessageRow.id == message_id,
                MessageRow.conversation_id == conversation_id,
            )
        )
        if branch_point is None:
            return None
        if branch_point.role != "assistant" or branch_point.status != "complete":
            raise ValueError("alternatives require a complete assistant message")

        user_rows = list(
            session.scalars(
                select(MessageRow)
                .where(
                    MessageRow.conversation_id == conversation_id,
                    MessageRow.parent_message_id == message_id,
                    MessageRow.role == "user",
                    MessageRow.status == "complete",
                )
                .order_by(MessageRow.created_at.asc(), MessageRow.id.asc())
            )
        )
        alternatives: list[ConversationAlternative] = []
        for user_row in user_rows:
            assistant_row = session.scalar(
                select(MessageRow).where(
                    MessageRow.conversation_id == conversation_id,
                    MessageRow.parent_message_id == user_row.id,
                    MessageRow.task_id == user_row.task_id,
                    MessageRow.role == "assistant",
                    MessageRow.status == "complete",
                )
            )
            if assistant_row is not None:
                alternatives.append(
                    ConversationAlternative(
                        user_message=message_from_row(user_row),
                        assistant_message=message_from_row(assistant_row),
                    )
                )
        return alternatives

def get_unstable_turn(session_factory: SessionFactory,
    conversation_id: str,
    *,
    user_id: str,
) -> ConversationTurn | None:
    with session_factory() as session:
        conversation_row = select_owned_conversation(
            session,
            conversation_id,
            user_id,
        )
        if conversation_row is None:
            return None
        parent_matches = (
            MessageRow.parent_message_id.is_(None)
            if conversation_row.head_message_id is None
            else MessageRow.parent_message_id == conversation_row.head_message_id
        )
        ownership_and_parent_filters = (
            MessageRow.conversation_id == conversation_id,
            MessageRow.role == "user",
            parent_matches,
            ResearchTaskRow.conversation_id == conversation_id,
            ResearchTaskRow.user_id == user_id,
        )
        result = session.execute(
            select(MessageRow, ResearchTaskRow)
            .join(ResearchTaskRow, ResearchTaskRow.id == MessageRow.task_id)
            .where(
                *ownership_and_parent_filters,
                ResearchTaskRow.status.in_(("pending", "running")),
            )
            .limit(1)
        ).first()
        if result is None:
            queued_order = (
                select(
                    TaskEventRow.task_id.label("task_id"),
                    func.max(TaskEventRow.id).label("queued_event_id"),
                )
                .where(TaskEventRow.type == "queued")
                .group_by(TaskEventRow.task_id)
                .subquery()
            )
            result = session.execute(
                select(MessageRow, ResearchTaskRow)
                .join(ResearchTaskRow, ResearchTaskRow.id == MessageRow.task_id)
                .join(queued_order, queued_order.c.task_id == ResearchTaskRow.id)
                .where(
                    *ownership_and_parent_filters,
                    ResearchTaskRow.status == "failed",
                )
                .order_by(queued_order.c.queued_event_id.desc())
                .limit(1)
            ).first()
        if result is None:
            return None
        message_row, task_row = result
        return ConversationTurn(
            user_message=message_from_row(message_row),
            task=task_from_row(task_row),
        )

def _new_task(
    *,
    question: str,
    depth: str,
    user_id: str | None,
    conversation_id: str | None = None,
    base_checkpoint_id: str | None = None,
) -> ResearchTask:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("question is required")
    if depth not in VALID_DEPTHS:
        raise ValueError(f"invalid depth: {depth!r}")
    now = utc_now()
    return ResearchTask(
        id=f"task_{uuid.uuid4().hex}",
        question=cleaned_question,
        depth=depth,
        status="pending",
        created_at=now,
        updated_at=now,
        user_id=user_id,
        conversation_id=conversation_id,
        base_checkpoint_id=base_checkpoint_id,
    )

def _raise_conversation_turn_conflict(
    session: Session,
    *,
    conversation_id: str,
    user_id: str,
    expected_head_message_id: str | None,
) -> None:
    conversation_row = select_owned_conversation(
        session,
        conversation_id,
        user_id,
    )
    if conversation_row is None:
        raise ValueError(f"conversation not found: {conversation_id}")
    if conversation_row.archived_at is not None:
        raise ValueError("conversation is archived")
    active_task_id = session.scalar(
        select(ResearchTaskRow.id).where(
            ResearchTaskRow.conversation_id == conversation_id,
            ResearchTaskRow.status.in_(("pending", "running")),
        )
    )
    if active_task_id is not None:
        raise ConversationBusyError("conversation already has an active task")
    if conversation_row.head_message_id != expected_head_message_id:
        raise StaleConversationHeadError("conversation head has changed")
    raise ConversationBusyError("conversation turn admission lost a concurrent race")

def _is_active_task_unique_conflict(exc: IntegrityError) -> bool:
    message = str(exc.orig).lower()
    return (
        "unique constraint failed" in message
        and "research_tasks.conversation_id" in message
    )

def _raise_broken_message_reference(
    session: Session,
    *,
    conversation_id: str,
    message_id: str,
    relation: str,
) -> None:
    referenced_row = session.get(MessageRow, message_id)
    if referenced_row is None:
        raise RuntimeError(
            f"message tree has a missing {relation}: {message_id}"
        )
    if referenced_row.conversation_id != conversation_id:
        raise RuntimeError(
            f"message tree {relation} belongs to another conversation: {message_id}"
        )
    raise RuntimeError(f"message tree could not resolve {relation}: {message_id}")

