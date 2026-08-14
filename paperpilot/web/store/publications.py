"""Conversation answer publication and terminal state transitions.

This module owns the Task result Assistant Message, result Artifact, completion/failure terminal states, stable message/checkpoint heads, and rollback head switches. It does not own HTTP validation or checkpoint storage. Publication, finalization, failure, and rollback each retain their existing single-transaction and idempotency/concurrency boundaries."""

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

def publish_conversation_result(session_factory: SessionFactory,
    *,
    task_id: str,
    content: str,
    metadata: dict,
    used_papers: list[UsedPaperInput],
) -> PublishedConversationResult:
    """Persist one immutable answer for a Task without moving the stable head."""
    cleaned_content = content.strip()
    if not cleaned_content:
        raise ValueError("assistant content is required")
    if not isinstance(metadata, dict):
        raise ValueError("assistant metadata must be a dictionary")
    valid_roles = {"comparison", "citation", "background", "follow_up"}
    if any(item.role not in valid_roles for item in used_papers):
        raise ValueError("invalid used paper role")

    metadata_json = json.dumps(metadata, ensure_ascii=False)
    now = utc_now()
    with session_factory.begin() as session:
        # SQLite has no row-level SELECT FOR UPDATE. A no-op write acquires
        # the database write reservation before any idempotency reads, so
        # concurrent redeliveries of the same Task cannot both observe a
        # missing Assistant/Artifact and publish duplicates.
        reservation = session.execute(
            update(ResearchTaskRow)
            .where(ResearchTaskRow.id == task_id)
            .values(updated_at=ResearchTaskRow.updated_at)
        )
        if reservation.rowcount != 1:
            raise ValueError(f"task not found: {task_id}")
        task_row = session.get(ResearchTaskRow, task_id)
        if task_row is None:
            raise RuntimeError("reserved task disappeared within transaction")
        if task_row.conversation_id is None or task_row.user_id is None:
            raise ValueError("task is not attached to a conversation")
        conversation_row = session.get(
            ConversationRow,
            task_row.conversation_id,
        )
        if conversation_row is None:
            raise RuntimeError("conversation task references a missing conversation")
        if conversation_row.user_id != task_row.user_id:
            raise ValueError("task owner does not match conversation owner")
        if conversation_row.archived_at is not None:
            raise ValueError("conversation is archived")

        user_message_row = session.scalar(
            select(MessageRow).where(
                MessageRow.task_id == task_id,
                MessageRow.role == "user",
            )
        )
        if user_message_row is None:
            raise ValueError("conversation task has no user message")
        if user_message_row.conversation_id != conversation_row.id:
            raise ValueError("user message belongs to another conversation")

        assistant_row = session.scalar(
            select(MessageRow).where(
                MessageRow.task_id == task_id,
                MessageRow.role == "assistant",
            )
        )
        if assistant_row is None:
            if task_row.status not in {"pending", "running"}:
                raise ValueError(
                    f"cannot publish result for {task_row.status} task"
                )
            assistant_row = MessageRow(
                id=f"msg_{uuid.uuid4().hex}",
                conversation_id=conversation_row.id,
                task_id=task_id,
                parent_message_id=user_message_row.id,
                role="assistant",
                content=cleaned_content,
                status="complete",
                metadata_json=metadata_json,
                created_at=now,
            )
            session.add(assistant_row)
            session.flush()
        else:
            if assistant_row.conversation_id != conversation_row.id:
                raise ValueError("assistant message belongs to another conversation")
            if assistant_row.parent_message_id != user_message_row.id:
                raise ValueError("assistant message has an invalid parent")
            if assistant_row.status != "complete":
                raise ValueError("assistant message is not complete")

        artifact_row = session.scalar(
            select(TaskArtifactRow)
            .where(
                TaskArtifactRow.task_id == task_id,
                TaskArtifactRow.kind == "result",
            )
            .order_by(TaskArtifactRow.id.asc())
            .limit(1)
        )
        if artifact_row is None:
            artifact_row = TaskArtifactRow(
                task_id=task_id,
                kind="result",
                title="Conversation result",
                content=cleaned_content,
                payload_json=metadata_json,
                created_at=now,
            )
            session.add(artifact_row)
            session.flush()

        active_rows = list(
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
        active_paper_ids = [conversation_row.primary_paper_id]
        active_paper_ids.extend(
            row.paper_id
            for row in active_rows
            if row.paper_id != conversation_row.primary_paper_id
        )

        seen_external_keys: set[tuple[str, str]] = set()
        for used in used_papers:
            paper = used.paper
            source = paper.source.strip()
            external_id = paper.external_id.strip()
            key = (source, external_id)
            if key in seen_external_keys:
                continue
            seen_external_keys.add(key)
            paper_row = session.scalar(
                sqlite_insert(PaperRow)
                .values(
                    id=f"paper_{uuid.uuid4().hex}",
                    source=source,
                    external_id=external_id,
                    title=paper.title.strip(),
                    authors_json=json.dumps(paper.authors, ensure_ascii=False),
                    abstract=paper.abstract,
                    source_url=paper.source_url.strip(),
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=[PaperRow.source, PaperRow.external_id],
                    set_={
                        "title": paper.title.strip(),
                        "authors_json": json.dumps(
                            paper.authors,
                            ensure_ascii=False,
                        ),
                        "abstract": paper.abstract,
                        "source_url": paper.source_url.strip(),
                        "updated_at": now,
                    },
                )
                .returning(PaperRow)
            )
            if paper_row is None:
                raise RuntimeError("used paper upsert did not return a row")
            if paper_row.id != conversation_row.primary_paper_id:
                session.execute(
                    sqlite_insert(ConversationPaperRow)
                    .values(
                        conversation_id=conversation_row.id,
                        paper_id=paper_row.id,
                        role=used.role,
                        added_by="agent",
                        source_task_id=task_id,
                        source_message_id=assistant_row.id,
                        is_active=False,
                        created_at=now,
                    )
                    .on_conflict_do_update(
                        index_elements=[
                            ConversationPaperRow.conversation_id,
                            ConversationPaperRow.paper_id,
                        ],
                        set_={
                            "role": used.role,
                            "added_by": "agent",
                        },
                    )
                )
            if paper_row.id not in active_paper_ids:
                active_paper_ids.append(paper_row.id)

        return PublishedConversationResult(
            message=message_from_row(assistant_row),
            artifact=artifact_from_row(artifact_row),
            active_paper_ids=active_paper_ids,
        )

def finalize_conversation_task(session_factory: SessionFactory,
    *,
    task_id: str,
    assistant_message_id: str,
    final_checkpoint_id: str,
    result_quality: str,
    active_paper_ids: list[str],
) -> FinalizedConversationTask:
    """Atomically make a published result the new stable conversation head."""
    checkpoint_id = final_checkpoint_id.strip()
    if not checkpoint_id:
        raise ValueError("final checkpoint id is required")
    if result_quality not in {"complete", "partial"}:
        raise ValueError(f"invalid result quality: {result_quality!r}")

    with session_factory.begin() as session:
        reservation = session.execute(
            update(ResearchTaskRow)
            .where(ResearchTaskRow.id == task_id)
            .values(updated_at=ResearchTaskRow.updated_at)
        )
        if reservation.rowcount != 1:
            raise ValueError(f"task not found: {task_id}")
        task_row = session.get(ResearchTaskRow, task_id)
        if task_row is None:
            raise RuntimeError("reserved task disappeared within transaction")
        if task_row.conversation_id is None:
            raise ValueError("task is not attached to a conversation")
        conversation_row = session.get(ConversationRow, task_row.conversation_id)
        if conversation_row is None:
            raise RuntimeError("conversation task references a missing conversation")
        if task_row.user_id != conversation_row.user_id:
            raise ValueError("task owner does not match conversation owner")
        assistant_row, user_message_row = _select_task_result_chain(
            session,
            task_row=task_row,
            assistant_message_id=assistant_message_id,
        )
        normalized_active_ids = validate_active_paper_ids(
            session,
            conversation_row=conversation_row,
            active_paper_ids=active_paper_ids,
        )

        if task_row.status == "completed":
            if (
                task_row.final_checkpoint_id != checkpoint_id
                or task_row.result_quality != result_quality
                or conversation_row.head_message_id != assistant_row.id
                or conversation_row.head_checkpoint_id != checkpoint_id
                or read_active_paper_ids(session, conversation_row)
                != normalized_active_ids
            ):
                raise ValueError("completed task finalization does not match stored result")
            return FinalizedConversationTask(
                task=task_from_row(task_row),
                conversation=conversation_from_row(conversation_row),
            )
        if task_row.status not in {"pending", "running"}:
            raise ValueError(f"cannot finalize {task_row.status} task")
        if conversation_row.archived_at is not None:
            raise ValueError("conversation is archived")
        if conversation_row.head_checkpoint_id != task_row.base_checkpoint_id:
            raise StaleConversationHeadError(
                "conversation checkpoint head has changed"
            )
        if user_message_row.parent_message_id != conversation_row.head_message_id:
            raise StaleConversationHeadError("conversation message head has changed")

        now = utc_now()
        task_row.status = "completed"
        task_row.final_checkpoint_id = checkpoint_id
        task_row.result_quality = result_quality
        task_row.updated_at = now
        set_active_paper_ids(
            session,
            conversation_row=conversation_row,
            active_paper_ids=normalized_active_ids,
        )
        conversation_row.head_message_id = assistant_row.id
        conversation_row.head_checkpoint_id = checkpoint_id
        conversation_row.updated_at = now
        session.add(
            TaskEventRow(
                task_id=task_id,
                type="completed",
                stage="finalize",
                message="Conversation result finalized.",
                payload_json=json.dumps(
                    {
                        "assistant_message_id": assistant_row.id,
                        "final_checkpoint_id": checkpoint_id,
                        "result_quality": result_quality,
                    },
                    ensure_ascii=False,
                ),
                created_at=now,
            )
        )
        session.flush()
        return FinalizedConversationTask(
            task=task_from_row(task_row),
            conversation=conversation_from_row(conversation_row),
        )

def fail_conversation_task(session_factory: SessionFactory,
    *,
    task_id: str,
    message: str,
    stage: str | None = None,
    payload: dict | None = None,
) -> ResearchTask | None:
    """Fail active conversation work once, preserving a first failure event."""
    cleaned_message = message.strip()
    if not cleaned_message:
        raise ValueError("failure message is required")
    with session_factory.begin() as session:
        reservation = session.execute(
            update(ResearchTaskRow)
            .where(ResearchTaskRow.id == task_id)
            .values(updated_at=ResearchTaskRow.updated_at)
        )
        if reservation.rowcount != 1:
            return None
        task_row = session.get(ResearchTaskRow, task_id)
        if task_row is None:
            raise RuntimeError("reserved task disappeared within transaction")
        if task_row.conversation_id is None:
            raise ValueError("task is not attached to a conversation")
        if task_row.status == "completed":
            raise ValueError("cannot fail completed task")
        if task_row.status not in {"pending", "running", "failed"}:
            raise ValueError(f"cannot fail {task_row.status} task")

        failure_event = session.scalar(
            select(TaskEventRow)
            .where(
                TaskEventRow.task_id == task_id,
                TaskEventRow.type == "failed",
            )
            .order_by(TaskEventRow.id.asc())
            .limit(1)
        )
        if task_row.status != "failed":
            task_row.status = "failed"
            task_row.updated_at = utc_now()
        if failure_event is None:
            session.add(
                TaskEventRow(
                    task_id=task_id,
                    type="failed",
                    stage=stage,
                    message=cleaned_message,
                    payload_json=json.dumps(payload or {}, ensure_ascii=False),
                    created_at=utc_now(),
                )
            )
        session.flush()
        return task_from_row(task_row)

def switch_conversation_head(session_factory: SessionFactory,
    conversation_id: str,
    *,
    user_id: str,
    expected_head_message_id: str | None,
    target_message_id: str,
    target_checkpoint_id: str,
    active_paper_ids: list[str],
) -> ConversationRecord | None:
    """Switch the stable business head and its active papers in one transaction."""
    checkpoint_id = target_checkpoint_id.strip()
    if not checkpoint_id:
        raise ValueError("target checkpoint id is required")
    with session_factory.begin() as session:
        reservation = session.execute(
            update(ConversationRow)
            .where(
                ConversationRow.id == conversation_id,
                ConversationRow.user_id == user_id,
            )
            .values(updated_at=ConversationRow.updated_at)
        )
        if reservation.rowcount != 1:
            return None
        conversation_row = session.get(ConversationRow, conversation_id)
        if conversation_row is None:
            raise RuntimeError(
                "reserved conversation disappeared within transaction"
            )
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

        target_row = session.scalar(
            select(MessageRow).where(
                MessageRow.id == target_message_id,
                MessageRow.conversation_id == conversation_id,
            )
        )
        if target_row is None:
            raise ValueError("target message not found in conversation")
        if target_row.role != "assistant" or target_row.status != "complete":
            raise ValueError("target must be a complete assistant message")
        normalized_active_ids = validate_active_paper_ids(
            session,
            conversation_row=conversation_row,
            active_paper_ids=active_paper_ids,
        )

        set_active_paper_ids(
            session,
            conversation_row=conversation_row,
            active_paper_ids=normalized_active_ids,
        )
        conversation_row.head_message_id = target_row.id
        conversation_row.head_checkpoint_id = checkpoint_id
        conversation_row.updated_at = utc_now()
        session.flush()
        return conversation_from_row(conversation_row)

def _select_task_result_chain(
    session: Session,
    *,
    task_row: ResearchTaskRow,
    assistant_message_id: str,
) -> tuple[MessageRow, MessageRow]:
    assistant_row = session.get(MessageRow, assistant_message_id)
    if assistant_row is None:
        raise ValueError("assistant message not found")
    if (
        assistant_row.task_id != task_row.id
        or assistant_row.conversation_id != task_row.conversation_id
        or assistant_row.role != "assistant"
        or assistant_row.status != "complete"
    ):
        raise ValueError("message is not this task's complete assistant result")
    user_message_row = session.scalar(
        select(MessageRow).where(
            MessageRow.task_id == task_row.id,
            MessageRow.role == "user",
        )
    )
    if user_message_row is None:
        raise ValueError("conversation task has no user message")
    if (
        user_message_row.conversation_id != task_row.conversation_id
        or user_message_row.status != "complete"
    ):
        raise ValueError("task user message is not a complete local message")
    if assistant_row.parent_message_id != user_message_row.id:
        raise ValueError("assistant message parent does not match task user message")
    return assistant_row, user_message_row

