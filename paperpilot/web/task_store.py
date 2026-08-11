"""SQLite-backed research task storage for the Web workbench."""
from __future__ import annotations

import json
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from sqlalchemy import (
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
from paperpilot.web.database import (
    DEFAULT_TASK_DB_PATH,
    create_session_factory,
    create_task_engine,
    resolve_task_db_path,
)
from paperpilot.web.db_migrations import ensure_database_current
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


class TaskStore:
    """Persist local research tasks in SQLite."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = resolve_task_db_path(db_path)
        ensure_database_current(self.db_path)
        self.engine = create_task_engine(self.db_path)
        self._session_factory = create_session_factory(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    def create_user(
        self,
        *,
        username: str,
        password_hash: str,
        password_salt: str,
    ) -> WebUser:
        username = username.strip()
        if not username:
            raise ValueError("username is required")
        row = UserRow(
            id=f"user_{uuid.uuid4().hex}",
            username=username,
            password_hash=password_hash,
            password_salt=password_salt,
            created_at=_utc_now(),
        )
        try:
            with self._session_factory.begin() as session:
                session.add(row)
        except IntegrityError as exc:
            raise DuplicateUsernameError("username already exists") from exc
        return _user_from_model(row)

    def get_user_by_username(self, username: str) -> WebUser | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(UserRow).where(UserRow.username == username)
            )
        return _user_from_model(row) if row is not None else None

    def get_user_by_id(self, user_id: str) -> WebUser | None:
        with self._session_factory() as session:
            row = session.get(UserRow, user_id)
        return _user_from_model(row) if row is not None else None

    def create_session(self, user_id: str) -> str:
        if self.get_user_by_id(user_id) is None:
            raise ValueError(f"user not found: {user_id}")
        token = f"session_{secrets.token_urlsafe(32)}"
        created_at = _utc_now()
        expires_at = _utc_in(days=7)
        with self._session_factory.begin() as session:
            session.add(
                LoginSessionRow(
                    token=token,
                    user_id=user_id,
                    created_at=created_at,
                    expires_at=expires_at,
                )
            )
        return token

    def get_user_for_session(self, token: str) -> WebUser | None:
        now = _utc_now()
        statement = (
            select(UserRow)
            .join(LoginSessionRow, LoginSessionRow.user_id == UserRow.id)
            .where(
                LoginSessionRow.token == token,
                LoginSessionRow.expires_at > now,
            )
        )
        with self._session_factory() as session:
            row = session.scalar(statement)
        return _user_from_model(row) if row is not None else None

    def delete_session(self, token: str) -> None:
        with self._session_factory.begin() as session:
            session.execute(
                delete(LoginSessionRow).where(LoginSessionRow.token == token)
            )

    def create_conversation(
        self,
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
        now = _utc_now()
        authors_json = json.dumps(paper.authors, ensure_ascii=False)
        source_url = paper.source_url.strip()

        with self._session_factory.begin() as session:
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
            return _conversation_from_model(conversation_row)

    def list_conversations(
        self,
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
        with self._session_factory() as session:
            rows = list(session.scalars(statement))
        return [_conversation_from_model(row) for row in rows]

    def get_conversation_detail(
        self,
        conversation_id: str,
        *,
        user_id: str,
    ) -> ConversationDetail | None:
        with self._session_factory() as session:
            conversation_row = _select_owned_conversation_model(
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
                conversation=_conversation_from_model(conversation_row),
                primary_paper=_paper_from_model(primary_paper_row),
                active_papers=[_paper_from_model(row[1]) for row in paper_rows],
                paper_associations=[
                    _conversation_paper_from_model(row[0]) for row in paper_rows
                ],
                active_task=(
                    _task_from_model(active_task_row)
                    if active_task_row is not None
                    else None
                ),
            )

    def update_conversation(
        self,
        conversation_id: str,
        *,
        user_id: str,
        title: str | None = None,
        archived: bool | None = None,
    ) -> ConversationRecord | None:
        with self._session_factory.begin() as session:
            row = _select_owned_conversation_model(
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
                row.archived_at = _utc_now() if archived else None
                changed = True
            if changed:
                row.updated_at = _utc_now()
            session.flush()
            return _conversation_from_model(row)

    def create_conversation_turn(
        self,
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
            with self._session_factory.begin() as session:
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
                task = _task_from_model(task_row)
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
                        created_at=_utc_now(),
                    )
                )
                session.flush()
                return ConversationTurn(
                    user_message=_message_from_model(user_message_row),
                    task=task,
                )
        except IntegrityError as exc:
            if _is_active_task_unique_conflict(exc):
                raise ConversationBusyError(
                    "conversation already has an active task"
                ) from exc
            raise

    def publish_conversation_result(
        self,
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
        now = _utc_now()
        with self._session_factory.begin() as session:
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
                message=_message_from_model(assistant_row),
                artifact=_artifact_from_model(artifact_row),
                active_paper_ids=active_paper_ids,
            )

    def finalize_conversation_task(
        self,
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

        with self._session_factory.begin() as session:
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
            normalized_active_ids = _validate_active_paper_ids(
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
                    or _read_active_paper_ids(session, conversation_row)
                    != normalized_active_ids
                ):
                    raise ValueError("completed task finalization does not match stored result")
                return FinalizedConversationTask(
                    task=_task_from_model(task_row),
                    conversation=_conversation_from_model(conversation_row),
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

            now = _utc_now()
            task_row.status = "completed"
            task_row.final_checkpoint_id = checkpoint_id
            task_row.result_quality = result_quality
            task_row.updated_at = now
            _set_active_paper_ids(
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
                task=_task_from_model(task_row),
                conversation=_conversation_from_model(conversation_row),
            )

    def fail_conversation_task(
        self,
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
        with self._session_factory.begin() as session:
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
                task_row.updated_at = _utc_now()
            if failure_event is None:
                session.add(
                    TaskEventRow(
                        task_id=task_id,
                        type="failed",
                        stage=stage,
                        message=cleaned_message,
                        payload_json=json.dumps(payload or {}, ensure_ascii=False),
                        created_at=_utc_now(),
                    )
                )
            session.flush()
            return _task_from_model(task_row)

    def switch_conversation_head(
        self,
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
        with self._session_factory.begin() as session:
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
            normalized_active_ids = _validate_active_paper_ids(
                session,
                conversation_row=conversation_row,
                active_paper_ids=active_paper_ids,
            )

            _set_active_paper_ids(
                session,
                conversation_row=conversation_row,
                active_paper_ids=normalized_active_ids,
            )
            conversation_row.head_message_id = target_row.id
            conversation_row.head_checkpoint_id = checkpoint_id
            conversation_row.updated_at = _utc_now()
            session.flush()
            return _conversation_from_model(conversation_row)

    def get_message(
        self,
        conversation_id: str,
        message_id: str,
        *,
        user_id: str,
    ) -> MessageRecord | None:
        with self._session_factory() as session:
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
        return _message_from_model(row) if row is not None else None

    def get_task_message(
        self,
        task_id: str,
        role: str,
    ) -> MessageRecord | None:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"invalid message role: {role!r}")
        with self._session_factory() as session:
            row = session.scalar(
                select(MessageRow).where(
                    MessageRow.task_id == task_id,
                    MessageRow.role == role,
                )
            )
        return _message_from_model(row) if row is not None else None

    def list_active_messages(
        self,
        conversation_id: str,
        *,
        user_id: str,
    ) -> list[MessageRecord] | None:
        with self._session_factory() as session:
            conversation_row = _select_owned_conversation_model(
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
            return [_message_from_model(row) for row in path]

    def list_message_alternatives(
        self,
        conversation_id: str,
        message_id: str,
        *,
        user_id: str,
    ) -> list[ConversationAlternative] | None:
        with self._session_factory() as session:
            if (
                _select_owned_conversation_model(
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
                            user_message=_message_from_model(user_row),
                            assistant_message=_message_from_model(assistant_row),
                        )
                    )
            return alternatives

    def get_unstable_turn(
        self,
        conversation_id: str,
        *,
        user_id: str,
    ) -> ConversationTurn | None:
        with self._session_factory() as session:
            conversation_row = _select_owned_conversation_model(
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
                user_message=_message_from_model(message_row),
                task=_task_from_model(task_row),
            )

    def check_health(self) -> None:
        with self.engine.connect() as connection:
            row = connection.execute(text("SELECT 1")).one_or_none()
        if row is None or int(row[0]) != 1:
            raise RuntimeError("SQLite health probe returned an invalid result")

    def get_task(
        self,
        task_id: str,
        *,
        user_id: str | None = None,
    ) -> ResearchTask | None:
        filters = [ResearchTaskRow.id == task_id]
        if user_id is not None:
            filters.append(ResearchTaskRow.user_id == user_id)
        with self._session_factory() as session:
            row = session.scalar(select(ResearchTaskRow).where(*filters))
        return _task_from_model(row) if row is not None else None

    def claim_task(
        self,
        task_id: str,
        *,
        allow_running: bool = False,
    ) -> ResearchTask | None:
        """Atomically claim pending work or recover a redelivered running task."""
        claimable_statuses = ("pending", "running") if allow_running else ("pending",)
        updated_at = _utc_now()
        with self._session_factory.begin() as session:
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
            return _task_from_model(row)

    def fail_pending_task(self, task_id: str) -> ResearchTask | None:
        """Mark an unclaimed task failed without overwriting active work."""
        with self._session_factory.begin() as session:
            result = session.execute(
                update(ResearchTaskRow)
                .where(
                    ResearchTaskRow.id == task_id,
                    ResearchTaskRow.status == "pending",
                )
                .values(status="failed", updated_at=_utc_now())
            )
            if result.rowcount != 1:
                return None
            row = session.get(ResearchTaskRow, task_id)
            if row is None:
                raise RuntimeError("failed task disappeared within transaction")
            return _task_from_model(row)

    def add_event(
        self,
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
            created_at=_utc_now(),
        )
        with self._session_factory.begin() as session:
            if session.get(ResearchTaskRow, task_id) is None:
                raise ValueError(f"task not found: {task_id}")
            session.add(row)
            session.flush()
        return _event_from_model(row)

    def list_events_page(
        self,
        task_id: str,
        *,
        user_id: str | None,
        after_id: int,
        limit: int,
    ) -> TaskEventBatch | None:
        _validate_incremental_page(after_id, limit)
        with self._session_factory() as session:
            if _select_owned_task_model(session, task_id, user_id) is None:
                return None
            return _read_event_batch(
                session,
                task_id,
                after_id=after_id,
                limit=limit,
            )

    def add_artifact(
        self,
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
            created_at=_utc_now(),
        )
        with self._session_factory.begin() as session:
            if session.get(ResearchTaskRow, task_id) is None:
                raise ValueError(f"task not found: {task_id}")
            session.add(row)
            session.flush()
        return _artifact_from_model(row)

    def list_artifacts_page(
        self,
        task_id: str,
        *,
        user_id: str | None,
        after_id: int,
        limit: int,
    ) -> TaskArtifactBatch | None:
        _validate_incremental_page(after_id, limit)
        with self._session_factory() as session:
            if _select_owned_task_model(session, task_id, user_id) is None:
                return None
            return _read_artifact_batch(
                session,
                task_id,
                after_id=after_id,
                limit=limit,
            )

    def get_conversation_task_updates(
        self,
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
        with self._session_factory.begin() as session:
            row = _select_owned_task_model(session, task_id, user_id)
            if row is None or row.conversation_id != conversation_id:
                return None
            return TaskUpdates(
                task=_task_from_model(row),
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
    now = _utc_now()
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


def _task_from_model(row: ResearchTaskRow) -> ResearchTask:
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


def _user_from_model(row: UserRow) -> WebUser:
    return WebUser(
        id=row.id,
        username=row.username,
        password_hash=row.password_hash,
        password_salt=row.password_salt,
        created_at=row.created_at,
    )


def _paper_from_model(row: PaperRow) -> PaperRecord:
    authors = _decode_json_list(row.authors_json)
    return PaperRecord(
        id=row.id,
        source=row.source,
        external_id=row.external_id,
        title=row.title,
        authors=authors,
        abstract=row.abstract,
        source_url=row.source_url,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _conversation_from_model(row: ConversationRow) -> ConversationRecord:
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


def _conversation_paper_from_model(
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


def _message_from_model(row: MessageRow) -> MessageRecord:
    return MessageRecord(
        id=row.id,
        conversation_id=row.conversation_id,
        task_id=row.task_id,
        parent_message_id=row.parent_message_id,
        role=row.role,
        content=row.content,
        status=row.status,
        metadata=_decode_payload(row.metadata_json),
        created_at=row.created_at,
    )


def _event_from_model(row: TaskEventRow) -> TaskEvent:
    return TaskEvent(
        id=row.id,
        task_id=row.task_id,
        type=row.type,
        stage=row.stage,
        message=row.message,
        payload=_decode_payload(row.payload_json),
        created_at=row.created_at,
    )


def _artifact_from_model(row: TaskArtifactRow) -> TaskArtifact:
    return TaskArtifact(
        id=row.id,
        task_id=row.task_id,
        kind=row.kind,
        title=row.title,
        content=row.content,
        payload=_decode_payload(row.payload_json),
        created_at=row.created_at,
    )


def _select_owned_task_model(
    session: Session,
    task_id: str,
    user_id: str | None,
) -> ResearchTaskRow | None:
    filters = [ResearchTaskRow.id == task_id]
    if user_id is not None:
        filters.append(ResearchTaskRow.user_id == user_id)
    return session.scalar(select(ResearchTaskRow).where(*filters))


def _select_owned_conversation_model(
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


def _validate_active_paper_ids(
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
            .where(
                ConversationPaperRow.conversation_id == conversation_row.id,
            )
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


def _read_active_paper_ids(
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


def _set_active_paper_ids(
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


def _raise_conversation_turn_conflict(
    session: Session,
    *,
    conversation_id: str,
    user_id: str,
    expected_head_message_id: str | None,
) -> None:
    conversation_row = _select_owned_conversation_model(
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
    items = [_event_from_model(row) for row in rows[:limit]]
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
    items = [_artifact_from_model(row) for row in rows[:limit]]
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


def _decode_payload(value: object) -> dict:
    if value is None:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _decode_json_list(value: object) -> list[str]:
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def _validate_conversation_title(title: str) -> str:
    cleaned = title.strip()
    if not 1 <= len(cleaned) <= 200:
        raise ValueError("conversation title must be between 1 and 200 characters")
    return cleaned


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utc_in(*, days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(
        timespec="seconds"
    )
