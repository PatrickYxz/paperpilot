"""SQLite-backed research task storage for the Web workbench."""
from __future__ import annotations

import json
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from sqlalchemy import and_, delete, or_, select, text, update
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
    PaperRow,
    ResearchTaskRow,
    TaskArtifactRow,
    TaskEventRow,
    UserRow,
)


TaskDepth = Literal["quick", "standard", "deep"]
TaskStatus = Literal["pending", "running", "completed", "failed"]

VALID_DEPTHS: set[str] = {"quick", "standard", "deep"}
VALID_STATUSES: set[str] = {"pending", "running", "completed", "failed"}


class DuplicateUsernameError(ValueError):
    """Raised when the users table rejects a duplicate username."""


@dataclass(frozen=True)
class ResearchTask:
    id: str
    question: str
    depth: str
    status: str
    created_at: str
    updated_at: str
    user_id: str | None = None

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
class TaskPage:
    items: list[ResearchTask]
    has_more: bool


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

    def create_task(
        self,
        *,
        question: str,
        depth: str = "standard",
        user_id: str | None = None,
    ) -> ResearchTask:
        task = _new_task(question=question, depth=depth, user_id=user_id)
        with self._session_factory.begin() as session:
            session.add(_task_to_model(task))
        return task

    def create_queued_task(
        self,
        *,
        question: str,
        depth: str,
        user_id: str,
        execution_mode: str,
    ) -> ResearchTask:
        if execution_mode not in {"simulated", "real"}:
            raise ValueError(f"invalid execution mode: {execution_mode!r}")

        task = _new_task(question=question, depth=depth, user_id=user_id)
        payload = {
            "depth": task.depth,
            "execution_mode": execution_mode,
            "simulated": execution_mode == "simulated",
        }
        task_row = _task_to_model(task)
        with self._session_factory.begin() as session:
            session.add(task_row)
            session.flush()
            session.add(
                TaskEventRow(
                    task_id=task.id,
                    type="queued",
                    stage="queue",
                    message=f"Task queued for {execution_mode} workflow.",
                    payload_json=json.dumps(payload, ensure_ascii=False),
                    created_at=_utc_now(),
                )
            )
        return task

    def check_health(self) -> None:
        with self.engine.connect() as connection:
            row = connection.execute(text("SELECT 1")).one_or_none()
        if row is None or int(row[0]) != 1:
            raise RuntimeError("SQLite health probe returned an invalid result")

    def list_tasks_page(
        self,
        *,
        user_id: str | None,
        limit: int,
        status: str | None = None,
        before_created_at: str | None = None,
        before_id: str | None = None,
    ) -> TaskPage:
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status!r}")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        if (before_created_at is None) != (before_id is None):
            raise ValueError("task page position requires created_at and id")

        filters = [
            ResearchTaskRow.user_id.is_(None)
            if user_id is None
            else ResearchTaskRow.user_id == user_id
        ]
        if status is not None:
            filters.append(ResearchTaskRow.status == status)
        if before_created_at is not None:
            assert before_id is not None
            filters.append(
                or_(
                    ResearchTaskRow.created_at < before_created_at,
                    and_(
                        ResearchTaskRow.created_at == before_created_at,
                        ResearchTaskRow.id < before_id,
                    ),
                )
            )
        statement = (
            select(ResearchTaskRow)
            .where(*filters)
            .order_by(
                ResearchTaskRow.created_at.desc(),
                ResearchTaskRow.id.desc(),
            )
            .limit(limit + 1)
        )
        with self._session_factory() as session:
            rows = list(session.scalars(statement))
        tasks = [_task_from_model(row) for row in rows]
        return TaskPage(
            items=tasks[:limit],
            has_more=len(tasks) > limit,
        )

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

    def update_status(self, task_id: str, status: str) -> ResearchTask | None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status!r}")

        with self._session_factory.begin() as session:
            result = session.execute(
                update(ResearchTaskRow)
                .where(ResearchTaskRow.id == task_id)
                .values(status=status, updated_at=_utc_now())
            )
            if result.rowcount != 1:
                return None
        return self.get_task(task_id)

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

    def get_task_updates(
        self,
        task_id: str,
        *,
        user_id: str | None,
        after_event_id: int,
        after_artifact_id: int,
        limit: int,
    ) -> TaskUpdates | None:
        _validate_incremental_page(after_event_id, limit)
        _validate_incremental_page(after_artifact_id, limit)
        with self._session_factory.begin() as session:
            row = _select_owned_task_model(session, task_id, user_id)
            if row is None:
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
    )


def _task_to_model(task: ResearchTask) -> ResearchTaskRow:
    return ResearchTaskRow(
        id=task.id,
        question=task.question,
        depth=task.depth,
        status=task.status,
        created_at=task.created_at,
        updated_at=task.updated_at,
        user_id=task.user_id,
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
