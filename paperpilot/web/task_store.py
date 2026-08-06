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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from paperpilot.web.database import (
    DEFAULT_TASK_DB_PATH,
    create_session_factory,
    create_task_engine,
    resolve_task_db_path,
)
from paperpilot.web.db_migrations import ensure_database_current
from paperpilot.web.db_models import (
    LoginSessionRow,
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utc_in(*, days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(
        timespec="seconds"
    )
