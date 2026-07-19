"""SQLite-backed research task storage for the Web workbench."""
from __future__ import annotations

import json
import os
import sqlite3
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

DEFAULT_TASK_DB_PATH = Path("data/web/tasks.sqlite3")
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
        configured_path = db_path or os.environ.get(
            "PAPERPILOT_TASK_DB_PATH",
            str(DEFAULT_TASK_DB_PATH),
        )
        self.db_path = Path(configured_path)
        self._ensure_schema()

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
        now = _utc_now()
        user = WebUser(
            id=f"user_{uuid.uuid4().hex}",
            username=username,
            password_hash=password_hash,
            password_salt=password_salt,
            created_at=now,
        )
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO users (
                        id, username, password_hash, password_salt, created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        user.id,
                        user.username,
                        user.password_hash,
                        user.password_salt,
                        user.created_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateUsernameError("username already exists") from exc
        return user

    def get_user_by_username(self, username: str) -> WebUser | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            return None
        return _user_from_row(row)

    def get_user_by_id(self, user_id: str) -> WebUser | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return _user_from_row(row)

    def create_session(self, user_id: str) -> str:
        if self.get_user_by_id(user_id) is None:
            raise ValueError(f"user not found: {user_id}")
        token = f"session_{secrets.token_urlsafe(32)}"
        created_at = _utc_now()
        expires_at = _utc_in(days=7)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (token, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (token, user_id, created_at, expires_at),
            )
        return token

    def get_user_for_session(self, token: str) -> WebUser | None:
        now = _utc_now()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT users.*
                FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token = ? AND sessions.expires_at > ?
                """,
                (token, now),
            ).fetchone()
        if row is None:
            return None
        return _user_from_row(row)

    def delete_session(self, token: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

    def create_task(
        self,
        *,
        question: str,
        depth: str = "standard",
        user_id: str | None = None,
    ) -> ResearchTask:
        task = _new_task(question=question, depth=depth, user_id=user_id)
        with self._connect() as conn:
            _insert_task(conn, task)
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
        with self._connect() as conn:
            conn.execute("BEGIN")
            _insert_task(conn, task)
            conn.execute(
                """
                INSERT INTO task_events (
                    task_id, type, stage, message, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    "queued",
                    "queue",
                    f"Task queued for {execution_mode} workflow.",
                    json.dumps(payload, ensure_ascii=False),
                    _utc_now(),
                ),
            )
        return task

    def check_health(self) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT 1").fetchone()
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

        clauses: list[str] = []
        params: list[object] = []
        if user_id is None:
            clauses.append("user_id IS NULL")
        else:
            clauses.append("user_id = ?")
            params.append(user_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if before_created_at is not None:
            clauses.append(
                "(created_at < ? OR (created_at = ? AND id < ?))"
            )
            params.extend([before_created_at, before_created_at, before_id])

        query = f"""
            SELECT *
            FROM research_tasks
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
        """
        params.append(limit + 1)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        tasks = [_task_from_row(row) for row in rows]
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
        query = "SELECT * FROM research_tasks WHERE id = ?"
        params: list[str] = [task_id]
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        with self._connect() as conn:
            row = conn.execute(query, tuple(params)).fetchone()
        if row is None:
            return None
        return _task_from_row(row)

    def update_status(self, task_id: str, status: str) -> ResearchTask | None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status!r}")

        updated_at = _utc_now()
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE research_tasks
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, updated_at, task_id),
            )
            if result.rowcount == 0:
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
        placeholders = ", ".join("?" for _ in claimable_statuses)
        updated_at = _utc_now()
        with self._connect() as conn:
            result = conn.execute(
                f"""
                UPDATE research_tasks
                SET status = 'running', updated_at = ?
                WHERE id = ? AND status IN ({placeholders})
                """,
                (updated_at, task_id, *claimable_statuses),
            )
            if result.rowcount == 0:
                return None
            row = conn.execute(
                "SELECT * FROM research_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        assert row is not None
        return _task_from_row(row)

    def fail_pending_task(self, task_id: str) -> ResearchTask | None:
        """Mark an unclaimed task failed without overwriting active work."""
        updated_at = _utc_now()
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE research_tasks
                SET status = 'failed', updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (updated_at, task_id),
            )
            if result.rowcount == 0:
                return None
            row = conn.execute(
                "SELECT * FROM research_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        assert row is not None
        return _task_from_row(row)

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
        created_at = _utc_now()
        payload_dict = payload or {}
        payload_json = json.dumps(payload_dict, ensure_ascii=False)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM research_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if existing is None:
                raise ValueError(f"task not found: {task_id}")
            cursor = conn.execute(
                """
                INSERT INTO task_events (
                    task_id, type, stage, message, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (task_id, type, stage, message.strip(), payload_json, created_at),
            )
            event_id = int(cursor.lastrowid)
        return TaskEvent(
            id=event_id,
            task_id=task_id,
            type=type,
            stage=stage,
            message=message.strip(),
            payload=payload_dict,
            created_at=created_at,
        )

    def list_events_page(
        self,
        task_id: str,
        *,
        user_id: str | None,
        after_id: int,
        limit: int,
    ) -> TaskEventBatch | None:
        _validate_incremental_page(after_id, limit)
        with self._connect() as conn:
            if _select_owned_task_row(conn, task_id, user_id) is None:
                return None
            return _read_event_batch(
                conn,
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
        created_at = _utc_now()
        payload_dict = payload or {}
        payload_json = json.dumps(payload_dict, ensure_ascii=False)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM research_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if existing is None:
                raise ValueError(f"task not found: {task_id}")
            cursor = conn.execute(
                """
                INSERT INTO task_artifacts (
                    task_id, kind, title, content, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    kind.strip(),
                    title.strip(),
                    content.strip(),
                    payload_json,
                    created_at,
                ),
            )
            artifact_id = int(cursor.lastrowid)
        return TaskArtifact(
            id=artifact_id,
            task_id=task_id,
            kind=kind.strip(),
            title=title.strip(),
            content=content.strip(),
            payload=payload_dict,
            created_at=created_at,
        )

    def list_artifacts_page(
        self,
        task_id: str,
        *,
        user_id: str | None,
        after_id: int,
        limit: int,
    ) -> TaskArtifactBatch | None:
        _validate_incremental_page(after_id, limit)
        with self._connect() as conn:
            if _select_owned_task_row(conn, task_id, user_id) is None:
                return None
            return _read_artifact_batch(
                conn,
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
        with self._connect() as conn:
            conn.execute("BEGIN")
            row = _select_owned_task_row(conn, task_id, user_id)
            if row is None:
                return None
            return TaskUpdates(
                task=_task_from_row(row),
                events=_read_event_batch(
                    conn,
                    task_id,
                    after_id=after_event_id,
                    limit=limit,
                ),
                artifacts=_read_artifact_batch(
                    conn,
                    task_id,
                    after_id=after_artifact_id,
                    limit=limit,
                ),
            )

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            journal_mode = str(
                conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            ).strip().lower()
            if journal_mode != "wal":
                raise RuntimeError(
                    f"SQLite WAL initialization failed: returned {journal_mode!r}"
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_tasks (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    depth TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    user_id TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            self._ensure_research_task_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    stage TEXT,
                    message TEXT NOT NULL,
                    payload_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES research_tasks(id)
                )
                """
            )
            self._ensure_task_event_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    payload_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES research_tasks(id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_user_created_id
                ON research_tasks(user_id, created_at DESC, id DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_user_status_created_id
                ON research_tasks(user_id, status, created_at DESC, id DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_task_id_id
                ON task_events(task_id, id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_artifacts_task_id_id
                ON task_artifacts(task_id, id)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _ensure_research_task_columns(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(research_tasks)").fetchall()
        columns = {str(row["name"]) for row in rows}
        if "user_id" not in columns:
            conn.execute("ALTER TABLE research_tasks ADD COLUMN user_id TEXT")

    def _ensure_task_event_columns(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(task_events)").fetchall()
        columns = {str(row["name"]) for row in rows}
        if "stage" not in columns:
            conn.execute("ALTER TABLE task_events ADD COLUMN stage TEXT")
        if "payload_json" not in columns:
            conn.execute("ALTER TABLE task_events ADD COLUMN payload_json TEXT")


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


def _insert_task(conn: sqlite3.Connection, task: ResearchTask) -> None:
    conn.execute(
        """
        INSERT INTO research_tasks (
            id, question, depth, status, created_at, updated_at, user_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task.id,
            task.question,
            task.depth,
            task.status,
            task.created_at,
            task.updated_at,
            task.user_id,
        ),
    )


def _task_from_row(row: sqlite3.Row) -> ResearchTask:
    return ResearchTask(
        id=str(row["id"]),
        question=str(row["question"]),
        depth=str(row["depth"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        user_id=_optional_str(row["user_id"]),
    )


def _user_from_row(row: sqlite3.Row) -> WebUser:
    return WebUser(
        id=str(row["id"]),
        username=str(row["username"]),
        password_hash=str(row["password_hash"]),
        password_salt=str(row["password_salt"]),
        created_at=str(row["created_at"]),
    )


def _event_from_row(row: sqlite3.Row) -> TaskEvent:
    payload_json = row["payload_json"]
    return TaskEvent(
        id=int(row["id"]),
        task_id=str(row["task_id"]),
        type=str(row["type"]),
        stage=_optional_str(row["stage"]),
        message=str(row["message"]),
        payload=_decode_payload(payload_json),
        created_at=str(row["created_at"]),
    )


def _artifact_from_row(row: sqlite3.Row) -> TaskArtifact:
    return TaskArtifact(
        id=int(row["id"]),
        task_id=str(row["task_id"]),
        kind=str(row["kind"]),
        title=str(row["title"]),
        content=str(row["content"]),
        payload=_decode_payload(row["payload_json"]),
        created_at=str(row["created_at"]),
    )


def _select_owned_task_row(
    conn: sqlite3.Connection,
    task_id: str,
    user_id: str | None,
) -> sqlite3.Row | None:
    if user_id is None:
        return conn.execute(
            "SELECT * FROM research_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    return conn.execute(
        "SELECT * FROM research_tasks WHERE id = ? AND user_id = ?",
        (task_id, user_id),
    ).fetchone()


def _read_event_batch(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    after_id: int,
    limit: int,
) -> TaskEventBatch:
    rows = conn.execute(
        """
        SELECT id, task_id, type, stage, message, payload_json, created_at
        FROM task_events
        WHERE task_id = ? AND id > ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (task_id, after_id, limit + 1),
    ).fetchall()
    items = [_event_from_row(row) for row in rows[:limit]]
    return TaskEventBatch(
        items=items,
        next_after_id=items[-1].id if items else after_id,
        has_more=len(rows) > limit,
    )


def _read_artifact_batch(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    after_id: int,
    limit: int,
) -> TaskArtifactBatch:
    rows = conn.execute(
        """
        SELECT id, task_id, kind, title, content, payload_json, created_at
        FROM task_artifacts
        WHERE task_id = ? AND id > ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (task_id, after_id, limit + 1),
    ).fetchall()
    items = [_artifact_from_row(row) for row in rows[:limit]]
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


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


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
