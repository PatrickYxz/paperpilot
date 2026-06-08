"""SQLite-backed research task storage for the Web workbench."""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

DEFAULT_TASK_DB_PATH = Path("data/web/tasks.sqlite3")
TaskDepth = Literal["quick", "standard", "deep"]
TaskStatus = Literal["pending", "running", "completed", "failed"]

VALID_DEPTHS: set[str] = {"quick", "standard", "deep"}
VALID_STATUSES: set[str] = {"pending", "running", "completed", "failed"}


@dataclass(frozen=True)
class ResearchTask:
    id: str
    question: str
    depth: str
    status: str
    created_at: str
    updated_at: str

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


class TaskStore:
    """Persist local research tasks in SQLite."""

    def __init__(self, db_path: Path | str = DEFAULT_TASK_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._ensure_schema()

    def create_task(self, *, question: str, depth: str = "standard") -> ResearchTask:
        question = question.strip()
        if not question:
            raise ValueError("question is required")
        if depth not in VALID_DEPTHS:
            raise ValueError(f"invalid depth: {depth!r}")

        now = _utc_now()
        task = ResearchTask(
            id=f"task_{uuid.uuid4().hex}",
            question=question,
            depth=depth,
            status="pending",
            created_at=now,
            updated_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_tasks (
                    id, question, depth, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.question,
                    task.depth,
                    task.status,
                    task.created_at,
                    task.updated_at,
                ),
            )
        return task

    def list_tasks(self, *, status: str | None = None) -> list[ResearchTask]:
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status!r}")

        query = "SELECT * FROM research_tasks"
        params: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC, rowid DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_task_from_row(row) for row in rows]

    def get_task(self, task_id: str) -> ResearchTask | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
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

    def list_events(self, task_id: str) -> list[TaskEvent] | None:
        if self.get_task(task_id) is None:
            return None
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, task_id, type, stage, message, payload_json, created_at
                FROM task_events
                WHERE task_id = ?
                ORDER BY id ASC
                """,
                (task_id,),
            ).fetchall()
        return [_event_from_row(row) for row in rows]

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

    def list_artifacts(self, task_id: str) -> list[TaskArtifact] | None:
        if self.get_task(task_id) is None:
            return None
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, task_id, kind, title, content, payload_json, created_at
                FROM task_artifacts
                WHERE task_id = ?
                ORDER BY id ASC
                """,
                (task_id,),
            ).fetchall()
        return [_artifact_from_row(row) for row in rows]

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_tasks (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    depth TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_task_event_columns(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(task_events)").fetchall()
        columns = {str(row["name"]) for row in rows}
        if "stage" not in columns:
            conn.execute("ALTER TABLE task_events ADD COLUMN stage TEXT")
        if "payload_json" not in columns:
            conn.execute("ALTER TABLE task_events ADD COLUMN payload_json TEXT")


def _task_from_row(row: sqlite3.Row) -> ResearchTask:
    return ResearchTask(
        id=str(row["id"]),
        question=str(row["question"]),
        depth=str(row["depth"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
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
