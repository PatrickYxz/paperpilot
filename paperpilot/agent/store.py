"""SQLite persistence for durable agent runs.

The run store shares the Web task database, but owns only the agent tables.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from paperpilot.agent.models import AgentRun, RunCheckpoint, RUN_ACTIVE_STATUSES, SCHEMA_VERSION
from paperpilot.agent.policy import RunPolicy
from paperpilot.message_codec import decode_messages, encode_messages


class ActiveRunExistsError(ValueError):
    """Raised when a task already has a run that has not reached a terminal state."""


class RunStore(Protocol):
    """Persistence surface required before the execution runtime is introduced."""

    def create_run(
        self,
        *,
        task_id: str,
        policy: RunPolicy,
        initial_messages: list[dict],
        runtime_state: dict,
    ) -> AgentRun: ...

    def get_run(self, run_id: str) -> AgentRun | None: ...

    def get_active_run_for_task(self, task_id: str) -> AgentRun | None: ...

    def get_latest_run_for_task(self, task_id: str) -> AgentRun | None: ...

    def get_latest_checkpoint(self, run_id: str) -> RunCheckpoint | None: ...

    def check_health(self) -> None: ...


class SQLiteRunStore:
    """Persist agent state in the SQLite file already used by ``TaskStore``."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        clock: Callable[[], str] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.clock = clock or utc_now
        self._ensure_schema()

    def create_run(
        self,
        *,
        task_id: str,
        policy: RunPolicy,
        initial_messages: list[dict],
        runtime_state: dict,
    ) -> AgentRun:
        now = self.clock()
        run = AgentRun(
            id=f"run_{uuid.uuid4().hex}",
            task_id=task_id,
            status="pending",
            attempt=1,
            current_step=0,
            cancel_requested_at=None,
            retry_at=None,
            failure_class=None,
            failure_message=None,
            policy=policy,
            owner_id=None,
            lease_expires_at=None,
            schema_version=SCHEMA_VERSION,
            created_at=now,
            updated_at=now,
            started_at=None,
            finished_at=None,
        )
        checkpoint = RunCheckpoint(
            id=f"checkpoint_{uuid.uuid4().hex}",
            run_id=run.id,
            step_sequence=0,
            messages=initial_messages,
            runtime_state=runtime_state,
            schema_version=SCHEMA_VERSION,
            created_at=now,
        )
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            conn.execute(
                """
                INSERT INTO agent_runs (
                    id, task_id, status, attempt, current_step, cancel_requested_at,
                    retry_at, failure_class, failure_message, policy_json, owner_id,
                    lease_expires_at, schema_version, created_at, updated_at, started_at,
                    finished_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.task_id,
                    run.status,
                    run.attempt,
                    run.current_step,
                    run.cancel_requested_at,
                    run.retry_at,
                    run.failure_class,
                    run.failure_message,
                    _encode_json(run.policy.to_dict()),
                    run.owner_id,
                    run.lease_expires_at,
                    run.schema_version,
                    run.created_at,
                    run.updated_at,
                    run.started_at,
                    run.finished_at,
                ),
            )
            conn.execute(
                """
                INSERT INTO run_checkpoints (
                    id, run_id, step_sequence, messages_json, runtime_state_json,
                    schema_version, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.id,
                    checkpoint.run_id,
                    checkpoint.step_sequence,
                    _encode_json(encode_messages(checkpoint.messages)),
                    _encode_json(checkpoint.runtime_state),
                    checkpoint.schema_version,
                    checkpoint.created_at,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            if "agent_runs.task_id" in str(exc):
                raise ActiveRunExistsError(
                    f"task already has an active run: {task_id}"
                ) from exc
            raise
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return run

    def get_run(self, run_id: str) -> AgentRun | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return _run_from_row(row) if row is not None else None

    def get_active_run_for_task(self, task_id: str) -> AgentRun | None:
        placeholders = ", ".join("?" for _ in RUN_ACTIVE_STATUSES)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM agent_runs
                WHERE task_id = ? AND status IN ({placeholders})
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (task_id, *sorted(RUN_ACTIVE_STATUSES)),
            ).fetchone()
        return _run_from_row(row) if row is not None else None

    def get_latest_run_for_task(self, task_id: str) -> AgentRun | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM agent_runs
                WHERE task_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        return _run_from_row(row) if row is not None else None

    def get_latest_checkpoint(self, run_id: str) -> RunCheckpoint | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM run_checkpoints
                WHERE run_id = ?
                ORDER BY step_sequence DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        return _checkpoint_from_row(row) if row is not None else None

    def check_health(self) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT 1").fetchone()
        if row is None or int(row[0]) != 1:
            raise RuntimeError("SQLite health probe returned an invalid result")

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    current_step INTEGER NOT NULL,
                    cancel_requested_at TEXT,
                    retry_at TEXT,
                    failure_class TEXT,
                    failure_message TEXT,
                    policy_json TEXT NOT NULL,
                    owner_id TEXT,
                    lease_expires_at TEXT,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY(task_id) REFERENCES research_tasks(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_steps (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT,
                    error_json TEXT,
                    schema_version INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(id),
                    UNIQUE(run_id, sequence, attempt)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_checkpoints (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_sequence INTEGER NOT NULL,
                    messages_json TEXT NOT NULL,
                    runtime_state_json TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(id),
                    UNIQUE(run_id, step_sequence)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_executions (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_preview TEXT,
                    failure_class TEXT,
                    failure_message TEXT,
                    duration_ms INTEGER,
                    schema_version INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(id),
                    FOREIGN KEY(step_id) REFERENCES agent_steps(id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_steps_run_sequence
                ON agent_steps(run_id, sequence)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_run_checkpoints_run_sequence
                ON run_checkpoints(run_id, step_sequence)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_runs_task_created
                ON agent_runs(task_id, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_runs_status_retry
                ON agent_runs(status, retry_at)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_runs_one_active_task
                ON agent_runs(task_id)
                WHERE status IN ('pending', 'running', 'waiting_retry', 'cancelling')
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _run_from_row(row: sqlite3.Row) -> AgentRun:
    return AgentRun(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        status=str(row["status"]),
        attempt=int(row["attempt"]),
        current_step=int(row["current_step"]),
        cancel_requested_at=row["cancel_requested_at"],
        retry_at=row["retry_at"],
        failure_class=row["failure_class"],
        failure_message=row["failure_message"],
        policy=RunPolicy.from_dict(json.loads(str(row["policy_json"]))),
        owner_id=row["owner_id"],
        lease_expires_at=row["lease_expires_at"],
        schema_version=int(row["schema_version"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def _checkpoint_from_row(row: sqlite3.Row) -> RunCheckpoint:
    return RunCheckpoint(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        step_sequence=int(row["step_sequence"]),
        messages=decode_messages(json.loads(str(row["messages_json"]))),
        runtime_state=json.loads(str(row["runtime_state_json"])),
        schema_version=int(row["schema_version"]),
        created_at=str(row["created_at"]),
    )
