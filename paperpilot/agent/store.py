"""SQLite persistence for durable agent runs.

The run store shares the Web task database, but owns only the agent tables.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from paperpilot.agent.models import (
    AgentRun,
    AgentStep,
    RunCheckpoint,
    StepPage,
    ToolExecution,
    RUN_ACTIVE_STATUSES,
    RUN_TERMINAL_STATUSES,
    SCHEMA_VERSION,
)
from paperpilot.agent.policy import RunPolicy
from paperpilot.message_codec import decode_messages, encode_messages


class ActiveRunExistsError(ValueError):
    """Raised when a task already has a run that has not reached a terminal state."""


class RunLeaseLostError(RuntimeError):
    """Raised when a worker tries to mutate a run it no longer owns."""


class ToolExecutionConflictError(RuntimeError):
    """Raised when a stable tool execution ID is reused inconsistently."""


ToolExecutionStartDisposition = Literal[
    "new", "retry", "in_progress", "failed", "completed", "blocked"
]


@dataclass(frozen=True)
class ToolExecutionStart:
    """Atomic decision about whether a stable tool execution may invoke a handler."""

    execution: ToolExecution
    disposition: ToolExecutionStartDisposition

    @property
    def should_invoke(self) -> bool:
        return self.disposition in {"new", "retry"}

    def __getattr__(self, name: str) -> Any:
        """Keep Task 3 callers source-compatible while exposing the disposition."""
        return getattr(self.execution, name)


_LOCKED_NOW = object()


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

    def claim_run(
        self, run_id: str, *, owner_id: str, lease_seconds: int
    ) -> AgentRun | None: ...

    def renew_lease(
        self, run_id: str, *, owner_id: str, lease_seconds: int
    ) -> bool: ...

    def has_valid_lease(self, run_id: str, owner_id: str) -> bool: ...

    def require_owned_run(self, run_id: str, owner_id: str) -> AgentRun: ...

    def start_step(
        self,
        *,
        run_id: str,
        owner_id: str,
        kind: str,
        input_data: dict[str, Any],
    ) -> AgentStep: ...

    def complete_step_and_checkpoint(
        self,
        *,
        step_id: str,
        run_id: str,
        owner_id: str,
        output_data: dict[str, Any],
        messages: list[dict],
        runtime_state: dict[str, Any],
        tool_completion: Mapping[str, Any] | Any | None = None,
    ) -> RunCheckpoint: ...

    def fail_step(
        self,
        step_id: str,
        *,
        run_id: str,
        owner_id: str,
        error_data: dict[str, Any],
    ) -> AgentStep: ...

    def cancel_step(
        self, step_id: str, *, run_id: str, owner_id: str
    ) -> AgentStep: ...

    def request_cancel(self, run_id: str) -> AgentRun | None: ...

    def schedule_retry(
        self,
        run_id: str,
        *,
        owner_id: str,
        retry_at: datetime,
        failure_class: str,
        failure_message: str,
    ) -> AgentRun: ...

    def mark_completed(self, run_id: str, *, owner_id: str) -> AgentRun: ...

    def mark_failed(
        self,
        run_id: str,
        *,
        owner_id: str,
        failure_class: str,
        failure_message: str,
    ) -> AgentRun: ...

    def mark_cancelled(self, run_id: str, *, owner_id: str) -> AgentRun: ...

    def list_steps(
        self, run_id: str, after_sequence: int = 0, limit: int = 100
    ) -> StepPage: ...

    def list_pending_runs(self, limit: int = 100) -> list[AgentRun]: ...

    def list_cancelling_runs(self, limit: int = 100) -> list[AgentRun]: ...

    def recover_due_runs(
        self, *, now: datetime, limit: int = 100
    ) -> list[AgentRun]: ...

    def start_tool_execution(
        self,
        *,
        run_id: str,
        step_id: str,
        owner_id: str,
        tool_use_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        classification: str,
    ) -> ToolExecutionStart: ...

    def get_tool_execution(self, execution_id: str) -> ToolExecution | None: ...

    def complete_tool_execution(self, **kwargs: Any) -> RunCheckpoint: ...

    def fail_tool_execution(
        self,
        execution_id: str,
        *,
        run_id: str,
        step_id: str,
        owner_id: str,
        failure_class: str,
        failure_message: str,
        duration_ms: int,
    ) -> ToolExecution: ...

    def check_health(self) -> None: ...


class SQLiteRunStore:
    """Persist agent state in the SQLite file already used by ``TaskStore``."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        clock: Callable[[], datetime] | None = None,
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
        now = _format_utc_datetime(self.clock())
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
            if (
                exc.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE
                and "agent_runs.task_id" in str(exc)
            ):
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
                ORDER BY created_at DESC, rowid DESC
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
                ORDER BY created_at DESC, rowid DESC
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

    def claim_run(
        self,
        run_id: str,
        *,
        owner_id: str,
        lease_seconds: int,
    ) -> AgentRun | None:
        _validate_owner_and_lease(owner_id, lease_seconds)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _format_utc_datetime(self.clock())
            expires_at = _format_utc_datetime(
                _parse_utc_datetime(now) + timedelta(seconds=lease_seconds)
            )
            cursor = conn.execute(
                """
                UPDATE agent_runs
                SET status = CASE WHEN status = 'pending' THEN 'running' ELSE status END,
                    owner_id = ?, lease_expires_at = ?, updated_at = ?,
                    started_at = COALESCE(started_at, ?)
                WHERE id = ? AND (
                    status = 'pending'
                    OR (status = 'running' AND lease_expires_at <= ?)
                    OR (
                        status = 'cancelling'
                        AND (owner_id IS NULL OR lease_expires_at IS NULL OR lease_expires_at <= ?)
                    )
                )
                """,
                (owner_id, expires_at, now, now, run_id, now, now),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                return None
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return _run_from_row(row)

    def renew_lease(
        self,
        run_id: str,
        *,
        owner_id: str,
        lease_seconds: int,
    ) -> bool:
        _validate_owner_and_lease(owner_id, lease_seconds)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _format_utc_datetime(self.clock())
            expires_at = _format_utc_datetime(
                _parse_utc_datetime(now) + timedelta(seconds=lease_seconds)
            )
            cursor = conn.execute(
                """
                UPDATE agent_runs
                SET lease_expires_at = ?, updated_at = ?
                WHERE id = ?
                  AND status IN ('running', 'cancelling')
                  AND owner_id = ?
                  AND lease_expires_at > ?
                """,
                (expires_at, now, run_id, owner_id, now),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return cursor.rowcount == 1

    def has_valid_lease(self, run_id: str, owner_id: str) -> bool:
        now = _format_utc_datetime(self.clock())
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM agent_runs
                WHERE id = ?
                  AND status IN ('running', 'cancelling')
                  AND owner_id = ?
                  AND lease_expires_at > ?
                """,
                (run_id, owner_id, now),
            ).fetchone()
        return row is not None

    def require_owned_run(self, run_id: str, owner_id: str) -> AgentRun:
        now = _format_utc_datetime(self.clock())
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM agent_runs
                WHERE id = ?
                  AND status IN ('running', 'cancelling')
                  AND owner_id = ?
                  AND lease_expires_at > ?
                """,
                (run_id, owner_id, now),
            ).fetchone()
        if row is None:
            raise RunLeaseLostError(f"run lease is not owned by {owner_id}: {run_id}")
        return _run_from_row(row)

    def start_step(
        self,
        *,
        run_id: str,
        owner_id: str,
        kind: str,
        input_data: dict[str, Any],
    ) -> AgentStep:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _format_utc_datetime(self.clock())
            run_row = self._owned_run_row(conn, run_id, owner_id, now, ("running",))
            sequence = int(run_row["current_step"]) + 1
            attempt_row = conn.execute(
                """
                SELECT COALESCE(MAX(attempt), 0) + 1
                FROM agent_steps WHERE run_id = ? AND sequence = ?
                """,
                (run_id, sequence),
            ).fetchone()
            step = AgentStep(
                id=f"step_{uuid.uuid4().hex}",
                run_id=run_id,
                sequence=sequence,
                kind=kind,
                status="started",
                attempt=int(attempt_row[0]),
                input=input_data,
                output=None,
                error=None,
                schema_version=SCHEMA_VERSION,
                started_at=now,
                finished_at=None,
            )
            conn.execute(
                """
                INSERT INTO agent_steps (
                    id, run_id, sequence, kind, status, attempt, input_json,
                    output_json, error_json, schema_version, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step.id, step.run_id, step.sequence, step.kind, step.status,
                    step.attempt, _encode_json(step.input), None, None,
                    step.schema_version, step.started_at, None,
                ),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return step

    def complete_step_and_checkpoint(
        self,
        *,
        step_id: str,
        run_id: str,
        owner_id: str,
        output_data: dict[str, Any],
        messages: list[dict],
        runtime_state: dict[str, Any],
        tool_completion: Mapping[str, Any] | Any | None = None,
    ) -> RunCheckpoint:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _format_utc_datetime(self.clock())
            run_row = self._owned_run_row(conn, run_id, owner_id, now, ("running",))
            step_row = conn.execute(
                """
                SELECT * FROM agent_steps
                WHERE id = ? AND run_id = ? AND status = 'started'
                """,
                (step_id, run_id),
            ).fetchone()
            if step_row is None or int(step_row["sequence"]) != int(run_row["current_step"]) + 1:
                raise ValueError(f"step is not the active sequence for run: {step_id}")
            is_tool_step = str(step_row["kind"]) == "tool"
            if is_tool_step and tool_completion is None:
                raise ValueError("tool step requires tool_completion")
            if not is_tool_step and tool_completion is not None:
                raise ValueError("non-tool step forbids tool_completion")
            if is_tool_step:
                execution_rows = conn.execute(
                    """
                    SELECT id FROM tool_executions
                    WHERE run_id = ? AND step_id = ?
                    ORDER BY id
                    """,
                    (run_id, step_id),
                ).fetchall()
                completion_id = _completion_value(tool_completion, "execution_id")
                if (
                    len(execution_rows) != 1
                    or str(execution_rows[0]["id"]) != str(completion_id)
                ):
                    raise ToolExecutionConflictError(
                        f"tool step must have one matching execution: {step_id}"
                    )
                self._complete_tool_execution_in_transaction(
                    conn, run_id=run_id, step_id=step_id,
                    completion=tool_completion, now=now,
                )
            cursor = conn.execute(
                """
                UPDATE agent_steps
                SET status = 'completed', output_json = ?, error_json = NULL,
                    finished_at = ?
                WHERE id = ? AND run_id = ? AND status = 'started'
                """,
                (_encode_json(output_data), now, step_id, run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"step is no longer started: {step_id}")
            checkpoint = RunCheckpoint(
                id=f"checkpoint_{uuid.uuid4().hex}",
                run_id=run_id,
                step_sequence=int(step_row["sequence"]),
                messages=messages,
                runtime_state=runtime_state,
                schema_version=SCHEMA_VERSION,
                created_at=now,
            )
            conn.execute(
                """
                INSERT INTO run_checkpoints (
                    id, run_id, step_sequence, messages_json, runtime_state_json,
                    schema_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.id, checkpoint.run_id, checkpoint.step_sequence,
                    _encode_json(encode_messages(checkpoint.messages)),
                    _encode_json(checkpoint.runtime_state), checkpoint.schema_version,
                    checkpoint.created_at,
                ),
            )
            cursor = conn.execute(
                """
                UPDATE agent_runs SET current_step = ?, updated_at = ?
                WHERE id = ? AND status = 'running' AND owner_id = ?
                  AND lease_expires_at > ?
                """,
                (checkpoint.step_sequence, now, run_id, owner_id, now),
            )
            if cursor.rowcount != 1:
                raise RunLeaseLostError(
                    f"run lease is not owned by {owner_id}: {run_id}"
                )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return checkpoint

    def complete_tool_execution(self, **kwargs: Any) -> RunCheckpoint:
        """Complete a tool only through the atomic step/checkpoint transaction."""
        execution_id = kwargs.pop("execution_id")
        tool_result = kwargs.pop("tool_result")
        result_preview = kwargs.pop("result_preview")
        duration_ms = kwargs.pop("duration_ms")
        return self.complete_step_and_checkpoint(
            **kwargs,
            tool_completion={
                "execution_id": execution_id,
                "tool_result": tool_result,
                "result_preview": result_preview,
                "duration_ms": duration_ms,
            },
        )

    def fail_step(
        self,
        step_id: str,
        *,
        run_id: str,
        owner_id: str,
        error_data: dict[str, Any],
    ) -> AgentStep:
        return self._finish_step(
            step_id, run_id=run_id, owner_id=owner_id,
            status="failed", error_data=error_data,
        )

    def cancel_step(self, step_id: str, *, run_id: str, owner_id: str) -> AgentStep:
        return self._finish_step(
            step_id, run_id=run_id, owner_id=owner_id,
            status="cancelled", error_data={"failure_class": "cancelled"},
            run_statuses=("running", "cancelling"),
        )

    def request_cancel(self, run_id: str) -> AgentRun | None:
        now = _format_utc_datetime(self.clock())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            if str(row["status"]) in RUN_TERMINAL_STATUSES:
                conn.commit()
                return _run_from_row(row)
            conn.execute(
                """
                UPDATE agent_runs
                SET status = 'cancelling',
                    cancel_requested_at = COALESCE(cancel_requested_at, ?),
                    updated_at = ?
                WHERE id = ?
                  AND status IN ('pending', 'running', 'waiting_retry', 'cancelling')
                """,
                (now, now, run_id),
            )
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return _run_from_row(row)

    def schedule_retry(
        self,
        run_id: str,
        *,
        owner_id: str,
        retry_at: datetime,
        failure_class: str,
        failure_message: str,
    ) -> AgentRun:
        retry_at_text = _format_utc_datetime(retry_at)
        return self._transition_owned_run(
            run_id,
            owner_id=owner_id,
            expected_statuses=("running",),
            target_status="waiting_retry",
            fields={
                "retry_at": retry_at_text,
                "failure_class": failure_class,
                "failure_message": failure_message,
                "owner_id": None,
                "lease_expires_at": None,
            },
        )

    def mark_completed(self, run_id: str, *, owner_id: str) -> AgentRun:
        return self._mark_terminal(
            run_id, owner_id=owner_id, expected_statuses=("running",),
            target_status="completed", failure_class=None, failure_message=None,
        )

    def mark_failed(
        self,
        run_id: str,
        *,
        owner_id: str,
        failure_class: str,
        failure_message: str,
    ) -> AgentRun:
        return self._mark_terminal(
            run_id, owner_id=owner_id, expected_statuses=("running",),
            target_status="failed", failure_class=failure_class,
            failure_message=failure_message,
        )

    def mark_cancelled(self, run_id: str, *, owner_id: str) -> AgentRun:
        return self._mark_terminal(
            run_id, owner_id=owner_id, expected_statuses=("cancelling",),
            target_status="cancelled", failure_class="cancelled",
            failure_message=None,
        )

    def list_steps(
        self, run_id: str, after_sequence: int = 0, limit: int = 100
    ) -> StepPage:
        _validate_limit(limit)
        with self._connect() as conn:
            sequences = [
                int(row[0])
                for row in conn.execute(
                    """
                    SELECT DISTINCT sequence FROM agent_steps
                    WHERE run_id = ? AND sequence > ?
                    ORDER BY sequence ASC LIMIT ?
                    """,
                    (run_id, after_sequence, limit + 1),
                )
            ]
            page_sequences = sequences[:limit]
            if page_sequences:
                placeholders = ", ".join("?" for _ in page_sequences)
                rows = conn.execute(
                    f"""
                    SELECT * FROM agent_steps
                    WHERE run_id = ? AND sequence IN ({placeholders})
                    ORDER BY sequence ASC, attempt ASC
                    """,
                    (run_id, *page_sequences),
                ).fetchall()
            else:
                rows = []
        has_more = len(sequences) > limit
        return StepPage(
            items=tuple(_step_from_row(row) for row in rows),
            next_after_sequence=page_sequences[-1] if has_more else None,
            has_more=has_more,
        )

    def list_pending_runs(self, limit: int = 100) -> list[AgentRun]:
        _validate_limit(limit)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM agent_runs WHERE status = 'pending'
                ORDER BY created_at ASC, rowid ASC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_run_from_row(row) for row in rows]

    def list_cancelling_runs(self, limit: int = 100) -> list[AgentRun]:
        _validate_limit(limit)
        now = _format_utc_datetime(self.clock())
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM agent_runs
                WHERE status = 'cancelling'
                  AND (owner_id IS NULL OR lease_expires_at IS NULL OR lease_expires_at <= ?)
                ORDER BY updated_at ASC, rowid ASC LIMIT ?
                """,
                (now, limit),
            ).fetchall()
        return [_run_from_row(row) for row in rows]

    def recover_due_runs(self, *, now: datetime, limit: int = 100) -> list[AgentRun]:
        _validate_limit(limit)
        now_text = _format_utc_datetime(now)
        conn = self._connect()
        changed: list[AgentRun] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM agent_runs
                WHERE (status = 'running' AND lease_expires_at <= ?)
                   OR (status = 'waiting_retry' AND retry_at <= ?)
                ORDER BY updated_at ASC, rowid ASC LIMIT ?
                """,
                (now_text, now_text, limit),
            ).fetchall()
            for row in rows:
                old_status = str(row["status"])
                target = "cancelling" if row["cancel_requested_at"] else "pending"
                cursor = conn.execute(
                    """
                    UPDATE agent_runs
                    SET status = ?, attempt = attempt + ?, owner_id = NULL,
                        lease_expires_at = NULL, retry_at = NULL, updated_at = ?
                    WHERE id = ? AND status = ?
                      AND ((? = 'running' AND lease_expires_at <= ?)
                           OR (? = 'waiting_retry' AND retry_at <= ?))
                    """,
                    (
                        target, 1 if old_status == "waiting_retry" else 0,
                        now_text, row["id"], old_status, old_status, now_text,
                        old_status, now_text,
                    ),
                )
                if cursor.rowcount == 1:
                    updated = conn.execute(
                        "SELECT * FROM agent_runs WHERE id = ?", (row["id"],)
                    ).fetchone()
                    changed.append(_run_from_row(updated))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return changed

    def start_tool_execution(
        self,
        *,
        run_id: str,
        step_id: str,
        owner_id: str,
        tool_use_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        classification: str,
    ) -> ToolExecutionStart:
        execution_id = stable_tool_execution_id(run_id, tool_use_id)
        arguments_json = _encode_json(arguments)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _format_utc_datetime(self.clock())
            self._require_current_started_tool_step(
                conn, run_id=run_id, step_id=step_id,
                owner_id=owner_id, now=now,
            )
            row = conn.execute(
                "SELECT * FROM tool_executions WHERE id = ?", (execution_id,)
            ).fetchone()
            step_execution = conn.execute(
                "SELECT id FROM tool_executions WHERE run_id = ? AND step_id = ?",
                (run_id, step_id),
            ).fetchone()
            if step_execution is not None and str(step_execution["id"]) != execution_id:
                raise ToolExecutionConflictError(
                    f"tool step already has an execution: {step_id}"
                )
            if row is not None:
                if (
                    str(row["run_id"]) != run_id
                    or str(row["tool_name"]) != tool_name
                    or str(row["arguments_json"]) != arguments_json
                    or str(row["classification"]) != classification
                ):
                    raise ToolExecutionConflictError(
                        f"tool execution ID reused with conflicting data: {execution_id}"
                    )
                current = _tool_execution_from_row(row)
                if current.status == "completed":
                    result = ToolExecutionStart(current, "completed")
                elif current.step_id == step_id:
                    disposition: ToolExecutionStartDisposition = (
                        "in_progress" if current.status == "started" else "failed"
                    )
                    result = ToolExecutionStart(current, disposition)
                elif current.classification == "non_retryable":
                    result = ToolExecutionStart(current, "blocked")
                else:
                    conn.execute(
                        """
                        UPDATE tool_executions
                        SET step_id = ?, status = 'started', result_preview = NULL,
                            failure_class = NULL, failure_message = NULL,
                            duration_ms = NULL, started_at = ?, finished_at = NULL
                        WHERE id = ? AND status IN ('started', 'failed')
                        """,
                        (step_id, now, execution_id),
                    )
                    row = conn.execute(
                        "SELECT * FROM tool_executions WHERE id = ?", (execution_id,)
                    ).fetchone()
                    result = ToolExecutionStart(_tool_execution_from_row(row), "retry")
            else:
                conn.execute(
                    """
                    INSERT INTO tool_executions (
                        id, run_id, step_id, tool_name, arguments_json,
                        classification, status, result_preview, failure_class,
                        failure_message, duration_ms, schema_version, started_at,
                        finished_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'started', NULL, NULL, NULL, NULL, ?, ?, NULL)
                    """,
                    (
                        execution_id, run_id, step_id, tool_name, arguments_json,
                        classification, SCHEMA_VERSION, now,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM tool_executions WHERE id = ?", (execution_id,)
                ).fetchone()
                result = ToolExecutionStart(_tool_execution_from_row(row), "new")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return result

    def get_tool_execution(self, execution_id: str) -> ToolExecution | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tool_executions WHERE id = ?", (execution_id,)
            ).fetchone()
        return _tool_execution_from_row(row) if row is not None else None

    def fail_tool_execution(
        self,
        execution_id: str,
        *,
        run_id: str,
        step_id: str,
        owner_id: str,
        failure_class: str,
        failure_message: str,
        duration_ms: int,
    ) -> ToolExecution:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _format_utc_datetime(self.clock())
            self._require_current_started_tool_step(
                conn, run_id=run_id, step_id=step_id,
                owner_id=owner_id, now=now,
            )
            row = conn.execute(
                """
                SELECT * FROM tool_executions
                WHERE id = ? AND run_id = ? AND step_id = ?
                """,
                (execution_id, run_id, step_id),
            ).fetchone()
            if row is None:
                raise ToolExecutionConflictError(
                    f"tool execution does not belong to current step: {execution_id}"
                )
            if str(row["status"]) == "failed":
                if (
                    row["failure_class"] == failure_class
                    and row["failure_message"] == failure_message
                    and int(row["duration_ms"]) == duration_ms
                ):
                    conn.commit()
                    return _tool_execution_from_row(row)
                raise ToolExecutionConflictError(
                    f"tool failure conflicts with persisted result: {execution_id}"
                )
            if str(row["status"]) != "started":
                raise ToolExecutionConflictError(
                    f"completed tool execution is immutable: {execution_id}"
                )
            cursor = conn.execute(
                """
                UPDATE tool_executions
                SET status = 'failed', failure_class = ?, failure_message = ?,
                    duration_ms = ?, finished_at = ?
                WHERE id = ? AND run_id = ? AND step_id = ? AND status = 'started'
                """,
                (
                    failure_class, failure_message, duration_ms, now,
                    execution_id, run_id, step_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ToolExecutionConflictError(
                    f"tool execution failure raced: {execution_id}"
                )
            row = conn.execute(
                "SELECT * FROM tool_executions WHERE id = ?", (execution_id,)
            ).fetchone()
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return _tool_execution_from_row(row)

    def _owned_run_row(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        owner_id: str,
        now: str,
        statuses: tuple[str, ...],
    ) -> sqlite3.Row:
        placeholders = ", ".join("?" for _ in statuses)
        row = conn.execute(
            f"""
            SELECT * FROM agent_runs
            WHERE id = ? AND status IN ({placeholders}) AND owner_id = ?
              AND lease_expires_at > ?
            """,
            (run_id, *statuses, owner_id, now),
        ).fetchone()
        if row is None:
            raise RunLeaseLostError(f"run lease is not owned by {owner_id}: {run_id}")
        return row

    def _finish_step(
        self,
        step_id: str,
        *,
        run_id: str,
        owner_id: str,
        status: str,
        error_data: dict[str, Any],
        run_statuses: tuple[str, ...] = ("running",),
    ) -> AgentStep:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _format_utc_datetime(self.clock())
            self._owned_run_row(conn, run_id, owner_id, now, run_statuses)
            cursor = conn.execute(
                """
                UPDATE agent_steps SET status = ?, error_json = ?, finished_at = ?
                WHERE id = ? AND run_id = ? AND status = 'started'
                """,
                (status, _encode_json(error_data), now, step_id, run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"step is no longer started: {step_id}")
            row = conn.execute(
                "SELECT * FROM agent_steps WHERE id = ?", (step_id,)
            ).fetchone()
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return _step_from_row(row)

    def _transition_owned_run(
        self,
        run_id: str,
        *,
        owner_id: str,
        expected_statuses: tuple[str, ...],
        target_status: str,
        fields: dict[str, Any],
    ) -> AgentRun:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            now = _format_utc_datetime(self.clock())
            self._owned_run_row(conn, run_id, owner_id, now, expected_statuses)
            assignments = ["status = ?", "updated_at = ?"]
            values: list[Any] = [target_status, now]
            for key, value in fields.items():
                if key not in {
                    "retry_at", "failure_class", "failure_message", "owner_id",
                    "lease_expires_at", "finished_at",
                }:
                    raise ValueError(f"unsupported run transition field: {key}")
                assignments.append(f"{key} = ?")
                values.append(now if value is _LOCKED_NOW else value)
            placeholders = ", ".join("?" for _ in expected_statuses)
            cursor = conn.execute(
                f"""
                UPDATE agent_runs SET {', '.join(assignments)}
                WHERE id = ? AND status IN ({placeholders}) AND owner_id = ?
                  AND lease_expires_at > ?
                """,
                (*values, run_id, *expected_statuses, owner_id, now),
            )
            if cursor.rowcount != 1:
                raise RunLeaseLostError(
                    f"run lease is not owned by {owner_id}: {run_id}"
                )
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return _run_from_row(row)

    def _mark_terminal(
        self,
        run_id: str,
        *,
        owner_id: str,
        expected_statuses: tuple[str, ...],
        target_status: str,
        failure_class: str | None,
        failure_message: str | None,
    ) -> AgentRun:
        return self._transition_owned_run(
            run_id,
            owner_id=owner_id,
            expected_statuses=expected_statuses,
            target_status=target_status,
            fields={
                "failure_class": failure_class,
                "failure_message": failure_message,
                "retry_at": None,
                "owner_id": None,
                "lease_expires_at": None,
                "finished_at": _LOCKED_NOW,
            },
        )

    def _require_current_started_tool_step(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        step_id: str,
        owner_id: str,
        now: str,
    ) -> sqlite3.Row:
        run_row = self._owned_run_row(
            conn, run_id, owner_id, now, ("running",)
        )
        sequence = int(run_row["current_step"]) + 1
        row = conn.execute(
            """
            SELECT * FROM agent_steps
            WHERE id = ? AND run_id = ? AND sequence = ?
              AND kind = 'tool' AND status = 'started'
              AND attempt = (
                  SELECT MAX(attempt) FROM agent_steps
                  WHERE run_id = ? AND sequence = ?
              )
            """,
            (step_id, run_id, sequence, run_id, sequence),
        ).fetchone()
        if row is None:
            raise ToolExecutionConflictError(
                f"tool execution requires current highest started attempt: {step_id}"
            )
        return row

    def _complete_tool_execution_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        step_id: str,
        completion: Mapping[str, Any] | Any,
        now: str,
    ) -> None:
        execution_id = _completion_value(completion, "execution_id")
        tool_result = _completion_value(completion, "tool_result")
        result_preview = _completion_value(completion, "result_preview")
        duration_ms = _completion_value(completion, "duration_ms")
        tool_result_id = _completion_value(tool_result, "id")
        if stable_tool_execution_id(run_id, str(tool_result_id)) != execution_id:
            raise ToolExecutionConflictError(
                f"tool result ID does not match execution: {execution_id}"
            )
        if bool(getattr(tool_result, "is_error", False)):
            raise ToolExecutionConflictError(
                f"error tool result cannot be committed as success: {execution_id}"
            )
        row = conn.execute(
            "SELECT * FROM tool_executions WHERE id = ?", (execution_id,)
        ).fetchone()
        if (
            row is None
            or str(row["run_id"]) != run_id
            or str(row["step_id"]) != step_id
            or str(row["status"]) != "started"
        ):
            raise ToolExecutionConflictError(
                f"tool execution is not started for active step: {execution_id}"
            )
        cursor = conn.execute(
            """
            UPDATE tool_executions
            SET status = 'completed', result_preview = ?, duration_ms = ?,
                failure_class = NULL, failure_message = NULL, finished_at = ?
            WHERE id = ? AND run_id = ? AND step_id = ? AND status = 'started'
            """,
            (result_preview, duration_ms, now, execution_id, run_id, step_id),
        )
        if cursor.rowcount != 1:
            raise ToolExecutionConflictError(
                f"tool execution completion raced: {execution_id}"
            )

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


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_tool_execution_id(run_id: str, tool_use_id: str) -> str:
    digest = hashlib.sha256(f"{run_id}:{tool_use_id}".encode("utf-8")).hexdigest()
    return f"tool_{digest[:32]}"


def _format_utc_datetime(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("clock must return a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_utc_datetime(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )


def _validate_owner_and_lease(owner_id: str, lease_seconds: int) -> None:
    if not owner_id:
        raise ValueError("owner_id must not be empty")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")


def _validate_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError("limit must be positive")


def _completion_value(completion: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(completion, Mapping):
        try:
            return completion[name]
        except KeyError as exc:
            raise ValueError(f"tool completion is missing {name}") from exc
    try:
        return getattr(completion, name)
    except AttributeError as exc:
        raise ValueError(f"tool completion is missing {name}") from exc


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


def _step_from_row(row: sqlite3.Row) -> AgentStep:
    return AgentStep(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        sequence=int(row["sequence"]),
        kind=str(row["kind"]),
        status=str(row["status"]),
        attempt=int(row["attempt"]),
        input=json.loads(str(row["input_json"])),
        output=(
            json.loads(str(row["output_json"]))
            if row["output_json"] is not None
            else None
        ),
        error=(
            json.loads(str(row["error_json"]))
            if row["error_json"] is not None
            else None
        ),
        schema_version=int(row["schema_version"]),
        started_at=str(row["started_at"]),
        finished_at=row["finished_at"],
    )


def _tool_execution_from_row(row: sqlite3.Row) -> ToolExecution:
    return ToolExecution(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        step_id=str(row["step_id"]),
        tool_name=str(row["tool_name"]),
        arguments=json.loads(str(row["arguments_json"])),
        classification=str(row["classification"]),
        status=str(row["status"]),
        result_preview=row["result_preview"],
        failure_class=row["failure_class"],
        failure_message=row["failure_message"],
        duration_ms=(
            int(row["duration_ms"]) if row["duration_ms"] is not None else None
        ),
        schema_version=int(row["schema_version"]),
        started_at=str(row["started_at"]),
        finished_at=row["finished_at"],
    )
