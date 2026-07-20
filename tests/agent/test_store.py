import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from paperpilot.agent.policy import RunPolicy
from paperpilot.agent.store import ActiveRunExistsError, SQLiteRunStore, utc_now
from paperpilot.web.task_store import TaskStore


def test_create_run_persists_policy_and_initial_checkpoint_for_owned_task(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    task_store = TaskStore(db_path)
    user = task_store.create_user(
        username="alice",
        password_hash="hash",
        password_salt="salt",
    )
    task = task_store.create_task(
        question="Compare two papers",
        user_id=user.id,
    )
    store = SQLiteRunStore(db_path)

    run = store.create_run(
        task_id=task.id,
        policy=RunPolicy.for_depth("standard"),
        initial_messages=[{"role": "user", "content": task.question}],
        runtime_state={"turn_count": 0, "tokens_used": 0},
    )

    checkpoint = store.get_latest_checkpoint(run.id)
    assert task_store.get_task(task.id, user_id=user.id) == task
    assert run.status == "pending"
    assert run.current_step == 0
    assert run.policy == RunPolicy.for_depth("standard")
    assert checkpoint is not None
    assert checkpoint.step_sequence == 0
    assert checkpoint.messages[0]["content"] == task.question
    assert checkpoint.runtime_state == {"turn_count": 0, "tokens_used": 0}
    assert store.get_run(run.id) == run
    assert store.get_active_run_for_task(task.id) == run
    assert store.get_latest_run_for_task(task.id) == run


def test_active_run_unique_constraint_is_translated_but_other_integrity_errors_remain(tmp_path, monkeypatch):
    db_path = tmp_path / "tasks.sqlite3"
    task = TaskStore(db_path).create_task(question="Question")
    store = SQLiteRunStore(db_path)
    arguments = {
        "task_id": task.id,
        "policy": RunPolicy.for_depth("quick"),
        "initial_messages": [{"role": "user", "content": "Question"}],
        "runtime_state": {"turn_count": 0, "tokens_used": 0},
    }
    run = store.create_run(**arguments)

    with pytest.raises(ActiveRunExistsError) as active_error:
        store.create_run(**arguments)
    assert active_error.value.__cause__.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE

    with pytest.raises(sqlite3.IntegrityError) as not_null_error:
        store.create_run(**(arguments | {"task_id": None}))
    assert not_null_error.value.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_NOTNULL

    with pytest.raises(sqlite3.IntegrityError) as foreign_key_error:
        store.create_run(**(arguments | {"task_id": "task_missing"}))
    assert foreign_key_error.value.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_FOREIGNKEY

    other_task = TaskStore(db_path).create_task(question="Other")
    third_task = TaskStore(db_path).create_task(question="Third")
    original_uuid4 = iter(("same-run", "same-checkpoint"))
    monkeypatch.setattr(
        "paperpilot.agent.store.uuid.uuid4",
        lambda: SimpleNamespace(hex=next(original_uuid4)),
    )
    store.create_run(**(arguments | {"task_id": other_task.id}))
    monkeypatch.setattr(
        "paperpilot.agent.store.uuid.uuid4",
        lambda: SimpleNamespace(hex="same-run"),
    )
    with pytest.raises(sqlite3.IntegrityError) as primary_key_error:
        store.create_run(**(arguments | {"task_id": third_task.id}))
    assert primary_key_error.value.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY


def test_latest_run_uses_rowid_when_fixed_clock_ties(tmp_path, monkeypatch):
    db_path = tmp_path / "tasks.sqlite3"
    task = TaskStore(db_path).create_task(question="Question")
    fixed_now = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    store = SQLiteRunStore(db_path, clock=lambda: fixed_now)
    identifiers = iter(("z", "checkpoint-z", "a", "checkpoint-a"))
    monkeypatch.setattr(
        "paperpilot.agent.store.uuid.uuid4",
        lambda: SimpleNamespace(hex=next(identifiers)),
    )
    arguments = {
        "task_id": task.id,
        "policy": RunPolicy.for_depth("quick"),
        "initial_messages": [{"role": "user", "content": "Question"}],
        "runtime_state": {"turn_count": 0, "tokens_used": 0},
    }
    old_run = store.create_run(**arguments)

    with store._connect() as conn:
        conn.execute("UPDATE agent_runs SET status = 'completed' WHERE id = ?", (old_run.id,))
    new_run = store.create_run(**arguments)

    assert store.get_active_run_for_task(task.id).id == new_run.id
    assert store.get_latest_run_for_task(task.id).id == new_run.id


def test_clock_requires_aware_datetimes_and_persists_lexically_ordered_utc_strings(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    task = TaskStore(db_path).create_task(question="Question")
    first = datetime(2026, 7, 20, 12, 0, tzinfo=timezone(timedelta(hours=8)))
    store = SQLiteRunStore(db_path, clock=lambda: first)
    run = store.create_run(
        task_id=task.id,
        policy=RunPolicy.for_depth("quick"),
        initial_messages=[{"role": "user", "content": "Question"}],
        runtime_state={"turn_count": 0, "tokens_used": 0},
    )

    assert utc_now().tzinfo is not None
    assert run.created_at == "2026-07-20T04:00:00.000000Z"
    assert run.created_at < "2026-07-20T04:00:01.000000Z"
    naive_store = SQLiteRunStore(db_path, clock=lambda: datetime(2026, 7, 20, 12, 0))
    with pytest.raises(ValueError, match="timezone-aware"):
        naive_store.create_run(
            task_id=task.id,
            policy=RunPolicy.for_depth("quick"),
            initial_messages=[{"role": "user", "content": "Question"}],
            runtime_state={"turn_count": 0, "tokens_used": 0},
        )


def test_schema_has_required_tables_columns_indexes_and_foreign_keys(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    TaskStore(db_path)
    store = SQLiteRunStore(db_path)

    with store._connect() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"agent_runs", "agent_steps", "run_checkpoints", "tool_executions"} <= tables
        columns_by_table = {
            table: {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for table in ("agent_runs", "agent_steps", "run_checkpoints", "tool_executions")
        }
        assert columns_by_table["agent_runs"] == {
            "id", "task_id", "status", "attempt", "current_step", "cancel_requested_at",
            "retry_at", "failure_class", "failure_message", "policy_json", "owner_id",
            "lease_expires_at", "schema_version", "created_at", "updated_at", "started_at",
            "finished_at",
        }
        assert columns_by_table["agent_steps"] == {
            "id", "run_id", "sequence", "kind", "status", "attempt", "input_json",
            "output_json", "error_json", "schema_version", "started_at", "finished_at",
        }
        assert columns_by_table["run_checkpoints"] == {
            "id", "run_id", "step_sequence", "messages_json", "runtime_state_json",
            "schema_version", "created_at",
        }
        assert columns_by_table["tool_executions"] == {
            "id", "run_id", "step_id", "tool_name", "arguments_json", "classification",
            "status", "result_preview", "failure_class", "failure_message", "duration_ms",
            "schema_version", "started_at", "finished_at",
        }
        indexes = {
            row[1]: row
            for row in conn.execute("PRAGMA index_list(agent_runs)")
        }
        active_index = indexes["idx_agent_runs_one_active_task"]
        assert active_index[2] == 1
        assert active_index[4] == 1
        assert [row[2] for row in conn.execute("PRAGMA index_info(idx_agent_runs_one_active_task)")] == ["task_id"]
        active_index_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            ("idx_agent_runs_one_active_task",),
        ).fetchone()[0]
        assert "WHERE status IN ('pending', 'running', 'waiting_retry', 'cancelling')" in active_index_sql
        assert "idx_agent_runs_task_created" in indexes
        assert "idx_agent_runs_status_retry" in indexes
        assert "idx_agent_steps_run_sequence" in {
            row[1] for row in conn.execute("PRAGMA index_list(agent_steps)")
        }
        assert "idx_run_checkpoints_run_sequence" in {
            row[1] for row in conn.execute("PRAGMA index_list(run_checkpoints)")
        }
        assert _index_key_columns(conn, "idx_agent_runs_task_created") == [
            ("task_id", 0),
            ("created_at", 1),
        ]
        assert _index_key_columns(conn, "idx_agent_runs_status_retry") == [
            ("status", 0),
            ("retry_at", 0),
        ]
        assert _index_key_columns(conn, "idx_agent_steps_run_sequence") == [
            ("run_id", 0),
            ("sequence", 0),
        ]
        assert _index_key_columns(conn, "idx_run_checkpoints_run_sequence") == [
            ("run_id", 0),
            ("step_sequence", 0),
        ]
        foreign_keys = {
            table: {
                (row[3], row[2])
                for row in conn.execute(f"PRAGMA foreign_key_list({table})")
            }
            for table in ("agent_runs", "agent_steps", "run_checkpoints", "tool_executions")
        }
        assert foreign_keys == {
            "agent_runs": {("task_id", "research_tasks")},
            "agent_steps": {("run_id", "agent_runs")},
            "run_checkpoints": {("run_id", "agent_runs")},
            "tool_executions": {
                ("run_id", "agent_runs"),
                ("step_id", "agent_steps"),
            },
        }


def test_create_run_requires_a_real_research_task_and_health_probe_passes(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    TaskStore(db_path)
    store = SQLiteRunStore(db_path)

    with pytest.raises(sqlite3.IntegrityError):
        store.create_run(
            task_id="task_missing",
            policy=RunPolicy.for_depth("quick"),
            initial_messages=[{"role": "user", "content": "Question"}],
            runtime_state={"turn_count": 0, "tokens_used": 0},
        )

    assert store.check_health() is None


def _index_key_columns(conn, index_name):
    return [
        (str(row["name"]), int(row["desc"]))
        for row in conn.execute(f"PRAGMA index_xinfo({index_name})")
        if int(row["key"]) == 1
    ]
