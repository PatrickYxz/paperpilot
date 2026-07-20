import sqlite3

import pytest

from paperpilot.agent.policy import RunPolicy
from paperpilot.agent.store import ActiveRunExistsError, SQLiteRunStore
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


def test_only_one_active_run_is_allowed_per_task_and_latest_includes_terminal_runs(tmp_path):
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

    with pytest.raises(ActiveRunExistsError):
        store.create_run(**arguments)

    with store._connect() as conn:
        conn.execute("UPDATE agent_runs SET status = 'completed' WHERE id = ?", (run.id,))

    assert store.get_active_run_for_task(task.id) is None
    assert store.get_latest_run_for_task(task.id).id == run.id


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
            row[1]
            for row in conn.execute("PRAGMA index_list(agent_runs)")
        }
        assert "idx_agent_runs_one_active_task" in indexes
        assert "idx_agent_runs_task_created" in indexes
        assert "idx_agent_runs_status_retry" in indexes
        assert "idx_agent_steps_run_sequence" in {
            row[1] for row in conn.execute("PRAGMA index_list(agent_steps)")
        }
        assert "idx_run_checkpoints_run_sequence" in {
            row[1] for row in conn.execute("PRAGMA index_list(run_checkpoints)")
        }
        assert conn.execute("PRAGMA foreign_key_list(agent_runs)").fetchone()[2] == "research_tasks"


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
