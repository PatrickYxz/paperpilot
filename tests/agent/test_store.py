import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from paperpilot.agent.policy import RunPolicy
from paperpilot.agent import store as agent_store
from paperpilot.agent.store import ActiveRunExistsError, SQLiteRunStore, utc_now
from paperpilot.core.adapter import ToolResult
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


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 20, 8, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.value

    def advance(self, *, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


@pytest.fixture
def clock():
    return MutableClock()


@pytest.fixture
def run_store(tmp_path, clock):
    db_path = tmp_path / "tasks.sqlite3"
    task = TaskStore(db_path).create_task(question="Question")
    store = SQLiteRunStore(db_path, clock=clock.now)
    run = store.create_run(
        task_id=task.id,
        policy=RunPolicy.for_depth("standard"),
        initial_messages=[{"role": "user", "content": task.question}],
        runtime_state={"turn_count": 0, "tokens_used": 0},
    )
    return store, run


def test_only_current_unexpired_lease_owner_can_start_and_commit_step(run_store, clock):
    store, run = run_store
    claimed = store.claim_run(run.id, owner_id="owner-a", lease_seconds=30)

    assert claimed is not None
    assert claimed.status == "running"
    assert claimed.owner_id == "owner-a"
    assert store.claim_run(run.id, owner_id="owner-b", lease_seconds=30) is None
    step = store.start_step(
        run_id=run.id,
        owner_id="owner-a",
        kind="llm",
        input_data={"message_count": 1},
    )
    with pytest.raises(agent_store.RunLeaseLostError):
        store.complete_step_and_checkpoint(
            step_id=step.id,
            run_id=run.id,
            owner_id="owner-b",
            output_data={"tool_calls": []},
            messages=[{"role": "assistant", "content": "answer"}],
            runtime_state={"turn_count": 1, "tokens_used": 10},
        )

    clock.advance(seconds=31)
    assert store.has_valid_lease(run.id, "owner-a") is False
    with pytest.raises(agent_store.RunLeaseLostError):
        store.start_step(
            run_id=run.id,
            owner_id="owner-a",
            kind="llm",
            input_data={},
        )


def test_lease_renewal_and_expired_reclaim_are_owner_safe(run_store, clock):
    store, run = run_store
    assert store.claim_run(run.id, owner_id="owner-a", lease_seconds=10) is not None
    clock.advance(seconds=5)
    assert store.renew_lease(run.id, owner_id="owner-b", lease_seconds=30) is False
    assert store.renew_lease(run.id, owner_id="owner-a", lease_seconds=30) is True
    clock.advance(seconds=29)
    assert store.claim_run(run.id, owner_id="owner-b", lease_seconds=30) is None
    clock.advance(seconds=2)

    reclaimed = store.claim_run(run.id, owner_id="owner-b", lease_seconds=30)

    assert reclaimed is not None
    assert reclaimed.owner_id == "owner-b"
    assert store.has_valid_lease(run.id, "owner-a") is False


def test_completed_step_checkpoint_and_tool_success_commit_atomically(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    step = store.start_step(
        run_id=run.id,
        owner_id="worker",
        kind="tool",
        input_data={"tool_name": "mcp__colbert__search"},
    )
    execution = store.start_tool_execution(
        run_id=run.id,
        step_id=step.id,
        owner_id="worker",
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"query": "attention"},
        classification="read_only",
    )
    store.complete_step_and_checkpoint(
        step_id=step.id,
        run_id=run.id,
        owner_id="worker",
        output_data={"content": "full result"},
        messages=[{"role": "user", "content": "full result"}],
        runtime_state={"turn_count": 1, "tokens_used": 10},
        tool_completion={
            "execution_id": execution.id,
            "tool_result": ToolResult(id="call-1", content="full result"),
            "result_preview": "full result",
            "duration_ms": 12,
        },
    )

    stored_run = store.get_run(run.id)
    stored_step = store.list_steps(run.id, after_sequence=0, limit=10).items[0]
    stored_execution = store.get_tool_execution(execution.id)
    assert stored_run.current_step == 1
    assert store.get_latest_checkpoint(run.id).step_sequence == 1
    assert stored_step.status == "completed"
    assert stored_execution.status == "completed"
    assert stored_execution.result_preview == "full result"
    assert stored_execution.duration_ms == 12


def test_tool_completion_conflict_rolls_back_step_checkpoint_and_run(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    step = store.start_step(
        run_id=run.id, owner_id="worker", kind="tool", input_data={}
    )
    execution = store.start_tool_execution(
        run_id=run.id,
        step_id=step.id,
        owner_id="worker",
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"query": "attention"},
        classification="read_only",
    )
    store.fail_tool_execution(
        execution.id,
        run_id=run.id,
        step_id=step.id,
        owner_id="worker",
        failure_class="transport",
        failure_message="closed",
        duration_ms=4,
    )

    with pytest.raises(agent_store.ToolExecutionConflictError):
        store.complete_step_and_checkpoint(
            step_id=step.id,
            run_id=run.id,
            owner_id="worker",
            output_data={"content": "late success"},
            messages=[{"role": "user", "content": "late success"}],
            runtime_state={"turn_count": 1},
            tool_completion={
                "execution_id": execution.id,
                "tool_result": ToolResult(id="call-1", content="late success"),
                "result_preview": "late success",
                "duration_ms": 5,
            },
        )

    assert store.get_run(run.id).current_step == 0
    assert store.get_latest_checkpoint(run.id).step_sequence == 0
    assert store.list_steps(run.id, after_sequence=0, limit=10).items[0].status == "started"


def test_checkpoint_write_failure_rolls_back_tool_step_and_run(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    step = store.start_step(
        run_id=run.id, owner_id="worker", kind="tool", input_data={}
    )
    execution = store.start_tool_execution(
        run_id=run.id,
        step_id=step.id,
        owner_id="worker",
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"query": "attention"},
        classification="read_only",
    )
    with store._connect() as conn:
        conn.execute(
            """
            CREATE TRIGGER reject_first_runtime_checkpoint
            BEFORE INSERT ON run_checkpoints
            WHEN NEW.step_sequence = 1
            BEGIN
                SELECT RAISE(ABORT, 'injected checkpoint failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected checkpoint failure"):
        store.complete_step_and_checkpoint(
            step_id=step.id,
            run_id=run.id,
            owner_id="worker",
            output_data={"content": "result"},
            messages=[{"role": "user", "content": "result"}],
            runtime_state={"turn_count": 1},
            tool_completion={
                "execution_id": execution.id,
                "tool_result": ToolResult(id="call-1", content="result"),
                "result_preview": "result",
                "duration_ms": 5,
            },
        )

    assert store.get_run(run.id).current_step == 0
    assert store.get_latest_checkpoint(run.id).step_sequence == 0
    assert store.list_steps(run.id, 0, 10).items[0].status == "started"
    assert store.get_tool_execution(execution.id).status == "started"


def test_tool_completion_rejects_mismatched_tool_result_id_atomically(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    step = store.start_step(
        run_id=run.id, owner_id="worker", kind="tool", input_data={}
    )
    execution = store.start_tool_execution(
        run_id=run.id,
        step_id=step.id,
        owner_id="worker",
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"query": "attention"},
        classification="read_only",
    )

    with pytest.raises(agent_store.ToolExecutionConflictError):
        store.complete_step_and_checkpoint(
            step_id=step.id,
            run_id=run.id,
            owner_id="worker",
            output_data={"content": "wrong result"},
            messages=[{"role": "user", "content": "wrong result"}],
            runtime_state={"turn_count": 1},
            tool_completion={
                "execution_id": execution.id,
                "tool_result": ToolResult(id="call-other", content="wrong result"),
                "result_preview": "wrong result",
                "duration_ms": 5,
            },
        )

    assert store.get_run(run.id).current_step == 0
    assert store.list_steps(run.id, 0, 10).items[0].status == "started"
    assert store.get_tool_execution(execution.id).status == "started"


def test_step_retry_uses_same_sequence_and_increments_attempt(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    first = store.start_step(
        run_id=run.id, owner_id="worker", kind="llm", input_data={}
    )
    store.fail_step(
        first.id,
        run_id=run.id,
        owner_id="worker",
        error_data={"failure_class": "transport", "message": "closed"},
    )
    second = store.start_step(
        run_id=run.id, owner_id="worker", kind="llm", input_data={}
    )

    assert (first.sequence, first.attempt) == (1, 1)
    assert (second.sequence, second.attempt) == (1, 2)


def test_cancel_request_is_idempotent_and_terminal_runs_do_not_change(run_store, clock):
    store, run = run_store
    first = store.request_cancel(run.id)
    clock.advance(seconds=5)
    second = store.request_cancel(run.id)

    assert first.status == "cancelling"
    assert second.cancel_requested_at == first.cancel_requested_at
    claimed = store.claim_run(run.id, owner_id="canceller", lease_seconds=30)
    assert claimed is not None
    cancelled = store.mark_cancelled(run.id, owner_id="canceller")
    assert cancelled.status == "cancelled"
    assert store.request_cancel(run.id) == cancelled


def test_retry_recovery_is_due_once_and_increments_run_attempt(run_store, clock):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    retry_at = clock.now() + timedelta(seconds=10)
    waiting = store.schedule_retry(
        run.id,
        owner_id="worker",
        retry_at=retry_at,
        failure_class="transport",
        failure_message="closed",
    )
    assert waiting.status == "waiting_retry"
    assert waiting.attempt == 1
    assert store.recover_due_runs(now=clock.now(), limit=10) == []
    clock.advance(seconds=10)

    recovered = store.recover_due_runs(now=clock.now(), limit=10)

    assert [item.id for item in recovered] == [run.id]
    assert recovered[0].status == "pending"
    assert recovered[0].attempt == 2
    assert recovered[0].retry_at is None
    assert store.recover_due_runs(now=clock.now(), limit=10) == []


def test_expired_running_recovery_releases_owner_once(run_store, clock):
    store, run = run_store
    store.claim_run(run.id, owner_id="dead", lease_seconds=10)
    clock.advance(seconds=11)

    recovered = store.recover_due_runs(now=clock.now(), limit=10)

    assert [item.id for item in recovered] == [run.id]
    assert recovered[0].status == "pending"
    assert recovered[0].owner_id is None
    assert recovered[0].attempt == 1
    assert store.recover_due_runs(now=clock.now(), limit=10) == []


def test_recovery_routes_cancel_requested_run_to_cancelling(run_store, clock):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=10)
    store.request_cancel(run.id)
    clock.advance(seconds=11)

    listed = store.list_cancelling_runs(limit=10)
    claimed = store.claim_run(run.id, owner_id="canceller", lease_seconds=30)

    assert [item.id for item in listed] == [run.id]
    assert claimed is not None
    assert claimed.status == "cancelling"
    assert claimed.owner_id == "canceller"


def test_pending_and_cancelling_reconciliation_lists_are_bounded_oldest_first(tmp_path, clock):
    db_path = tmp_path / "tasks.sqlite3"
    task_store = TaskStore(db_path)
    store = SQLiteRunStore(db_path, clock=clock.now)
    runs = []
    for index in range(3):
        task = task_store.create_task(question=f"Question {index}")
        runs.append(
            store.create_run(
                task_id=task.id,
                policy=RunPolicy.for_depth("quick"),
                initial_messages=[{"role": "user", "content": task.question}],
                runtime_state={},
            )
        )
        clock.advance(seconds=1)

    assert [item.id for item in store.list_pending_runs(limit=2)] == [
        runs[0].id,
        runs[1].id,
    ]
    store.request_cancel(runs[0].id)
    store.request_cancel(runs[1].id)
    assert [item.id for item in store.list_cancelling_runs(limit=1)] == [runs[0].id]


def test_step_paging_uses_sequence_cursor_and_reports_more(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    for sequence in range(1, 4):
        step = store.start_step(
            run_id=run.id, owner_id="worker", kind="llm", input_data={}
        )
        store.complete_step_and_checkpoint(
            step_id=step.id,
            run_id=run.id,
            owner_id="worker",
            output_data={"sequence": sequence},
            messages=[{"role": "assistant", "content": str(sequence)}],
            runtime_state={"turn_count": sequence},
        )

    first = store.list_steps(run.id, after_sequence=0, limit=2)
    second = store.list_steps(run.id, after_sequence=first.next_after_sequence, limit=2)
    assert [step.sequence for step in first.items] == [1, 2]
    assert first.has_more is True
    assert first.next_after_sequence == 2
    assert [step.sequence for step in second.items] == [3]
    assert second.has_more is False
    assert second.next_after_sequence is None


def test_tool_execution_id_arguments_and_completed_record_are_stable(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    step = store.start_step(
        run_id=run.id, owner_id="worker", kind="tool", input_data={}
    )
    expected_id = agent_store.stable_tool_execution_id(run.id, "call-1")
    first = store.start_tool_execution(
        run_id=run.id,
        step_id=step.id,
        owner_id="worker",
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"paper_id": "1", "query": "attention"},
        classification="read_only",
    )
    repeated = store.start_tool_execution(
        run_id=run.id,
        step_id=step.id,
        owner_id="worker",
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"query": "attention", "paper_id": "1"},
        classification="read_only",
    )
    assert first.id == expected_id
    assert first.disposition == "new"
    assert repeated.disposition == "in_progress"
    assert repeated.execution == first.execution

    with pytest.raises(agent_store.ToolExecutionConflictError):
        store.start_tool_execution(
            run_id=run.id,
            step_id=step.id,
            owner_id="worker",
            tool_use_id="call-1",
            tool_name="mcp__colbert__search",
            arguments={"query": "different"},
            classification="read_only",
        )

    store.complete_step_and_checkpoint(
        step_id=step.id,
        run_id=run.id,
        owner_id="worker",
        output_data={},
        messages=[{"role": "user", "content": "result"}],
        runtime_state={},
        tool_completion={
            "execution_id": first.id,
            "tool_result": ToolResult(id="call-1", content="result"),
            "result_preview": "result",
            "duration_ms": 1,
        },
    )
    next_step = store.start_step(
        run_id=run.id, owner_id="worker", kind="tool", input_data={}
    )
    completed = store.start_tool_execution(
        run_id=run.id,
        step_id=next_step.id,
        owner_id="worker",
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"query": "attention", "paper_id": "1"},
        classification="read_only",
    )
    assert completed.step_id == step.id
    assert completed.status == "completed"
    assert completed.disposition == "completed"


def test_failed_tool_execution_is_idempotent_and_retry_moves_step(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    first_step = store.start_step(
        run_id=run.id, owner_id="worker", kind="tool", input_data={}
    )
    execution = store.start_tool_execution(
        run_id=run.id,
        step_id=first_step.id,
        owner_id="worker",
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"query": "attention"},
        classification="read_only",
    )
    first_failure = store.fail_tool_execution(
        execution.id,
        run_id=run.id,
        step_id=first_step.id,
        owner_id="worker",
        failure_class="transport",
        failure_message="closed",
        duration_ms=4,
    )
    assert store.fail_tool_execution(
        execution.id,
        run_id=run.id,
        step_id=first_step.id,
        owner_id="worker",
        failure_class="transport",
        failure_message="closed",
        duration_ms=4,
    ) == first_failure
    with pytest.raises(agent_store.ToolExecutionConflictError):
        store.fail_tool_execution(
            execution.id,
            run_id=run.id,
            step_id=first_step.id,
            owner_id="worker",
            failure_class="timeout",
            failure_message="slow",
            duration_ms=5,
        )
    store.fail_step(
        first_step.id,
        run_id=run.id,
        owner_id="worker",
        error_data={"failure_class": "transport"},
    )
    retry_step = store.start_step(
        run_id=run.id, owner_id="worker", kind="tool", input_data={}
    )
    retried = store.start_tool_execution(
        run_id=run.id,
        step_id=retry_step.id,
        owner_id="worker",
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"query": "attention"},
        classification="read_only",
    )
    assert retried.id == execution.id
    assert retried.step_id == retry_step.id
    assert retried.status == "started"
    assert retried.failure_class is None
    assert retried.disposition == "retry"


@pytest.mark.parametrize(
    ("tool_name", "classification", "expected_disposition"),
    (
        ("mcp__colbert__search", "read_only", "retry"),
        ("mcp__colbert__build_index", "idempotent_write", "retry"),
        ("paper_deep_read", "non_retryable", "blocked"),
    ),
)
@pytest.mark.parametrize("stale_started", (False, True))
def test_tool_start_disposition_enforces_replay_safety_across_attempts(
    run_store, clock, tool_name, classification, expected_disposition, stale_started
):
    store, run = run_store
    owner_id = "owner-a"
    store.claim_run(run.id, owner_id=owner_id, lease_seconds=10)
    first_step = store.start_step(
        run_id=run.id, owner_id=owner_id, kind="tool", input_data={}
    )
    first = store.start_tool_execution(
        run_id=run.id,
        step_id=first_step.id,
        owner_id=owner_id,
        tool_use_id="call-1",
        tool_name=tool_name,
        arguments={"query": "attention"},
        classification=classification,
    )

    if stale_started:
        clock.advance(seconds=11)
        assert [item.id for item in store.recover_due_runs(now=clock.now())] == [run.id]
        owner_id = "owner-b"
        assert store.claim_run(run.id, owner_id=owner_id, lease_seconds=30) is not None
    else:
        store.fail_tool_execution(
            first.id,
            run_id=run.id,
            step_id=first_step.id,
            owner_id=owner_id,
            failure_class="transport",
            failure_message="closed",
            duration_ms=1,
        )
        store.fail_step(
            first_step.id,
            run_id=run.id,
            owner_id=owner_id,
            error_data={"failure_class": "transport"},
        )

    retry_step = store.start_step(
        run_id=run.id, owner_id=owner_id, kind="tool", input_data={}
    )
    restarted = store.start_tool_execution(
        run_id=run.id,
        step_id=retry_step.id,
        owner_id=owner_id,
        tool_use_id="call-1",
        tool_name=tool_name,
        arguments={"query": "attention"},
        classification=classification,
    )

    assert restarted.disposition == expected_disposition
    if expected_disposition == "retry":
        assert restarted.step_id == retry_step.id
        assert restarted.status == "started"
    else:
        assert restarted.step_id == first_step.id
        assert restarted.status == ("started" if stale_started else "failed")


def test_owner_guarded_terminal_transitions_clear_lease(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    with pytest.raises(agent_store.RunLeaseLostError):
        store.mark_completed(run.id, owner_id="other")

    completed = store.mark_completed(run.id, owner_id="worker")

    assert completed.status == "completed"
    assert completed.owner_id is None
    assert completed.lease_expires_at is None
    assert completed.finished_at is not None


@pytest.mark.parametrize(
    "mutation_name",
    (
        "start",
        "complete",
        "renew",
        "finish",
        "run_transition",
        "tool_start",
        "tool_fail",
    ),
)
def test_owner_mutation_resamples_clock_after_write_lock_wait(
    run_store, clock, monkeypatch, mutation_name
):
    store, run = run_store
    store.claim_run(run.id, owner_id="owner-a", lease_seconds=10)
    step = None
    if mutation_name in {"complete", "finish", "tool_start", "tool_fail"}:
        step = store.start_step(
            run_id=run.id,
            owner_id="owner-a",
            kind="tool" if mutation_name.startswith("tool_") else "llm",
            input_data={},
        )
    execution = None
    if mutation_name == "tool_fail":
        execution = store.start_tool_execution(
            run_id=run.id,
            step_id=step.id,
            owner_id="owner-a",
            tool_use_id="call-1",
            tool_name="mcp__colbert__search",
            arguments={"query": "attention"},
            classification="read_only",
        )

    if mutation_name == "start":
        mutation = lambda: store.start_step(
            run_id=run.id, owner_id="owner-a", kind="llm", input_data={}
        )
    elif mutation_name == "complete":
        mutation = lambda: store.complete_step_and_checkpoint(
            step_id=step.id,
            run_id=run.id,
            owner_id="owner-a",
            output_data={},
            messages=[{"role": "assistant", "content": "late"}],
            runtime_state={},
        )
    elif mutation_name == "renew":
        mutation = lambda: store.renew_lease(
            run.id, owner_id="owner-a", lease_seconds=30
        )
    elif mutation_name == "finish":
        mutation = lambda: store.fail_step(
            step.id,
            run_id=run.id,
            owner_id="owner-a",
            error_data={"failure_class": "transport"},
        )
    elif mutation_name == "tool_start":
        mutation = lambda: store.start_tool_execution(
            run_id=run.id,
            step_id=step.id,
            owner_id="owner-a",
            tool_use_id="call-1",
            tool_name="mcp__colbert__search",
            arguments={"query": "attention"},
            classification="read_only",
        )
    elif mutation_name == "tool_fail":
        mutation = lambda: store.fail_tool_execution(
            execution.id,
            run_id=run.id,
            step_id=step.id,
            owner_id="owner-a",
            failure_class="transport",
            failure_message="late",
            duration_ms=1,
        )
    else:
        mutation = lambda: store.mark_completed(run.id, owner_id="owner-a")

    result, error = _invoke_across_locked_deadline(
        store, clock, monkeypatch, mutation
    )

    if mutation_name == "renew":
        assert error is None
        assert result is False
    else:
        assert isinstance(error, agent_store.RunLeaseLostError)
    persisted = store.get_run(run.id)
    assert persisted.status == "running"
    assert persisted.current_step == 0
    if step is not None:
        assert store.list_steps(run.id, 0, 10).items[0].status == "started"
    if execution is not None:
        assert store.get_tool_execution(execution.id).status == "started"


def test_claim_calculates_lease_expiry_after_write_lock_wait(
    run_store, clock, monkeypatch
):
    store, run = run_store

    result, error = _invoke_across_locked_deadline(
        store,
        clock,
        monkeypatch,
        lambda: store.claim_run(run.id, owner_id="owner-a", lease_seconds=10),
    )

    assert error is None
    assert result is not None
    assert result.lease_expires_at == "2026-07-20T08:00:21.000000Z"


@pytest.mark.parametrize("new_owner", ("owner-b", "owner-a"))
def test_old_attempt_cannot_fail_tool_execution_after_new_attempt_takes_over(
    run_store, clock, new_owner
):
    store, run = run_store
    store.claim_run(run.id, owner_id="owner-a", lease_seconds=10)
    old_step = store.start_step(
        run_id=run.id, owner_id="owner-a", kind="tool", input_data={}
    )
    execution = store.start_tool_execution(
        run_id=run.id,
        step_id=old_step.id,
        owner_id="owner-a",
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"query": "attention"},
        classification="read_only",
    )
    clock.advance(seconds=11)
    assert [item.id for item in store.recover_due_runs(now=clock.now())] == [run.id]
    store.claim_run(run.id, owner_id=new_owner, lease_seconds=30)
    new_step = store.start_step(
        run_id=run.id, owner_id=new_owner, kind="tool", input_data={}
    )
    moved = store.start_tool_execution(
        run_id=run.id,
        step_id=new_step.id,
        owner_id=new_owner,
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"query": "attention"},
        classification="read_only",
    )
    assert moved.step_id == new_step.id

    expected_error = (
        agent_store.RunLeaseLostError
        if new_owner == "owner-b"
        else agent_store.ToolExecutionConflictError
    )
    with pytest.raises(expected_error):
        store.fail_tool_execution(
            execution.id,
            run_id=run.id,
            step_id=old_step.id,
            owner_id="owner-a",
            failure_class="transport",
            failure_message="late callback",
            duration_ms=12,
        )

    current = store.get_tool_execution(execution.id)
    assert current.step_id == new_step.id
    assert current.status == "started"


def test_tool_execution_start_requires_current_highest_started_attempt(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    old_step = store.start_step(
        run_id=run.id, owner_id="worker", kind="tool", input_data={}
    )
    newest_step = store.start_step(
        run_id=run.id, owner_id="worker", kind="tool", input_data={}
    )
    assert newest_step.attempt == old_step.attempt + 1

    with pytest.raises(agent_store.ToolExecutionConflictError):
        store.start_tool_execution(
            run_id=run.id,
            step_id=old_step.id,
            owner_id="worker",
            tool_use_id="call-old",
            tool_name="mcp__colbert__search",
            arguments={"query": "attention"},
            classification="read_only",
        )


def test_tool_step_requires_completion_before_any_durable_mutation(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    step = store.start_step(
        run_id=run.id, owner_id="worker", kind="tool", input_data={}
    )
    execution = store.start_tool_execution(
        run_id=run.id,
        step_id=step.id,
        owner_id="worker",
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"query": "attention"},
        classification="read_only",
    )

    with pytest.raises(ValueError, match="tool step requires tool_completion"):
        store.complete_step_and_checkpoint(
            step_id=step.id,
            run_id=run.id,
            owner_id="worker",
            output_data={},
            messages=[{"role": "user", "content": "result"}],
            runtime_state={},
        )

    assert store.get_run(run.id).current_step == 0
    assert store.get_latest_checkpoint(run.id).step_sequence == 0
    assert store.list_steps(run.id, 0, 10).items[0].status == "started"
    assert store.get_tool_execution(execution.id).status == "started"


def test_non_tool_step_rejects_tool_completion_before_any_durable_mutation(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    step = store.start_step(
        run_id=run.id, owner_id="worker", kind="llm", input_data={}
    )

    with pytest.raises(ValueError, match="non-tool step forbids tool_completion"):
        store.complete_step_and_checkpoint(
            step_id=step.id,
            run_id=run.id,
            owner_id="worker",
            output_data={},
            messages=[{"role": "assistant", "content": "answer"}],
            runtime_state={},
            tool_completion={
                "execution_id": "tool_missing",
                "tool_result": ToolResult(id="call-1", content="result"),
                "result_preview": "result",
                "duration_ms": 1,
            },
        )

    assert store.get_run(run.id).current_step == 0
    assert store.get_latest_checkpoint(run.id).step_sequence == 0
    assert store.list_steps(run.id, 0, 10).items[0].status == "started"


def test_tool_step_allows_only_one_execution_record(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    step = store.start_step(
        run_id=run.id, owner_id="worker", kind="tool", input_data={}
    )
    store.start_tool_execution(
        run_id=run.id,
        step_id=step.id,
        owner_id="worker",
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"query": "attention"},
        classification="read_only",
    )

    with pytest.raises(agent_store.ToolExecutionConflictError):
        store.start_tool_execution(
            run_id=run.id,
            step_id=step.id,
            owner_id="worker",
            tool_use_id="call-2",
            tool_name="mcp__colbert__search",
            arguments={"query": "transformer"},
            classification="read_only",
        )


def test_tool_completion_rejects_multiple_records_for_same_step_atomically(run_store):
    store, run = run_store
    store.claim_run(run.id, owner_id="worker", lease_seconds=30)
    step = store.start_step(
        run_id=run.id, owner_id="worker", kind="tool", input_data={}
    )
    execution = store.start_tool_execution(
        run_id=run.id,
        step_id=step.id,
        owner_id="worker",
        tool_use_id="call-1",
        tool_name="mcp__colbert__search",
        arguments={"query": "attention"},
        classification="read_only",
    )
    with store._connect() as conn:
        conn.execute(
            """
            INSERT INTO tool_executions (
                id, run_id, step_id, tool_name, arguments_json, classification,
                status, schema_version, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'started', ?, ?)
            """,
            (
                "tool_orphan",
                run.id,
                step.id,
                "mcp__colbert__search",
                '{"query": "orphan"}',
                "read_only",
                1,
                "2026-07-20T08:00:00.000000Z",
            ),
        )

    with pytest.raises(agent_store.ToolExecutionConflictError):
        store.complete_step_and_checkpoint(
            step_id=step.id,
            run_id=run.id,
            owner_id="worker",
            output_data={},
            messages=[{"role": "user", "content": "result"}],
            runtime_state={},
            tool_completion={
                "execution_id": execution.id,
                "tool_result": ToolResult(id="call-1", content="result"),
                "result_preview": "result",
                "duration_ms": 1,
            },
        )

    assert store.get_run(run.id).current_step == 0
    assert store.get_latest_checkpoint(run.id).step_sequence == 0
    assert store.list_steps(run.id, 0, 10).items[0].status == "started"
    assert store.get_tool_execution(execution.id).status == "started"


class _BeginImmediateSignalConnection:
    def __init__(self, connection, attempted):
        self._connection = connection
        self._attempted = attempted

    def execute(self, sql, parameters=()):
        if sql.strip() == "BEGIN IMMEDIATE":
            self._attempted.set()
        return self._connection.execute(sql, parameters)

    def __enter__(self):
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._connection.__exit__(exc_type, exc_value, traceback)

    def __getattr__(self, name):
        return getattr(self._connection, name)


def _invoke_across_locked_deadline(store, clock, monkeypatch, mutation):
    original_connect = store._connect
    lock_connection = original_connect()
    lock_connection.execute("BEGIN IMMEDIATE")
    attempted = threading.Event()

    def connect_with_signal():
        return _BeginImmediateSignalConnection(original_connect(), attempted)

    monkeypatch.setattr(store, "_connect", connect_with_signal)
    outcome = {}

    def invoke():
        try:
            outcome["result"] = mutation()
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    assert attempted.wait(timeout=2), "mutation did not reach BEGIN IMMEDIATE"
    clock.advance(seconds=11)
    lock_connection.commit()
    lock_connection.close()
    thread.join(timeout=2)
    assert not thread.is_alive(), "mutation remained blocked after lock release"
    return outcome.get("result"), outcome.get("error")


def _index_key_columns(conn, index_name):
    return [
        (str(row["name"]), int(row["desc"]))
        for row in conn.execute(f"PRAGMA index_xinfo({index_name})")
        if int(row["key"]) == 1
    ]
