"""TaskStore tests."""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import event
from sqlalchemy.exc import DatabaseError, IntegrityError, OperationalError

from paperpilot.web.db_migrations import get_database_heads, get_script_heads
from paperpilot.web.task_store import TaskStore


def _create_test_user(store, username):
    return store.create_user(
        username=username,
        password_hash="hash",
        password_salt="salt",
    )


def test_constructor_upgrades_blank_database_to_alembic_head(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    store = TaskStore(db_path)
    try:
        assert get_database_heads(db_path) == get_script_heads()
        assert store.check_health() is None
    finally:
        store.close()


def test_close_is_idempotent(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")

    store.close()
    store.close()


def test_user_and_session_persist_across_store_instances(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    first = TaskStore(db_path)
    user = _create_test_user(first, "persistent-user")
    token = first.create_session(user.id)
    first.close()

    second = TaskStore(db_path)
    try:
        assert second.get_user_by_username(user.username) == user
        assert second.get_user_for_session(token) == user
    finally:
        second.close()


def test_create_and_get_task_persists_to_sqlite(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    store = TaskStore(db_path)

    created = store.create_task(
        question="What are long-context RAG retrieval strategies?",
        depth="standard",
    )
    reloaded = TaskStore(db_path).get_task(created.id)

    assert reloaded == created
    assert created.status == "pending"
    assert created.id.startswith("task_")


def test_research_task_maps_conversation_fields_without_expanding_legacy_dict(
    tmp_path,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user = _create_test_user(store, "task-shape-user")
    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            """
            INSERT INTO research_tasks (
                id, question, depth, status, created_at, updated_at, user_id,
                conversation_id, base_checkpoint_id, final_checkpoint_id,
                result_quality
            ) VALUES (
                'task_shape', 'Shape test', 'standard', 'completed', 't1', 't2',
                ?, NULL, 'cp-base', 'cp-final', 'partial'
            )
            """,
            (user.id,),
        )

    task = store.get_task("task_shape", user_id=user.id)

    assert task is not None
    assert task.conversation_id is None
    assert task.base_checkpoint_id == "cp-base"
    assert task.final_checkpoint_id == "cp-final"
    assert task.result_quality == "partial"
    assert task.to_dict() == {
        "id": "task_shape",
        "question": "Shape test",
        "depth": "standard",
        "status": "completed",
        "created_at": "t1",
        "updated_at": "t2",
    }


def test_store_enables_wal_and_connection_pragmas(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")

    with store.engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA journal_mode").scalar_one() == "wal"
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        assert (
            connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one()
            == 30_000
        )
        assert connection.exec_driver_sql("PRAGMA synchronous").scalar_one() == 1


def test_store_creates_query_indexes(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")

    with store.engine.connect() as connection:
        names = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).all()
        }

    assert {
        "idx_tasks_user_created_id",
        "idx_tasks_user_status_created_id",
        "idx_events_task_id_id",
        "idx_artifacts_task_id_id",
    } <= names


def test_task_store_uses_environment_default_path(tmp_path, monkeypatch):
    db_path = tmp_path / "shared" / "tasks.sqlite3"
    monkeypatch.setenv("PAPERPILOT_TASK_DB_PATH", str(db_path))

    store = TaskStore()
    task = store.create_task(question="shared worker task")

    assert db_path.exists()
    assert TaskStore().get_task(task.id) == task


def test_list_tasks_page_returns_newest_first(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")

    first = store.create_task(question="first", depth="quick")
    second = store.create_task(question="second", depth="deep")

    page = store.list_tasks_page(user_id=None, limit=100)
    expected = sorted(
        [first, second],
        key=lambda task: (task.created_at, task.id),
        reverse=True,
    )

    assert page.items == expected


def test_list_tasks_page_filters_by_status(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.create_task(question="pending task", depth="quick")

    pending = store.list_tasks_page(user_id=None, status="pending", limit=100)
    failed = store.list_tasks_page(user_id=None, status="failed", limit=100)

    assert len(pending.items) == 1
    assert failed.items == []


def test_list_tasks_page_uses_stable_keyset_with_equal_timestamps(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user = _create_test_user(store, "alice")
    created = [
        store.create_task(question=f"task {index}", user_id=user.id)
        for index in range(5)
    ]
    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE research_tasks SET created_at = ? WHERE user_id = ?",
            ("2026-07-14T08:00:00+00:00", user.id),
        )

    first = store.list_tasks_page(user_id=user.id, limit=2)
    second = store.list_tasks_page(
        user_id=user.id,
        limit=2,
        before_created_at=first.items[-1].created_at,
        before_id=first.items[-1].id,
    )
    third = store.list_tasks_page(
        user_id=user.id,
        limit=2,
        before_created_at=second.items[-1].created_at,
        before_id=second.items[-1].id,
    )

    actual = [task.id for task in first.items + second.items + third.items]
    assert actual == sorted((task.id for task in created), reverse=True)
    assert len(set(actual)) == 5
    assert [first.has_more, second.has_more, third.has_more] == [True, True, False]


def test_list_tasks_page_filters_status_with_user_isolation(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice")
    bob = _create_test_user(store, "bob")
    alice_pending = store.create_task(question="alice pending", user_id=alice.id)
    alice_failed = store.create_task(question="alice failed", user_id=alice.id)
    bob_pending = store.create_task(question="bob pending", user_id=bob.id)
    store.update_status(alice_failed.id, "failed")

    page = store.list_tasks_page(
        user_id=alice.id,
        status="pending",
        limit=10,
    )

    assert [task.id for task in page.items] == [alice_pending.id]
    assert bob_pending.id not in [task.id for task in page.items]
    assert page.has_more is False


@pytest.mark.parametrize(
    ("status", "expected_index"),
    [
        (None, "idx_tasks_user_created_id"),
        ("pending", "idx_tasks_user_status_created_id"),
    ],
)
def test_list_tasks_page_query_plans_use_business_indexes(
    tmp_path,
    status,
    expected_index,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user = _create_test_user(store, "alice")
    for index in range(200):
        store.create_task(question=f"task {index}", user_id=user.id)

    if status is None:
        query = """
            EXPLAIN QUERY PLAN
            SELECT *
            FROM research_tasks
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
        """
        params = (user.id, 11)
    else:
        query = """
            EXPLAIN QUERY PLAN
            SELECT *
            FROM research_tasks
            WHERE user_id = ? AND status = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
        """
        params = (user.id, status, 11)

    with store.engine.connect() as connection:
        plan = connection.exec_driver_sql(query, params).all()

    details = " ".join(str(row[3]) for row in plan)
    assert expected_index in details
    assert "USE TEMP B-TREE" not in details


def test_create_task_rejects_empty_question(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")

    with pytest.raises(ValueError, match="question is required"):
        store.create_task(question="   ", depth="standard")


def test_create_task_rejects_invalid_depth(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")

    with pytest.raises(ValueError, match="invalid depth"):
        store.create_task(question="test", depth="huge")


def test_create_queued_task_writes_task_and_event_together(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user = _create_test_user(store, "atomic-user")

    task = store.create_queued_task(
        question="Atomic queue",
        depth="quick",
        user_id=user.id,
        execution_mode="real",
    )

    events = store.list_events_page(
        task.id,
        user_id=user.id,
        after_id=0,
        limit=100,
    )
    assert task.status == "pending"
    assert events is not None
    assert [event.type for event in events.items] == ["queued"]
    assert events.items[0].stage == "queue"
    assert events.items[0].message == "Task queued for real workflow."
    assert events.items[0].payload == {
        "depth": "quick",
        "execution_mode": "real",
        "simulated": False,
    }


def test_create_queued_task_rolls_back_task_when_event_insert_fails(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user = _create_test_user(store, "rollback-user")
    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TRIGGER reject_queued_event
            BEFORE INSERT ON task_events
            BEGIN
                SELECT RAISE(ABORT, 'event insert rejected');
            END
            """
        )

    with pytest.raises(IntegrityError, match="event insert rejected"):
        store.create_queued_task(
            question="Must roll back",
            depth="standard",
            user_id=user.id,
            execution_mode="simulated",
        )

    page = store.list_tasks_page(user_id=user.id, limit=100)
    assert page.items == []


def test_create_queued_task_rejects_invalid_execution_mode(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user = _create_test_user(store, "mode-user")

    with pytest.raises(ValueError, match="invalid execution mode"):
        store.create_queued_task(
            question="Invalid mode",
            depth="standard",
            user_id=user.id,
            execution_mode="later",
        )


def test_check_health_executes_a_lightweight_select(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    statements: list[str] = []

    def capture_statement(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(statement.strip().upper())

    event.listen(store.engine, "before_cursor_execute", capture_statement)
    try:
        assert store.check_health() is None
    finally:
        event.remove(store.engine, "before_cursor_execute", capture_statement)

    assert [statement for statement in statements if statement != "BEGIN"] == [
        "SELECT 1"
    ]


def test_check_health_propagates_connection_error(tmp_path, monkeypatch):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    connection_error = OperationalError(
        "connect",
        {},
        sqlite3.OperationalError("connection failed"),
    )

    def failing_connect():
        raise connection_error

    monkeypatch.setattr(store.engine, "connect", failing_connect)

    with pytest.raises(OperationalError) as exc_info:
        store.check_health()

    assert exc_info.value is connection_error


def test_check_health_propagates_select_error(tmp_path, monkeypatch):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    select_error = DatabaseError(
        "SELECT 1",
        {},
        sqlite3.DatabaseError("select failed"),
    )

    class FailingConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def execute(self, statement):
            assert str(statement) == "SELECT 1"
            raise select_error

    monkeypatch.setattr(store.engine, "connect", FailingConnection)

    with pytest.raises(DatabaseError) as exc_info:
        store.check_health()

    assert exc_info.value is select_error


def test_add_and_list_task_events(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="track progress", depth="standard")

    event = store.add_event(
        task_id=task.id,
        type="queued",
        stage="queue",
        message="Task queued.",
        payload={"depth": "standard", "simulated": True},
    )

    event_page = store.list_events_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    )
    assert event_page is not None
    assert event_page.items == [event]
    assert event.id == 1
    assert event.task_id == task.id
    assert event.stage == "queue"
    assert event.payload == {"depth": "standard", "simulated": True}


def test_update_status_changes_updated_task(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="status update", depth="standard")

    updated = store.update_status(task.id, "running")

    assert updated is not None
    assert updated.status == "running"
    assert updated.updated_at >= task.updated_at


def test_claim_task_allows_only_one_fresh_worker(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    task = TaskStore(db_path).create_task(question="claim once")

    def claim():
        return TaskStore(db_path).claim_task(task.id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(lambda _: claim(), range(2)))

    assert sum(result is not None for result in claimed) == 1
    assert TaskStore(db_path).get_task(task.id).status == "running"


def test_claim_task_allows_redelivered_worker_to_recover_running_task(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="recover worker loss")
    store.update_status(task.id, "running")

    assert store.claim_task(task.id) is None
    recovered = store.claim_task(task.id, allow_running=True)

    assert recovered is not None
    assert recovered.status == "running"


def test_claim_task_never_reopens_failed_or_completed_task(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    failed = store.create_task(question="failed")
    completed = store.create_task(question="completed")
    store.update_status(failed.id, "failed")
    store.update_status(completed.id, "completed")

    assert store.claim_task(failed.id, allow_running=True) is None
    assert store.claim_task(completed.id, allow_running=True) is None


def test_fail_pending_task_does_not_overwrite_claimed_work(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    pending = store.create_task(question="pending")
    claimed = store.create_task(question="claimed")
    store.claim_task(claimed.id)

    failed = store.fail_pending_task(pending.id)
    not_failed = store.fail_pending_task(claimed.id)

    assert failed is not None
    assert failed.status == "failed"
    assert not_failed is None
    assert store.get_task(claimed.id).status == "running"


def test_events_for_missing_task_return_none(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")

    assert store.list_events_page(
        "task_missing",
        user_id=None,
        after_id=0,
        limit=100,
    ) is None


def test_event_and_artifact_pages_advance_watermarks(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="updates")
    events = [
        store.add_event(task_id=task.id, type="progress", message=f"event {index}")
        for index in range(3)
    ]
    artifacts = [
        store.add_artifact(
            task_id=task.id,
            kind="result",
            title=f"artifact {index}",
            content=f"content {index}",
        )
        for index in range(2)
    ]

    event_page = store.list_events_page(
        task.id,
        user_id=None,
        after_id=events[0].id,
        limit=1,
    )
    artifact_page = store.list_artifacts_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=1,
    )

    assert event_page is not None
    assert event_page.items == [events[1]]
    assert event_page.next_after_id == events[1].id
    assert event_page.has_more is True
    assert artifact_page is not None
    assert artifact_page.items == [artifacts[0]]
    assert artifact_page.next_after_id == artifacts[0].id
    assert artifact_page.has_more is True


def test_task_updates_returns_one_owned_snapshot(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice")
    bob = _create_test_user(store, "bob")
    task = store.create_task(question="alice updates", user_id=alice.id)
    event = store.add_event(task_id=task.id, type="queued", message="queued")
    artifact = store.add_artifact(
        task_id=task.id,
        kind="result",
        title="result",
        content="answer",
    )

    updates = store.get_task_updates(
        task.id,
        user_id=alice.id,
        after_event_id=0,
        after_artifact_id=0,
        limit=50,
    )

    assert updates is not None
    assert updates.task == task
    assert updates.events.items == [event]
    assert updates.artifacts.items == [artifact]
    assert store.get_task_updates(
        task.id,
        user_id=bob.id,
        after_event_id=0,
        after_artifact_id=0,
        limit=50,
    ) is None


@pytest.mark.parametrize(
    ("after_id", "limit", "message"),
    [
        (-1, 1, "after_id must be non-negative"),
        (0, 0, "limit must be between 1 and 100"),
        (0, 101, "limit must be between 1 and 100"),
    ],
)
def test_incremental_pages_reject_invalid_watermarks_and_limits(
    tmp_path,
    after_id,
    limit,
    message,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="bounded updates")

    with pytest.raises(ValueError, match=message):
        store.list_events_page(
            task.id,
            user_id=None,
            after_id=after_id,
            limit=limit,
        )


def test_task_updates_starts_explicit_transaction_before_selects(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="snapshot transaction")
    statements: list[str] = []

    def capture_statement(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(statement)

    event.listen(store.engine, "before_cursor_execute", capture_statement)
    try:
        updates = store.get_task_updates(
            task.id,
            user_id=None,
            after_event_id=0,
            after_artifact_id=0,
            limit=1,
        )
    finally:
        event.remove(store.engine, "before_cursor_execute", capture_statement)

    assert updates is not None
    begin_index = next(
        index
        for index, statement in enumerate(statements)
        if statement.strip().upper() == "BEGIN"
    )
    first_select_index = next(
        index
        for index, statement in enumerate(statements)
        if statement.lstrip().upper().startswith("SELECT")
    )
    assert begin_index < first_select_index


def test_task_updates_keeps_one_snapshot_across_event_and_artifact_reads(
    tmp_path,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="snapshot consistency")
    store.add_event(task_id=task.id, type="progress", message="event")
    inserted_artifacts = []

    def insert_artifact_after_event_read(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        if inserted_artifacts or "FROM task_events" not in statement:
            return
        inserted_artifacts.append(
            store.add_artifact(
                task_id=task.id,
                kind="late",
                title="Late artifact",
                content="Committed after the snapshot started.",
            )
        )

    event.listen(
        store.engine,
        "after_cursor_execute",
        insert_artifact_after_event_read,
    )
    try:
        updates = store.get_task_updates(
            task.id,
            user_id=None,
            after_event_id=0,
            after_artifact_id=0,
            limit=10,
        )
    finally:
        event.remove(
            store.engine,
            "after_cursor_execute",
            insert_artifact_after_event_read,
        )

    assert inserted_artifacts
    assert updates is not None
    assert updates.artifacts.items == []

    later_updates = store.get_task_updates(
        task.id,
        user_id=None,
        after_event_id=0,
        after_artifact_id=0,
        limit=10,
    )
    assert later_updates is not None
    assert later_updates.artifacts.items == inserted_artifacts


@pytest.mark.parametrize(
    ("query", "index_name"),
    [
        (
            """
            SELECT id, task_id, type, stage, message, payload_json, created_at
            FROM task_events
            WHERE task_id = ? AND id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            "idx_events_task_id_id",
        ),
        (
            """
            SELECT id, task_id, kind, title, content, payload_json, created_at
            FROM task_artifacts
            WHERE task_id = ? AND id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            "idx_artifacts_task_id_id",
        ),
    ],
)
def test_watermark_page_query_plans_use_indexes(tmp_path, query, index_name):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="query plan")

    with store.engine.connect() as connection:
        plan = connection.exec_driver_sql(
            f"EXPLAIN QUERY PLAN {query}",
            (task.id, 0, 2),
        ).all()

    details = " ".join(str(row[3]) for row in plan)
    assert index_name in details
    assert "USE TEMP B-TREE" not in details


def test_store_migrates_legacy_task_events_table(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE research_tasks (
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
            CREATE TABLE task_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                type TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO research_tasks (
                id, question, depth, status, created_at, updated_at
            )
            VALUES ('task_legacy', 'legacy', 'standard', 'pending', 't1', 't1')
            """
        )
        conn.execute(
            """
            INSERT INTO task_events (task_id, type, message, created_at)
            VALUES ('task_legacy', 'queued', 'Legacy event.', 't1')
            """
        )

    store = TaskStore(db_path)
    store.add_event(
        task_id="task_legacy",
        type="progress",
        stage="prepare",
        message="Migrated event.",
        payload={"migrated": True},
    )

    event_page = store.list_events_page(
        "task_legacy",
        user_id=None,
        after_id=0,
        limit=100,
    )
    assert event_page is not None
    events = event_page.items
    assert events[0].stage is None
    assert events[0].payload == {}
    assert events[1].stage == "prepare"
    assert events[1].payload == {"migrated": True}


def test_add_and_list_task_artifacts(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="artifact task", depth="standard")

    artifact = store.add_artifact(
        task_id=task.id,
        kind="result",
        title="Result",
        content="A compact result.",
        payload={"simulated": True},
    )

    artifact_page = store.list_artifacts_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    )
    assert artifact_page is not None
    assert artifact_page.items == [artifact]
    assert artifact.id == 1
    assert artifact.kind == "result"
    assert artifact.payload == {"simulated": True}


def test_artifacts_for_missing_task_return_none(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")

    assert store.list_artifacts_page(
        "task_missing",
        user_id=None,
        after_id=0,
        limit=100,
    ) is None
