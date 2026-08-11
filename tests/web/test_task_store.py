"""TaskStore tests."""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import event
from sqlalchemy.exc import DatabaseError, OperationalError

from paperpilot.papers import PaperCandidate
from paperpilot.web.db_migrations import get_database_heads, get_script_heads
from paperpilot.web.task_store import TaskStore


def _create_test_user(store, username):
    return store.create_user(
        username=username,
        password_hash="hash",
        password_salt="salt",
    )


def _create_conversation_task(
    store: TaskStore,
    *,
    username: str,
    question: str,
    depth: str = "standard",
):
    user = _create_test_user(store, username)
    conversation = store.create_conversation(
        user_id=user.id,
        paper=PaperCandidate(
            external_id="2401.12345v1",
            title="A Test Paper",
            authors=["Ada Lovelace"],
            abstract="abstract",
            source_url="https://arxiv.org/abs/2401.12345v1",
        ),
    )
    turn = store.create_conversation_turn(
        user_id=user.id,
        conversation_id=conversation.id,
        content=question,
        depth=depth,
        expected_head_message_id=None,
    )
    return user, conversation, turn.task


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

    user, _conversation, created = _create_conversation_task(
        store,
        username="persisted-task-user",
        question="What are long-context RAG retrieval strategies?",
        depth="standard",
    )
    reloaded = TaskStore(db_path).get_task(created.id, user_id=user.id)

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
    user, _conversation, task = _create_conversation_task(
        store,
        username="shared-worker-user",
        question="shared worker task",
    )

    assert db_path.exists()
    assert TaskStore().get_task(task.id, user_id=user.id) == task


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
    user, _conversation, task = _create_conversation_task(
        store,
        username="event-user",
        question="track progress",
    )

    event = store.add_event(
        task_id=task.id,
        type="progress",
        stage="queue",
        message="Task queued.",
        payload={"depth": "standard", "simulated": True},
    )

    event_page = store.list_events_page(
        task.id,
        user_id=user.id,
        after_id=0,
        limit=100,
    )
    assert event_page is not None
    assert event_page.items[-1] == event
    assert [item.type for item in event_page.items] == ["queued", "progress"]
    assert event.task_id == task.id
    assert event.stage == "queue"
    assert event.payload == {"depth": "standard", "simulated": True}


def test_claim_task_allows_only_one_fresh_worker(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    seed = TaskStore(db_path)
    _user, _conversation, task = _create_conversation_task(
        seed,
        username="claim-once-user",
        question="claim once",
    )
    seed.close()

    def claim():
        return TaskStore(db_path).claim_task(task.id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(lambda _: claim(), range(2)))

    assert sum(result is not None for result in claimed) == 1
    assert TaskStore(db_path).get_task(task.id).status == "running"


def test_claim_task_allows_redelivered_worker_to_recover_running_task(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    _user, _conversation, task = _create_conversation_task(
        store,
        username="recover-worker-user",
        question="recover worker loss",
    )
    assert store.claim_task(task.id) is not None

    assert store.claim_task(task.id) is None
    recovered = store.claim_task(task.id, allow_running=True)

    assert recovered is not None
    assert recovered.status == "running"


def test_claim_task_never_reopens_failed_or_completed_task(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    _failed_user, _failed_conversation, failed = _create_conversation_task(
        store,
        username="failed-task-user",
        question="failed",
    )
    _completed_user, _completed_conversation, completed = (
        _create_conversation_task(
            store,
            username="completed-task-user",
            question="completed",
        )
    )
    assert store.fail_pending_task(failed.id) is not None
    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE research_tasks SET status = 'completed' WHERE id = ?",
            (completed.id,),
        )

    assert store.claim_task(failed.id, allow_running=True) is None
    assert store.claim_task(completed.id, allow_running=True) is None


def test_fail_pending_task_does_not_overwrite_claimed_work(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    _pending_user, _pending_conversation, pending = _create_conversation_task(
        store,
        username="pending-task-user",
        question="pending",
    )
    _claimed_user, _claimed_conversation, claimed = _create_conversation_task(
        store,
        username="claimed-task-user",
        question="claimed",
    )
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
    user, _conversation, task = _create_conversation_task(
        store,
        username="watermark-user",
        question="updates",
    )
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
        user_id=user.id,
        after_id=events[0].id,
        limit=1,
    )
    artifact_page = store.list_artifacts_page(
        task.id,
        user_id=user.id,
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


def test_conversation_task_updates_returns_one_owned_snapshot(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice, conversation, task = _create_conversation_task(
        store,
        username="alice",
        question="alice updates",
    )
    bob = _create_test_user(store, "bob")
    event = store.add_event(task_id=task.id, type="queued", message="queued")
    artifact = store.add_artifact(
        task_id=task.id,
        kind="result",
        title="result",
        content="answer",
    )

    updates = store.get_conversation_task_updates(
        conversation.id,
        task.id,
        user_id=alice.id,
        after_event_id=0,
        after_artifact_id=0,
        limit=50,
    )

    assert updates is not None
    assert updates.task == task
    assert updates.events.items[-1] == event
    assert updates.artifacts.items == [artifact]
    assert store.get_conversation_task_updates(
        conversation.id,
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
    user, _conversation, task = _create_conversation_task(
        store,
        username="bounded-update-user",
        question="bounded updates",
    )

    with pytest.raises(ValueError, match=message):
        store.list_events_page(
            task.id,
            user_id=user.id,
            after_id=after_id,
            limit=limit,
        )


def test_conversation_task_updates_starts_transaction_before_selects(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user, conversation, task = _create_conversation_task(
        store,
        username="snapshot-transaction-user",
        question="snapshot transaction",
    )
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
        updates = store.get_conversation_task_updates(
            conversation.id,
            task.id,
            user_id=user.id,
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


def test_conversation_task_updates_keeps_one_snapshot_across_reads(
    tmp_path,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user, conversation, task = _create_conversation_task(
        store,
        username="snapshot-consistency-user",
        question="snapshot consistency",
    )
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
        updates = store.get_conversation_task_updates(
            conversation.id,
            task.id,
            user_id=user.id,
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

    later_updates = store.get_conversation_task_updates(
        conversation.id,
        task.id,
        user_id=user.id,
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
    _user, _conversation, task = _create_conversation_task(
        store,
        username="query-plan-user",
        question="query plan",
    )

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
    user, _conversation, task = _create_conversation_task(
        store,
        username="artifact-user",
        question="artifact task",
    )

    artifact = store.add_artifact(
        task_id=task.id,
        kind="result",
        title="Result",
        content="A compact result.",
        payload={"simulated": True},
    )

    artifact_page = store.list_artifacts_page(
        task.id,
        user_id=user.id,
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
