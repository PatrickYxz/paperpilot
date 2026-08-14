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
