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


def _paper(**overrides: object) -> PaperCandidate:
    values: dict[str, object] = {
        "external_id": "2401.12345v2",
        "title": "A Test Paper",
        "authors": ["Ada Lovelace"],
        "abstract": "abstract",
        "source_url": "https://arxiv.org/abs/2401.12345v2",
    }
    values.update(overrides)
    return PaperCandidate(**values)

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


def test_conversation_task_updates_require_matching_owner_conversation_and_task(
    tmp_path,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-task-updates")
    bob = _create_test_user(store, "bob-task-updates")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20029v1"),
    )
    other_conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20030v1"),
    )
    task = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Scope this task update.",
        depth="standard",
        expected_head_message_id=None,
    ).task

    updates = store.get_conversation_task_updates(
        conversation.id,
        task.id,
        user_id=alice.id,
    )

    assert updates is not None
    assert updates.task.id == task.id
    assert store.get_conversation_task_updates(
        other_conversation.id,
        task.id,
        user_id=alice.id,
    ) is None
    assert store.get_conversation_task_updates(
        conversation.id,
        task.id,
        user_id=bob.id,
    ) is None
