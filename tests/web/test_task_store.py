"""TaskStore tests."""
from __future__ import annotations

import sqlite3

import pytest

from paperpilot.web.task_store import TaskStore


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


def test_list_tasks_returns_newest_first(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")

    first = store.create_task(question="first", depth="quick")
    second = store.create_task(question="second", depth="deep")

    assert [task.id for task in store.list_tasks()] == [second.id, first.id]


def test_list_tasks_filters_by_status(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.create_task(question="pending task", depth="quick")

    assert len(store.list_tasks(status="pending")) == 1
    assert store.list_tasks(status="failed") == []


def test_create_task_rejects_empty_question(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")

    with pytest.raises(ValueError, match="question is required"):
        store.create_task(question="   ", depth="standard")


def test_create_task_rejects_invalid_depth(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")

    with pytest.raises(ValueError, match="invalid depth"):
        store.create_task(question="test", depth="huge")


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

    events = store.list_events(task.id)
    assert events == [event]
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


def test_events_for_missing_task_return_none(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")

    assert store.list_events("task_missing") is None


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

    events = store.list_events("task_legacy")
    assert events is not None
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

    artifacts = store.list_artifacts(task.id)
    assert artifacts == [artifact]
    assert artifact.id == 1
    assert artifact.kind == "result"
    assert artifact.payload == {"simulated": True}


def test_artifacts_for_missing_task_return_none(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")

    assert store.list_artifacts("task_missing") is None
