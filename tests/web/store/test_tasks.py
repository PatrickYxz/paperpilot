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
