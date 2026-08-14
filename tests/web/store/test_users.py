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
