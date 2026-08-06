"""Alembic adoption tests against real temporary SQLite files."""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from alembic import command

from paperpilot.web.db_migrations import (
    build_alembic_config,
    ensure_database_current,
    get_database_heads,
    get_script_heads,
    upgrade_database,
)


MANAGED_TABLES = {
    "users",
    "sessions",
    "research_tasks",
    "task_events",
    "task_artifacts",
}

EXPECTED_INDEXES = {
    "idx_tasks_user_created_id",
    "idx_tasks_user_status_created_id",
    "idx_events_task_id_id",
    "idx_artifacts_task_id_id",
}


def _table_names(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as connection:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }


def _column_names(db_path: Path, table_name: str) -> set[str]:
    with sqlite3.connect(db_path) as connection:
        return {
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({table_name})")
        }


def _index_names(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as connection:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }


def _foreign_key_targets(db_path: Path, table_name: str) -> set[tuple[str, str, str]]:
    with sqlite3.connect(db_path) as connection:
        return {
            (str(row[3]), str(row[2]), str(row[4]))
            for row in connection.execute(f"PRAGMA foreign_key_list({table_name})")
        }


def _create_legacy_database(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE research_tasks (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                depth TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE task_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                type TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE agent_runs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                status TEXT NOT NULL
            );
            INSERT INTO research_tasks VALUES (
                'task-1', 'question', 'quick', 'pending',
                '2026-08-06T00:00:00+00:00', '2026-08-06T00:00:00+00:00'
            );
            INSERT INTO task_events(task_id, type, message, created_at)
            VALUES (
                'task-1', 'queued', 'queued', '2026-08-06T00:00:00+00:00'
            );
            INSERT INTO agent_runs VALUES ('run-1', 'task-1', 'running');
            """
        )


def test_blank_database_upgrades_to_complete_managed_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "blank.sqlite3"

    upgrade_database(db_path)

    assert MANAGED_TABLES <= _table_names(db_path)
    assert "alembic_version" in _table_names(db_path)
    assert EXPECTED_INDEXES <= _index_names(db_path)
    assert _column_names(db_path, "users") == {
        "id",
        "username",
        "password_hash",
        "password_salt",
        "created_at",
    }
    assert _column_names(db_path, "task_artifacts") == {
        "id",
        "task_id",
        "kind",
        "title",
        "content",
        "payload_json",
        "created_at",
    }
    assert ("user_id", "users", "id") in _foreign_key_targets(
        db_path,
        "research_tasks",
    )
    assert get_database_heads(db_path) == get_script_heads()


def test_legacy_database_is_adopted_without_losing_rows_or_unknown_tables(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    _create_legacy_database(db_path)

    upgrade_database(db_path)

    assert MANAGED_TABLES <= _table_names(db_path)
    assert "agent_runs" in _table_names(db_path)
    assert "user_id" in _column_names(db_path, "research_tasks")
    assert {"stage", "payload_json"} <= _column_names(db_path, "task_events")
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT question FROM research_tasks WHERE id = 'task-1'"
        ).fetchone() == ("question",)
        assert connection.execute(
            "SELECT status FROM agent_runs WHERE id = 'run-1'"
        ).fetchone() == ("running",)


def test_legacy_task_table_is_not_rebuilt_to_add_a_foreign_key(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    _create_legacy_database(db_path)

    upgrade_database(db_path)

    assert _foreign_key_targets(db_path, "research_tasks") == set()


def test_upgrade_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "tasks.sqlite3"
    ensure_database_current(db_path)
    first_heads = get_database_heads(db_path)

    ensure_database_current(db_path)

    assert get_database_heads(db_path) == first_heads == get_script_heads()


def test_initial_revision_refuses_destructive_downgrade(tmp_path: Path) -> None:
    db_path = tmp_path / "tasks.sqlite3"
    upgrade_database(db_path)

    with pytest.raises(RuntimeError, match="downgrade is intentionally unsupported"):
        command.downgrade(build_alembic_config(db_path), "base")


def test_adoption_revision_refuses_offline_sql(tmp_path: Path) -> None:
    config = build_alembic_config(tmp_path / "offline.sqlite3")

    with pytest.raises(RuntimeError, match="Offline SQL is unsupported"):
        command.upgrade(config, "head", sql=True)


def test_alembic_cli_uses_configured_database_path(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cli.sqlite3"
    environment = os.environ.copy()
    environment["PAPERPILOT_TASK_DB_PATH"] = str(db_path)
    project_root = Path(__file__).resolve().parents[2]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "-c",
            str(project_root / "alembic.ini"),
            "upgrade",
            "head",
        ],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert MANAGED_TABLES <= _table_names(db_path)
    assert get_database_heads(db_path) == get_script_heads()
