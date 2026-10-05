"""Alembic adoption tests against real temporary SQLite files."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from threading import Barrier

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
    "papers",
    "conversations",
    "conversation_papers",
    "messages",
    "context_artifacts",
    "turn_archives",
    "compression_states",
    "user_memories",
    "user_memory_profiles",
}

EXPECTED_INDEXES = {
    "idx_tasks_user_created_id",
    "idx_tasks_user_status_created_id",
    "idx_events_task_id_id",
    "idx_artifacts_task_id_id",
    "idx_conversations_user_updated_id",
    "idx_conversation_papers_conversation_active",
    "idx_messages_conversation_parent_created",
    "uq_tasks_one_active_per_conversation",
    "idx_context_artifacts_conversation_sha256",
    "idx_turn_archives_conversation_created",
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


def _unique_column_sets(db_path: Path, table_name: str) -> set[tuple[str, ...]]:
    with sqlite3.connect(db_path) as connection:
        unique_indexes = [
            str(row[1])
            for row in connection.execute(f"PRAGMA index_list({table_name})")
            if bool(row[2])
        ]
        return {
            tuple(
                str(row[2])
                for row in connection.execute(f'PRAGMA index_info("{index_name}")')
            )
            for index_name in unique_indexes
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
    assert _column_names(db_path, "research_tasks") == {
        "id",
        "question",
        "depth",
        "status",
        "created_at",
        "updated_at",
        "user_id",
        "conversation_id",
        "base_checkpoint_id",
        "final_checkpoint_id",
        "result_quality",
    }
    assert _column_names(db_path, "papers") == {
        "id",
        "source",
        "external_id",
        "title",
        "authors_json",
        "abstract",
        "source_url",
        "created_at",
        "updated_at",
    }
    assert _column_names(db_path, "conversations") == {
        "id",
        "user_id",
        "primary_paper_id",
        "title",
        "head_message_id",
        "head_checkpoint_id",
        "created_at",
        "updated_at",
        "archived_at",
    }
    assert _column_names(db_path, "conversation_papers") == {
        "conversation_id",
        "paper_id",
        "role",
        "added_by",
        "source_task_id",
        "source_message_id",
        "is_active",
        "created_at",
    }
    assert _column_names(db_path, "messages") == {
        "id",
        "conversation_id",
        "task_id",
        "parent_message_id",
        "role",
        "content",
        "status",
        "metadata_json",
        "created_at",
    }
    assert _column_names(db_path, "context_artifacts") == {
        "artifact_id",
        "conversation_id",
        "task_id",
        "tool_call_id",
        "tool_name",
        "kind",
        "storage_key",
        "sha256",
        "byte_size",
        "token_estimate",
        "preview",
        "initial_action",
        "future_retention",
        "created_at",
    }
    assert _column_names(db_path, "turn_archives") == {
        "archive_id",
        "conversation_id",
        "task_id",
        "user_message_id",
        "terminal_status",
        "archive_version",
        "seed_json",
        "narrative_summary",
        "narrative_status",
        "supersedes_json",
        "created_at",
        "updated_at",
    }
    assert _column_names(db_path, "compression_states") == {
        "conversation_id",
        "compressor_version",
        "state",
        "consecutive_failures",
        "last_failure_type",
        "last_input_digest",
        "opened_at",
        "updated_at",
    }
    assert ("source", "external_id") in _unique_column_sets(db_path, "papers")
    assert ("task_id", "role") in _unique_column_sets(db_path, "messages")
    assert ("task_id", "tool_call_id", "sha256") in _unique_column_sets(
        db_path, "context_artifacts"
    )
    assert ("conversation_id", "user_message_id", "archive_version") in _unique_column_sets(
        db_path, "turn_archives"
    )
    assert ("conversation_id", "compressor_version") in _unique_column_sets(
        db_path, "compression_states"
    )
    assert ("user_id", "users", "id") in _foreign_key_targets(
        db_path,
        "research_tasks",
    )
    assert get_database_heads(db_path) == get_script_heads()


def test_current_five_table_database_upgrades_to_conversation_schema(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "current.sqlite3"
    upgrade_database(db_path, "20260806_0001")
    assert MANAGED_TABLES - {
        "papers",
        "conversations",
        "conversation_papers",
        "messages",
        "context_artifacts",
        "turn_archives",
        "compression_states",
        "user_memories",
        "user_memory_profiles",
    } <= _table_names(db_path)

    upgrade_database(db_path)

    assert MANAGED_TABLES <= _table_names(db_path)
    assert {
        "conversation_id",
        "base_checkpoint_id",
        "final_checkpoint_id",
        "result_quality",
    } <= _column_names(db_path, "research_tasks")
    assert get_database_heads(db_path) == {"20261005_0005"}


def test_legacy_database_is_adopted_without_losing_rows_or_unknown_tables(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    _create_legacy_database(db_path)

    upgrade_database(db_path)

    assert MANAGED_TABLES <= _table_names(db_path)
    assert "agent_runs" in _table_names(db_path)
    assert "user_id" in _column_names(db_path, "research_tasks")
    assert {
        "conversation_id",
        "base_checkpoint_id",
        "final_checkpoint_id",
        "result_quality",
    } <= _column_names(db_path, "research_tasks")
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


def test_new_tables_enforce_sqlite_foreign_keys(tmp_path: Path) -> None:
    db_path = tmp_path / "foreign-keys.sqlite3"
    upgrade_database(db_path)

    assert _foreign_key_targets(db_path, "conversations") == {
        ("user_id", "users", "id"),
        ("primary_paper_id", "papers", "id"),
        ("head_message_id", "messages", "id"),
    }
    assert _foreign_key_targets(db_path, "messages") == {
        ("conversation_id", "conversations", "id"),
        ("task_id", "research_tasks", "id"),
        ("parent_message_id", "messages", "id"),
    }
    assert _foreign_key_targets(db_path, "conversation_papers") == {
        ("conversation_id", "conversations", "id"),
        ("paper_id", "papers", "id"),
        ("source_task_id", "research_tasks", "id"),
        ("source_message_id", "messages", "id"),
    }
    assert _foreign_key_targets(db_path, "context_artifacts") == {
        ("conversation_id", "conversations", "id"),
        ("task_id", "research_tasks", "id"),
    }
    assert _foreign_key_targets(db_path, "turn_archives") == {
        ("conversation_id", "conversations", "id"),
        ("task_id", "research_tasks", "id"),
        ("user_message_id", "messages", "id"),
    }
    assert _foreign_key_targets(db_path, "compression_states") == {
        ("conversation_id", "conversations", "id"),
    }
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                """
                INSERT INTO conversations(
                    id, user_id, primary_paper_id, title, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "conversation-with-missing-parents",
                    "missing-user",
                    "missing-paper",
                    "Conversation",
                    "2026-08-07T00:00:00Z",
                    "2026-08-07T00:00:00Z",
                ),
            )


def test_context_tables_upgrade_from_0002_and_preserve_business_rows(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "from-0002.sqlite3"
    upgrade_database(db_path, "20260807_0002")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
            ("user-1", "user", "hash", "salt", "2026-08-07T00:00:00Z"),
        )
        connection.execute(
            "INSERT INTO research_tasks(id, question, depth, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("task-1", "question", "quick", "pending", "now", "now"),
        )
    upgrade_database(db_path)
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT question FROM research_tasks WHERE id = 'task-1'"
        ).fetchone() == ("question",)
    assert get_database_heads(db_path) == {"20261005_0005"}


def test_context_migration_downgrade_is_explicitly_unsupported(tmp_path: Path) -> None:
    db_path = tmp_path / "no-downgrade.sqlite3"
    upgrade_database(db_path)
    with pytest.raises(RuntimeError, match="downgrade"):
        command.downgrade(build_alembic_config(db_path), "20260807_0002")


def test_partial_unique_index_rejects_concurrent_active_tasks(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "active-task-race.sqlite3"
    upgrade_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
            ("user-1", "user", "hash", "salt", "2026-08-07T00:00:00Z"),
        )
        connection.execute(
            "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "paper-1",
                "arxiv",
                "2608.00001",
                "Paper",
                "[]",
                None,
                "https://arxiv.org/abs/2608.00001",
                "2026-08-07T00:00:00Z",
                "2026-08-07T00:00:00Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO conversations(
                id, user_id, primary_paper_id, title, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "conversation-1",
                "user-1",
                "paper-1",
                "Conversation",
                "2026-08-07T00:00:00Z",
                "2026-08-07T00:00:00Z",
            ),
        )

    ready = Barrier(2)

    def insert_active_task(task_id: str) -> str:
        try:
            with sqlite3.connect(db_path, timeout=5) as connection:
                connection.execute("PRAGMA busy_timeout = 5000")
                connection.execute("BEGIN DEFERRED")
                ready.wait()
                connection.execute(
                    """
                    INSERT INTO research_tasks(
                        id, question, depth, status, created_at, updated_at,
                        conversation_id
                    ) VALUES (?, ?, ?, 'pending', ?, ?, ?)
                    """,
                    (
                        task_id,
                        "question",
                        "quick",
                        "2026-08-07T00:00:00Z",
                        "2026-08-07T00:00:00Z",
                        "conversation-1",
                    ),
                )
            return "committed"
        except sqlite3.IntegrityError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(insert_active_task, ("task-1", "task-2")))

    assert sorted(results) == ["committed", "rejected"]
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            """
            SELECT count(*) FROM research_tasks
            WHERE conversation_id = 'conversation-1'
              AND status IN ('pending', 'running')
            """
        ).fetchone() == (1,)


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


def test_upgrade_does_not_disable_existing_application_loggers(
    tmp_path: Path,
) -> None:
    logger = logging.getLogger("paperpilot.test.migration-logging")
    handler = logging.NullHandler()
    previous_disabled = logger.disabled
    logger.disabled = False
    logger.addHandler(handler)
    try:
        upgrade_database(tmp_path / "logging.sqlite3")

        assert logger.disabled is False
        assert handler in logger.handlers
    finally:
        logger.removeHandler(handler)
        logger.disabled = previous_disabled
