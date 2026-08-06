"""ORM metadata contract tests for the existing Web schema."""
from __future__ import annotations

from paperpilot.web.db_models import Base


def test_web_metadata_owns_only_the_five_existing_tables() -> None:
    assert set(Base.metadata.tables) == {
        "users",
        "sessions",
        "research_tasks",
        "task_events",
        "task_artifacts",
    }


def test_web_table_columns_match_the_existing_schema() -> None:
    expected_columns = {
        "users": [
            "id",
            "username",
            "password_hash",
            "password_salt",
            "created_at",
        ],
        "sessions": ["token", "user_id", "created_at", "expires_at"],
        "research_tasks": [
            "id",
            "question",
            "depth",
            "status",
            "created_at",
            "updated_at",
            "user_id",
        ],
        "task_events": [
            "id",
            "task_id",
            "type",
            "stage",
            "message",
            "payload_json",
            "created_at",
        ],
        "task_artifacts": [
            "id",
            "task_id",
            "kind",
            "title",
            "content",
            "payload_json",
            "created_at",
        ],
    }

    assert {
        table_name: [column.name for column in Base.metadata.tables[table_name].columns]
        for table_name in expected_columns
    } == expected_columns


def test_timestamp_and_json_columns_remain_sqlite_text() -> None:
    for table_name, column_names in {
        "users": ["created_at"],
        "sessions": ["created_at", "expires_at"],
        "research_tasks": ["created_at", "updated_at"],
        "task_events": ["payload_json", "created_at"],
        "task_artifacts": ["payload_json", "created_at"],
    }.items():
        table = Base.metadata.tables[table_name]
        assert {str(table.c[column_name].type) for column_name in column_names} == {
            "TEXT"
        }


def test_metadata_declares_expected_indexes() -> None:
    index_names = {
        index.name
        for table in Base.metadata.sorted_tables
        for index in table.indexes
    }

    assert index_names == {
        "idx_tasks_user_created_id",
        "idx_tasks_user_status_created_id",
        "idx_events_task_id_id",
        "idx_artifacts_task_id_id",
    }


def test_metadata_declares_expected_foreign_keys_and_autoincrement() -> None:
    metadata = Base.metadata
    foreign_keys = {
        (
            foreign_key.parent.table.name,
            foreign_key.parent.name,
            foreign_key.target_fullname,
        )
        for table in metadata.sorted_tables
        for foreign_key in table.foreign_keys
    }

    assert foreign_keys == {
        ("sessions", "user_id", "users.id"),
        ("research_tasks", "user_id", "users.id"),
        ("task_events", "task_id", "research_tasks.id"),
        ("task_artifacts", "task_id", "research_tasks.id"),
    }
    assert metadata.tables["task_events"].dialect_options["sqlite"]["autoincrement"]
    assert metadata.tables["task_artifacts"].dialect_options["sqlite"][
        "autoincrement"
    ]
