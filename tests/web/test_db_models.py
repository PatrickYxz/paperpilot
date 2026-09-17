"""ORM metadata contract tests for the Web business schema."""
from __future__ import annotations

from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects import sqlite
from sqlalchemy.schema import CreateIndex

from paperpilot.web.db_models import Base


def test_web_metadata_owns_the_twelve_business_tables() -> None:
    assert set(Base.metadata.tables) == {
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
            "conversation_id",
            "base_checkpoint_id",
            "final_checkpoint_id",
            "result_quality",
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
        "papers": [
            "id",
            "source",
            "external_id",
            "title",
            "authors_json",
            "abstract",
            "source_url",
            "created_at",
            "updated_at",
        ],
        "conversations": [
            "id",
            "user_id",
            "primary_paper_id",
            "title",
            "head_message_id",
            "head_checkpoint_id",
            "created_at",
            "updated_at",
            "archived_at",
        ],
        "conversation_papers": [
            "conversation_id",
            "paper_id",
            "role",
            "added_by",
            "source_task_id",
            "source_message_id",
            "is_active",
            "created_at",
        ],
        "messages": [
            "id",
            "conversation_id",
            "task_id",
            "parent_message_id",
            "role",
            "content",
            "status",
            "metadata_json",
            "created_at",
        ],
        "context_artifacts": [
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
        ],
        "turn_archives": [
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
        ],
        "compression_states": [
            "conversation_id",
            "compressor_version",
            "state",
            "consecutive_failures",
            "last_failure_type",
            "last_input_digest",
            "opened_at",
            "updated_at",
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
        "papers": ["authors_json", "abstract", "created_at", "updated_at"],
        "conversations": ["created_at", "updated_at", "archived_at"],
        "conversation_papers": ["created_at"],
        "messages": ["metadata_json", "created_at"],
        "context_artifacts": ["storage_key", "sha256", "preview", "created_at"],
        "turn_archives": [
            "archive_version",
            "seed_json",
            "narrative_summary",
            "supersedes_json",
            "created_at",
            "updated_at",
        ],
        "compression_states": [
            "compressor_version",
            "last_failure_type",
            "last_input_digest",
            "opened_at",
            "updated_at",
        ],
    }.items():
        table = Base.metadata.tables[table_name]
        assert {str(table.c[column_name].type) for column_name in column_names} == {
            "TEXT"
        }


def test_metadata_declares_expected_indexes() -> None:
    index_names = {
        index.name
        for table in Base.metadata.tables.values()
        for index in table.indexes
    }

    assert index_names == {
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

    unique_indexes = {
        index.name
        for table in Base.metadata.tables.values()
        for index in table.indexes
        if index.unique
    }
    assert unique_indexes == {"uq_tasks_one_active_per_conversation"}
    unique_constraints = {
        constraint.name
        for table in Base.metadata.tables.values()
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint) and constraint.name is not None
    }
    assert unique_constraints == {
        "uq_papers_source_external_id",
        "uq_messages_task_role",
        "uq_context_artifacts_task_call_hash",
        "uq_turn_archives_conversation_message_version",
    }
    active_task_index = next(
        index
        for index in Base.metadata.tables["research_tasks"].indexes
        if index.name == "uq_tasks_one_active_per_conversation"
    )
    rendered_index = str(
        CreateIndex(active_task_index).compile(dialect=sqlite.dialect())
    )
    assert "WHERE conversation_id IS NOT NULL" in rendered_index
    assert "status IN ('pending', 'running')" in rendered_index


def test_metadata_declares_expected_foreign_keys_and_autoincrement() -> None:
    metadata = Base.metadata
    foreign_keys = {
        (
            foreign_key.parent.table.name,
            foreign_key.parent.name,
            foreign_key.target_fullname,
        )
        for table in metadata.tables.values()
        for foreign_key in table.foreign_keys
    }

    assert foreign_keys == {
        ("sessions", "user_id", "users.id"),
        ("research_tasks", "user_id", "users.id"),
        ("research_tasks", "conversation_id", "conversations.id"),
        ("task_events", "task_id", "research_tasks.id"),
        ("task_artifacts", "task_id", "research_tasks.id"),
        ("conversations", "user_id", "users.id"),
        ("conversations", "primary_paper_id", "papers.id"),
        ("conversations", "head_message_id", "messages.id"),
        ("conversation_papers", "conversation_id", "conversations.id"),
        ("conversation_papers", "paper_id", "papers.id"),
        ("conversation_papers", "source_task_id", "research_tasks.id"),
        ("conversation_papers", "source_message_id", "messages.id"),
        ("messages", "conversation_id", "conversations.id"),
        ("messages", "task_id", "research_tasks.id"),
        ("messages", "parent_message_id", "messages.id"),
        ("context_artifacts", "conversation_id", "conversations.id"),
        ("context_artifacts", "task_id", "research_tasks.id"),
        ("turn_archives", "conversation_id", "conversations.id"),
        ("turn_archives", "task_id", "research_tasks.id"),
        ("turn_archives", "user_message_id", "messages.id"),
        ("compression_states", "conversation_id", "conversations.id"),
    }
    assert metadata.tables["task_events"].dialect_options["sqlite"]["autoincrement"]
    assert metadata.tables["task_artifacts"].dialect_options["sqlite"][
        "autoincrement"
    ]
    head_message_fk = next(
        iter(metadata.tables["conversations"].c.head_message_id.foreign_keys)
    )
    message_conversation_fk = next(
        iter(metadata.tables["messages"].c.conversation_id.foreign_keys)
    )
    assert (
        head_message_fk.constraint.name
        == "fk_conversations_head_message_id_messages"
    )
    assert (
        message_conversation_fk.constraint.name
        == "fk_messages_conversation_id_conversations"
    )


def test_new_business_columns_match_required_nullability_and_primary_keys() -> None:
    metadata = Base.metadata

    assert {
        column.name
        for column in metadata.tables["papers"].columns
        if column.nullable
    } == {"abstract"}
    assert {
        column.name
        for column in metadata.tables["conversations"].columns
        if column.nullable
    } == {"head_message_id", "head_checkpoint_id", "archived_at"}
    assert {
        column.name
        for column in metadata.tables["conversation_papers"].columns
        if column.nullable
    } == {"source_task_id", "source_message_id"}
    assert {
        column.name
        for column in metadata.tables["messages"].columns
        if column.nullable
    } == {"task_id", "parent_message_id"}
    assert {
        column.name
        for column in metadata.tables["research_tasks"].columns
        if column.nullable
    } == {
        "user_id",
        "conversation_id",
        "base_checkpoint_id",
        "final_checkpoint_id",
        "result_quality",
    }
    assert [
        column.name
        for column in metadata.tables["conversation_papers"].primary_key.columns
    ] == ["conversation_id", "paper_id"]
    assert str(metadata.tables["conversation_papers"].c.is_active.type) == "BOOLEAN"
