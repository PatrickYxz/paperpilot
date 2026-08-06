"""Adopt the existing PaperPilot Web schema without destructive rewrites."""
from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260806_0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table_names(bind) -> set[str]:
    return set(sa.inspect(bind).get_table_names())


def _column_names(bind, table_name: str) -> set[str]:
    return {column["name"] for column in sa.inspect(bind).get_columns(table_name)}


def _index_names(bind, table_name: str) -> set[str]:
    return {index["name"] for index in sa.inspect(bind).get_indexes(table_name)}


def _create_index_if_missing(
    bind,
    table_name: str,
    index_name: str,
    ddl: str,
) -> None:
    if index_name not in _index_names(bind, table_name):
        op.execute(sa.text(ddl))


def upgrade() -> None:
    bind = op.get_bind()
    tables = _table_names(bind)

    if "users" not in tables:
        op.create_table(
            "users",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("username", sa.Text(), nullable=False, unique=True),
            sa.Column("password_hash", sa.Text(), nullable=False),
            sa.Column("password_salt", sa.Text(), nullable=False),
            sa.Column("created_at", sa.Text(), nullable=False),
        )

    tables = _table_names(bind)
    if "sessions" not in tables:
        op.create_table(
            "sessions",
            sa.Column("token", sa.Text(), primary_key=True),
            sa.Column(
                "user_id",
                sa.Text(),
                sa.ForeignKey("users.id"),
                nullable=False,
            ),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("expires_at", sa.Text(), nullable=False),
        )

    tables = _table_names(bind)
    if "research_tasks" not in tables:
        op.create_table(
            "research_tasks",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("question", sa.Text(), nullable=False),
            sa.Column("depth", sa.Text(), nullable=False),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("created_at", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.Text(), nullable=False),
            sa.Column(
                "user_id",
                sa.Text(),
                sa.ForeignKey("users.id"),
                nullable=True,
            ),
        )
    elif "user_id" not in _column_names(bind, "research_tasks"):
        # SQLite cannot add the FK without rebuilding the historical table.
        op.add_column(
            "research_tasks",
            sa.Column("user_id", sa.Text(), nullable=True),
        )

    tables = _table_names(bind)
    if "task_events" not in tables:
        op.create_table(
            "task_events",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "task_id",
                sa.Text(),
                sa.ForeignKey("research_tasks.id"),
                nullable=False,
            ),
            sa.Column("type", sa.Text(), nullable=False),
            sa.Column("stage", sa.Text(), nullable=True),
            sa.Column("message", sa.Text(), nullable=False),
            sa.Column("payload_json", sa.Text(), nullable=True),
            sa.Column("created_at", sa.Text(), nullable=False),
            sqlite_autoincrement=True,
        )
    else:
        event_columns = _column_names(bind, "task_events")
        if "stage" not in event_columns:
            op.add_column(
                "task_events",
                sa.Column("stage", sa.Text(), nullable=True),
            )
        if "payload_json" not in event_columns:
            op.add_column(
                "task_events",
                sa.Column("payload_json", sa.Text(), nullable=True),
            )

    tables = _table_names(bind)
    if "task_artifacts" not in tables:
        op.create_table(
            "task_artifacts",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "task_id",
                sa.Text(),
                sa.ForeignKey("research_tasks.id"),
                nullable=False,
            ),
            sa.Column("kind", sa.Text(), nullable=False),
            sa.Column("title", sa.Text(), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("payload_json", sa.Text(), nullable=True),
            sa.Column("created_at", sa.Text(), nullable=False),
            sqlite_autoincrement=True,
        )

    _create_index_if_missing(
        bind,
        "research_tasks",
        "idx_tasks_user_created_id",
        "CREATE INDEX idx_tasks_user_created_id "
        "ON research_tasks(user_id, created_at DESC, id DESC)",
    )
    _create_index_if_missing(
        bind,
        "research_tasks",
        "idx_tasks_user_status_created_id",
        "CREATE INDEX idx_tasks_user_status_created_id "
        "ON research_tasks(user_id, status, created_at DESC, id DESC)",
    )
    _create_index_if_missing(
        bind,
        "task_events",
        "idx_events_task_id_id",
        "CREATE INDEX idx_events_task_id_id ON task_events(task_id, id)",
    )
    _create_index_if_missing(
        bind,
        "task_artifacts",
        "idx_artifacts_task_id_id",
        "CREATE INDEX idx_artifacts_task_id_id ON task_artifacts(task_id, id)",
    )


def downgrade() -> None:
    raise RuntimeError(
        "PaperPilot schema downgrade is intentionally unsupported; "
        "restore a verified database backup instead"
    )
