"""Add append-only user long-term memory table."""
from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20261005_0004"
down_revision: str | Sequence[str] | None = "20260901_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "user_memories",
        sa.Column("memory_id", sa.Text(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Text(),
            sa.ForeignKey("users.id", name="fk_user_memories_user_id_users"),
            nullable=False,
        ),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("context_json", sa.Text(), nullable=False),
        sa.Column(
            "source_conversation_id",
            sa.Text(),
            sa.ForeignKey("conversations.id"),
            nullable=False,
        ),
        sa.Column(
            "source_task_id",
            sa.Text(),
            sa.ForeignKey("research_tasks.id"),
            nullable=False,
        ),
        sa.Column(
            "source_message_id",
            sa.Text(),
            sa.ForeignKey("messages.id"),
            nullable=False,
        ),
        sa.Column("support_span", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_index(
        "idx_user_memories_user_created_id",
        "user_memories",
        ["user_id", "created_at", "memory_id"],
    )


def downgrade() -> None:
    raise RuntimeError(
        "downgrade is intentionally unsupported for 20261005_0004 user memories"
    )
