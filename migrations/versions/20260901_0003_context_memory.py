"""Add internal context artifacts, turn archives, and breaker state."""
from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260901_0003"
down_revision: str | Sequence[str] | None = "20260807_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "context_artifacts",
        sa.Column("artifact_id", sa.Text(), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.Text(),
            sa.ForeignKey("conversations.id"),
            nullable=False,
        ),
        sa.Column(
            "task_id",
            sa.Text(),
            sa.ForeignKey("research_tasks.id"),
            nullable=False,
        ),
        sa.Column("tool_call_id", sa.Text(), nullable=False),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False, unique=True),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("token_estimate", sa.Integer(), nullable=False),
        sa.Column("preview", sa.Text(), nullable=False),
        sa.Column("initial_action", sa.Text(), nullable=False),
        sa.Column("future_retention", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "task_id",
            "tool_call_id",
            "sha256",
            name="uq_context_artifacts_task_call_hash",
        ),
    )
    op.create_index(
        "idx_context_artifacts_conversation_sha256",
        "context_artifacts",
        ["conversation_id", "sha256"],
    )

    op.create_table(
        "turn_archives",
        sa.Column("archive_id", sa.Text(), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.Text(),
            sa.ForeignKey("conversations.id"),
            nullable=False,
        ),
        sa.Column(
            "task_id",
            sa.Text(),
            sa.ForeignKey("research_tasks.id"),
            nullable=False,
        ),
        sa.Column(
            "user_message_id",
            sa.Text(),
            sa.ForeignKey("messages.id"),
            nullable=False,
        ),
        sa.Column("terminal_status", sa.Text(), nullable=False),
        sa.Column("archive_version", sa.Text(), nullable=False),
        sa.Column("seed_json", sa.Text(), nullable=False),
        sa.Column("narrative_summary", sa.Text(), nullable=True),
        sa.Column("narrative_status", sa.Text(), nullable=False),
        sa.Column("supersedes_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "narrative_status IN ('pending', 'running', 'complete', 'failed')",
            name="ck_turn_archives_narrative_status",
        ),
        sa.UniqueConstraint(
            "conversation_id",
            "user_message_id",
            "archive_version",
            name="uq_turn_archives_conversation_message_version",
        ),
    )
    op.create_index(
        "idx_turn_archives_conversation_created",
        "turn_archives",
        ["conversation_id", "created_at", "archive_id"],
    )

    op.create_table(
        "compression_states",
        sa.Column(
            "conversation_id",
            sa.Text(),
            sa.ForeignKey("conversations.id"),
            primary_key=True,
        ),
        sa.Column("compressor_version", sa.Text(), primary_key=True),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False),
        sa.Column("last_failure_type", sa.Text(), nullable=True),
        sa.Column("last_input_digest", sa.Text(), nullable=True),
        sa.Column("opened_at", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "state IN ('CLOSED', 'OPEN', 'HALF_OPEN')",
            name="ck_compression_states_state",
        ),
    )


def downgrade() -> None:
    raise RuntimeError(
        "downgrade is intentionally unsupported for 20260901_0003 context memory"
    )
