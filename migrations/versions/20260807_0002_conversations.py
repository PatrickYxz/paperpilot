"""Add conversations, papers, messages, and checkpoint task fields."""
from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260807_0002"
down_revision: str | Sequence[str] | None = "20260806_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "papers",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("authors_json", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "source",
            "external_id",
            name="uq_papers_source_external_id",
        ),
    )

    # SQLite permits a foreign key to a table created later in the same revision.
    # Creating conversations first lets messages carry the reverse FK as well.
    op.create_table(
        "conversations",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Text(),
            sa.ForeignKey("users.id", name="fk_conversations_user_id_users"),
            nullable=False,
        ),
        sa.Column(
            "primary_paper_id",
            sa.Text(),
            sa.ForeignKey(
                "papers.id",
                name="fk_conversations_primary_paper_id_papers",
            ),
            nullable=False,
        ),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column(
            "head_message_id",
            sa.Text(),
            sa.ForeignKey(
                "messages.id",
                name="fk_conversations_head_message_id_messages",
            ),
            nullable=True,
        ),
        sa.Column("head_checkpoint_id", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("archived_at", sa.Text(), nullable=True),
    )
    op.create_index(
        "idx_conversations_user_updated_id",
        "conversations",
        ["user_id", sa.text("updated_at DESC"), sa.text("id DESC")],
    )

    # Do not rebuild the existing table: legacy databases may contain unknown
    # objects and rows. The ORM declares the FK; business writes and the partial
    # unique index enforce the new-conversation task invariant on SQLite.
    op.add_column(
        "research_tasks",
        sa.Column("conversation_id", sa.Text(), nullable=True),
    )
    op.add_column(
        "research_tasks",
        sa.Column("base_checkpoint_id", sa.Text(), nullable=True),
    )
    op.add_column(
        "research_tasks",
        sa.Column("final_checkpoint_id", sa.Text(), nullable=True),
    )
    op.add_column(
        "research_tasks",
        sa.Column("result_quality", sa.Text(), nullable=True),
    )
    op.create_index(
        "uq_tasks_one_active_per_conversation",
        "research_tasks",
        ["conversation_id"],
        unique=True,
        sqlite_where=sa.text(
            "conversation_id IS NOT NULL "
            "AND status IN ('pending', 'running')"
        ),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.Text(),
            sa.ForeignKey(
                "conversations.id",
                name="fk_messages_conversation_id_conversations",
            ),
            nullable=False,
        ),
        sa.Column(
            "task_id",
            sa.Text(),
            sa.ForeignKey(
                "research_tasks.id",
                name="fk_messages_task_id_research_tasks",
            ),
            nullable=True,
        ),
        sa.Column(
            "parent_message_id",
            sa.Text(),
            sa.ForeignKey(
                "messages.id",
                name="fk_messages_parent_message_id_messages",
            ),
            nullable=True,
        ),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "task_id",
            "role",
            name="uq_messages_task_role",
        ),
    )
    op.create_index(
        "idx_messages_conversation_parent_created",
        "messages",
        ["conversation_id", "parent_message_id", "created_at"],
    )

    op.create_table(
        "conversation_papers",
        sa.Column(
            "conversation_id",
            sa.Text(),
            sa.ForeignKey(
                "conversations.id",
                name="fk_conversation_papers_conversation_id_conversations",
            ),
            primary_key=True,
        ),
        sa.Column(
            "paper_id",
            sa.Text(),
            sa.ForeignKey(
                "papers.id",
                name="fk_conversation_papers_paper_id_papers",
            ),
            primary_key=True,
        ),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("added_by", sa.Text(), nullable=False),
        sa.Column(
            "source_task_id",
            sa.Text(),
            sa.ForeignKey(
                "research_tasks.id",
                name="fk_conversation_papers_source_task_id_research_tasks",
            ),
            nullable=True,
        ),
        sa.Column(
            "source_message_id",
            sa.Text(),
            sa.ForeignKey(
                "messages.id",
                name="fk_conversation_papers_source_message_id_messages",
            ),
            nullable=True,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_index(
        "idx_conversation_papers_conversation_active",
        "conversation_papers",
        ["conversation_id", "is_active"],
    )


def downgrade() -> None:
    raise RuntimeError(
        "PaperPilot schema downgrade is intentionally unsupported; "
        "restore a verified database backup instead"
    )
