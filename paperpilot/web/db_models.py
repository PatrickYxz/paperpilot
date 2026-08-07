"""SQLAlchemy row models for PaperPilot's Web business tables."""
from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, Index, Integer, Text, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative metadata owned by the Web business database."""


class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    username: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    password_salt: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


class LoginSessionRow(Base):
    __tablename__ = "sessions"

    token: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[str] = mapped_column(Text, nullable=False)


class ResearchTaskRow(Base):
    __tablename__ = "research_tasks"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    depth: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
    )
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "conversations.id",
            name="fk_research_tasks_conversation_id_conversations",
        ),
        nullable=True,
    )
    base_checkpoint_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_checkpoint_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_quality: Mapped[str | None] = mapped_column(Text, nullable=True)


class TaskEventRow(Base):
    __tablename__ = "task_events"
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(
        ForeignKey("research_tasks.id"),
        nullable=False,
    )
    type: Mapped[str] = mapped_column(Text, nullable=False)
    stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


class TaskArtifactRow(Base):
    __tablename__ = "task_artifacts"
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(
        ForeignKey("research_tasks.id"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


class PaperRow(Base):
    __tablename__ = "papers"
    __table_args__ = (
        UniqueConstraint(
            "source",
            "external_id",
            name="uq_papers_source_external_id",
        ),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors_json: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


class ConversationRow(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", name="fk_conversations_user_id_users"),
        nullable=False,
    )
    primary_paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", name="fk_conversations_primary_paper_id_papers"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    head_message_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "messages.id",
            name="fk_conversations_head_message_id_messages",
        ),
        nullable=True,
    )
    head_checkpoint_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)
    archived_at: Mapped[str | None] = mapped_column(Text, nullable=True)


class MessageRow(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("task_id", "role", name="uq_messages_task_role"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey(
            "conversations.id",
            name="fk_messages_conversation_id_conversations",
        ),
        nullable=False,
    )
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_tasks.id", name="fk_messages_task_id_research_tasks"),
        nullable=True,
    )
    parent_message_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.id", name="fk_messages_parent_message_id_messages"),
        nullable=True,
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


class ConversationPaperRow(Base):
    __tablename__ = "conversation_papers"

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey(
            "conversations.id",
            name="fk_conversation_papers_conversation_id_conversations",
        ),
        primary_key=True,
    )
    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", name="fk_conversation_papers_paper_id_papers"),
        primary_key=True,
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    added_by: Mapped[str] = mapped_column(Text, nullable=False)
    source_task_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "research_tasks.id",
            name="fk_conversation_papers_source_task_id_research_tasks",
        ),
        nullable=True,
    )
    source_message_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "messages.id",
            name="fk_conversation_papers_source_message_id_messages",
        ),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


Index(
    "idx_tasks_user_created_id",
    ResearchTaskRow.user_id,
    ResearchTaskRow.created_at.desc(),
    ResearchTaskRow.id.desc(),
)
Index(
    "idx_tasks_user_status_created_id",
    ResearchTaskRow.user_id,
    ResearchTaskRow.status,
    ResearchTaskRow.created_at.desc(),
    ResearchTaskRow.id.desc(),
)
Index("idx_events_task_id_id", TaskEventRow.task_id, TaskEventRow.id)
Index("idx_artifacts_task_id_id", TaskArtifactRow.task_id, TaskArtifactRow.id)
Index(
    "idx_conversations_user_updated_id",
    ConversationRow.user_id,
    ConversationRow.updated_at.desc(),
    ConversationRow.id.desc(),
)
Index(
    "idx_conversation_papers_conversation_active",
    ConversationPaperRow.conversation_id,
    ConversationPaperRow.is_active,
)
Index(
    "idx_messages_conversation_parent_created",
    MessageRow.conversation_id,
    MessageRow.parent_message_id,
    MessageRow.created_at,
)
Index(
    "uq_tasks_one_active_per_conversation",
    ResearchTaskRow.conversation_id,
    unique=True,
    sqlite_where=text(
        "conversation_id IS NOT NULL AND status IN ('pending', 'running')"
    ),
)
