"""Pure Conversation API presenters.

This module converts Store records into response dictionaries only.  It does
not query the database, inspect HTTP requests, or make business decisions.
"""
from __future__ import annotations

from paperpilot.web.task_store import (
    ConversationDetail,
    ConversationRecord,
    MessageRecord,
    PaperRecord,
    ResearchTask,
)


def paper_dict(record: PaperRecord) -> dict:
    return {
        "source": record.source,
        "external_id": record.external_id,
        "title": record.title,
        "authors": list(record.authors),
        "abstract": record.abstract,
        "source_url": record.source_url,
    }


def conversation_dict(record: ConversationRecord) -> dict:
    return {
        "id": record.id,
        "primary_paper_id": record.primary_paper_id,
        "title": record.title,
        "head_message_id": record.head_message_id,
        "head_checkpoint_id": record.head_checkpoint_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "archived_at": record.archived_at,
    }


def task_dict(task: ResearchTask) -> dict:
    return task.to_dict()


def message_dict(message: MessageRecord) -> dict:
    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "task_id": message.task_id,
        "parent_message_id": message.parent_message_id,
        "role": message.role,
        "content": message.content,
        "status": message.status,
        "metadata": message.metadata,
        "created_at": message.created_at,
    }


def detail_dict(detail: ConversationDetail) -> dict:
    return {
        "conversation": conversation_dict(detail.conversation),
        "primary_paper": paper_dict(detail.primary_paper),
        "active_papers": [paper_dict(paper) for paper in detail.active_papers],
        "active_task": (
            task_dict(detail.active_task) if detail.active_task is not None else None
        ),
    }
