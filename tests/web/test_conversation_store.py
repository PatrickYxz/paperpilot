"""Paper and conversation persistence tests."""
from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy.exc import IntegrityError

from paperpilot.papers import PaperCandidate
from paperpilot.web.task_store import TaskStore


def _create_test_user(store: TaskStore, username: str):
    return store.create_user(
        username=username,
        password_hash="hash",
        password_salt="salt",
    )


def _paper(**overrides: object) -> PaperCandidate:
    values: dict[str, object] = {
        "external_id": "2401.12345v2",
        "title": "A Test Paper",
        "authors": ["Ada Lovelace"],
        "abstract": "abstract",
        "source_url": "https://arxiv.org/abs/2401.12345v2",
    }
    values.update(overrides)
    return PaperCandidate(**values)


def _table_count(db_path, table: str) -> int:
    with sqlite3.connect(db_path) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_create_conversation_persists_primary_paper_and_association(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    store = TaskStore(db_path)
    alice = _create_test_user(store, "alice")

    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(),
        title=None,
    )
    detail = store.get_conversation_detail(conversation.id, user_id=alice.id)

    assert conversation.id.startswith("conv_")
    assert conversation.user_id == alice.id
    assert conversation.primary_paper_id.startswith("paper_")
    assert conversation.title == "A Test Paper"
    assert conversation.head_message_id is None
    assert conversation.head_checkpoint_id is None
    assert detail is not None
    assert detail.conversation == conversation
    assert detail.primary_paper.id == conversation.primary_paper_id
    assert detail.primary_paper.authors == ["Ada Lovelace"]
    assert detail.active_papers == [detail.primary_paper]
    assert len(detail.paper_associations) == 1
    association = detail.paper_associations[0]
    assert association.conversation_id == conversation.id
    assert association.paper_id == conversation.primary_paper_id
    assert association.role == "primary"
    assert association.added_by == "user"
    assert association.source_task_id is None
    assert association.source_message_id is None
    assert association.is_active is True
    assert detail.active_task is None
    assert _table_count(db_path, "papers") == 1
    assert _table_count(db_path, "conversations") == 1
    assert _table_count(db_path, "conversation_papers") == 1


def test_create_conversation_reuses_paper_id_and_refreshes_public_metadata(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    store = TaskStore(db_path)
    alice = _create_test_user(store, "alice")
    first = store.create_conversation(user_id=alice.id, paper=_paper(), title=None)

    second = store.create_conversation(
        user_id=alice.id,
        paper=_paper(
            title="Updated Paper Title",
            authors=["Ada Lovelace", "Grace Hopper"],
            abstract="updated abstract",
            source_url="https://arxiv.org/abs/2401.12345v2",
        ),
        title="A separate conversation",
    )
    detail = store.get_conversation_detail(second.id, user_id=alice.id)

    assert second.primary_paper_id == first.primary_paper_id
    assert detail is not None
    assert detail.primary_paper.title == "Updated Paper Title"
    assert detail.primary_paper.authors == ["Ada Lovelace", "Grace Hopper"]
    assert detail.primary_paper.abstract == "updated abstract"
    assert _table_count(db_path, "papers") == 1
    assert _table_count(db_path, "conversations") == 2


def test_create_conversation_rolls_back_paper_and_conversation_on_link_failure(
    tmp_path,
):
    db_path = tmp_path / "tasks.sqlite3"
    store = TaskStore(db_path)
    alice = _create_test_user(store, "alice")
    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TRIGGER reject_primary_conversation_paper
            BEFORE INSERT ON conversation_papers
            BEGIN
                SELECT RAISE(ABORT, 'reject primary association');
            END
            """
        )

    with pytest.raises(IntegrityError, match="reject primary association"):
        store.create_conversation(user_id=alice.id, paper=_paper(), title=None)

    assert _table_count(db_path, "papers") == 0
    assert _table_count(db_path, "conversations") == 0
    assert _table_count(db_path, "conversation_papers") == 0


def test_conversation_reads_and_updates_are_scoped_to_owner(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice")
    bob = _create_test_user(store, "bob")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(),
        title=None,
    )

    assert store.get_conversation_detail(conversation.id, user_id=bob.id) is None
    assert store.update_conversation(
        conversation.id,
        user_id=bob.id,
        title="Bob's title",
    ) is None
    assert store.list_conversations(user_id=bob.id) == []
    assert store.get_conversation_detail(
        conversation.id,
        user_id=alice.id,
    ).conversation.title == "A Test Paper"


def test_list_conversations_orders_stably_and_hides_archived_by_default(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice")
    conversations = [
        store.create_conversation(
            user_id=alice.id,
            paper=_paper(external_id=f"2401.1234{index}v1"),
            title=f"Conversation {index}",
        )
        for index in range(3)
    ]
    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE conversations SET updated_at = ?",
            ("2026-08-07T08:00:00+00:00",),
        )
    archived = conversations[1]
    updated = store.update_conversation(
        archived.id,
        user_id=alice.id,
        archived=True,
    )

    visible = store.list_conversations(user_id=alice.id)
    all_conversations = store.list_conversations(
        user_id=alice.id,
        include_archived=True,
    )

    assert updated is not None
    assert updated.archived_at is not None
    assert archived.id not in [item.id for item in visible]
    assert [item.id for item in visible] == sorted(
        (item.id for item in conversations if item.id != archived.id),
        reverse=True,
    )
    assert all_conversations[0].id == archived.id
    assert {item.id for item in all_conversations} == {
        item.id for item in conversations
    }


def test_update_conversation_allows_title_but_rejects_archive_with_active_task(
    tmp_path,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(),
        title=None,
    )
    task = store.create_task(question="active research", user_id=alice.id)
    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE research_tasks SET conversation_id = ? WHERE id = ?",
            (conversation.id, task.id),
        )

    renamed = store.update_conversation(
        conversation.id,
        user_id=alice.id,
        title="Renamed while running",
    )
    detail = store.get_conversation_detail(conversation.id, user_id=alice.id)

    assert renamed is not None
    assert renamed.title == "Renamed while running"
    assert detail is not None
    assert detail.active_task is not None
    assert detail.active_task.id == task.id
    with pytest.raises(ValueError, match="active task"):
        store.update_conversation(
            conversation.id,
            user_id=alice.id,
            archived=True,
        )
    assert store.get_conversation_detail(
        conversation.id,
        user_id=alice.id,
    ).conversation.archived_at is None


@pytest.mark.parametrize("title", ["   ", "x" * 201])
def test_create_conversation_rejects_invalid_explicit_title(tmp_path, title):
    db_path = tmp_path / "tasks.sqlite3"
    store = TaskStore(db_path)
    alice = _create_test_user(store, "alice")

    with pytest.raises(ValueError, match="between 1 and 200"):
        store.create_conversation(user_id=alice.id, paper=_paper(), title=title)

    assert _table_count(db_path, "papers") == 0
    assert _table_count(db_path, "conversations") == 0


def test_explicit_conversation_title_allows_long_paper_metadata_title(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice")

    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(title="P" * 201),
        title="Readable conversation title",
    )
    detail = store.get_conversation_detail(conversation.id, user_id=alice.id)

    assert conversation.title == "Readable conversation title"
    assert detail is not None
    assert detail.primary_paper.title == "P" * 201


@pytest.mark.parametrize("title", ["   ", "x" * 201])
def test_update_conversation_rejects_invalid_title_without_changing_row(
    tmp_path,
    title,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(),
        title=None,
    )

    with pytest.raises(ValueError, match="between 1 and 200"):
        store.update_conversation(
            conversation.id,
            user_id=alice.id,
            title=title,
        )

    assert store.get_conversation_detail(
        conversation.id,
        user_id=alice.id,
    ).conversation.title == "A Test Paper"
