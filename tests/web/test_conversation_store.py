"""Paper and conversation persistence tests."""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, BrokenBarrierError
from types import SimpleNamespace

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError

from paperpilot.papers import PaperCandidate
from paperpilot.web import task_store as task_store_module
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


def _queued_event_id(store: TaskStore, task_id: str) -> int:
    with sqlite3.connect(store.db_path) as connection:
        return int(
            connection.execute(
                "SELECT id FROM task_events WHERE task_id = ? AND type = 'queued'",
                (task_id,),
            ).fetchone()[0]
        )


def _fix_turn_ids_and_time(monkeypatch, *hex_values: str) -> None:
    values = iter(hex_values)
    monkeypatch.setattr(
        task_store_module.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex=next(values)),
    )
    monkeypatch.setattr(
        task_store_module,
        "_utc_now",
        lambda: "2026-08-07T09:00:00+00:00",
    )


def _insert_completed_turn(
    store: TaskStore,
    *,
    conversation_id: str,
    suffix: str,
    parent_message_id: str | None,
    make_head: bool = False,
    assistant_status: str = "complete",
) -> tuple[str, str, str]:
    task_id = f"task_{suffix}"
    user_message_id = f"msg_{suffix}_user"
    assistant_message_id = f"msg_{suffix}_assistant"
    with store.engine.begin() as connection:
        user_id = connection.exec_driver_sql(
            "SELECT user_id FROM conversations WHERE id = ?",
            (conversation_id,),
        ).scalar_one()
        connection.exec_driver_sql(
            """
            INSERT INTO research_tasks (
                id, question, depth, status, created_at, updated_at, user_id,
                conversation_id, base_checkpoint_id, final_checkpoint_id,
                result_quality
            ) VALUES (?, ?, 'standard', 'completed', ?, ?, ?, ?, ?, ?, 'complete')
            """,
            (
                task_id,
                f"Question {suffix}",
                f"2026-08-07T08:00:{suffix[-1]}0+00:00",
                f"2026-08-07T08:00:{suffix[-1]}1+00:00",
                user_id,
                conversation_id,
                f"cp_base_{suffix}" if parent_message_id else None,
                f"cp_final_{suffix}",
            ),
        )
        connection.exec_driver_sql(
            """
            INSERT INTO messages (
                id, conversation_id, task_id, parent_message_id, role,
                content, status, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, 'user', ?, 'complete', '{}', ?)
            """,
            (
                user_message_id,
                conversation_id,
                task_id,
                parent_message_id,
                f"Question {suffix}",
                f"2026-08-07T08:00:{suffix[-1]}0+00:00",
            ),
        )
        connection.exec_driver_sql(
            """
            INSERT INTO messages (
                id, conversation_id, task_id, parent_message_id, role,
                content, status, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, 'assistant', ?, ?, '{}', ?)
            """,
            (
                assistant_message_id,
                conversation_id,
                task_id,
                user_message_id,
                f"Answer {suffix}",
                assistant_status,
                f"2026-08-07T08:00:{suffix[-1]}1+00:00",
            ),
        )
        if make_head:
            connection.exec_driver_sql(
                """
                UPDATE conversations
                SET head_message_id = ?, head_checkpoint_id = ?
                WHERE id = ?
                """,
                (assistant_message_id, f"cp_final_{suffix}", conversation_id),
            )
    return task_id, user_message_id, assistant_message_id


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


def test_concurrent_conversation_creation_upserts_one_shared_paper(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    store = TaskStore(db_path)
    users = [
        _create_test_user(store, "alice"),
        _create_test_user(store, "bob"),
    ]
    paper_writers_ready = Barrier(2)

    def synchronize_paper_writers(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        normalized = " ".join(statement.split())
        if normalized.startswith("INSERT INTO papers"):
            try:
                paper_writers_ready.wait(timeout=5)
            except BrokenBarrierError as exc:
                raise AssertionError(
                    "both paper writers did not reach INSERT within 5 seconds"
                ) from exc

    def create_for(user_id: str):
        return store.create_conversation(
            user_id=user_id,
            paper=_paper(),
            title=None,
        )

    event.listen(
        store.engine,
        "before_cursor_execute",
        synchronize_paper_writers,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            conversations = list(
                executor.map(create_for, (user.id for user in users))
            )
    finally:
        event.remove(
            store.engine,
            "before_cursor_execute",
            synchronize_paper_writers,
        )

    assert len(conversations) == 2
    assert len({item.primary_paper_id for item in conversations}) == 1
    assert _table_count(db_path, "papers") == 1
    assert _table_count(db_path, "conversations") == 2
    assert _table_count(db_path, "conversation_papers") == 2


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


def test_create_conversation_turn_writes_user_task_and_queued_event_atomically(
    tmp_path,
):
    db_path = tmp_path / "tasks.sqlite3"
    store = TaskStore(db_path)
    alice = _create_test_user(store, "alice-turn")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20001v1"),
    )

    turn = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content=" Explain the main contribution. ",
        depth="standard",
        expected_head_message_id=None,
    )
    events = store.list_events_page(
        turn.task.id,
        user_id=alice.id,
        after_id=0,
        limit=100,
    )
    detail = store.get_conversation_detail(conversation.id, user_id=alice.id)

    assert turn.user_message.parent_message_id is None
    assert turn.user_message.task_id == turn.task.id
    assert turn.user_message.role == "user"
    assert turn.user_message.status == "complete"
    assert turn.user_message.content == "Explain the main contribution."
    assert turn.task.conversation_id == conversation.id
    assert turn.task.base_checkpoint_id is None
    assert turn.task.final_checkpoint_id is None
    assert turn.task.result_quality is None
    assert events is not None
    assert [item.type for item in events.items] == ["queued"]
    assert detail is not None
    assert detail.conversation.head_message_id is None
    assert _table_count(db_path, "messages") == 1
    assert _table_count(db_path, "research_tasks") == 1
    assert _table_count(db_path, "task_events") == 1


def test_create_conversation_turn_rolls_back_message_and_task_when_event_fails(
    tmp_path,
):
    db_path = tmp_path / "tasks.sqlite3"
    store = TaskStore(db_path)
    alice = _create_test_user(store, "alice-event-rollback")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20002v1"),
    )
    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TRIGGER reject_conversation_queued_event
            BEFORE INSERT ON task_events
            BEGIN
                SELECT RAISE(ABORT, 'reject queued event');
            END
            """
        )

    with pytest.raises(IntegrityError, match="reject queued event"):
        store.create_conversation_turn(
            user_id=alice.id,
            conversation_id=conversation.id,
            content="Will roll back",
            depth="quick",
            expected_head_message_id=None,
        )

    assert _table_count(db_path, "messages") == 0
    assert _table_count(db_path, "research_tasks") == 0
    assert _table_count(db_path, "task_events") == 0


def test_create_conversation_turn_reports_busy_before_stale_head(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-busy")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20003v1"),
    )
    store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="First",
        depth="standard",
        expected_head_message_id=None,
    )

    with pytest.raises(task_store_module.ConversationBusyError):
        store.create_conversation_turn(
            user_id=alice.id,
            conversation_id=conversation.id,
            content="Second",
            depth="standard",
            expected_head_message_id="msg_stale",
        )


def test_create_conversation_turn_rejects_stale_stable_head(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-stale")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20004v1"),
    )

    with pytest.raises(task_store_module.StaleConversationHeadError):
        store.create_conversation_turn(
            user_id=alice.id,
            conversation_id=conversation.id,
            content="Stale tab",
            depth="standard",
            expected_head_message_id="msg_stale",
        )

    assert _table_count(store.db_path, "messages") == 0
    assert _table_count(store.db_path, "research_tasks") == 0


def test_concurrent_turns_share_real_sqlite_and_map_unique_race_to_busy(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    first_store = TaskStore(db_path)
    alice = _create_test_user(first_store, "alice-race")
    conversation = first_store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20005v1"),
    )
    second_store = TaskStore(db_path)
    writers_ready = Barrier(2)

    def synchronize_task_insert(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        if not " ".join(statement.split()).startswith("INSERT INTO research_tasks"):
            return
        try:
            writers_ready.wait(timeout=5)
        except BrokenBarrierError as exc:
            raise AssertionError("both TaskStore writers did not reach INSERT") from exc

    for store in (first_store, second_store):
        event.listen(store.engine, "before_cursor_execute", synchronize_task_insert)

    def create(store: TaskStore, content: str):
        try:
            return store.create_conversation_turn(
                user_id=alice.id,
                conversation_id=conversation.id,
                content=content,
                depth="standard",
                expected_head_message_id=None,
            )
        except Exception as exc:  # return both race outcomes for exact assertions
            return exc

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = [
                executor.submit(create, first_store, "First writer"),
                executor.submit(create, second_store, "Second writer"),
            ]
            results = [future.result() for future in outcomes]
    finally:
        for store in (first_store, second_store):
            event.remove(store.engine, "before_cursor_execute", synchronize_task_insert)

    assert sum(
        isinstance(result, task_store_module.ConversationTurn) for result in results
    ) == 1
    assert sum(
        isinstance(result, task_store_module.ConversationBusyError)
        for result in results
    ) == 1
    assert _table_count(db_path, "messages") == 1
    assert _table_count(db_path, "research_tasks") == 1
    assert _table_count(db_path, "task_events") == 1


def test_two_stores_can_create_turns_for_different_conversations_in_parallel(
    tmp_path,
):
    db_path = tmp_path / "tasks.sqlite3"
    first_store = TaskStore(db_path)
    alice = _create_test_user(first_store, "alice-parallel")
    conversations = [
        first_store.create_conversation(
            user_id=alice.id,
            paper=_paper(external_id=f"2401.2001{index}v1"),
        )
        for index in range(2)
    ]
    second_store = TaskStore(db_path)
    writers_ready = Barrier(2)

    def synchronize_task_insert(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        if " ".join(statement.split()).startswith("INSERT INTO research_tasks"):
            writers_ready.wait(timeout=5)

    for store in (first_store, second_store):
        event.listen(store.engine, "before_cursor_execute", synchronize_task_insert)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    store.create_conversation_turn,
                    user_id=alice.id,
                    conversation_id=conversation.id,
                    content=f"Question {index}",
                    depth="quick",
                    expected_head_message_id=None,
                )
                for index, (store, conversation) in enumerate(
                    zip((first_store, second_store), conversations, strict=True)
                )
            ]
            turns = [future.result() for future in futures]
    finally:
        for store in (first_store, second_store):
            event.remove(store.engine, "before_cursor_execute", synchronize_task_insert)

    assert {turn.task.conversation_id for turn in turns} == {
        conversation.id for conversation in conversations
    }
    assert _table_count(db_path, "messages") == 2
    assert _table_count(db_path, "research_tasks") == 2


def test_active_path_and_alternatives_follow_only_direct_complete_turns(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-tree")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20020v1"),
    )
    _, root_user, root_assistant = _insert_completed_turn(
        store,
        conversation_id=conversation.id,
        suffix="1",
        parent_message_id=None,
    )
    _, branch_a_user, branch_a_assistant = _insert_completed_turn(
        store,
        conversation_id=conversation.id,
        suffix="2",
        parent_message_id=root_assistant,
    )
    _, branch_b_user, branch_b_assistant = _insert_completed_turn(
        store,
        conversation_id=conversation.id,
        suffix="3",
        parent_message_id=root_assistant,
        make_head=True,
    )
    _, incomplete_user, _ = _insert_completed_turn(
        store,
        conversation_id=conversation.id,
        suffix="4",
        parent_message_id=root_assistant,
        assistant_status="failed",
    )

    active = store.list_active_messages(conversation.id, user_id=alice.id)
    alternatives = store.list_message_alternatives(
        conversation.id,
        root_assistant,
        user_id=alice.id,
    )

    assert active is not None
    assert [message.id for message in active] == [
        root_user,
        root_assistant,
        branch_b_user,
        branch_b_assistant,
    ]
    assert alternatives is not None
    assert {
        (item.user_message.id, item.assistant_message.id) for item in alternatives
    } == {
        (branch_a_user, branch_a_assistant),
        (branch_b_user, branch_b_assistant),
    }
    assert incomplete_user not in {
        item.user_message.id for item in alternatives
    }
    assert store.get_message(
        conversation.id,
        branch_a_assistant,
        user_id=alice.id,
    ).id == branch_a_assistant


def test_message_reads_and_tree_queries_are_scoped_to_owner(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-owner")
    bob = _create_test_user(store, "bob-owner")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20021v1"),
    )
    _, _, assistant = _insert_completed_turn(
        store,
        conversation_id=conversation.id,
        suffix="5",
        parent_message_id=None,
        make_head=True,
    )

    assert store.get_message(conversation.id, assistant, user_id=bob.id) is None
    assert store.list_active_messages(conversation.id, user_id=bob.id) is None
    assert store.list_message_alternatives(
        conversation.id,
        assistant,
        user_id=bob.id,
    ) is None
    assert store.get_unstable_turn(conversation.id, user_id=bob.id) is None


def test_task_message_and_unstable_turn_keep_stable_head_unchanged(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-unstable")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20022v1"),
    )
    _, _, stable_head = _insert_completed_turn(
        store,
        conversation_id=conversation.id,
        suffix="6",
        parent_message_id=None,
        make_head=True,
    )

    turn = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Unstable follow-up",
        depth="deep",
        expected_head_message_id=stable_head,
    )

    assert store.get_task_message(turn.task.id, "user") == turn.user_message
    assert store.get_task_message(turn.task.id, "assistant") is None
    assert store.get_unstable_turn(conversation.id, user_id=alice.id) == turn
    assert turn.user_message.parent_message_id == stable_head
    assert turn.task.base_checkpoint_id == "cp_final_6"
    assert store.get_conversation_detail(
        conversation.id,
        user_id=alice.id,
    ).conversation.head_message_id == stable_head

    store.update_status(turn.task.id, "failed")
    failed = store.get_unstable_turn(conversation.id, user_id=alice.id)
    assert failed is not None
    assert failed.task.status == "failed"
    assert failed.user_message == turn.user_message


def test_unstable_turn_prefers_active_retry_over_larger_same_second_failed_id(
    tmp_path,
    monkeypatch,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-active-retry-order")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20027v1"),
    )
    _fix_turn_ids_and_time(
        monkeypatch,
        "f" * 32,
        "1" * 32,
        "0" * 32,
        "2" * 32,
    )
    older_failed = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Older failed attempt",
        depth="standard",
        expected_head_message_id=None,
    )
    store.update_status(older_failed.task.id, "failed")
    newer_active = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="New active retry",
        depth="standard",
        expected_head_message_id=None,
    )
    store.update_status(newer_active.task.id, "running")

    assert older_failed.task.id > newer_active.task.id
    assert older_failed.task.created_at == newer_active.task.created_at
    assert _queued_event_id(store, older_failed.task.id) < _queued_event_id(
        store,
        newer_active.task.id,
    )
    unstable = store.get_unstable_turn(conversation.id, user_id=alice.id)
    assert unstable is not None
    assert unstable.task.id == newer_active.task.id
    assert unstable.task.status == "running"


def test_unstable_turn_uses_queued_event_order_for_same_second_failed_tasks(
    tmp_path,
    monkeypatch,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-failed-retry-order")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20028v1"),
    )
    _fix_turn_ids_and_time(
        monkeypatch,
        "f" * 32,
        "3" * 32,
        "0" * 32,
        "4" * 32,
    )
    older_failed = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Older failed attempt",
        depth="quick",
        expected_head_message_id=None,
    )
    store.update_status(older_failed.task.id, "failed")
    newer_failed = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Newer failed attempt",
        depth="quick",
        expected_head_message_id=None,
    )
    store.update_status(newer_failed.task.id, "failed")
    later_old_diagnostic = store.add_event(
        task_id=older_failed.task.id,
        type="diagnostic",
        message="A late diagnostic for the older failed turn.",
    )

    assert older_failed.task.id > newer_failed.task.id
    assert older_failed.task.created_at == newer_failed.task.created_at
    older_queued_id = _queued_event_id(store, older_failed.task.id)
    newer_queued_id = _queued_event_id(store, newer_failed.task.id)
    assert older_queued_id < newer_queued_id
    assert newer_queued_id < later_old_diagnostic.id
    unstable = store.get_unstable_turn(conversation.id, user_id=alice.id)
    assert unstable is not None
    assert unstable.task.id == newer_failed.task.id
    assert unstable.task.status == "failed"


def test_active_path_rejects_missing_parent_with_diagnostic_error(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-missing-parent")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20023v1"),
    )
    _, _, head = _insert_completed_turn(
        store,
        conversation_id=conversation.id,
        suffix="7",
        parent_message_id=None,
        make_head=True,
    )
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE messages SET parent_message_id = 'msg_missing' WHERE id = ?",
            (head,),
        )

    with pytest.raises(RuntimeError, match="missing parent"):
        store.list_active_messages(conversation.id, user_id=alice.id)


def test_active_path_rejects_parent_from_another_conversation(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-cross-parent")
    first = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20024v1"),
    )
    second = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20025v1"),
    )
    _, _, foreign_head = _insert_completed_turn(
        store,
        conversation_id=second.id,
        suffix="8",
        parent_message_id=None,
        make_head=True,
    )
    _, _, local_head = _insert_completed_turn(
        store,
        conversation_id=first.id,
        suffix="9",
        parent_message_id=None,
        make_head=True,
    )
    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE messages SET parent_message_id = ? WHERE id = ?",
            (foreign_head, local_head),
        )

    with pytest.raises(RuntimeError, match="another conversation"):
        store.list_active_messages(first.id, user_id=alice.id)


def test_active_path_detects_parent_cycle_instead_of_truncating(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-cycle")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.20026v1"),
    )
    _, user_message, assistant_message = _insert_completed_turn(
        store,
        conversation_id=conversation.id,
        suffix="0",
        parent_message_id=None,
        make_head=True,
    )
    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE messages SET parent_message_id = ? WHERE id = ?",
            (assistant_message, user_message),
        )

    with pytest.raises(RuntimeError, match="cycle"):
        store.list_active_messages(conversation.id, user_id=alice.id)
