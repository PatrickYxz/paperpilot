"""Persistence tests for the append-only user memory table."""
from __future__ import annotations

from paperpilot.papers import PaperCandidate
from paperpilot.web.store.records import NewUserMemory
from paperpilot.web.task_store import TaskStore

PAPER = PaperCandidate(
    external_id="2401.12345v1",
    title="Paper",
    authors=["Author"],
    abstract="Abstract",
    source_url="https://arxiv.org/abs/2401.12345v1",
)


def _turn(store: TaskStore, username: str = "alice"):
    user = store.create_user(
        username=username, password_hash="hash", password_salt="salt"
    )
    conversation = store.create_conversation(user_id=user.id, paper=PAPER)
    detail = store.get_conversation_detail(conversation.id, user_id=user.id)
    assert detail is not None
    turn = store.create_conversation_turn(
        user_id=user.id,
        conversation_id=conversation.id,
        content="question",
        depth="standard",
        expected_head_message_id=detail.conversation.head_message_id,
    )
    return user, conversation, turn


def _memory(turn, *, memory_id="mem-1", user_id=None, created_at="2026-10-05T10:00:00Z"):
    user_id = user_id or _user_of(turn)
    return NewUserMemory(
        memory_id=memory_id,
        user_id=user_id,
        kind="preference",
        content="Prefers papers with released code.",
        context={},
        source_conversation_id=turn.task.conversation_id,
        source_task_id=turn.task.id,
        source_message_id=turn.user_message.id,
        support_span="papers with released code",
        created_at=created_at,
    )


def _user_of(turn) -> str:
    # turn.task.user_id is set for conversation-bound turns in this fixture.
    return turn.task.user_id


def test_append_and_list_roundtrip(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user, _conversation, turn = _turn(store)
    record = store.append_user_memory(record=_memory(turn, user_id=user.id))

    assert record.status == "active"
    listed = store.list_user_memories(user.id)
    assert len(listed) == 1
    assert listed[0].content == "Prefers papers with released code."
    assert listed[0].context == {}
    assert store.count_task_memories(turn.task.id) == 1


def test_append_is_idempotent_on_memory_id(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user, _conversation, turn = _turn(store)
    first = store.append_user_memory(record=_memory(turn, user_id=user.id))
    second = store.append_user_memory(record=_memory(turn, user_id=user.id))

    assert second == first
    assert len(store.list_user_memories(user.id)) == 1


def test_listing_is_user_scoped_and_newest_first(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user_a, _conv_a, turn_a = _turn(store, "alice")
    user_b, _conv_b, turn_b = _turn(store, "bob")

    store.append_user_memory(
        record=_memory(turn_a, memory_id="a-old", user_id=user_a.id,
                       created_at="2026-09-01T00:00:00Z")
    )
    store.append_user_memory(
        record=_memory(turn_a, memory_id="a-new", user_id=user_a.id,
                       created_at="2026-10-01T00:00:00Z")
    )
    store.append_user_memory(record=_memory(turn_b, user_id=user_b.id))

    assert [m.memory_id for m in store.list_user_memories(user_a.id)] == [
        "a-new",
        "a-old",
    ]
    assert len(store.list_user_memories(user_b.id)) == 1


def test_count_scoped_to_source_task(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user, _conversation, turn = _turn(store)
    store.append_user_memory(record=_memory(turn, user_id=user.id))

    assert store.count_task_memories(turn.task.id) == 1
    assert store.count_task_memories("task-that-never-extracted") == 0
