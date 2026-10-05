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


def test_profile_upsert_roundtrip(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user, _conversation, _turn_rec = _turn(store)

    assert store.get_user_profile(user.id) is None
    record = store.upsert_user_profile(
        user_id=user.id,
        profile_text="PhD student working on model distillation.",
        source_memory_count=3,
    )
    assert record.source_memory_count == 3

    updated = store.upsert_user_profile(
        user_id=user.id,
        profile_text="Updated profile.",
        source_memory_count=8,
    )
    assert updated.profile_text == "Updated profile."
    assert store.get_user_profile(user.id).source_memory_count == 8


def test_list_user_turn_summaries_is_user_scoped_newest_first(tmp_path):
    from paperpilot.web.store.records import TurnArchiveSeedRecord

    store = TaskStore(tmp_path / "tasks.sqlite3")
    user_a, conv_a, turn_a = _turn(store, "alice")
    user_b, _conv_b, _turn_b = _turn(store, "bob")

    def _seed(conversation_id, turn, archive_id, created_at, narrative=None):
        from paperpilot.web.store.context_memory import (
            claim_turn_archive_narrative,
            complete_turn_archive_narrative,
            seed_turn_archive,
        )
        seed_turn_archive(
            session_factory=store._session_factory,
            seed=TurnArchiveSeedRecord(
                archive_id=archive_id,
                conversation_id=conversation_id,
                task_id=turn.task.id,
                user_message_id=turn.user_message.id,
                terminal_status="success",
                archive_version="turn-archive-v1",
                seed_json={"question": turn.user_message.content},
                supersedes_json=[],
                created_at=created_at,
            ),
        )
        if narrative is not None:
            claim_turn_archive_narrative(store._session_factory, archive_id)
            complete_turn_archive_narrative(
                store._session_factory,
                archive_id,
                narrative_summary=narrative,
            )

    _seed(conv_a.id, turn_a, "arch-1", "2026-09-01T00:00:00Z",
          narrative="Compared LoRA ranks.")
    from paperpilot.web.store.records import TurnArchiveSeedRecord as _SR
    from paperpilot.web.store.context_memory import seed_turn_archive as _sta
    _sta(
        store._session_factory,
        seed=_SR(
            archive_id="arch-2",
            conversation_id=conv_a.id,
            task_id=turn_a.task.id,
            user_message_id=turn_a.user_message.id,
            terminal_status="success",
            archive_version="turn-archive-v2",
            seed_json={"question": turn_a.user_message.content},
            supersedes_json=[],
            created_at="2026-09-20T00:00:00Z",
        ),
    )

    summaries = store.list_user_turn_summaries(user_a.id)
    assert [s.user_message_id for s in summaries] == [
        turn_a.user_message.id,
        turn_a.user_message.id,
    ] or len(summaries) == 2
    assert [s.created_at for s in summaries] == sorted(
        [s.created_at for s in summaries], reverse=True
    )
    assert summaries[0].narrative is None  # newest has no narrative yet
    assert "Compared LoRA ranks." in {
        s.narrative for s in summaries if s.narrative
    }
    assert store.list_user_turn_summaries(user_b.id) == []
