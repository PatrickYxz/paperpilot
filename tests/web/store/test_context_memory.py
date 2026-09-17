"""Persistence tests for internal context memory tables."""
from __future__ import annotations

import pytest

from paperpilot.papers import PaperCandidate
from paperpilot.web.store.records import (
    CompressionOutcomeRecord,
    NewContextArtifact,
    TurnArchiveSeedRecord,
)
from paperpilot.web.task_store import TaskStore


PAPER = PaperCandidate(
    external_id="2401.12345v1",
    title="Paper",
    authors=["Author"],
    abstract="Abstract",
    source_url="https://arxiv.org/abs/2401.12345v1",
)


def _turn(store: TaskStore):
    user = store.create_user(
        username="alice",
        password_hash="hash",
        password_salt="salt",
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


def _artifact(turn) -> NewContextArtifact:
    return NewContextArtifact(
        artifact_id="artifact-1",
        conversation_id=turn.task.conversation_id,
        task_id=turn.task.id,
        tool_call_id="call-1",
        tool_name="search_related_papers",
        kind="search",
        storage_key="artifact-1.json",
        sha256="a" * 64,
        byte_size=10,
        token_estimate=4,
        preview='{"items":[]}',
        initial_action="EXTERNALIZE_NOW",
        future_retention="CLEARABLE_AFTER_USE",
        created_at="2026-09-01T00:00:00+00:00",
    )


def _seed(
    turn,
    archive_id: str,
    message_id: str,
    *,
    archive_version: str = "turn-archive-v1",
) -> TurnArchiveSeedRecord:
    return TurnArchiveSeedRecord(
        archive_id=archive_id,
        conversation_id=turn.task.conversation_id,
        task_id=turn.task.id,
        user_message_id=message_id,
        terminal_status="success",
        archive_version=archive_version,
        seed_json={"user_goal": "question", "verification": ["pass"]},
        supersedes_json=[],
        created_at="2026-09-01T00:00:00+00:00",
    )


def test_context_artifact_can_be_found_by_id_and_digest(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    _user, _conversation, turn = _turn(store)
    record = store.create_context_artifact(record=_artifact(turn))

    assert store.get_context_artifact(record.artifact_id, conversation_id=_conversation.id) == record
    assert store.find_context_artifact_by_digest(
        conversation_id=_conversation.id,
        tool_name="search_related_papers",
        sha256="a" * 64,
    ) == record
    assert store.get_context_artifact(record.artifact_id, conversation_id="other") is None
    store.close()


def test_archive_seed_is_idempotent_append_only_and_narrative_lifecycle(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    _user, conversation, turn = _turn(store)
    first = _seed(turn, "archive-1", turn.user_message.id)
    assert store.seed_turn_archive(seed=first) == store.seed_turn_archive(seed=first)
    redelivered = TurnArchiveSeedRecord(
        **{**first.__dict__, "created_at": "2026-09-01T00:00:01+00:00"}
    )
    assert store.seed_turn_archive(seed=redelivered).archive_id == first.archive_id
    assert [item.archive_id for item in store.list_turn_archives(conversation.id)] == [
        "archive-1"
    ]

    claimed = store.claim_turn_archive_narrative("archive-1")
    assert claimed is not None and claimed.narrative_status == "running"
    completed = store.complete_turn_archive_narrative(
        "archive-1", narrative_summary="A bounded summary"
    )
    assert completed.narrative_status == "complete"
    assert completed.narrative_summary == "A bounded summary"

    second = _seed(
        turn,
        "archive-2",
        turn.user_message.id,
        archive_version="turn-archive-v2",
    )
    assert store.seed_turn_archive(seed=second).archive_id == "archive-2"
    assert [item.archive_id for item in store.list_turn_archives(conversation.id)] == [
        "archive-1",
        "archive-2",
    ]
    assert store.claim_turn_archive_narrative("archive-2") is not None
    failed = store.fail_turn_archive_narrative("archive-2")
    assert failed.narrative_status == "failed"
    store.close()


def test_archive_seed_conflict_is_rejected(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    _user, _conversation, turn = _turn(store)
    store.seed_turn_archive(seed=_seed(turn, "archive-1", turn.user_message.id))
    conflicting = _seed(turn, "archive-other", turn.user_message.id)
    with pytest.raises(ValueError, match="conflict"):
        store.seed_turn_archive(seed=conflicting)
    store.close()


def test_compression_state_starts_closed_and_outcome_event_is_atomic(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    _user, conversation, turn = _turn(store)
    initial = store.get_compression_state(conversation.id, "compressor-v1")
    assert initial.state == "CLOSED"
    assert initial.consecutive_failures == 0

    outcome = CompressionOutcomeRecord(
        conversation_id=conversation.id,
        compressor_version="compressor-v1",
        success=False,
        failure_type="schema_invalid",
        input_digest="b" * 64,
        stage="full",
        reason="candidate_rejected",
        before_tokens=100,
        after_tokens=100,
        reclaimed_tokens=0,
        protected_item_count=1,
        archive_ref_count=0,
        artifact_ref_count=0,
    )
    updated = store.record_compression_outcome(outcome=outcome)
    assert updated.consecutive_failures == 1
    events = store.list_events_page(turn.task.id, user_id=None, after_id=0, limit=100)
    assert events is not None
    assert any(event.type == "compression_candidate_rejected" for event in events.items)
    store.close()


def test_compression_state_uses_configured_failure_threshold(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    _user, conversation, turn = _turn(store)
    outcome = CompressionOutcomeRecord(
        conversation_id=conversation.id,
        compressor_version="compressor-threshold-2",
        success=False,
        failure_type="schema_invalid",
        input_digest="b" * 64,
        stage="full",
        reason="candidate_rejected",
        before_tokens=100,
        after_tokens=100,
        reclaimed_tokens=0,
        protected_item_count=1,
        archive_ref_count=0,
        artifact_ref_count=0,
        task_id=turn.task.id,
        failure_threshold=2,
    )

    first = store.record_compression_outcome(outcome=outcome)
    second = store.record_compression_outcome(outcome=outcome)

    assert first.state == "CLOSED"
    assert second.state == "OPEN"
    store.close()


def test_breaker_state_updates_do_not_duplicate_candidate_events_or_open_events(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    _user, conversation, turn = _turn(store)
    outcome = CompressionOutcomeRecord(
        conversation_id=conversation.id,
        compressor_version="compressor-no-duplicate-events",
        success=False,
        failure_type="schema_invalid",
        input_digest="b" * 64,
        stage="full",
        reason="compression_failed",
        before_tokens=0,
        after_tokens=0,
        reclaimed_tokens=0,
        protected_item_count=0,
        archive_ref_count=0,
        artifact_ref_count=0,
        task_id=turn.task.id,
        failure_threshold=2,
    )
    for _ in range(3):
        store.record_compression_outcome(outcome=outcome)

    events = store.list_events_page(
        turn.task.id,
        user_id=None,
        after_id=0,
        limit=100,
    )
    assert events is not None
    assert [item.type for item in events.items].count("compression_candidate_rejected") == 0
    assert [item.type for item in events.items].count("compression_circuit_open") == 1
    store.close()


def test_compression_probe_is_a_single_atomic_half_open_transition(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    _user, conversation, turn = _turn(store)
    for index in range(3):
        store.record_compression_outcome(
            outcome=CompressionOutcomeRecord(
                conversation_id=conversation.id,
                compressor_version="compressor-v1",
                success=False,
                failure_type="transient",
                input_digest=f"digest-{index}",
                stage="full",
                reason="candidate_rejected",
                before_tokens=100,
                after_tokens=100,
                reclaimed_tokens=0,
                protected_item_count=0,
                archive_ref_count=0,
                artifact_ref_count=0,
                task_id=turn.task.id,
            )
        )
    assert store.get_compression_state(conversation.id, "compressor-v1").state == "OPEN"
    assert store.claim_compression_probe(conversation.id, "compressor-v1").state == "HALF_OPEN"
    assert store.claim_compression_probe(conversation.id, "compressor-v1") is None
    store.close()
