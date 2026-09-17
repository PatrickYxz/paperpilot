"""Atomic and bounded local ContextArtifactStore tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paperpilot.deep_reading.context_management.budget import ModelAwareTokenCounter
from paperpilot.papers import PaperCandidate
from paperpilot.web.context_artifacts import (
    ArtifactPutRequest,
    ArtifactIntegrityError,
    LocalContextArtifactStore,
)
from paperpilot.web.task_store import TaskStore


PAPER = PaperCandidate(
    external_id="2401.12345v1",
    title="Paper",
    authors=["Author"],
    abstract="Abstract",
    source_url="https://arxiv.org/abs/2401.12345v1",
)


def _context(tmp_path: Path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
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
    artifact_store = LocalContextArtifactStore(
        root=tmp_path / "context-artifacts",
        task_store=store,
        token_counter=ModelAwareTokenCounter(),
        read_max_tokens=20,
    )
    return store, conversation, turn, artifact_store


def _request(turn, payload=None) -> ArtifactPutRequest:
    return ArtifactPutRequest(
        conversation_id=turn.task.conversation_id,
        task_id=turn.task.id,
        tool_call_id="call-1",
        tool_name="search_related_papers",
        kind="search",
        payload=payload or {"items": [{"external_id": "p1", "title": "Paper 1"}]},
        preview='{"items":[{"external_id":"p1","title":"Paper 1"}]}',
        initial_action="EXTERNALIZE_NOW",
        future_retention="CLEARABLE_AFTER_USE",
    )


def test_put_writes_canonical_json_hash_and_reuses_duplicate_digest(tmp_path):
    store, conversation, turn, artifacts = _context(tmp_path)
    first = artifacts.put(_request(turn, {"z": 1, "a": "中文"}))
    path = tmp_path / "context-artifacts" / first.storage_key
    assert path.name == f"{first.artifact_id}.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": "中文", "z": 1}
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert first.sha256 == expected
    assert first.byte_size == path.stat().st_size
    assert artifacts.put(_request(turn, {"a": "中文", "z": 1})) == first
    assert len(list(path.parent.glob("*"))) == 1
    store.close()


def test_artifact_reads_are_bounded_and_reconstructable_by_cursor(tmp_path):
    store, conversation, turn, artifacts = _context(tmp_path)
    request = _request(turn, {"text": "abcdef中文" * 20})
    record = artifacts.put(request)
    path = tmp_path / "context-artifacts" / record.storage_key
    full_text = path.read_text(encoding="utf-8")

    cursor = 0
    chunks: list[str] = []
    while True:
        result = artifacts.read_slice(
            record.artifact_id,
            conversation_id=conversation.id,
            cursor=cursor,
            max_tokens=5,
        )
        chunks.append(result.text)
        assert result.actual_tokens <= 5
        if result.next_cursor is None:
            break
        cursor = result.next_cursor
    assert "".join(chunks) == full_text
    store.close()


def test_search_is_casefolded_ordered_and_budgeted(tmp_path):
    store, conversation, turn, artifacts = _context(tmp_path)
    record = artifacts.put(_request(turn, {"text": "Alpha beta ALPHA gamma alpha"}))
    matches = artifacts.search(
        record.artifact_id,
        conversation_id=conversation.id,
        query="alpha",
        max_matches=10,
    )
    assert [item.start for item in matches.matches] == sorted(
        item.start for item in matches.matches
    )
    assert matches.matches[0].start == 9
    assert matches.actual_tokens <= 20
    store.close()


@pytest.mark.parametrize(
    ("operation", "kwargs", "message"),
    [
        ("read", {"cursor": -1, "max_tokens": 1}, "cursor"),
        ("read", {"cursor": 0, "max_tokens": 21}, "max_tokens"),
        ("search", {"query": "", "max_matches": 1}, "query"),
        ("search", {"query": "x", "max_matches": 11}, "max_matches"),
    ],
)
def test_artifact_reads_reject_invalid_bounds(tmp_path, operation, kwargs, message):
    store, conversation, turn, artifacts = _context(tmp_path)
    record = artifacts.put(_request(turn))
    with pytest.raises(ValueError, match=message):
        if operation == "read":
            artifacts.read_slice(
                record.artifact_id,
                conversation_id=conversation.id,
                **kwargs,
            )
        else:
            artifacts.search(
                record.artifact_id,
                conversation_id=conversation.id,
                **kwargs,
            )
    store.close()


def test_artifact_store_rejects_cross_conversation_unknown_and_tampered_files(tmp_path):
    store, conversation, turn, artifacts = _context(tmp_path)
    record = artifacts.put(_request(turn))
    with pytest.raises(ValueError, match="conversation"):
        artifacts.read_slice(
            record.artifact_id,
            conversation_id="other",
            cursor=0,
            max_tokens=1,
        )
    with pytest.raises(ValueError, match="unknown"):
        artifacts.read_slice(
            "not-an-artifact",
            conversation_id=conversation.id,
            cursor=0,
            max_tokens=1,
        )
    path = tmp_path / "context-artifacts" / record.storage_key
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="hash"):
        artifacts.read_slice(
            record.artifact_id,
            conversation_id=conversation.id,
            cursor=0,
            max_tokens=1,
        )
    store.close()


def test_artifact_write_failure_does_not_publish_file_or_metadata(tmp_path, monkeypatch):
    store, _conversation, turn, artifacts = _context(tmp_path)
    def fail_create(**_kwargs):
        raise RuntimeError("metadata write failed")

    monkeypatch.setattr(store, "create_context_artifact", fail_create)
    with pytest.raises(RuntimeError, match="metadata"):
        artifacts.put(_request(turn))
    root = tmp_path / "context-artifacts"
    assert not list(root.glob("*.json"))
    assert store.find_context_artifact_by_digest(
        conversation_id=turn.task.conversation_id,
        tool_name="search_related_papers",
        sha256="a" * 64,
    ) is None
    store.close()
