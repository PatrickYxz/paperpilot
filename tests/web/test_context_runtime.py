"""Web composition tests for the optional context-management runtime."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from paperpilot.web.config import ContextManagementConfig, WebRuntimeConfig
from paperpilot.web.context_runtime import build_context_management_runtime
from paperpilot.web.task_store import TaskStore
from paperpilot.deep_reading.nodes.prepare_context import _migrate_legacy_summary_once
from paperpilot.papers import PaperCandidate


class _Model:
    pass


def test_disabled_context_runtime_has_no_artifact_root_or_database_side_effect(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    root = tmp_path / "artifacts"
    config = WebRuntimeConfig(
        context_management=ContextManagementConfig(
            enabled=False,
            artifact_root=root,
        )
    )
    try:
        assert build_context_management_runtime(store, config, _Model()) is None
        assert not root.exists()
    finally:
        store.close()


def test_enabled_runtime_shares_store_config_artifact_and_ports(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    config = WebRuntimeConfig(
        context_management=ContextManagementConfig(
            enabled=True,
            artifact_root=tmp_path / "artifacts",
        )
    )
    runtime = build_context_management_runtime(store, config, _Model())
    try:
        assert runtime is not None
        assert runtime.task_store is store
        assert runtime.config is config
        assert runtime.archive_store is store
        assert runtime.compression_state_port is store
        assert runtime.artifact_store._task_store is store
        with pytest.raises(FrozenInstanceError):
            runtime.config = config  # type: ignore[misc]
    finally:
        store.close()


def test_enabled_runtime_wires_compaction_retry_threshold_and_recent_turns(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    settings = ContextManagementConfig(
        enabled=True,
        full_compaction_recent_turns=4,
        compression_failure_threshold=5,
        compression_transient_retry_count=2,
        artifact_root=tmp_path / "artifacts",
    )
    runtime = build_context_management_runtime(
        store,
        WebRuntimeConfig(context_management=settings),
        _Model(),
        conversation_id="conversation-1",
    )
    try:
        assert runtime is not None
        assert runtime.compression_coordinator._recent_turns == 4
        assert runtime.compression_coordinator._transient_retry_count == 2
        assert runtime.breaker is not None
        assert runtime.breaker._failure_threshold == 5
    finally:
        store.close()


def test_legacy_summary_is_migrated_once_as_untrusted_archive(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user = store.create_user(
        username="legacy",
        password_hash="hash",
        password_salt="salt",
    )
    paper = PaperCandidate(
        external_id="2401.12345v1",
        title="Paper",
        authors=["Author"],
        abstract="Abstract",
        source_url="https://arxiv.org/abs/2401.12345v1",
    )
    conversation = store.create_conversation(user_id=user.id, paper=paper)
    detail = store.get_conversation_detail(conversation.id, user_id=user.id)
    assert detail is not None
    turn = store.create_conversation_turn(
        user_id=user.id,
        conversation_id=conversation.id,
        content="new goal",
        depth="standard",
        expected_head_message_id=detail.conversation.head_message_id,
    )
    context = SimpleNamespace(
        task_store=store,
        conversation_id=conversation.id,
        task_id=turn.task.id,
        current_user_message_id=turn.user_message.id,
    )
    state = {"conversation_summary": {"confirmed_facts": ["candidate"]}}
    try:
        _migrate_legacy_summary_once(state, context, turn.user_message)
        _migrate_legacy_summary_once(state, context, turn.user_message)
        archives = store.list_turn_archives(conversation.id)
        assert len(archives) == 1
        assert archives[0].archive_version == "legacy-summary-v1"
        assert archives[0].seed_json["untrusted"] is True
        assert archives[0].seed_json["legacy_summary"] == state["conversation_summary"]
    finally:
        store.close()
