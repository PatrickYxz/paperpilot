from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest

from paperpilot.web.task_store import (
    ConversationAlternative,
    ConversationBusyError,
    ConversationDetail,
    ConversationPaperRecord,
    ConversationRecord,
    ConversationTurn,
    DuplicateUsernameError,
    FinalizedConversationTask,
    MessageRecord,
    PaperRecord,
    PublishedConversationResult,
    ResearchTask,
    StaleConversationHeadError,
    TaskArtifact,
    TaskArtifactBatch,
    TaskEvent,
    TaskEventBatch,
    TaskStore,
    TaskUpdates,
    UsedPaperInput,
    VALID_DEPTHS,
    WebUser,
    CompressionOutcomeRecord,
    CompressionStateRecord,
    ContextArtifactRecord,
    NewContextArtifact,
    TurnArchiveRecord,
    TurnArchiveSeedRecord,
    NewUserMemory,
    UserMemoryRecord,
    UserProfileRecord,
    UserTurnSummary,
)


EXPECTED_TASK_STORE_METHODS = {
    "close",
    "create_user",
    "get_user_by_username",
    "get_user_by_id",
    "create_session",
    "get_user_for_session",
    "delete_session",
    "create_conversation",
    "list_conversations",
    "get_conversation_detail",
    "update_conversation",
    "create_conversation_turn",
    "publish_conversation_result",
    "finalize_conversation_task",
    "fail_conversation_task",
    "switch_conversation_head",
    "get_message",
    "get_task_message",
    "list_active_messages",
    "list_message_alternatives",
    "get_unstable_turn",
    "check_health",
    "get_task",
    "claim_task",
    "fail_pending_task",
    "add_event",
    "list_events_page",
    "add_artifact",
    "list_artifacts_page",
    "get_conversation_task_updates",
    "create_context_artifact",
    "get_context_artifact",
    "find_context_artifact_by_digest",
    "seed_turn_archive",
    "list_turn_archives",
    "claim_turn_archive_narrative",
    "complete_turn_archive_narrative",
    "fail_turn_archive_narrative",
    "get_compression_state",
    "claim_compression_probe",
    "record_compression_outcome",
    "is_context_artifact_referenced",
    "append_user_memory",
    "list_user_memories",
    "count_task_memories",
    "get_user_profile",
    "upsert_user_profile",
    "list_user_turn_summaries",
}


EXPECTED_TASK_STORE_SIGNATURES = {
    "close": "(self) -> 'None'",
    "create_user": "(self, *, username: 'str', password_hash: 'str', password_salt: 'str') -> 'WebUser'",
    "get_user_by_username": "(self, username: 'str') -> 'WebUser | None'",
    "get_user_by_id": "(self, user_id: 'str') -> 'WebUser | None'",
    "create_session": "(self, user_id: 'str') -> 'str'",
    "get_user_for_session": "(self, token: 'str') -> 'WebUser | None'",
    "delete_session": "(self, token: 'str') -> 'None'",
    "create_conversation": "(self, *, user_id: 'str', paper: 'PaperCandidate', title: 'str | None' = None) -> 'ConversationRecord'",
    "list_conversations": "(self, *, user_id: 'str', include_archived: 'bool' = False, limit: 'int' = 100) -> 'list[ConversationRecord]'",
    "get_conversation_detail": "(self, conversation_id: 'str', *, user_id: 'str') -> 'ConversationDetail | None'",
    "update_conversation": "(self, conversation_id: 'str', *, user_id: 'str', title: 'str | None' = None, archived: 'bool | None' = None) -> 'ConversationRecord | None'",
    "create_conversation_turn": "(self, *, user_id: 'str', conversation_id: 'str', content: 'str', depth: 'str', expected_head_message_id: 'str | None') -> 'ConversationTurn'",
    "publish_conversation_result": "(self, *, task_id: 'str', content: 'str', metadata: 'dict', used_papers: 'list[UsedPaperInput]') -> 'PublishedConversationResult'",
    "finalize_conversation_task": "(self, *, task_id: 'str', assistant_message_id: 'str', final_checkpoint_id: 'str', result_quality: 'str', active_paper_ids: 'list[str]') -> 'FinalizedConversationTask'",
    "fail_conversation_task": "(self, *, task_id: 'str', message: 'str', stage: 'str | None' = None, payload: 'dict | None' = None) -> 'ResearchTask | None'",
    "switch_conversation_head": "(self, conversation_id: 'str', *, user_id: 'str', expected_head_message_id: 'str | None', target_message_id: 'str', target_checkpoint_id: 'str', active_paper_ids: 'list[str]') -> 'ConversationRecord | None'",
    "get_message": "(self, conversation_id: 'str', message_id: 'str', *, user_id: 'str') -> 'MessageRecord | None'",
    "get_task_message": "(self, task_id: 'str', role: 'str') -> 'MessageRecord | None'",
    "list_active_messages": "(self, conversation_id: 'str', *, user_id: 'str') -> 'list[MessageRecord] | None'",
    "list_message_alternatives": "(self, conversation_id: 'str', message_id: 'str', *, user_id: 'str') -> 'list[ConversationAlternative] | None'",
    "get_unstable_turn": "(self, conversation_id: 'str', *, user_id: 'str') -> 'ConversationTurn | None'",
    "check_health": "(self) -> 'None'",
    "get_task": "(self, task_id: 'str', *, user_id: 'str | None' = None) -> 'ResearchTask | None'",
    "claim_task": "(self, task_id: 'str', *, allow_running: 'bool' = False) -> 'ResearchTask | None'",
    "fail_pending_task": "(self, task_id: 'str') -> 'ResearchTask | None'",
    "add_event": "(self, *, task_id: 'str', type: 'str', message: 'str', stage: 'str | None' = None, payload: 'dict | None' = None) -> 'TaskEvent'",
    "list_events_page": "(self, task_id: 'str', *, user_id: 'str | None', after_id: 'int', limit: 'int') -> 'TaskEventBatch | None'",
    "add_artifact": "(self, *, task_id: 'str', kind: 'str', title: 'str', content: 'str', payload: 'dict | None' = None) -> 'TaskArtifact'",
    "list_artifacts_page": "(self, task_id: 'str', *, user_id: 'str | None', after_id: 'int', limit: 'int') -> 'TaskArtifactBatch | None'",
    "get_conversation_task_updates": "(self, conversation_id: 'str', task_id: 'str', *, user_id: 'str', after_event_id: 'int' = 0, after_artifact_id: 'int' = 0, limit: 'int' = 50) -> 'TaskUpdates | None'",
    "create_context_artifact": "(self, *, record: 'NewContextArtifact') -> 'ContextArtifactRecord'",
    "get_context_artifact": "(self, artifact_id: 'str', *, conversation_id: 'str') -> 'ContextArtifactRecord | None'",
    "find_context_artifact_by_digest": "(self, *, conversation_id: 'str', tool_name: 'str', sha256: 'str') -> 'ContextArtifactRecord | None'",
    "seed_turn_archive": "(self, *, seed: 'TurnArchiveSeedRecord') -> 'TurnArchiveRecord'",
    "list_turn_archives": "(self, conversation_id: 'str') -> 'list[TurnArchiveRecord]'",
    "claim_turn_archive_narrative": "(self, archive_id: 'str') -> 'TurnArchiveRecord | None'",
    "complete_turn_archive_narrative": "(self, archive_id: 'str', *, narrative_summary: 'str') -> 'TurnArchiveRecord'",
    "fail_turn_archive_narrative": "(self, archive_id: 'str') -> 'TurnArchiveRecord'",
    "get_compression_state": "(self, conversation_id: 'str', compressor_version: 'str') -> 'CompressionStateRecord'",
    "claim_compression_probe": "(self, conversation_id: 'str', compressor_version: 'str') -> 'CompressionStateRecord | None'",
    "record_compression_outcome": "(self, *, outcome: 'CompressionOutcomeRecord') -> 'CompressionStateRecord'",
    "is_context_artifact_referenced": "(self, *, artifact_id: 'str') -> 'bool'",
    "append_user_memory": "(self, *, record: 'NewUserMemory') -> 'UserMemoryRecord'",
    "list_user_memories": "(self, user_id: 'str', *, active_only: 'bool' = True) -> 'list[UserMemoryRecord]'",
    "count_task_memories": "(self, task_id: 'str') -> 'int'",
    "get_user_profile": "(self, user_id: 'str') -> 'UserProfileRecord | None'",
    "upsert_user_profile": "(self, *, user_id: 'str', profile_text: 'str', source_memory_count: 'int') -> 'UserProfileRecord'",
    "list_user_turn_summaries": "(self, user_id: 'str', *, limit: 'int' = 50) -> 'list[UserTurnSummary]'",
}


def test_task_store_keeps_explicit_public_methods() -> None:
    actual = {
        name
        for name, value in TaskStore.__dict__.items()
        if not name.startswith("_") and callable(value)
    }
    assert actual == EXPECTED_TASK_STORE_METHODS
    assert TaskStore.__bases__ == (object,)
    assert "__getattr__" not in TaskStore.__dict__
    assert {
        name: str(inspect.signature(TaskStore.__dict__[name]))
        for name in sorted(EXPECTED_TASK_STORE_METHODS)
    } == EXPECTED_TASK_STORE_SIGNATURES


def test_task_store_reexports_canonical_records_and_errors() -> None:
    records = pytest.importorskip("paperpilot.web.store.records")
    assert VALID_DEPTHS is records.VALID_DEPTHS
    assert DuplicateUsernameError is records.DuplicateUsernameError
    assert ConversationBusyError is records.ConversationBusyError
    assert StaleConversationHeadError is records.StaleConversationHeadError
    assert WebUser is records.WebUser
    assert ResearchTask is records.ResearchTask
    assert TaskEvent is records.TaskEvent
    assert TaskArtifact is records.TaskArtifact
    assert TaskEventBatch is records.TaskEventBatch
    assert TaskArtifactBatch is records.TaskArtifactBatch
    assert TaskUpdates is records.TaskUpdates
    assert PaperRecord is records.PaperRecord
    assert ConversationRecord is records.ConversationRecord
    assert ConversationPaperRecord is records.ConversationPaperRecord
    assert ConversationDetail is records.ConversationDetail
    assert MessageRecord is records.MessageRecord
    assert ConversationTurn is records.ConversationTurn
    assert ConversationAlternative is records.ConversationAlternative
    assert UsedPaperInput is records.UsedPaperInput
    assert PublishedConversationResult is records.PublishedConversationResult
    assert FinalizedConversationTask is records.FinalizedConversationTask
    assert CompressionOutcomeRecord is records.CompressionOutcomeRecord
    assert CompressionStateRecord is records.CompressionStateRecord
    assert ContextArtifactRecord is records.ContextArtifactRecord
    assert NewContextArtifact is records.NewContextArtifact
    assert TurnArchiveRecord is records.TurnArchiveRecord
    assert TurnArchiveSeedRecord is records.TurnArchiveSeedRecord
    assert NewUserMemory is records.NewUserMemory
    assert UserMemoryRecord is records.UserMemoryRecord
    assert UserProfileRecord is records.UserProfileRecord
    assert UserTurnSummary is records.UserTurnSummary


def test_conversation_router_keeps_compatibility_entrypoint() -> None:
    module = importlib.import_module("paperpilot.web.routes.conversations")
    assert callable(module.build_conversation_router)
    assert str(inspect.signature(module.build_conversation_router)) == (
        "(*, store: 'TaskStore', executor: 'TaskExecutorLike', "
        "require_user: 'RequireUser', deep_reading_runner: "
        "'DeepReadingRunner', paper_search: 'PaperSearch', "
        "overload_retry_after_seconds: 'int') -> 'APIRouter'"
    )


def test_task_store_explicitly_delegates_user_and_conversation_operations(
    monkeypatch,
) -> None:
    from paperpilot.web.store import conversations, users

    store = object.__new__(TaskStore)
    session_factory = object()
    store._session_factory = session_factory
    sentinel_user = object()
    sentinel_conversation = object()
    calls: list[tuple[str, object, object]] = []

    def fake_get_user_by_id(factory, user_id):
        calls.append(("get_user_by_id", factory, user_id))
        return sentinel_user

    def fake_list_conversations(factory, **kwargs):
        calls.append(("list_conversations", factory, kwargs))
        return [sentinel_conversation]

    monkeypatch.setattr(users, "get_user_by_id", fake_get_user_by_id)
    monkeypatch.setattr(
        conversations,
        "list_conversations",
        fake_list_conversations,
    )
    assert store.get_user_by_id("user_1") is sentinel_user
    assert store.list_conversations(user_id="user_1") == [sentinel_conversation]
    assert calls == [
        ("get_user_by_id", session_factory, "user_1"),
        ("list_conversations", session_factory, {"user_id": "user_1"}),
    ]


def test_task_store_delegates_conversation_turn_to_message_store(
    monkeypatch,
) -> None:
    from paperpilot.web.store import messages

    store = object.__new__(TaskStore)
    session_factory = object()
    store._session_factory = session_factory
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_create_turn(factory, **kwargs):
        captured["factory"] = factory
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(messages, "create_conversation_turn", fake_create_turn)
    result = store.create_conversation_turn(
        user_id="user_1",
        conversation_id="conversation_1",
        content="follow up",
        depth="standard",
        expected_head_message_id="message_1",
    )

    assert result is sentinel
    assert captured == {
        "factory": session_factory,
        "user_id": "user_1",
        "conversation_id": "conversation_1",
        "content": "follow up",
        "depth": "standard",
        "expected_head_message_id": "message_1",
    }


def test_task_store_delegates_publication_and_head_switch_operations(
    monkeypatch,
) -> None:
    from paperpilot.web.store import publications

    store = object.__new__(TaskStore)
    session_factory = object()
    store._session_factory = session_factory
    published = object()
    switched = object()
    calls: list[tuple[str, object, dict[str, object]]] = []

    def fake_publish(factory, **kwargs):
        calls.append(("publish", factory, kwargs))
        return published

    def fake_switch(factory, conversation_id, **kwargs):
        calls.append(("switch", factory, {"conversation_id": conversation_id, **kwargs}))
        return switched

    monkeypatch.setattr(publications, "publish_conversation_result", fake_publish)
    monkeypatch.setattr(publications, "switch_conversation_head", fake_switch)

    assert store.publish_conversation_result(
        task_id="task_1",
        content="answer",
        metadata={"quality": "complete"},
        used_papers=[],
    ) is published
    assert store.switch_conversation_head(
        "conversation_1",
        user_id="user_1",
        expected_head_message_id="message_1",
        target_message_id="message_2",
        target_checkpoint_id="checkpoint_2",
        active_paper_ids=["paper_1"],
    ) is switched
    assert calls == [
        (
            "publish",
            session_factory,
            {
                "task_id": "task_1",
                "content": "answer",
                "metadata": {"quality": "complete"},
                "used_papers": [],
            },
        ),
        (
            "switch",
            session_factory,
            {
                "conversation_id": "conversation_1",
                "user_id": "user_1",
                "expected_head_message_id": "message_1",
                "target_message_id": "message_2",
                "target_checkpoint_id": "checkpoint_2",
                "active_paper_ids": ["paper_1"],
            },
        ),
    ]


def test_task_store_delegates_task_and_update_operations(
    monkeypatch,
) -> None:
    from paperpilot.web.store import tasks, updates

    store = object.__new__(TaskStore)
    session_factory = object()
    engine = object()
    store._session_factory = session_factory
    store.engine = engine
    claimed = object()
    update_snapshot = object()
    calls: list[tuple[str, object, object]] = []

    def fake_claim(factory, task_id, **kwargs):
        calls.append(("claim", factory, {"task_id": task_id, **kwargs}))
        return claimed

    def fake_updates(factory, conversation_id, task_id, **kwargs):
        calls.append(
            (
                "updates",
                factory,
                {"conversation_id": conversation_id, "task_id": task_id, **kwargs},
            )
        )
        return update_snapshot

    health_calls: list[object] = []
    monkeypatch.setattr(tasks, "claim_task", fake_claim)
    monkeypatch.setattr(updates, "get_conversation_task_updates", fake_updates)
    monkeypatch.setattr(tasks, "check_health", health_calls.append)

    assert store.claim_task("task_1", allow_running=True) is claimed
    assert store.get_conversation_task_updates(
        "conversation_1",
        "task_1",
        user_id="user_1",
    ) is update_snapshot
    assert store.check_health() is None
    assert health_calls == [engine]
    assert calls == [
        ("claim", session_factory, {"task_id": "task_1", "allow_running": True}),
        (
            "updates",
            session_factory,
            {
                "conversation_id": "conversation_1",
                "task_id": "task_1",
                "user_id": "user_1",
            },
        ),
    ]


FORBIDDEN_STORE_IMPORT_PREFIXES = (
    "fastapi",
    "paperpilot.web.app",
    "paperpilot.web.routes",
    "paperpilot.web.task_store",
)


def _resolved_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            raw_name = "." * node.level + (node.module or "")
            base = (
                importlib.util.resolve_name(raw_name, "paperpilot.web.store")
                if node.level
                else raw_name
            )
            imported.add(base)
            imported.update(
                f"{base}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )
    return imported


def test_store_modules_do_not_depend_on_web_edges() -> None:
    root = Path(__file__).parents[2] / "paperpilot" / "web" / "store"
    assert root.is_dir()
    violations = {
        path.relative_to(root).as_posix(): sorted(
            name
            for name in _resolved_imports(path)
            if name.startswith(FORBIDDEN_STORE_IMPORT_PREFIXES)
        )
        for path in root.glob("*.py")
    }
    assert {path: names for path, names in violations.items() if names} == {}


def test_split_conversation_route_modules_are_importable() -> None:
    for name in (
        "paperpilot.web.routes.conversations.router",
        "paperpilot.web.routes.conversations.crud",
        "paperpilot.web.routes.conversations.messages",
        "paperpilot.web.routes.conversations.rollback",
        "paperpilot.web.routes.conversations.presenters",
    ):
        pytest.importorskip(name)
