"""Compatibility facade for PaperPilot Web persistence operations."""
from __future__ import annotations

from pathlib import Path

from paperpilot.papers import PaperCandidate
from paperpilot.web.database import (
    create_session_factory,
    create_task_engine,
    resolve_task_db_path,
)
from paperpilot.web.db_migrations import ensure_database_current
from paperpilot.web.store import (
    context_memory,
    conversations,
    messages,
    publications,
    tasks,
    updates,
    user_memories,
    users,
)
from paperpilot.web.store.records import (
    ConversationAlternative,
    ConversationBusyError,
    ConversationDetail,
    ConversationPaperRecord,
    ConversationRecord,
    ConversationTurn,
    DuplicateUsernameError,
    FinalizedConversationTask,
    MessageRecord,
    NewUserMemory,
    PaperRecord,
    PublishedConversationResult,
    ResearchTask,
    StaleConversationHeadError,
    TaskArtifact,
    TaskArtifactBatch,
    TaskEvent,
    TaskEventBatch,
    TaskUpdates,
    UsedPaperInput,
    UserMemoryRecord,
    VALID_DEPTHS,
    WebUser,
    CompressionOutcomeRecord,
    CompressionStateRecord,
    ContextArtifactRecord,
    NewContextArtifact,
    TurnArchiveRecord,
    TurnArchiveSeedRecord,
    UserProfileRecord,
    UserTurnSummary,
)

class TaskStore:
    """Own the Web database resources and expose the stable public API."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = resolve_task_db_path(db_path)
        ensure_database_current(self.db_path)
        self.engine = create_task_engine(self.db_path)
        self._session_factory = create_session_factory(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    def create_user(
        self,
        *,
        username: str,
        password_hash: str,
        password_salt: str,
    ) -> WebUser:
        return users.create_user(
            self._session_factory,
            username=username,
            password_hash=password_hash,
            password_salt=password_salt,
        )
    def get_user_by_username(self, username: str) -> WebUser | None:
        return users.get_user_by_username(self._session_factory, username)

    def get_user_by_id(self, user_id: str) -> WebUser | None:
        return users.get_user_by_id(self._session_factory, user_id)

    def create_session(self, user_id: str) -> str:
        return users.create_session(self._session_factory, user_id)

    def get_user_for_session(self, token: str) -> WebUser | None:
        return users.get_user_for_session(self._session_factory, token)

    def delete_session(self, token: str) -> None:
        users.delete_session(self._session_factory, token)

    def create_conversation(
        self,
        *,
        user_id: str,
        paper: PaperCandidate,
        title: str | None = None,
    ) -> ConversationRecord:
        return conversations.create_conversation(
            self._session_factory,
            user_id=user_id,
            paper=paper,
            title=title,
        )

    def list_conversations(
        self,
        *,
        user_id: str,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[ConversationRecord]:
        kwargs: dict[str, object] = {"user_id": user_id}
        if include_archived:
            kwargs["include_archived"] = include_archived
        if limit != 100:
            kwargs["limit"] = limit
        return conversations.list_conversations(self._session_factory, **kwargs)

    def get_conversation_detail(
        self,
        conversation_id: str,
        *,
        user_id: str,
    ) -> ConversationDetail | None:
        return conversations.get_conversation_detail(
            self._session_factory,
            conversation_id,
            user_id=user_id,
        )

    def update_conversation(
        self,
        conversation_id: str,
        *,
        user_id: str,
        title: str | None = None,
        archived: bool | None = None,
    ) -> ConversationRecord | None:
        return conversations.update_conversation(
            self._session_factory,
            conversation_id,
            user_id=user_id,
            title=title,
            archived=archived,
        )

    def create_conversation_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        content: str,
        depth: str,
        expected_head_message_id: str | None,
    ) -> ConversationTurn:
        return messages.create_conversation_turn(
            self._session_factory,
            user_id=user_id,
            conversation_id=conversation_id,
            content=content,
            depth=depth,
            expected_head_message_id=expected_head_message_id,
        )

    def publish_conversation_result(
        self,
        *,
        task_id: str,
        content: str,
        metadata: dict,
        used_papers: list[UsedPaperInput],
    ) -> PublishedConversationResult:
        return publications.publish_conversation_result(
            self._session_factory,
            task_id=task_id,
            content=content,
            metadata=metadata,
            used_papers=used_papers,
        )

    def finalize_conversation_task(
        self,
        *,
        task_id: str,
        assistant_message_id: str,
        final_checkpoint_id: str,
        result_quality: str,
        active_paper_ids: list[str],
    ) -> FinalizedConversationTask:
        return publications.finalize_conversation_task(
            self._session_factory,
            task_id=task_id,
            assistant_message_id=assistant_message_id,
            final_checkpoint_id=final_checkpoint_id,
            result_quality=result_quality,
            active_paper_ids=active_paper_ids,
        )

    def fail_conversation_task(
        self,
        *,
        task_id: str,
        message: str,
        stage: str | None = None,
        payload: dict | None = None,
    ) -> ResearchTask | None:
        return publications.fail_conversation_task(
            self._session_factory,
            task_id=task_id,
            message=message,
            stage=stage,
            payload=payload,
        )

    def switch_conversation_head(
        self,
        conversation_id: str,
        *,
        user_id: str,
        expected_head_message_id: str | None,
        target_message_id: str,
        target_checkpoint_id: str,
        active_paper_ids: list[str],
    ) -> ConversationRecord | None:
        return publications.switch_conversation_head(
            self._session_factory,
            conversation_id,
            user_id=user_id,
            expected_head_message_id=expected_head_message_id,
            target_message_id=target_message_id,
            target_checkpoint_id=target_checkpoint_id,
            active_paper_ids=active_paper_ids,
        )

    def get_message(
        self,
        conversation_id: str,
        message_id: str,
        *,
        user_id: str,
    ) -> MessageRecord | None:
        return messages.get_message(
            self._session_factory,
            conversation_id,
            message_id,
            user_id=user_id,
        )

    def get_task_message(self, task_id: str, role: str) -> MessageRecord | None:
        return messages.get_task_message(self._session_factory, task_id, role)

    def list_active_messages(
        self,
        conversation_id: str,
        *,
        user_id: str,
    ) -> list[MessageRecord] | None:
        return messages.list_active_messages(
            self._session_factory,
            conversation_id,
            user_id=user_id,
        )

    def list_message_alternatives(
        self,
        conversation_id: str,
        message_id: str,
        *,
        user_id: str,
    ) -> list[ConversationAlternative] | None:
        return messages.list_message_alternatives(
            self._session_factory,
            conversation_id,
            message_id,
            user_id=user_id,
        )

    def get_unstable_turn(
        self,
        conversation_id: str,
        *,
        user_id: str,
    ) -> ConversationTurn | None:
        return messages.get_unstable_turn(
            self._session_factory,
            conversation_id,
            user_id=user_id,
        )

    def check_health(self) -> None:
        tasks.check_health(self.engine)

    def get_task(
        self,
        task_id: str,
        *,
        user_id: str | None = None,
    ) -> ResearchTask | None:
        return tasks.get_task(self._session_factory, task_id, user_id=user_id)

    def claim_task(
        self,
        task_id: str,
        *,
        allow_running: bool = False,
    ) -> ResearchTask | None:
        return tasks.claim_task(
            self._session_factory,
            task_id,
            allow_running=allow_running,
        )

    def fail_pending_task(self, task_id: str) -> ResearchTask | None:
        return tasks.fail_pending_task(self._session_factory, task_id)

    def add_event(
        self,
        *,
        task_id: str,
        type: str,
        message: str,
        stage: str | None = None,
        payload: dict | None = None,
    ) -> TaskEvent:
        return updates.add_event(
            self._session_factory,
            task_id=task_id,
            type=type,
            message=message,
            stage=stage,
            payload=payload,
        )

    def list_events_page(
        self,
        task_id: str,
        *,
        user_id: str | None,
        after_id: int,
        limit: int,
    ) -> TaskEventBatch | None:
        return updates.list_events_page(
            self._session_factory,
            task_id,
            user_id=user_id,
            after_id=after_id,
            limit=limit,
        )

    def add_artifact(
        self,
        *,
        task_id: str,
        kind: str,
        title: str,
        content: str,
        payload: dict | None = None,
    ) -> TaskArtifact:
        return updates.add_artifact(
            self._session_factory,
            task_id=task_id,
            kind=kind,
            title=title,
            content=content,
            payload=payload,
        )

    def list_artifacts_page(
        self,
        task_id: str,
        *,
        user_id: str | None,
        after_id: int,
        limit: int,
    ) -> TaskArtifactBatch | None:
        return updates.list_artifacts_page(
            self._session_factory,
            task_id,
            user_id=user_id,
            after_id=after_id,
            limit=limit,
        )

    def get_conversation_task_updates(
        self,
        conversation_id: str,
        task_id: str,
        *,
        user_id: str,
        after_event_id: int = 0,
        after_artifact_id: int = 0,
        limit: int = 50,
    ) -> TaskUpdates | None:
        kwargs: dict[str, object] = {"user_id": user_id}
        if after_event_id != 0:
            kwargs["after_event_id"] = after_event_id
        if after_artifact_id != 0:
            kwargs["after_artifact_id"] = after_artifact_id
        if limit != 50:
            kwargs["limit"] = limit
        return updates.get_conversation_task_updates(
            self._session_factory,
            conversation_id,
            task_id,
            **kwargs,
        )

    def create_context_artifact(
        self,
        *,
        record: NewContextArtifact,
    ) -> ContextArtifactRecord:
        return context_memory.create_context_artifact(
            self._session_factory,
            record=record,
        )

    def get_context_artifact(
        self,
        artifact_id: str,
        *,
        conversation_id: str,
    ) -> ContextArtifactRecord | None:
        return context_memory.get_context_artifact(
            self._session_factory,
            artifact_id,
            conversation_id=conversation_id,
        )

    def find_context_artifact_by_digest(
        self,
        *,
        conversation_id: str,
        tool_name: str,
        sha256: str,
    ) -> ContextArtifactRecord | None:
        return context_memory.find_context_artifact_by_digest(
            self._session_factory,
            conversation_id=conversation_id,
            tool_name=tool_name,
            sha256=sha256,
        )

    def seed_turn_archive(
        self,
        *,
        seed: TurnArchiveSeedRecord,
    ) -> TurnArchiveRecord:
        return context_memory.seed_turn_archive(self._session_factory, seed=seed)

    def list_turn_archives(self, conversation_id: str) -> list[TurnArchiveRecord]:
        return context_memory.list_turn_archives(
            self._session_factory,
            conversation_id,
        )

    def claim_turn_archive_narrative(
        self,
        archive_id: str,
    ) -> TurnArchiveRecord | None:
        return context_memory.claim_turn_archive_narrative(
            self._session_factory,
            archive_id,
        )

    def complete_turn_archive_narrative(
        self,
        archive_id: str,
        *,
        narrative_summary: str,
    ) -> TurnArchiveRecord:
        return context_memory.complete_turn_archive_narrative(
            self._session_factory,
            archive_id,
            narrative_summary=narrative_summary,
        )

    def fail_turn_archive_narrative(self, archive_id: str) -> TurnArchiveRecord:
        return context_memory.fail_turn_archive_narrative(
            self._session_factory,
            archive_id,
        )

    def append_user_memory(self, *, record: NewUserMemory) -> UserMemoryRecord:
        return user_memories.append_user_memory(
            self._session_factory,
            record=record,
        )

    def list_user_memories(
        self,
        user_id: str,
        *,
        active_only: bool = True,
    ) -> list[UserMemoryRecord]:
        return user_memories.list_user_memories(
            self._session_factory,
            user_id,
            active_only=active_only,
        )

    def count_task_memories(self, task_id: str) -> int:
        return user_memories.count_task_memories(
            self._session_factory,
            task_id,
        )

    def get_user_profile(self, user_id: str) -> UserProfileRecord | None:
        return user_memories.get_user_profile(self._session_factory, user_id)

    def upsert_user_profile(
        self,
        *,
        user_id: str,
        profile_text: str,
        source_memory_count: int,
    ) -> UserProfileRecord:
        return user_memories.upsert_user_profile(
            self._session_factory,
            user_id=user_id,
            profile_text=profile_text,
            source_memory_count=source_memory_count,
        )

    def list_user_turn_summaries(
        self,
        user_id: str,
        *,
        limit: int = 50,
    ) -> list[UserTurnSummary]:
        return user_memories.list_user_turn_summaries(
            self._session_factory,
            user_id,
            limit=limit,
        )

    def get_compression_state(
        self,
        conversation_id: str,
        compressor_version: str,
    ) -> CompressionStateRecord:
        return context_memory.get_compression_state(
            self._session_factory,
            conversation_id,
            compressor_version,
        )

    def claim_compression_probe(
        self,
        conversation_id: str,
        compressor_version: str,
    ) -> CompressionStateRecord | None:
        return context_memory.claim_compression_probe(
            self._session_factory,
            conversation_id,
            compressor_version,
        )

    def record_compression_outcome(
        self,
        *,
        outcome: CompressionOutcomeRecord,
    ) -> CompressionStateRecord:
        return context_memory.record_compression_outcome(
            self._session_factory,
            outcome=outcome,
        )

    def is_context_artifact_referenced(self, *, artifact_id: str) -> bool:
        return context_memory.is_context_artifact_referenced(
            self._session_factory,
            artifact_id=artifact_id,
        )
