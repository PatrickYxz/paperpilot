"""Strict, checkpoint-safe context management domain models."""
from __future__ import annotations

from enum import Enum
import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InitialAction(str, Enum):
    DROP_NOW = "DROP_NOW"
    EXTERNALIZE_NOW = "EXTERNALIZE_NOW"
    KEEP_INLINE = "KEEP_INLINE"


class FutureRetention(str, Enum):
    PROTECTED = "PROTECTED"
    CLEARABLE_AFTER_USE = "CLEARABLE_AFTER_USE"


class StrictContextModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def reject_blank_text(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{name} must not be blank")
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and not item.strip():
                        raise ValueError(f"{name} must not contain blank text")
        return self


class ArtifactRef(StrictContextModel):
    artifact_id: str
    sha256: str = Field(min_length=64, max_length=64)
    token_estimate: int = Field(ge=0)
    preview: str


class ToolResultDisposition(StrictContextModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_call_id: str
    content_sha256: str = Field(min_length=64, max_length=64)
    initial_action: InitialAction
    future_retention: FutureRetention
    preview: str
    artifact_ref: ArtifactRef | None = None
    result_id: str = "result-unknown"
    reason: str = "dropped"


class ProtectedText(StrictContextModel):
    protected_id: str
    exact_text: str
    sha256: str = Field(min_length=64, max_length=64)
    source_message_id: str | None = None

    @model_validator(mode="after")
    def validate_hash(self):
        expected = hashlib.sha256(self.exact_text.encode("utf-8")).hexdigest()
        if self.sha256 != expected:
            raise ValueError("sha256 does not match exact_text")
        return self


class ArchiveSupersession(StrictContextModel):
    target_protected_id: str
    source_message_id: str
    exact_text: str
    sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_hash(self):
        expected = hashlib.sha256(self.exact_text.encode("utf-8")).hexdigest()
        if self.sha256 != expected:
            raise ValueError("sha256 does not match exact_text")
        return self


class TurnArchiveSeed(StrictContextModel):
    archive_id: str
    conversation_id: str
    task_id: str
    user_message_id: str
    terminal_status: Literal["success", "failed", "cancelled"]
    user_goal: ProtectedText
    constraints: list[ProtectedText] = Field(default_factory=list)
    decisions: list[ProtectedText] = Field(default_factory=list)
    paper_findings: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    failed_paths: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    unresolved_todos: list[str] = Field(default_factory=list)
    rollback_notes: list[str] = Field(default_factory=list)
    supersedes: list[ArchiveSupersession] = Field(default_factory=list)
    archive_version: str = "turn-archive-v1"
    created_at: str

    @model_validator(mode="after")
    def validate_unique_refs(self):
        for name in (
            "evidence_refs",
            "paper_findings",
            "failed_paths",
            "verification",
            "unresolved_todos",
            "rollback_notes",
        ):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        return self


class ActiveProjection(StrictContextModel):
    current_goal: str
    active_constraints: list[str] = Field(default_factory=list)
    active_decisions: list[str] = Field(default_factory=list)
    active_paper_ids: list[str] = Field(default_factory=list)
    open_todos: list[str] = Field(default_factory=list)
    failed_verifications: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    protected_items: list[ProtectedText] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_projection_refs(self):
        for name in ("active_paper_ids", "open_todos", "failed_verifications"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        return self


class RetrievedArchiveView(StrictContextModel):
    archive_id: str
    terminal_status: Literal["success", "failed", "cancelled"]
    seed: dict[str, Any]
    narrative_summary: str | None = None
    superseded: bool = False


class ContextView(StrictContextModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    active_projection: ActiveProjection | None = None
    retrieved_archives: list[RetrievedArchiveView] = Field(default_factory=list)
    continuation_capsule: dict[str, Any] | None = None
    input_tokens: int = Field(default=0, ge=0)
    authority_artifact_ids: list[str] = Field(default_factory=list)
    authority_artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    authority_evidence_ids: list[str] = Field(default_factory=list)
    authority_archive_ids: list[str] = Field(default_factory=list)
    authority_verification: list[str] = Field(default_factory=list)
    authority_failed_paths: list[str] = Field(default_factory=list)
    authority_rollback_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_authority_refs(self):
        for name in (
            "authority_artifact_ids",
            "authority_evidence_ids",
            "authority_archive_ids",
            "authority_verification",
            "authority_failed_paths",
            "authority_rollback_notes",
        ):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        ref_ids = [item.artifact_id for item in self.authority_artifact_refs]
        if len(ref_ids) != len(set(ref_ids)):
            raise ValueError("authority_artifact_refs must be unique")
        if set(ref_ids) != set(self.authority_artifact_ids):
            raise ValueError("authority artifact IDs and refs must match")
        return self


class SessionMemoryCandidate(StrictContextModel):
    active_projection: ActiveProjection
    source_archive_ids: list[str] = Field(default_factory=list)
    candidate_version: str = "session-memory-v1"

    @model_validator(mode="after")
    def validate_unique_archives(self):
        if len(self.source_archive_ids) != len(set(self.source_archive_ids)):
            raise ValueError("source_archive_ids must be unique")
        return self


class ContinuationCapsule(StrictContextModel):
    current_goal: str
    exact_constraints: list[str] = Field(default_factory=list)
    decisions_and_rationales: list[str] = Field(default_factory=list)
    active_papers: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    completed_work: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    failed_paths: list[str] = Field(default_factory=list)
    unresolved_todos: list[str] = Field(default_factory=list)
    rollback_notes: list[str] = Field(default_factory=list)
    source_archive_ids: list[str] = Field(default_factory=list)
    capsule_version: str = "continuation-v1"

    @model_validator(mode="after")
    def validate_unique_refs(self):
        for name in ("active_papers", "evidence_refs"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        if len(self.artifact_refs) != len({item.artifact_id for item in self.artifact_refs}):
            raise ValueError("artifact_refs must be unique")
        if len(self.source_archive_ids) != len(set(self.source_archive_ids)):
            raise ValueError("source_archive_ids must be unique")
        return self


class CompressionAttempt(StrictContextModel):
    stage: Literal["session_memory", "full"]
    compressor_version: str
    reason: str
    before_tokens: int = Field(ge=0)
    after_tokens: int = Field(ge=0)
    reclaimed_tokens: int = Field(ge=0)
    protected_item_count: int = Field(ge=0)
    archive_ref_count: int = Field(ge=0)
    artifact_ref_count: int = Field(ge=0)
    validation_failure_type: str | None = None
    breaker_state: Literal["CLOSED", "OPEN", "HALF_OPEN"]
    cache_hit_tokens: int | None = Field(default=None, ge=0)
    cache_miss_tokens: int | None = Field(default=None, ge=0)


class CompressionDecision(StrictContextModel):
    accepted: bool
    stage: Literal["none", "session_memory", "full"]
    reason: str
    view: ContextView
    attempt: CompressionAttempt | None = None
