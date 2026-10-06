"""Deterministic per-turn Archive seeds and bounded narrative contracts."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..schemas import AnswerDraft, ResearchResult
from .models import (
    ArchiveSupersession,
    ArtifactRef,
    ProtectedText,
    TurnArchiveSeed,
)


class ContextDeltaSpan(BaseModel):
    """A model-proposed exact substring from a user message."""

    model_config = ConfigDict(extra="forbid")

    source_message_id: str
    exact_text: str
    kind: str


class ContextDeltaSupersession(BaseModel):
    """A model-proposed scoped correction of a protected item."""

    model_config = ConfigDict(extra="forbid")

    target_protected_id: str
    source_message_id: str
    exact_text: str


class AgentResearchContextDelta(BaseModel):
    """Untrusted structured context additions returned by the Research Agent."""

    model_config = ConfigDict(extra="forbid")

    constraints: list[ContextDeltaSpan] = Field(default_factory=list)
    decisions: list[ContextDeltaSpan] = Field(default_factory=list)
    supersedes: list[ContextDeltaSupersession] = Field(default_factory=list)


ResearchContextDelta = AgentResearchContextDelta


@dataclass(frozen=True)
class ValidatedContextDelta:
    """Server-authorized context additions with hashes computed by the server."""

    constraints: tuple[ProtectedText, ...] = ()
    decisions: tuple[ProtectedText, ...] = ()
    supersedes: tuple[ArchiveSupersession, ...] = ()


def validate_context_delta(
    delta: AgentResearchContextDelta,
    *,
    conversation_id: str,
    messages: Sequence[object],
    authority_set: Iterable[object] | Mapping[str, object],
) -> ValidatedContextDelta:
    """Validate exact user spans and calculate protected IDs and hashes.

    The model cannot supply a hash or protected ID for new text.  A failed
    validation raises ``ValueError('invalid_exact_span')`` so callers can
    discard only the delta while retaining the validated research result.
    """

    message_by_id = {
        _message_value(message, "id"): message
        for message in messages
        if _message_value(message, "id")
    }
    authority_ids = _authority_ids(authority_set)
    constraints = tuple(
        _protected_span(
            span,
            conversation_id=conversation_id,
            message_by_id=message_by_id,
        )
        for span in delta.constraints
    )
    decisions = tuple(
        _protected_span(
            span,
            conversation_id=conversation_id,
            message_by_id=message_by_id,
        )
        for span in delta.decisions
    )
    supersedes: list[ArchiveSupersession] = []
    for span in delta.supersedes:
        if span.target_protected_id not in authority_ids:
            raise ValueError("invalid_exact_span")
        message = _validated_source_message(
            span.source_message_id,
            span.exact_text,
            conversation_id=conversation_id,
            message_by_id=message_by_id,
        )
        del message
        supersedes.append(
            ArchiveSupersession(
                target_protected_id=span.target_protected_id,
                source_message_id=span.source_message_id,
                exact_text=span.exact_text,
                sha256=_sha256(span.exact_text),
            )
        )
    return ValidatedContextDelta(
        constraints=constraints,
        decisions=decisions,
        supersedes=tuple(supersedes),
    )


@dataclass(frozen=True)
class ResearchTrace:
    """Safe, non-transcript observations needed by the Archive builder."""

    todos: tuple[str, ...] = ()
    tool_outcomes: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    verification: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "todos": list(self.todos),
            "tool_outcomes": list(self.tool_outcomes),
            "artifact_ids": list(self.artifact_ids),
            "verification": list(self.verification),
        }


@dataclass(frozen=True)
class ResearchExecutionOutcome:
    """Internal Research Agent result preserving the old result surface."""

    result: ResearchResult
    trace: ResearchTrace
    context_delta: ValidatedContextDelta | None = None
    user_memory_context: str = ""

    def __getattr__(self, name: str) -> Any:
        # Existing graph and tests consume ResearchResult attributes directly.
        return getattr(self.result, name)


class TurnArchiveNarrative(BaseModel):
    """The narrative model may return prose only; Seed fields are immutable."""

    model_config = ConfigDict(extra="forbid")

    summary: str


class TurnArchiveSeedBuilder:
    """Build an append-only, deterministic Seed from authoritative values."""

    def build(
        self,
        *,
        archive_id: str,
        conversation_id: str,
        task_id: str,
        user_message: object,
        terminal_status: Literal["success", "failed", "cancelled"],
        research_result: ResearchResult | None,
        answer_draft: AnswerDraft | None = None,
        context_delta: ValidatedContextDelta | None = None,
        trace: ResearchTrace | None = None,
        artifact_refs: Sequence[ArtifactRef] = (),
        verification: Sequence[str] = (),
        legacy_summary: Mapping[str, Any] | None = None,
        created_at: str,
        supersedes: Sequence[ArchiveSupersession] = (),
    ) -> TurnArchiveSeed:
        del legacy_summary
        trace = trace or ResearchTrace()
        user_message_id = _message_value(user_message, "id")
        user_goal = _protected_text(
            exact_text=_message_value(user_message, "content"),
            source_message_id=user_message_id,
            conversation_id=conversation_id,
        )
        constraints = list(context_delta.constraints) if context_delta else []
        decisions = list(context_delta.decisions) if context_delta else []

        evidence_refs: list[str] = []
        paper_findings: list[str] = []
        limitations: list[str] = []
        if research_result is not None:
            evidence_refs = [item.id for item in research_result.evidence_items]
            limitations.extend(research_result.limitations)
            for paper_use in research_result.used_papers:
                evidence_refs.extend(paper_use.evidence_ids)
                paper_findings.append(
                    f"{paper_use.paper.external_id}:{paper_use.role}:"
                    + ",".join(paper_use.evidence_ids)
                )

        safe_verification = list(verification) or list(trace.verification)
        if answer_draft is not None and not safe_verification:
            safe_verification.append("citation:pass")
        if not safe_verification:
            safe_verification.append("citation:not_run")
        terminal_verification = {
            "success": "terminal:pass",
            "failed": "terminal:fail",
            "cancelled": "terminal:not_run",
        }[terminal_status]
        if terminal_verification not in safe_verification:
            safe_verification.append(terminal_verification)

        for outcome in trace.tool_outcomes:
            if outcome not in {"success", "pass"} and outcome not in limitations:
                limitations.append(outcome)
        return TurnArchiveSeed(
            archive_id=archive_id,
            conversation_id=conversation_id,
            task_id=task_id,
            user_message_id=user_message_id,
            terminal_status=terminal_status,
            user_goal=user_goal,
            constraints=constraints,
            decisions=decisions,
            paper_findings=_unique(paper_findings),
            evidence_refs=_unique(evidence_refs),
            artifact_refs=_unique_artifacts([*artifact_refs, *_artifact_refs(trace)]),
            failed_paths=_unique(limitations),
            verification=_unique(safe_verification),
            unresolved_todos=_unresolved_todos(trace.todos),
            rollback_notes=[],
            supersedes=[*supersedes, *(context_delta.supersedes if context_delta else ())],
            created_at=created_at,
        )


def _protected_span(
    span: ContextDeltaSpan,
    *,
    conversation_id: str,
    message_by_id: Mapping[str, object],
) -> ProtectedText:
    _validated_source_message(
        span.source_message_id,
        span.exact_text,
        conversation_id=conversation_id,
        message_by_id=message_by_id,
    )
    return _protected_text(
        exact_text=span.exact_text,
        source_message_id=span.source_message_id,
        conversation_id=conversation_id,
    )


def _validated_source_message(
    message_id: str,
    exact_text: str,
    *,
    conversation_id: str,
    message_by_id: Mapping[str, object],
) -> object:
    message = message_by_id.get(message_id)
    if message is None:
        raise ValueError("invalid_exact_span")
    if (
        _message_value(message, "conversation_id") != conversation_id
        or _message_value(message, "role") != "user"
        or not exact_text
        or exact_text not in _message_value(message, "content")
    ):
        raise ValueError("invalid_exact_span")
    return message


def _protected_text(*, exact_text: str, source_message_id: str, conversation_id: str) -> ProtectedText:
    return ProtectedText(
        protected_id=_protected_id(conversation_id, source_message_id, exact_text),
        exact_text=exact_text,
        sha256=_sha256(exact_text),
        source_message_id=source_message_id,
    )


def _protected_id(conversation_id: str, message_id: str, exact_text: str) -> str:
    identity = f"{conversation_id}\x00{message_id}\x00{_sha256(exact_text)}"
    return "protected_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _message_value(message: object, name: str) -> str:
    value = getattr(message, name, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid_exact_span")
    return value


def _authority_ids(authority_set: Iterable[object] | Mapping[str, object]) -> set[str]:
    values = authority_set.keys() if isinstance(authority_set, Mapping) else authority_set
    result: set[str] = set()
    for value in values:
        if isinstance(value, str):
            result.add(value)
        else:
            protected_id = getattr(value, "protected_id", None)
            if isinstance(protected_id, str):
                result.add(protected_id)
    return result


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value.strip()))


def _unique_artifacts(values: Iterable[ArtifactRef]) -> list[ArtifactRef]:
    result: list[ArtifactRef] = []
    seen: set[str] = set()
    for value in values:
        if value.artifact_id not in seen:
            seen.add(value.artifact_id)
            result.append(value)
    return result


def _unresolved_todos(values: Iterable[str]) -> list[str]:
    return _unique(
        value
        for value in values
        if value.rsplit(":", 1)[-1] != "completed"
    )


def _artifact_refs(trace: ResearchTrace) -> list[ArtifactRef]:
    # Trace stores IDs only.  The authoritative ArtifactRef payload must be
    # supplied by the caller; IDs are therefore not fabricated into refs.
    return []
