"""Validated two-stage context compaction with atomic derived-view adoption."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from enum import Enum
import hashlib
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .models import (
    CompressionAttempt,
    CompressionDecision,
    ContextView,
    ContinuationCapsule,
    SessionMemoryCandidate,
)
from .views import ContextCapacityExhaustedError, render_context_view, select_recent_turns


_SESSION_MEMORY_PROMPT = """\
<session_memory_compressor>
<objective>Return exactly one SessionMemoryCandidate that removes only compressible narrative duplication.</objective>
<rules>
1. Copy active_projection exactly, including every list item and ProtectedText field.
2. Set source_archive_ids to exactly the archive IDs present in retrieved_archives.
3. Do not invent, rewrite, merge, or omit any goal, constraint, decision, paper ID, TODO, verification, question, protected ID, text, hash, or reference.
4. Return only the structured candidate. Never modify an Archive, Artifact, authority field, or message.
</rules>
</session_memory_compressor>
"""


_CONTINUATION_PROMPT = """\
<continuation_compressor>
<objective>Return exactly one ContinuationCapsule that preserves all authoritative state while removing replaceable conversation prose.</objective>
<rules>
1. Copy current_goal exactly from active_projection.current_goal.
2. Copy active_papers exactly from active_projection.active_paper_ids.
3. Copy exact_constraints, decisions_and_rationales, unresolved_todos, verification, failed_paths, and rollback_notes into their matching fields; never move an item to another field.
4. Copy evidence_refs and every ArtifactRef exactly from the authority fields, including artifact ID, SHA-256, token estimate, and preview.
5. Set source_archive_ids to exactly the retrieved archive IDs plus any source archive IDs already present in the input capsule.
6. Preserve every protected exact text verbatim. Do not invent, rewrite, merge, or omit IDs, hashes, negation, numbers, verification state, or rollback information.
7. Return only the structured candidate. Never modify an Archive, Artifact, authority field, or message.
</rules>
</continuation_compressor>
"""


class CompressionFailureType(str, Enum):
    SCHEMA_INVALID = "schema_invalid"
    INVALID_REFERENCE = "invalid_reference"
    PROTECTED_CONTEXT_LOST = "protected_context_lost"
    HASH_MISMATCH = "hash_mismatch"
    INSUFFICIENT_REDUCTION = "insufficient_reduction"
    TRANSIENT = "transient"
    PROTECTED_CONTEXT_OVERSIZED = "protected_context_oversized"


class _CandidateRejected(ValueError):
    def __init__(self, reason: CompressionFailureType) -> None:
        super().__init__(reason.value)
        self.reason = reason


class ContinuationValidator:
    """Validate candidates without mutating the input view or authority stores."""

    def validate_session(
        self,
        candidate: SessionMemoryCandidate | Mapping[str, Any],
        view: ContextView,
        *,
        authority_archive_ids: set[str],
        authority_artifact_ids: set[str],
        authority_paper_ids: set[str],
        target_tokens: int,
        actual_tokens: int,
    ) -> SessionMemoryCandidate:
        parsed = self._parse(candidate, SessionMemoryCandidate)
        self._validate_projection(
            parsed.active_projection,
            view,
            authority_paper_ids=authority_paper_ids,
        )
        self._validate_archive_refs(
            parsed.source_archive_ids,
            authority_archive_ids,
            required={item.archive_id for item in view.retrieved_archives},
        )
        self._validate_view_artifact_authority(view, authority_artifact_ids)
        self._validate_reduction(actual_tokens, target_tokens)
        return parsed

    def validate_continuation(
        self,
        candidate: ContinuationCapsule | Mapping[str, Any],
        view: ContextView,
        *,
        authority_archive_ids: set[str],
        authority_artifact_ids: set[str],
        authority_paper_ids: set[str],
        target_tokens: int,
        actual_tokens: int,
    ) -> ContinuationCapsule:
        parsed = self._parse(candidate, ContinuationCapsule)
        projection = view.active_projection
        if projection is None or parsed.current_goal != projection.current_goal:
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)
        required_archives = {item.archive_id for item in view.retrieved_archives}
        capsule = view.continuation_capsule
        if isinstance(capsule, Mapping):
            refs = capsule.get("source_archive_ids", [])
            if isinstance(refs, list):
                required_archives.update(item for item in refs if isinstance(item, str))
        self._validate_archive_refs(
            parsed.source_archive_ids,
            authority_archive_ids,
            required=required_archives,
        )
        self._validate_continuation_artifact_refs(
            parsed,
            view,
            authority_artifact_ids,
        )
        self._validate_exact_refs(
            parsed.evidence_refs,
            set(view.authority_evidence_ids),
        )
        self._validate_exact_refs(parsed.active_papers, authority_paper_ids)
        protected_texts = {item.exact_text for item in projection.protected_items}
        if set(parsed.exact_constraints) != set(projection.active_constraints):
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)
        if set(parsed.decisions_and_rationales) != set(projection.active_decisions):
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)
        if not protected_texts.issubset(
            set(parsed.exact_constraints) | set(parsed.decisions_and_rationales)
        ):
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)
        expected_todos = set(projection.open_todos)
        expected_verification = (
            set(view.authority_verification)
            | set(projection.failed_verifications)
        )
        expected_failed_paths = (
            set(view.authority_failed_paths)
            | set(projection.unresolved_questions)
        )
        if set(parsed.unresolved_todos) != expected_todos:
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)
        if set(parsed.verification) != expected_verification:
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)
        if set(parsed.failed_paths) != expected_failed_paths:
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)
        if set(parsed.rollback_notes) != set(view.authority_rollback_notes):
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)
        self._validate_reduction(actual_tokens, target_tokens)
        return parsed

    @staticmethod
    def _parse(candidate: object, schema):
        try:
            return schema.model_validate(candidate)
        except ValidationError as exc:
            raise _CandidateRejected(CompressionFailureType.SCHEMA_INVALID) from exc

    @staticmethod
    def _validate_projection(
        candidate,
        view: ContextView,
        *,
        authority_paper_ids: set[str],
    ) -> None:
        original = view.active_projection
        if original is None:
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)
        if candidate.current_goal != original.current_goal:
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)
        candidate_papers = set(
            candidate.active_papers
            if hasattr(candidate, "active_papers")
            else candidate.active_paper_ids
        )
        ContinuationValidator._validate_exact_refs(
            list(candidate_papers),
            authority_paper_ids,
        )
        for field_name in (
            "active_constraints",
            "active_decisions",
            "open_todos",
            "failed_verifications",
            "unresolved_questions",
        ):
            if set(getattr(original, field_name)) != set(getattr(candidate, field_name)):
                raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)
        before = {item.protected_id: item for item in original.protected_items}
        after = {item.protected_id: item for item in candidate.protected_items}
        if set(before) != set(after):
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)
        for protected_id, item in before.items():
            if after[protected_id].exact_text != item.exact_text or after[protected_id].sha256 != item.sha256:
                raise _CandidateRejected(CompressionFailureType.HASH_MISMATCH)

    @staticmethod
    def _validate_archive_refs(
        refs: list[str],
        authority: set[str],
        *,
        required: set[str],
    ) -> None:
        if not set(refs).issubset(authority):
            raise _CandidateRejected(CompressionFailureType.INVALID_REFERENCE)
        if set(refs) != required:
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)

    @staticmethod
    def _validate_view_artifact_authority(view: ContextView, authority: set[str]) -> None:
        if not set(view.authority_artifact_ids).issubset(authority):
            raise _CandidateRejected(CompressionFailureType.INVALID_REFERENCE)

    @staticmethod
    def _validate_exact_refs(refs: list[str], authority: set[str]) -> None:
        values = set(refs)
        if not values.issubset(authority):
            raise _CandidateRejected(CompressionFailureType.INVALID_REFERENCE)
        if values != authority:
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)

    @staticmethod
    def _validate_continuation_artifact_refs(
        candidate: ContinuationCapsule,
        view: ContextView,
        authority: set[str],
    ) -> None:
        expected = {item.artifact_id: item for item in view.authority_artifact_refs}
        if set(expected) != set(view.authority_artifact_ids) or set(expected) != authority:
            raise _CandidateRejected(CompressionFailureType.INVALID_REFERENCE)
        actual = {item.artifact_id: item for item in candidate.artifact_refs}
        if not set(actual).issubset(authority):
            raise _CandidateRejected(CompressionFailureType.INVALID_REFERENCE)
        if set(actual) != authority:
            raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)
        for artifact_id, expected_ref in expected.items():
            actual_ref = actual[artifact_id]
            if actual_ref.sha256 != expected_ref.sha256:
                raise _CandidateRejected(CompressionFailureType.HASH_MISMATCH)
            if actual_ref != expected_ref:
                raise _CandidateRejected(CompressionFailureType.PROTECTED_CONTEXT_LOST)

    @staticmethod
    def _validate_reduction(actual_tokens: int, target_tokens: int) -> None:
        if actual_tokens > target_tokens:
            raise _CandidateRejected(CompressionFailureType.INSUFFICIENT_REDUCTION)


class CompressionCoordinator:
    """Run Stage A then, only when necessary, Stage B and adopt one view copy."""

    def __init__(
        self,
        *,
        token_counter,
        usable_input_budget: int,
        full_compaction_enabled: bool,
        model: Any | None = None,
        session_compressor: Callable[[ContextView], object] | None = None,
        continuation_compressor: Callable[[ContextView], object] | None = None,
        event_sink: Callable[[str, dict[str, object]], None] | None = None,
        trigger_ratio: float = 0.80,
        session_target_ratio: float = 0.65,
        full_target_ratio: float = 0.50,
        recent_turns: int = 2,
        transient_retry_count: int = 1,
        breaker=None,
    ) -> None:
        if (
            usable_input_budget < 1
            or recent_turns < 1
            or transient_retry_count < 0
            or not 0 < full_target_ratio < session_target_ratio < trigger_ratio < 1
        ):
            raise ValueError("invalid compaction ratios or budget")
        self._token_counter = token_counter
        self._usable_input_budget = usable_input_budget
        self._full_compaction_enabled = full_compaction_enabled
        self._model = model
        self._session_compressor = session_compressor
        self._continuation_compressor = continuation_compressor
        self._event_sink = event_sink or (lambda _type, _payload: None)
        self._trigger_ratio = trigger_ratio
        self._session_target_ratio = session_target_ratio
        self._full_target_ratio = full_target_ratio
        self._recent_turns = recent_turns
        self._transient_retry_count = transient_retry_count
        self._validator = ContinuationValidator()
        self._breaker = breaker

    def prepare(
        self,
        view: ContextView,
        *,
        task_id: str,
        conversation_id: str,
        view_token_counter: Callable[[ContextView], int] | None = None,
    ) -> CompressionDecision:
        del task_id, conversation_id
        original = view.model_copy(deep=True)
        count_view = view_token_counter or self._input_tokens
        before_tokens = count_view(original)
        original = original.model_copy(update={"input_tokens": before_tokens})
        trigger_tokens = self._usable_input_budget * self._trigger_ratio
        if before_tokens < trigger_tokens:
            return self._decision(original, stage="none", reason="below_trigger")
        if not self._full_compaction_enabled:
            return self._decision(original, stage="none", reason="full_compaction_disabled")
        input_digest = hashlib.sha256(
            render_context_view(original).encode("utf-8")
        ).hexdigest()
        if self._breaker is not None and not self._breaker.before_attempt(
            input_digest=input_digest
        ):
            return self._decision(original, stage="none", reason="compression_circuit_open")
        protected_tokens = self._protected_tokens(original)
        session_target = int(self._usable_input_budget * self._session_target_ratio)
        full_target = int(self._usable_input_budget * self._full_target_ratio)
        if protected_tokens > full_target:
            self._record_breaker_failure(input_digest, CompressionFailureType.PROTECTED_CONTEXT_OVERSIZED.value)
            return self._decision(
                original,
                stage="none",
                reason=CompressionFailureType.PROTECTED_CONTEXT_OVERSIZED.value,
            )

        authority_archives = set(original.authority_archive_ids)
        authority_artifacts = set(original.authority_artifact_ids)
        authority_papers = set(
            original.active_projection.active_paper_ids
            if original.active_projection is not None
            else []
        )
        session_candidate, session_error = self._candidate(
            stage="session_memory",
            view=original,
            before_tokens=before_tokens,
        )
        if session_candidate is not None:
            try:
                session_view = self._session_view(original, session_candidate)
                actual = count_view(session_view)
                session_view = session_view.model_copy(update={"input_tokens": actual})
                parsed = self._validator.validate_session(
                    session_candidate,
                    original,
                    authority_archive_ids=authority_archives,
                    authority_artifact_ids=authority_artifacts,
                    authority_paper_ids=authority_papers,
                    target_tokens=session_target,
                    actual_tokens=actual,
                )
                del parsed
                self._emit_success("session_memory", before_tokens, actual, original)
                self._record_breaker_success(input_digest)
                return self._decision(
                    session_view,
                    stage="session_memory",
                    reason="session_memory_compacted",
                    before_tokens=before_tokens,
                    after_tokens=actual,
                )
            except _CandidateRejected as exc:
                session_error = exc.reason.value
        if session_error is not None:
            self._emit_rejected(
                "session_memory",
                session_error,
                before_tokens,
                original,
            )

        full_candidate, full_error = self._candidate(
            stage="full",
            view=original,
            before_tokens=before_tokens,
        )
        if full_candidate is None:
            self._emit_rejected(
                "full",
                full_error or "candidate_missing",
                before_tokens,
                original,
            )
            self._record_breaker_failure(input_digest, full_error or "candidate_missing")
            return self._decision(original, stage="none", reason=full_error or "candidate_missing")
        try:
            full_view = self._full_view(original, full_candidate)
            actual = count_view(full_view)
            full_view = full_view.model_copy(update={"input_tokens": actual})
            parsed = self._validator.validate_continuation(
                full_candidate,
                original,
                authority_archive_ids=authority_archives,
                authority_artifact_ids=authority_artifacts,
                authority_paper_ids=authority_papers,
                target_tokens=full_target,
                actual_tokens=actual,
            )
            del parsed
            self._emit_success("full", before_tokens, actual, original)
            self._record_breaker_success(input_digest)
            return self._decision(
                full_view,
                stage="full",
                reason="full_compacted",
                before_tokens=before_tokens,
                after_tokens=actual,
            )
        except _CandidateRejected as exc:
            self._emit_rejected(
                "full",
                exc.reason.value,
                before_tokens,
                original,
            )
            self._record_breaker_failure(input_digest, exc.reason.value)
            return self._decision(original, stage="none", reason=exc.reason.value)

    @property
    def usable_input_budget(self) -> int:
        return self._usable_input_budget

    def _record_breaker_success(self, input_digest: str) -> None:
        if self._breaker is None:
            return
        try:
            self._breaker.record_success(input_digest=input_digest)
        except Exception:
            return

    def _record_breaker_failure(self, input_digest: str, reason: str) -> None:
        if self._breaker is None:
            return
        try:
            self._breaker.record_failure(
                failure_type=reason,
                input_digest=input_digest,
            )
        except Exception:
            return

    def _candidate(
        self,
        *,
        stage: str,
        view: ContextView,
        before_tokens: int,
    ) -> tuple[object | None, str | None]:
        compressor = self._session_compressor if stage == "session_memory" else self._continuation_compressor
        schema = SessionMemoryCandidate if stage == "session_memory" else ContinuationCapsule
        self._emit_start(stage, view, before_tokens)
        try:
            candidate = self._invoke_with_retry(compressor, schema, view, stage)
            return candidate, None
        except _CandidateRejected as exc:
            return None, exc.reason.value
        except Exception as exc:
            return None, _classify_provider_error(exc).value

    def _invoke_with_retry(self, compressor, schema, view: ContextView, stage: str) -> object:
        last_error: Exception | None = None
        attempt_count = self._transient_retry_count + 1
        for attempt in range(attempt_count):
            try:
                if compressor is not None:
                    return compressor(view.model_copy(deep=True))
                if self._model is None:
                    raise _CandidateRejected(CompressionFailureType.SCHEMA_INVALID)
                structured = self._model.with_structured_output(schema)
                response = structured.invoke(
                    [
                        SystemMessage(
                            content=(
                                _SESSION_MEMORY_PROMPT
                                if stage == "session_memory"
                                else _CONTINUATION_PROMPT
                            )
                        ),
                        HumanMessage(content=render_context_view(view)),
                    ],
                    config={
                        "tags": ["paperpilot:model"],
                        "metadata": {
                            "paperpilot_stage": f"{stage}_compaction",
                            "prompt_version": (
                                "session-memory-compressor-v1"
                                if stage == "session_memory"
                                else "continuation-compressor-v1"
                            ),
                        },
                    },
                )
                if isinstance(response, Mapping) and "parsed" in response:
                    return response["parsed"]
                return response
            except Exception as exc:
                last_error = exc
                if not _is_transient(exc) or attempt == attempt_count - 1:
                    raise
        assert last_error is not None
        raise last_error

    @staticmethod
    def _session_view(view: ContextView, candidate: SessionMemoryCandidate) -> ContextView:
        archives = [item.model_copy(update={"narrative_summary": None}) for item in view.retrieved_archives]
        return view.model_copy(
            deep=True,
            update={
                "active_projection": candidate.active_projection,
                "retrieved_archives": archives,
                "input_tokens": 0,
            },
        )

    def _full_view(self, view: ContextView, candidate: ContinuationCapsule) -> ContextView:
        recent_messages = select_recent_turns(
            view.messages,
            current_goal=candidate.current_goal,
            completed_turn_count=self._recent_turns,
        )
        return view.model_copy(
            deep=True,
            update={
                "messages": recent_messages,
                "continuation_capsule": candidate.model_dump(mode="json"),
                "retrieved_archives": [],
                "input_tokens": 0,
            },
        )

    def _input_tokens(self, view: ContextView) -> int:
        if view.input_tokens > 0:
            return view.input_tokens
        return self._token_counter.count_text(render_context_view(view))

    @staticmethod
    def _protected_tokens(view: ContextView) -> int:
        projection = view.active_projection
        if projection is None:
            return 0
        payload = {
            "current_goal": projection.current_goal,
            "constraints": projection.active_constraints,
            "decisions": projection.active_decisions,
            "todos": projection.open_todos,
            "failed": projection.failed_verifications,
            "questions": projection.unresolved_questions,
            "protected": [item.model_dump(mode="json") for item in projection.protected_items],
        }
        return max(1, len(json_dumps(payload)) // 4)

    def _decision(
        self,
        view: ContextView,
        *,
        stage: str,
        reason: str,
        before_tokens: int | None = None,
        after_tokens: int | None = None,
    ) -> CompressionDecision:
        before = self._input_tokens(view) if before_tokens is None else before_tokens
        after = self._input_tokens(view) if after_tokens is None else after_tokens
        attempt = None
        if stage != "none":
            attempt = CompressionAttempt(
                stage=stage,
                compressor_version="context-compression-v1",
                reason=reason,
                before_tokens=before,
                after_tokens=after,
                reclaimed_tokens=max(0, before - after),
                protected_item_count=(
                    len(view.active_projection.protected_items)
                    if view.active_projection is not None
                    else 0
                ),
                archive_ref_count=len(view.authority_archive_ids),
                artifact_ref_count=len(view.authority_artifact_ids),
                breaker_state=self._breaker_state(),
            )
        return CompressionDecision(
            accepted=stage != "none",
            stage=stage,
            reason=reason,
            view=view.model_copy(deep=True),
            attempt=attempt,
        )

    def _event_payload(
        self,
        *,
        stage: str,
        reason: str,
        before_tokens: int,
        after_tokens: int,
        view: ContextView,
        validation_failure_type: str | None,
    ) -> dict[str, object]:
        projection = view.active_projection
        return {
            "stage": stage,
            "compressor_version": "context-compression-v1",
            "reason": reason,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "reclaimed_tokens": max(0, before_tokens - after_tokens),
            "protected_item_count": (
                len(projection.protected_items) if projection is not None else 0
            ),
            "archive_ref_count": len(view.authority_archive_ids),
            "artifact_ref_count": len(view.authority_artifact_ids),
            "validation_failure_type": validation_failure_type,
            "breaker_state": self._breaker_state(),
        }

    def _breaker_state(self) -> str:
        if self._breaker is None:
            return "CLOSED"
        try:
            return self._breaker.state().value
        except Exception:
            return "CLOSED"

    def _emit_start(self, stage: str, view: ContextView, before_tokens: int) -> None:
        self._event_sink(
            "session_memory_compaction_started" if stage == "session_memory" else "full_compaction_started",
            self._event_payload(
                stage=stage,
                reason="candidate_generation_started",
                before_tokens=before_tokens,
                after_tokens=before_tokens,
                view=view,
                validation_failure_type=None,
            ),
        )

    def _emit_rejected(
        self,
        stage: str,
        reason: str,
        before_tokens: int,
        view: ContextView,
    ) -> None:
        self._event_sink(
            "compression_candidate_rejected",
            self._event_payload(
                stage=stage,
                reason=reason,
                before_tokens=before_tokens,
                after_tokens=before_tokens,
                view=view,
                validation_failure_type=reason,
            ),
        )

    def _emit_success(
        self,
        stage: str,
        before_tokens: int,
        after_tokens: int,
        view: ContextView,
    ) -> None:
        self._event_sink(
            "compression_completed",
            self._event_payload(
                stage=stage,
                reason="compression_succeeded",
                before_tokens=before_tokens,
                after_tokens=after_tokens,
                view=view,
                validation_failure_type=None,
            ),
        )


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or 500 <= status <= 599)


def _classify_provider_error(exc: Exception) -> CompressionFailureType:
    return CompressionFailureType.TRANSIENT if _is_transient(exc) else CompressionFailureType.SCHEMA_INVALID


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
