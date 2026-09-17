"""Two-stage context compaction and candidate validation contracts."""
from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from paperpilot.deep_reading.context_management.compaction import (
    CompressionCoordinator,
    CompressionFailureType,
    ContinuationValidator,
)
from paperpilot.deep_reading.context_management.models import (
    ActiveProjection,
    ArtifactRef,
    ContextView,
    ContinuationCapsule,
    ProtectedText,
    RetrievedArchiveView,
    SessionMemoryCandidate,
)


class _Counter:
    def count_text(self, value: str) -> int:
        if '"current_goal":"large-input"' in value:
            return 90
        if '"current_goal":"small-input"' in value:
            return 40
        return 40


def _projection(goal: str = "goal") -> ActiveProjection:
    return ActiveProjection(
        current_goal=goal,
        active_constraints=[],
        active_decisions=[],
        active_paper_ids=["paper-1"],
        open_todos=[],
        failed_verifications=[],
        unresolved_questions=[],
    )


def _view(*, input_tokens: int = 90, goal: str = "goal") -> ContextView:
    return ContextView(
        messages=[{"role": "user", "content": "large-input"}],
        active_projection=_projection(goal),
        retrieved_archives=[],
        input_tokens=input_tokens,
        authority_artifact_ids=[],
        authority_archive_ids=[],
    )


def test_below_eighty_percent_does_not_call_a_compressor() -> None:
    calls: list[str] = []
    coordinator = CompressionCoordinator(
        token_counter=_Counter(),
        usable_input_budget=100,
        full_compaction_enabled=True,
        session_compressor=lambda _view: calls.append("a") or None,
        continuation_compressor=lambda _view: calls.append("b") or None,
    )

    decision = coordinator.prepare(_view(input_tokens=79), task_id="t", conversation_id="c")

    assert calls == []
    assert decision.accepted is False
    assert decision.stage == "none"
    assert decision.reason == "below_trigger"


def test_stage_a_success_stops_before_stage_b_and_is_atomic() -> None:
    calls: list[str] = []
    candidate = SessionMemoryCandidate(
        active_projection=_projection(),
        source_archive_ids=[],
    )
    coordinator = CompressionCoordinator(
        token_counter=_Counter(),
        usable_input_budget=100,
        full_compaction_enabled=True,
        session_compressor=lambda _view: calls.append("a") or candidate,
        continuation_compressor=lambda _view: calls.append("b") or None,
    )
    original = _view(input_tokens=80)

    decision = coordinator.prepare(original, task_id="t", conversation_id="c")

    assert calls == ["a"]
    assert decision.accepted is True
    assert decision.stage == "session_memory"
    assert decision.view is not original
    assert original.messages == [{"role": "user", "content": "large-input"}]


def test_candidate_recount_uses_complete_request_counter_and_stores_actual_tokens() -> None:
    candidate = SessionMemoryCandidate(
        active_projection=_projection(),
        source_archive_ids=["archive-1"],
    )
    original = _view(input_tokens=1).model_copy(
        update={
            "retrieved_archives": [
                RetrievedArchiveView(
                    archive_id="archive-1",
                    terminal_status="success",
                    seed={"archive_id": "archive-1"},
                    narrative_summary="large narrative",
                )
            ],
            "authority_archive_ids": ["archive-1"],
        }
    )
    counted: list[bool] = []

    def count_complete_request(view: ContextView) -> int:
        has_narrative = any(
            item.narrative_summary is not None for item in view.retrieved_archives
        )
        counted.append(has_narrative)
        return 80 if has_narrative else 40

    decision = CompressionCoordinator(
        token_counter=_Counter(),
        usable_input_budget=100,
        full_compaction_enabled=True,
        session_compressor=lambda _view: candidate,
    ).prepare(
        original,
        task_id="t",
        conversation_id="c",
        view_token_counter=count_complete_request,
    )

    assert counted == [True, False]
    assert decision.accepted is True
    assert decision.view.input_tokens == 40
    assert decision.attempt is not None
    assert decision.attempt.before_tokens == 80
    assert decision.attempt.after_tokens == 40


def test_compaction_events_include_safe_metrics_and_no_context_text() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    candidate = SessionMemoryCandidate(
        active_projection=_projection(),
        source_archive_ids=[],
    )
    coordinator = CompressionCoordinator(
        token_counter=_Counter(),
        usable_input_budget=100,
        full_compaction_enabled=True,
        session_compressor=lambda _view: candidate,
        event_sink=lambda kind, payload: events.append((kind, payload)),
    )

    coordinator.prepare(_view(input_tokens=80), task_id="t", conversation_id="c")

    assert events
    required = {
        "stage",
        "compressor_version",
        "reason",
        "before_tokens",
        "after_tokens",
        "reclaimed_tokens",
        "protected_item_count",
        "archive_ref_count",
        "artifact_ref_count",
        "validation_failure_type",
        "breaker_state",
    }
    for _kind, payload in events:
        assert required <= payload.keys()
        assert "large-input" not in str(payload)


def test_stage_b_runs_only_when_stage_a_remains_at_trigger() -> None:
    calls: list[str] = []
    a_candidate = SessionMemoryCandidate(
        active_projection=ActiveProjection(
            current_goal="large-input",
            active_constraints=[],
            active_decisions=[],
            active_paper_ids=["paper-1"],
            open_todos=[],
            failed_verifications=[],
            unresolved_questions=[],
        ),
        source_archive_ids=[],
    )
    b_candidate = ContinuationCapsule(
        current_goal="goal",
        exact_constraints=[],
        decisions_and_rationales=[],
        active_papers=["paper-1"],
        evidence_refs=[],
        artifact_refs=[],
        completed_work=[],
        verification=[],
        failed_paths=[],
        unresolved_todos=[],
        rollback_notes=[],
        source_archive_ids=[],
    )
    coordinator = CompressionCoordinator(
        token_counter=_Counter(),
        usable_input_budget=100,
        full_compaction_enabled=True,
        session_compressor=lambda _view: calls.append("a") or a_candidate,
        continuation_compressor=lambda _view: calls.append("b") or b_candidate,
    )

    decision = coordinator.prepare(_view(input_tokens=80), task_id="t", conversation_id="c")

    assert calls == ["a", "b"]
    assert decision.accepted is True
    assert decision.stage == "full"


def test_full_compaction_keeps_configured_completed_turns_plus_current() -> None:
    candidate = ContinuationCapsule(
        current_goal="current",
        active_papers=["paper-1"],
    )
    view = _view(input_tokens=80, goal="current").model_copy(
        update={
            "messages": [
                {"role": "user", "id": "u1", "content": "q1"},
                {"role": "assistant", "id": "a1", "content": "a1"},
                {"role": "user", "id": "u2", "content": "q2"},
                {"role": "assistant", "id": "a2", "content": "a2"},
                {"role": "user", "id": "u3", "content": "current"},
            ]
        }
    )
    decision = CompressionCoordinator(
        token_counter=_Counter(),
        usable_input_budget=100,
        full_compaction_enabled=True,
        session_compressor=lambda _view: None,
        continuation_compressor=lambda _view: candidate,
        recent_turns=1,
    ).prepare(view, task_id="t", conversation_id="c")

    assert decision.accepted is True
    assert [item["id"] for item in decision.view.messages] == ["u2", "a2", "u3"]


@pytest.mark.parametrize(
    "candidate, reason",
    [
        ({"active_projection": _projection().model_dump(), "source_archive_ids": [], "extra": 1}, "schema_invalid"),
        (SessionMemoryCandidate(active_projection=_projection(), source_archive_ids=["archive-missing"]), "invalid_reference"),
    ],
)
def test_validator_rejects_untrusted_candidates(candidate, reason: str) -> None:
    validator = ContinuationValidator()
    with pytest.raises(ValueError, match=reason):
        validator.validate_session(
            candidate,
            _view(input_tokens=80),
            authority_archive_ids=set(),
            authority_artifact_ids=set(),
            authority_paper_ids={"paper-1"},
            target_tokens=65,
            actual_tokens=40,
        )


def test_continuation_validator_rejects_fabricated_refs_and_dropped_active_papers() -> None:
    artifact = ArtifactRef(
        artifact_id="artifact-1",
        sha256="a" * 64,
        token_estimate=12,
        preview="trusted preview",
    )
    view = _view(input_tokens=80).model_copy(
        update={
            "active_projection": _projection().model_copy(
                update={"active_paper_ids": ["paper-1", "paper-2"]}
            ),
            "authority_artifact_ids": [artifact.artifact_id],
            "authority_artifact_refs": [artifact],
            "authority_evidence_ids": ["evidence-1"],
        }
    )
    valid = ContinuationCapsule(
        current_goal="goal",
        active_papers=["paper-1", "paper-2"],
        evidence_refs=["evidence-1"],
        artifact_refs=[artifact],
    )
    validator = ContinuationValidator()

    invalid_candidates = [
        valid.model_copy(update={"active_papers": ["paper-1"]}),
        valid.model_copy(update={"evidence_refs": ["evidence-invented"]}),
        valid.model_copy(
            update={
                "artifact_refs": [
                    artifact.model_copy(update={"artifact_id": "artifact-invented"})
                ]
            }
        ),
    ]
    for candidate in invalid_candidates:
        with pytest.raises(ValueError):
            validator.validate_continuation(
                candidate,
                view,
                authority_archive_ids=set(),
                authority_artifact_ids={"artifact-1"},
                authority_paper_ids={"paper-1", "paper-2"},
                target_tokens=50,
                actual_tokens=40,
            )


def test_continuation_validator_rejects_changed_artifact_hash() -> None:
    artifact = ArtifactRef(
        artifact_id="artifact-1",
        sha256="a" * 64,
        token_estimate=12,
        preview="trusted preview",
    )
    view = _view(input_tokens=80).model_copy(
        update={
            "authority_artifact_ids": [artifact.artifact_id],
            "authority_artifact_refs": [artifact],
        }
    )
    candidate = ContinuationCapsule(
        current_goal="goal",
        active_papers=["paper-1"],
        artifact_refs=[artifact.model_copy(update={"sha256": "b" * 64})],
    )

    with pytest.raises(ValueError, match="hash_mismatch"):
        ContinuationValidator().validate_continuation(
            candidate,
            view,
            authority_archive_ids=set(),
            authority_artifact_ids={"artifact-1"},
            authority_paper_ids={"paper-1"},
            target_tokens=50,
            actual_tokens=40,
        )


def test_continuation_validator_preserves_todos_verification_failures_and_rollback_by_field() -> None:
    projection = _projection().model_copy(
        update={
            "open_todos": ["todo-open"],
            "failed_verifications": ["citation:fail"],
            "unresolved_questions": ["missing comparison"],
        }
    )
    view = _view(input_tokens=80).model_copy(
        update={
            "active_projection": projection,
            "authority_verification": ["research:pass"],
            "authority_failed_paths": ["no_candidate"],
            "authority_rollback_notes": ["retry from checkpoint-1"],
        }
    )
    valid = ContinuationCapsule(
        current_goal="goal",
        active_papers=["paper-1"],
        verification=["research:pass", "citation:fail"],
        failed_paths=["no_candidate", "missing comparison"],
        unresolved_todos=["todo-open"],
        rollback_notes=["retry from checkpoint-1"],
    )
    validator = ContinuationValidator()
    validator.validate_continuation(
        valid,
        view,
        authority_archive_ids=set(),
        authority_artifact_ids=set(),
        authority_paper_ids={"paper-1"},
        target_tokens=50,
        actual_tokens=40,
    )

    misplaced_todo = valid.model_copy(
        update={
            "unresolved_todos": [],
            "verification": ["research:pass", "citation:fail", "todo-open"],
        }
    )
    with pytest.raises(ValueError, match="protected_context_lost"):
        validator.validate_continuation(
            misplaced_todo,
            view,
            authority_archive_ids=set(),
            authority_artifact_ids=set(),
            authority_paper_ids={"paper-1"},
            target_tokens=50,
            actual_tokens=40,
        )


def test_full_compaction_disabled_never_calls_compressor() -> None:
    calls: list[str] = []
    coordinator = CompressionCoordinator(
        token_counter=_Counter(),
        usable_input_budget=100,
        full_compaction_enabled=False,
        session_compressor=lambda _view: calls.append("a"),
        continuation_compressor=lambda _view: calls.append("b"),
    )

    decision = coordinator.prepare(_view(input_tokens=90), task_id="t", conversation_id="c")

    assert calls == []
    assert decision.reason == "full_compaction_disabled"


def test_transient_provider_failure_gets_only_one_extra_attempt_per_stage() -> None:
    calls = 0
    candidate = SessionMemoryCandidate(active_projection=_projection(), source_archive_ids=[])

    def compressor(_view):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary timeout")
        return candidate

    decision = CompressionCoordinator(
        token_counter=_Counter(),
        usable_input_budget=100,
        full_compaction_enabled=True,
        session_compressor=compressor,
    ).prepare(_view(input_tokens=80), task_id="t", conversation_id="c")

    assert decision.accepted is True
    assert calls == 2


def test_configured_transient_retry_count_controls_extra_attempts() -> None:
    calls = 0
    candidate = SessionMemoryCandidate(active_projection=_projection(), source_archive_ids=[])

    def compressor(_view):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TimeoutError("temporary timeout")
        return candidate

    decision = CompressionCoordinator(
        token_counter=_Counter(),
        usable_input_budget=100,
        full_compaction_enabled=True,
        session_compressor=compressor,
        transient_retry_count=2,
    ).prepare(_view(input_tokens=80), task_id="t", conversation_id="c")

    assert decision.accepted is True
    assert calls == 3


def test_protected_context_oversized_skips_both_compressor_stages() -> None:
    exact = "必须保留的约束 " * 100
    protected = ProtectedText(
        protected_id="protected-1",
        exact_text=exact,
        sha256=hashlib.sha256(exact.encode()).hexdigest(),
    )
    view = _view(input_tokens=90).model_copy(
        deep=True,
        update={
            "active_projection": _projection().model_copy(
                update={"active_constraints": [exact], "protected_items": [protected]}
            )
        },
    )
    calls: list[str] = []
    decision = CompressionCoordinator(
        token_counter=_Counter(),
        usable_input_budget=100,
        full_compaction_enabled=True,
        session_compressor=lambda _view: calls.append("a"),
        continuation_compressor=lambda _view: calls.append("b"),
    ).prepare(view, task_id="t", conversation_id="c")

    assert decision.accepted is False
    assert decision.reason == "protected_context_oversized"
    assert calls == []
