"""Deterministic TurnArchive seed and Research Agent outcome contracts."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from langchain.messages import ToolMessage
from pydantic import ValidationError

from paperpilot.deep_reading.context_management.archives import (
    AgentResearchContextDelta,
    ResearchExecutionOutcome,
    ResearchTrace,
    TurnArchiveNarrative,
    TurnArchiveSeedBuilder,
    validate_context_delta,
)
from paperpilot.deep_reading.context_management.models import ArtifactRef
from paperpilot.deep_reading.schemas import AnswerDraft, ResearchResult


def _message(
    *,
    message_id: str,
    conversation_id: str = "conversation-1",
    role: str = "user",
    content: str = "Please compare the primary method and preserve this constraint.",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
    )


def _result() -> ResearchResult:
    return ResearchResult(evidence_items=[], used_papers=[], limitations=["no_direct_evidence: one gap"])


def test_context_delta_accepts_exact_user_span_and_server_computes_hash() -> None:
    message = _message(message_id="message-1")
    delta = AgentResearchContextDelta(
        constraints=[
            {
                "source_message_id": message.id,
                "exact_text": "preserve this constraint",
                "kind": "constraint",
            }
        ],
        decisions=[],
        supersedes=[],
    )

    validated = validate_context_delta(
        delta,
        conversation_id="conversation-1",
        messages=[message],
        authority_set=set(),
    )

    protected = validated.constraints[0]
    assert protected.exact_text == "preserve this constraint"
    assert protected.sha256 == hashlib.sha256(protected.exact_text.encode()).hexdigest()
    assert protected.protected_id.startswith("protected_")


@pytest.mark.parametrize(
    "message, exact_text",
    [
        (_message(message_id="message-1"), "preserve this constriant"),
        (_message(message_id="message-1", role="assistant"), "preserve this constraint"),
        (_message(message_id="message-1", conversation_id="other"), "preserve this constraint"),
    ],
)
def test_invalid_context_delta_is_rejected_without_mutating_result(message, exact_text) -> None:
    delta = AgentResearchContextDelta(
        constraints=[
            {
                "source_message_id": "message-1",
                "exact_text": exact_text,
                "kind": "constraint",
            }
        ]
    )

    with pytest.raises(ValueError, match="invalid_exact_span"):
        validate_context_delta(
            delta,
            conversation_id="conversation-1",
            messages=[message],
            authority_set=set(),
        )


def test_unknown_supersession_target_is_rejected() -> None:
    delta = AgentResearchContextDelta(
        supersedes=[
            {
                "target_protected_id": "protected-missing",
                "source_message_id": "message-1",
                "exact_text": "use the corrected method",
            }
        ]
    )

    with pytest.raises(ValueError, match="invalid_exact_span"):
        validate_context_delta(
            delta,
            conversation_id="conversation-1",
            messages=[_message(message_id="message-1", content="use the corrected method")],
            authority_set=set(),
        )


def test_seed_builder_preserves_authoritative_refs_and_ignores_legacy_summary() -> None:
    user_message = _message(message_id="message-1")
    delta = validate_context_delta(
        AgentResearchContextDelta(
            constraints=[
                {
                    "source_message_id": user_message.id,
                    "exact_text": "preserve this constraint",
                    "kind": "constraint",
                }
            ]
        ),
        conversation_id="conversation-1",
        messages=[user_message],
        authority_set=set(),
    )
    trace = ResearchTrace(
        todos=("todo_1:completed", "todo_2:pending"),
        tool_outcomes=("no_direct_evidence",),
        artifact_ids=("artifact-1",),
        verification=("citation:pass",),
    )
    artifact = ArtifactRef(
        artifact_id="artifact-1",
        sha256="a" * 64,
        token_estimate=12,
        preview="bounded preview",
    )

    seed = TurnArchiveSeedBuilder().build(
        archive_id="archive-1",
        conversation_id="conversation-1",
        task_id="task-1",
        user_message=user_message,
        terminal_status="success",
        research_result=_result(),
        answer_draft=AnswerDraft(content="answer", citations=[], result_quality="partial"),
        context_delta=delta,
        trace=trace,
        artifact_refs=[artifact],
        legacy_summary={"confirmed_facts": ["must not become a fact"]},
        created_at="2026-09-01T00:00:00+00:00",
    )

    assert seed.user_goal.exact_text == user_message.content
    assert seed.user_goal.sha256 == hashlib.sha256(user_message.content.encode()).hexdigest()
    assert seed.constraints[0].exact_text == "preserve this constraint"
    assert seed.artifact_refs[0].artifact_id == "artifact-1"
    assert seed.unresolved_todos == ["todo_2:pending"]
    assert seed.rollback_notes == []
    assert "must not become a fact" not in json.dumps(seed.model_dump(mode="json"))
    assert "citation:pass" in seed.verification


def test_narrative_only_contains_summary_and_cannot_change_seed() -> None:
    narrative = TurnArchiveNarrative(summary="A bounded narrative.")
    assert narrative.model_dump() == {"summary": "A bounded narrative."}
    with pytest.raises(ValidationError):
        TurnArchiveNarrative(summary="ok", decisions=["rewrite"])


def test_execution_outcome_keeps_legacy_result_access_and_redacts_trace() -> None:
    outcome = ResearchExecutionOutcome(
        result=_result(),
        trace=ResearchTrace(
            todos=("todo_1:completed",),
            tool_outcomes=("success",),
            artifact_ids=("artifact-1",),
            verification=("research:pass",),
        ),
        context_delta=None,
    )

    assert outcome.result == _result()
    assert outcome.evidence_items == outcome.result.evidence_items
    trace_json = json.dumps(outcome.trace.to_dict())
    assert "ToolMessage" not in trace_json
    assert "Please compare" not in trace_json
    assert "chunk text" not in trace_json
