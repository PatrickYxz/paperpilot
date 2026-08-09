"""Validation contracts for deep-reading structured data."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from paperpilot.deep_reading.schemas import (
    AnswerCitation,
    AnswerDraft,
    ConversationSummary,
    EvidenceItem,
    PaperUse,
    ResearchResult,
)
from paperpilot.deep_reading.state import DeepReadingState
from paperpilot.papers import PaperCandidate


def _paper(external_id: str = "2401.00001") -> PaperCandidate:
    return PaperCandidate(
        external_id=external_id,
        title=f"Paper {external_id}",
        authors=["Ada Lovelace"],
        source_url=f"https://arxiv.org/abs/{external_id}",
    )


def _evidence(
    evidence_id: str,
    *,
    paper_external_id: str = "2401.00001",
    chunk_text: str = "The measured effect is statistically significant.",
    score: float = 0.8,
) -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        paper_external_id=paper_external_id,
        paper_title=f"Paper {paper_external_id}",
        chunk_text=chunk_text,
        score=score,
        supports=["The method improves the reported result."],
    )


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_evidence_score_accepts_inclusive_boundaries(score: float) -> None:
    assert _evidence("ev-1", score=score).score == score


@pytest.mark.parametrize("score", [-0.0001, 1.0001])
def test_evidence_score_rejects_values_outside_unit_interval(score: float) -> None:
    with pytest.raises(ValidationError):
        _evidence("ev-1", score=score)


def test_evidence_rejects_blank_chunk_text() -> None:
    with pytest.raises(ValidationError):
        _evidence("ev-1", chunk_text="   ")


def test_research_result_rejects_duplicate_evidence_ids() -> None:
    with pytest.raises(ValidationError):
        ResearchResult(
            evidence_items=[_evidence("ev-1"), _evidence("ev-1")],
            used_papers=[],
            limitations=[],
        )


def test_research_result_rejects_dangling_paper_evidence_reference() -> None:
    with pytest.raises(ValidationError):
        ResearchResult(
            evidence_items=[_evidence("ev-1")],
            used_papers=[
                PaperUse(
                    paper=_paper(),
                    role="comparison",
                    evidence_ids=["missing"],
                )
            ],
            limitations=[],
        )


def test_research_result_rejects_evidence_owned_by_another_paper() -> None:
    with pytest.raises(ValidationError):
        ResearchResult(
            evidence_items=[
                _evidence("ev-1", paper_external_id="2401.00002")
            ],
            used_papers=[
                PaperUse(
                    paper=_paper("2401.00001"),
                    role="citation",
                    evidence_ids=["ev-1"],
                )
            ],
            limitations=[],
        )


def test_full_research_result_round_trips_through_json_safe_state_payload() -> None:
    result = ResearchResult(
        evidence_items=[_evidence("ev-1")],
        used_papers=[
            PaperUse(
                paper=_paper(),
                role="comparison",
                evidence_ids=["ev-1"],
            )
        ],
        limitations=["Only one benchmark was reported."],
    )

    payload = result.model_dump(mode="json")
    state: DeepReadingState = {"research_result": payload}

    assert json.loads(json.dumps(state["research_result"])) == payload
    assert ResearchResult.model_validate(state["research_result"]) == result


def test_paper_use_rejects_duplicate_or_empty_evidence_references() -> None:
    with pytest.raises(ValidationError):
        PaperUse(
            paper=_paper(),
            role="background",
            evidence_ids=["ev-1", "ev-1"],
        )

    with pytest.raises(ValidationError):
        PaperUse(paper=_paper(), role="background", evidence_ids=[])


def test_paper_use_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        PaperUse(
            paper=_paper(),
            role="primary",  # type: ignore[arg-type]
            evidence_ids=["ev-1"],
        )


def test_answer_draft_rejects_duplicate_citations_and_unknown_quality() -> None:
    with pytest.raises(ValidationError):
        AnswerDraft(
            content="Answer",
            citations=[
                AnswerCitation(evidence_id="ev-1", label="[1]"),
                AnswerCitation(evidence_id="ev-1", label="[again]"),
            ],
            result_quality="complete",
        )

    with pytest.raises(ValidationError):
        AnswerDraft(
            content="Answer",
            citations=[],
            result_quality="uncertain",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("quality", ["complete", "partial"])
def test_answer_draft_accepts_supported_quality(quality: str) -> None:
    draft = AnswerDraft(
        content="Evidence-backed answer",
        citations=[AnswerCitation(evidence_id="ev-1", label="[1]")],
        result_quality=quality,
    )

    assert draft.result_quality == quality


def test_conversation_summary_json_round_trip() -> None:
    summary = ConversationSummary(
        confirmed_facts=["The user selected the primary paper."],
        paper_findings=["The method uses sparse attention."],
        comparison_context=["A baseline uses dense attention."],
        open_questions=["Does the result transfer to larger datasets?"],
    )

    payload = summary.model_dump(mode="json")

    assert ConversationSummary.model_validate(payload) == summary
    assert payload == {
        "confirmed_facts": ["The user selected the primary paper."],
        "paper_findings": ["The method uses sparse attention."],
        "comparison_context": ["A baseline uses dense attention."],
        "open_questions": ["Does the result transfer to larger datasets?"],
    }
