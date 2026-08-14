"""Structured state and citation validation shared by answer nodes."""
from __future__ import annotations

from pydantic import ValidationError

from ..research_agent import ResearchContractError
from ..schemas import AnswerDraft, ResearchResult


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchContractError(f"{field_name} must be a non-blank string")
    return value.strip()


def _validated_research_result(value: object) -> ResearchResult:
    try:
        return ResearchResult.model_validate(value)
    except ValidationError as exc:
        raise ResearchContractError("state research_result is invalid") from exc


def _validated_answer_draft(value: object) -> AnswerDraft:
    try:
        return AnswerDraft.model_validate(value)
    except ValidationError as exc:
        raise ResearchContractError("state answer_draft is invalid") from exc


def _validate_answer_citations(
    draft: AnswerDraft,
    result: ResearchResult,
) -> None:
    evidence_ids = {item.id for item in result.evidence_items}
    for citation in draft.citations:
        if citation.evidence_id not in evidence_ids:
            raise ResearchContractError(
                f"answer citation references unknown evidence: {citation.evidence_id}"
            )
