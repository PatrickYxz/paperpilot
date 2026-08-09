"""Pydantic contracts shared by deep-reading graph nodes."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from paperpilot.papers import PaperCandidate


def _require_non_blank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _require_non_blank_items(values: list[str], field_name: str) -> list[str]:
    for value in values:
        _require_non_blank(value, field_name)
    return values


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    paper_external_id: str
    paper_title: str
    chunk_text: str
    score: float = Field(ge=0.0, le=1.0)
    supports: list[str]

    @field_validator("id", "paper_external_id", "paper_title", "chunk_text")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_non_blank(value, "evidence text")

    @field_validator("supports")
    @classmethod
    def validate_supports(cls, values: list[str]) -> list[str]:
        return _require_non_blank_items(values, "support")


class PaperUse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper: PaperCandidate
    role: Literal["comparison", "citation", "background", "follow_up"]
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, values: list[str]) -> list[str]:
        _require_non_blank_items(values, "evidence id")
        if len(values) != len(set(values)):
            raise ValueError("evidence_ids must be unique")
        return values


class ResearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_items: list[EvidenceItem]
    used_papers: list[PaperUse]
    limitations: list[str]

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, values: list[str]) -> list[str]:
        return _require_non_blank_items(values, "limitation")

    @model_validator(mode="after")
    def validate_evidence_references(self) -> "ResearchResult":
        evidence_by_id: dict[str, EvidenceItem] = {}
        for evidence in self.evidence_items:
            if evidence.id in evidence_by_id:
                raise ValueError("evidence item ids must be unique")
            evidence_by_id[evidence.id] = evidence

        used_paper_keys: set[tuple[str, str]] = set()
        for paper_use in self.used_papers:
            paper_key = (paper_use.paper.source, paper_use.paper.external_id)
            if paper_key in used_paper_keys:
                raise ValueError("used papers must be unique")
            used_paper_keys.add(paper_key)
            for evidence_id in paper_use.evidence_ids:
                evidence = evidence_by_id.get(evidence_id)
                if evidence is None:
                    raise ValueError(f"unknown evidence id: {evidence_id}")
                if evidence.paper_external_id != paper_use.paper.external_id:
                    raise ValueError(
                        f"evidence {evidence_id} does not belong to used paper "
                        f"{paper_use.paper.external_id}"
                    )
        return self


class AnswerCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    label: str

    @field_validator("evidence_id", "label")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        return _require_non_blank(value, "citation text")


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    citations: list[AnswerCitation]
    result_quality: Literal["complete", "partial"]

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return _require_non_blank(value, "answer content")

    @field_validator("citations")
    @classmethod
    def validate_citations(
        cls, values: list[AnswerCitation]
    ) -> list[AnswerCitation]:
        evidence_ids = [citation.evidence_id for citation in values]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("citation evidence ids must be unique")
        return values


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmed_facts: list[str]
    paper_findings: list[str]
    comparison_context: list[str]
    open_questions: list[str]

    @field_validator(
        "confirmed_facts",
        "paper_findings",
        "comparison_context",
        "open_questions",
    )
    @classmethod
    def validate_summary_items(cls, values: list[str]) -> list[str]:
        return _require_non_blank_items(values, "summary item")
