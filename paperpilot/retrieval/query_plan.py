"""Structured query planning schema for planned retrieval."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PLAN_VERSION = "query_plan_v1"


@dataclass(frozen=True)
class QueryConstraints:
    needs_numbers: bool = False
    needs_comparison: bool = False
    needs_table_or_figure: bool = False
    polarity: str = "neutral"


@dataclass(frozen=True)
class EvidenceRequirement:
    id: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class PlannedQuery:
    id: str
    role: str
    query: str
    targets: list[str]
    priority: int = 1


@dataclass(frozen=True)
class ExpansionHints:
    neighbor_window: int = 1
    prefer_tables: bool = False
    prefer_captions: bool = False


@dataclass(frozen=True)
class QueryPlan:
    version: str
    question: str
    question_type: str
    answer_shape: str
    intent_summary: str
    focus_terms: list[str]
    constraints: QueryConstraints
    evidence_requirements: list[EvidenceRequirement]
    queries: list[PlannedQuery]
    avoid: list[str] = field(default_factory=list)
    expansion_hints: ExpansionHints = field(default_factory=ExpansionHints)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def minimal_fallback_plan(question: str) -> QueryPlan:
    """Return a safe literal-query-only plan."""
    requirement = EvidenceRequirement(
        id="req_direct",
        description="direct evidence answering the question",
        required=True,
    )
    return QueryPlan(
        version=PLAN_VERSION,
        question=question,
        question_type="other",
        answer_shape="freeform",
        intent_summary="Find direct evidence that answers the user question.",
        focus_terms=[],
        constraints=QueryConstraints(),
        evidence_requirements=[requirement],
        queries=[
            PlannedQuery(
                id="q_lit",
                role="literal",
                query=question,
                targets=[requirement.id],
                priority=1,
            )
        ],
    )
