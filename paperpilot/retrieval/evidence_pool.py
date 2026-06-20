"""Evidence pooling for planned retrieval results."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from paperpilot.retrieval.query_plan import PlannedQuery, QueryPlan


@dataclass(frozen=True)
class RawSearchHit:
    paper_id: str
    chunk_id: str | None
    chunk_text: str
    score: float
    query: PlannedQuery
    rank: int


@dataclass
class MatchedQuery:
    query_id: str
    query: str
    role: str
    rank: int
    score: float
    targets: list[str]


@dataclass
class EvidenceItem:
    id: str
    paper_id: str
    chunk_id: str | None
    chunk_text: str
    best_score: float
    matched_queries: list[MatchedQuery] = field(default_factory=list)

    def targets(self) -> set[str]:
        out: set[str] = set()
        for match in self.matched_queries:
            out.update(match.targets)
        return out


@dataclass(frozen=True)
class MissingRequirement:
    requirement_id: str
    description: str
    reason: str


@dataclass
class EvidencePool:
    plan_id: str
    question: str
    query_plan: dict[str, Any]
    items: list[EvidenceItem]
    summary_items: list[EvidenceItem]
    missing_requirements: list[MissingRequirement]
    stats: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "question": self.question,
            "query_plan": self.query_plan,
            "items": [asdict(item) for item in self.items],
            "summary_items": [item.id for item in self.summary_items],
            "missing_requirements": [
                asdict(item) for item in self.missing_requirements
            ],
            "stats": dict(self.stats),
        }


def build_evidence_pool(
    plan_id: str,
    plan: QueryPlan,
    hits: list[RawSearchHit],
    *,
    summary_k: int = 8,
    overlap_threshold: float = 0.75,
) -> EvidencePool:
    deduped = _dedupe_hits(hits)
    summary, missing = _select_summary(
        plan,
        deduped,
        summary_k=summary_k,
        overlap_threshold=overlap_threshold,
    )
    return EvidencePool(
        plan_id=plan_id,
        question=plan.question,
        query_plan=plan.to_dict(),
        items=deduped,
        summary_items=summary,
        missing_requirements=missing,
        stats={
            "query_count": len(plan.queries),
            "raw_result_count": len(hits),
            "deduped_count": len(deduped),
        },
    )


def format_evidence_summary(pool: EvidencePool) -> str:
    missing = (
        "none"
        if not pool.missing_requirements
        else "\n".join(
            f"- {item.requirement_id}: {item.description} ({item.reason})"
            for item in pool.missing_requirements
        )
    )
    blocks = [
        "Planned retrieval completed.",
        f"Queries executed: {pool.stats['query_count']}",
        (
            "Evidence chunks: "
            f"{pool.stats['raw_result_count']} raw, "
            f"{pool.stats['deduped_count']} deduped"
        ),
        f"Missing requirements: {missing}",
        "",
        "Top evidence:",
    ]
    for item in pool.summary_items:
        query_ids = ", ".join(match.query_id for match in item.matched_queries)
        blocks.append(
            f"[{item.id}] Found by {query_ids}. Score {item.best_score:.2f}.\n"
            f"{item.chunk_text}"
        )
    if not pool.summary_items:
        blocks.append("No retrieved evidence.")
    return "\n\n".join(blocks)


def _dedupe_hits(hits: list[RawSearchHit]) -> list[EvidenceItem]:
    by_key: dict[str, EvidenceItem] = {}
    order: list[str] = []
    for hit in hits:
        key = _dedupe_key(hit)
        match = MatchedQuery(
            query_id=hit.query.id,
            query=hit.query.query,
            role=hit.query.role,
            rank=hit.rank,
            score=float(hit.score),
            targets=list(hit.query.targets),
        )
        existing = by_key.get(key)
        if existing is None:
            item = EvidenceItem(
                id=f"ev_{len(order) + 1}",
                paper_id=hit.paper_id,
                chunk_id=hit.chunk_id,
                chunk_text=hit.chunk_text,
                best_score=float(hit.score),
                matched_queries=[match],
            )
            by_key[key] = item
            order.append(key)
        else:
            existing.best_score = max(existing.best_score, float(hit.score))
            existing.matched_queries.append(match)
    return sorted(
        (by_key[key] for key in order),
        key=lambda item: item.best_score,
        reverse=True,
    )


def _select_summary(
    plan: QueryPlan,
    items: list[EvidenceItem],
    *,
    summary_k: int,
    overlap_threshold: float,
) -> tuple[list[EvidenceItem], list[MissingRequirement]]:
    selected: list[EvidenceItem] = []
    missing: list[MissingRequirement] = []
    query_targets = {target for query in plan.queries for target in query.targets}
    for requirement in plan.evidence_requirements:
        if not requirement.required:
            continue
        if requirement.id not in query_targets:
            missing.append(
                MissingRequirement(
                    requirement.id,
                    requirement.description,
                    "no_targeted_query",
                )
            )
            continue
        candidates = [item for item in items if requirement.id in item.targets()]
        candidate = _first_diverse(candidates, selected, overlap_threshold)
        if candidate is None:
            missing.append(
                MissingRequirement(
                    requirement.id,
                    requirement.description,
                    "no_candidates",
                )
            )
        else:
            selected.append(candidate)
        if len(selected) >= summary_k:
            return selected[:summary_k], missing
    for item in items:
        if len(selected) >= summary_k:
            break
        if item in selected:
            continue
        if _is_diverse(item, selected, overlap_threshold):
            selected.append(item)
    return selected, missing


def _first_diverse(
    candidates: list[EvidenceItem],
    selected: list[EvidenceItem],
    overlap_threshold: float,
) -> EvidenceItem | None:
    for candidate in candidates:
        if candidate not in selected and _is_diverse(
            candidate, selected, overlap_threshold
        ):
            return candidate
    for candidate in candidates:
        if candidate not in selected:
            return candidate
    return None


def _is_diverse(
    candidate: EvidenceItem, selected: list[EvidenceItem], threshold: float
) -> bool:
    return all(
        _token_overlap(candidate.chunk_text, item.chunk_text) <= threshold
        for item in selected
    )


def _token_overlap(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"[A-Za-z0-9_]+", a.lower()))
    tokens_b = set(re.findall(r"[A-Za-z0-9_]+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


def _dedupe_key(hit: RawSearchHit) -> str:
    if hit.chunk_id:
        return f"{hit.paper_id}::{hit.chunk_id}"
    normalized = " ".join(hit.chunk_text.split()).lower()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return f"{hit.paper_id}::{digest}"
