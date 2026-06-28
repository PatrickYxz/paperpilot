"""Execute query plans against a paper-scoped search function."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from paperpilot.retrieval.evidence_pool import (
    EvidencePool,
    RawSearchHit,
    build_evidence_pool,
    format_evidence_summary,
)
from paperpilot.retrieval.evidence_verifier import run_evidence_verification
from paperpilot.retrieval.query_plan import QueryPlan

SearchFn = Callable[[str, str, int], list[dict[str, Any]]]


@dataclass(frozen=True)
class PlannedRetrievalResult:
    evidence_pool: EvidencePool
    summary_text: str
    query_errors: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_text": self.summary_text,
            "evidence_pool": self.evidence_pool.to_dict(),
            "query_errors": [dict(item) for item in self.query_errors],
        }


def run_planned_retrieval(
    *,
    plan_id: str,
    plan: QueryPlan,
    paper_id: str,
    search: SearchFn,
    top_k_each: int = 5,
    summary_k: int = 8,
    verify_evidence: bool = False,
    verifier_candidate_k: int = 6,
    verifier_client: Any | None = None,
) -> PlannedRetrievalResult:
    hits: list[RawSearchHit] = []
    query_errors: list[dict[str, str]] = []
    for planned_query in sorted(plan.queries, key=lambda item: item.priority):
        try:
            results = search(planned_query.query, paper_id, top_k_each)
        except Exception as exc:  # noqa: BLE001
            query_errors.append(
                {
                    "query_id": planned_query.id,
                    "query": planned_query.query,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        for rank, item in enumerate(results, start=1):
            hits.append(
                RawSearchHit(
                    paper_id=str(item.get("paper_id") or paper_id),
                    chunk_id=_optional_string(item.get("chunk_id")),
                    chunk_text=str(item.get("chunk_text") or ""),
                    score=float(item.get("score") or 0.0),
                    query=planned_query,
                    rank=rank,
                )
            )
    pool = build_evidence_pool(
        plan_id,
        plan,
        hits,
        summary_k=summary_k,
    )
    if verify_evidence:
        pool.verification = run_evidence_verification(
            plan=plan,
            pool=pool,
            client=verifier_client,
            summary_k=summary_k,
            verifier_candidate_k=verifier_candidate_k,
        )
    return PlannedRetrievalResult(
        evidence_pool=pool,
        summary_text=format_evidence_summary(pool, use_verified=verify_evidence),
        query_errors=query_errors,
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
