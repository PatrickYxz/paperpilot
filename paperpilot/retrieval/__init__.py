"""Retrieval planning and evidence pooling helpers."""

from paperpilot.retrieval.bm25 import BM25Index, tokenize
from paperpilot.retrieval.evidence_pool import (
    EvidencePool,
    RawSearchHit,
    build_evidence_pool,
    format_evidence_summary,
)
from paperpilot.retrieval.evidence_verifier import (
    EvidenceVerificationDecision,
    EvidenceVerificationResult,
    run_evidence_verification,
)
from paperpilot.retrieval.fusion import rrf_fuse
from paperpilot.retrieval.metrics import mrr, ndcg, recall_at_k
from paperpilot.retrieval.llm_query_planner import (
    build_planner_prompt,
    plan_with_llm,
)
from paperpilot.retrieval.planned_retrieval import (
    PlannedRetrievalResult,
    run_planned_retrieval,
)
from paperpilot.retrieval.query_plan import (
    EvidenceRequirement,
    ExpansionHints,
    PlannedQuery,
    QueryConstraints,
    QueryPlan,
    minimal_fallback_plan,
)
from paperpilot.retrieval.query_plan_validator import (
    parse_query_plan_json,
    validate_query_plan,
)

__all__ = [
    "BM25Index",
    "EvidencePool",
    "EvidenceVerificationDecision",
    "EvidenceVerificationResult",
    "EvidenceRequirement",
    "ExpansionHints",
    "PlannedQuery",
    "PlannedRetrievalResult",
    "QueryConstraints",
    "QueryPlan",
    "RawSearchHit",
    "build_evidence_pool",
    "build_planner_prompt",
    "format_evidence_summary",
    "minimal_fallback_plan",
    "mrr",
    "ndcg",
    "parse_query_plan_json",
    "plan_with_llm",
    "recall_at_k",
    "rrf_fuse",
    "run_evidence_verification",
    "run_planned_retrieval",
    "tokenize",
    "validate_query_plan",
]
