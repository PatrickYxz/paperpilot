from paperpilot.retrieval.evidence_pool import (
    RawSearchHit,
    build_evidence_pool,
    format_evidence_summary,
)
from paperpilot.retrieval.query_plan import (
    EvidenceRequirement,
    PlannedQuery,
    QueryConstraints,
    QueryPlan,
)


def _plan() -> QueryPlan:
    return QueryPlan(
        version="query_plan_v1",
        question="What datasets and baselines were used?",
        question_type="other",
        answer_shape="list",
        intent_summary="Find datasets and baselines.",
        focus_terms=[],
        constraints=QueryConstraints(),
        evidence_requirements=[
            EvidenceRequirement("req_dataset", "datasets used by the paper", True),
            EvidenceRequirement(
                "req_baseline", "baselines compared by the paper", True
            ),
        ],
        queries=[
            PlannedQuery(
                "q_dataset",
                "focused_rewrite",
                "dataset used experiment",
                ["req_dataset"],
                1,
            ),
            PlannedQuery(
                "q_baseline",
                "focused_rewrite",
                "baseline compared method",
                ["req_baseline"],
                2,
            ),
        ],
    )


def test_exact_duplicate_chunks_merge_matched_queries() -> None:
    plan = _plan()
    hits = [
        RawSearchHit(
            paper_id="p1",
            chunk_id=None,
            chunk_text="The dataset is WikiHop.",
            score=10.0,
            query=plan.queries[0],
            rank=1,
        ),
        RawSearchHit(
            paper_id="p1",
            chunk_id=None,
            chunk_text="The dataset   is WikiHop.",
            score=8.0,
            query=plan.queries[1],
            rank=2,
        ),
    ]

    pool = build_evidence_pool("plan-1", plan, hits, summary_k=4)

    assert pool.stats["raw_result_count"] == 2
    assert pool.stats["deduped_count"] == 1
    assert len(pool.items) == 1
    assert pool.items[0].best_score == 10.0
    assert [match.query_id for match in pool.items[0].matched_queries] == [
        "q_dataset",
        "q_baseline",
    ]


def test_summary_selection_covers_required_requirements_before_score_fill() -> None:
    plan = _plan()
    hits = [
        RawSearchHit("p1", None, "Dataset evidence A.", 30.0, plan.queries[0], 1),
        RawSearchHit("p1", None, "Dataset evidence B.", 29.0, plan.queries[0], 2),
        RawSearchHit("p1", None, "Baseline evidence.", 20.0, plan.queries[1], 1),
    ]

    pool = build_evidence_pool("plan-1", plan, hits, summary_k=2)
    selected_texts = [item.chunk_text for item in pool.summary_items]

    assert selected_texts == ["Dataset evidence A.", "Baseline evidence."]
    assert pool.missing_requirements == []


def test_missing_requirement_records_no_candidates() -> None:
    plan = _plan()
    hits = [
        RawSearchHit("p1", None, "Dataset evidence A.", 30.0, plan.queries[0], 1),
    ]

    pool = build_evidence_pool("plan-1", plan, hits, summary_k=4)

    assert len(pool.missing_requirements) == 1
    assert pool.missing_requirements[0].requirement_id == "req_baseline"
    assert pool.missing_requirements[0].reason == "no_candidates"


def test_format_evidence_summary_includes_missing_requirements() -> None:
    plan = _plan()
    pool = build_evidence_pool("plan-1", plan, [], summary_k=4)

    text = format_evidence_summary(pool)

    assert "Planned retrieval completed." in text
    assert "Missing requirements:" in text
    assert "req_dataset" in text
    assert "req_baseline" in text
