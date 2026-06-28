from paperpilot.mcp_servers.colbert import server
from paperpilot.retrieval.query_plan import (
    EvidenceRequirement,
    PlannedQuery,
    QueryConstraints,
    QueryPlan,
)


class FakeManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def search(self, query: str, paper_id: str, top_k: int) -> list[dict]:
        self.calls.append((query, paper_id, top_k))
        return [
            {
                "paper_id": paper_id,
                "chunk_id": f"chunk-{len(self.calls)}",
                "chunk_text": f"Evidence for {query}.",
                "score": 9.0,
            }
        ]


def _plan(question: str) -> QueryPlan:
    return QueryPlan(
        version="query_plan_v1",
        question=question,
        question_type="other",
        answer_shape="freeform",
        intent_summary="Find direct evidence.",
        focus_terms=[],
        constraints=QueryConstraints(),
        evidence_requirements=[
            EvidenceRequirement("req_direct", "direct evidence", True),
        ],
        queries=[
            PlannedQuery(
                "q_1",
                "literal",
                question,
                ["req_direct"],
                1,
            ),
            PlannedQuery(
                "q_2",
                "focused_rewrite",
                "direct evidence",
                ["req_direct"],
                2,
            ),
        ],
    )


def test_planned_retrieval_impl_returns_evidence_pool(monkeypatch) -> None:
    manager = FakeManager()
    monkeypatch.setattr(server, "_manager", manager)

    def fake_plan_with_llm(**kwargs):
        return _plan(kwargs["question"]), {"fallback_used": False}

    monkeypatch.setattr(server, "plan_with_llm", fake_plan_with_llm)

    payload = server._planned_retrieval_impl(
        question="What dataset was used?",
        paper_id="paper-1",
        paper_title="Title",
        abstract="Abstract",
        top_k_each=3,
        summary_k=2,
    )

    assert manager.calls == [
        ("What dataset was used?", "paper-1", 3),
        ("direct evidence", "paper-1", 3),
    ]
    assert payload["query_plan_meta"] == {"fallback_used": False}
    assert payload["evidence_pool"]["stats"]["raw_result_count"] == 2
    assert payload["evidence_pool"]["summary_items"]
    assert "Planned retrieval completed." in payload["summary_text"]


def test_planned_retrieval_impl_passes_verification_options(monkeypatch) -> None:
    manager = FakeManager()
    monkeypatch.setattr(server, "_manager", manager)

    def fake_plan_with_llm(**kwargs):
        return _plan(kwargs["question"]), {"fallback_used": False}

    captured: dict = {}

    def fake_run_planned_retrieval(**kwargs):
        captured.update(kwargs)

        class Result:
            def to_dict(self):
                return {
                    "summary_text": "Verified planned retrieval completed.",
                    "evidence_pool": {
                        "stats": {},
                        "summary_items": [],
                        "verified_summary_items": ["ev_1"],
                        "verification": {
                            "enabled": True,
                            "method": "llm_requirement_verifier_v1",
                            "decisions": [],
                            "missing_verified_requirements": [],
                            "conflicts": [],
                            "stats": {
                                "decision_count": 1,
                                "direct_count": 1,
                                "partial_count": 0,
                                "no_count": 0,
                            },
                        },
                    },
                    "query_errors": [],
                }

        return Result()

    monkeypatch.setattr(server, "plan_with_llm", fake_plan_with_llm)
    monkeypatch.setattr(server, "run_planned_retrieval", fake_run_planned_retrieval)

    payload = server._planned_retrieval_impl(
        question="What dataset was used?",
        paper_id="paper-1",
        paper_title="Title",
        abstract="Abstract",
        top_k_each=3,
        summary_k=2,
        verify_evidence=True,
        verifier_candidate_k=4,
    )

    assert captured["verify_evidence"] is True
    assert captured["verifier_candidate_k"] == 4
    assert payload["query_plan_meta"] == {"fallback_used": False}
    assert payload["evidence_pool"]["verification"]["enabled"] is True
