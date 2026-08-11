from paperpilot.retrieval.evidence_verifier import (
    EvidenceVerificationDecisionOutput,
    EvidenceVerificationOutput,
)
from paperpilot.retrieval.planned_retrieval import run_planned_retrieval
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
                "q_baseline",
                "focused_rewrite",
                "baseline compared method",
                ["req_baseline"],
                2,
            ),
            PlannedQuery(
                "q_dataset",
                "focused_rewrite",
                "dataset used experiment",
                ["req_dataset"],
                1,
            ),
        ],
    )


def test_run_planned_retrieval_executes_all_plan_queries() -> None:
    calls: list[tuple[str, str, int]] = []

    def search(query: str, paper_id: str, top_k: int) -> list[dict]:
        calls.append((query, paper_id, top_k))
        return [
            {
                "paper_id": paper_id,
                "chunk_id": f"chunk-{len(calls)}",
                "chunk_text": f"Evidence for {query}",
                "score": 10.0 - len(calls),
            }
        ]

    result = run_planned_retrieval(
        plan_id="plan-1",
        plan=_plan(),
        paper_id="paper-1",
        search=search,
        top_k_each=3,
        summary_k=4,
    )

    assert calls == [
        ("dataset used experiment", "paper-1", 3),
        ("baseline compared method", "paper-1", 3),
    ]
    assert result.query_errors == []
    assert result.evidence_pool.stats["raw_result_count"] == 2
    assert "Planned retrieval completed." in result.summary_text
    assert result.to_dict()["evidence_pool"]["plan_id"] == "plan-1"


def test_run_planned_retrieval_records_query_errors_and_continues() -> None:
    def search(query: str, paper_id: str, top_k: int) -> list[dict]:
        if "dataset" in query:
            raise RuntimeError("search unavailable")
        return [
            {
                "paper_id": paper_id,
                "chunk_id": "chunk-baseline",
                "chunk_text": "Baseline evidence.",
                "score": 7.0,
            }
        ]

    result = run_planned_retrieval(
        plan_id="plan-1",
        plan=_plan(),
        paper_id="paper-1",
        search=search,
    )

    assert len(result.query_errors) == 1
    assert result.query_errors[0]["query_id"] == "q_dataset"
    assert result.evidence_pool.stats["raw_result_count"] == 1
    assert result.evidence_pool.missing_requirements[0].requirement_id == (
        "req_dataset"
    )


class FakeStructuredVerifierRunnable:
    def __init__(self) -> None:
        self.calls: list[list] = []

    def invoke(self, messages: list) -> dict:
        self.calls.append(messages)
        return {
            "raw": object(),
            "parsed": EvidenceVerificationOutput(decisions=[
                EvidenceVerificationDecisionOutput(
                    requirement_id="req_dataset",
                    evidence_id="ev_1",
                    support="direct",
                    confidence="high",
                    answer_atoms=["dataset"],
                    risks=[],
                    reason="Direct evidence.",
                )
            ]),
            "parsing_error": None,
        }


class FakeVerifierModel:
    def __init__(self) -> None:
        self.runnable = FakeStructuredVerifierRunnable()

    def with_structured_output(self, schema: type, *, include_raw: bool):
        assert schema is EvidenceVerificationOutput
        assert include_raw is True
        return self.runnable


class UnexpectedVerifierModel:
    def with_structured_output(self, schema: type, *, include_raw: bool):
        raise AssertionError("verifier model must not be used when verification is disabled")


def test_run_planned_retrieval_does_not_verify_by_default() -> None:
    verifier = UnexpectedVerifierModel()

    def search(query: str, paper_id: str, top_k: int) -> list[dict]:
        return [
            {
                "paper_id": paper_id,
                "chunk_id": "chunk-1",
                "chunk_text": "Evidence for dataset used.",
                "score": 9.0,
            }
        ]

    result = run_planned_retrieval(
        plan_id="plan-1",
        plan=_plan(),
        paper_id="paper-1",
        search=search,
        verifier_model=verifier,
    )

    payload = result.to_dict()
    assert "verification" not in payload["evidence_pool"]
    assert "verified_summary_items" not in payload["evidence_pool"]


def test_run_planned_retrieval_can_verify_evidence() -> None:
    verifier = FakeVerifierModel()

    def search(query: str, paper_id: str, top_k: int) -> list[dict]:
        return [
            {
                "paper_id": paper_id,
                "chunk_id": "chunk-dataset",
                "chunk_text": "Evidence for dataset used.",
                "score": 9.0,
            }
        ]

    result = run_planned_retrieval(
        plan_id="plan-1",
        plan=_plan(),
        paper_id="paper-1",
        search=search,
        verify_evidence=True,
        verifier_model=verifier,
        verifier_candidate_k=3,
    )

    assert verifier.runnable.calls
    payload = result.to_dict()
    assert payload["evidence_pool"]["verification"]["enabled"] is True
    assert payload["evidence_pool"]["verified_summary_items"]
    assert "Verified top evidence:" in payload["summary_text"]
