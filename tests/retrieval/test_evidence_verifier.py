from types import SimpleNamespace

from paperpilot.retrieval.evidence_pool import EvidenceItem, EvidencePool, MatchedQuery
from paperpilot.retrieval.evidence_verifier import (
    EvidenceVerificationDecision,
    build_verifier_prompt,
    candidate_items_for_requirement,
    parse_verifier_output,
    run_evidence_verification,
    select_verified_summary,
)
from paperpilot.retrieval.query_plan import (
    EvidenceRequirement,
    PlannedQuery,
    QueryConstraints,
    QueryPlan,
)


def test_parse_verifier_output_normalizes_valid_json() -> None:
    text = """
    {
      "decisions": [
        {
          "requirement_id": "req_dataset",
          "evidence_id": "ev_2",
          "support": "direct",
          "confidence": "high",
          "answer_atoms": ["WikiHop"],
          "risks": ["dataset_role_clear"],
          "reason": "The chunk states the dataset used."
        }
      ]
    }
    """

    decisions, error = parse_verifier_output(text)

    assert error is None
    assert decisions == [
        EvidenceVerificationDecision(
            requirement_id="req_dataset",
            evidence_id="ev_2",
            support="direct",
            confidence="high",
            answer_atoms=["WikiHop"],
            risks=["dataset_role_clear"],
            reason="The chunk states the dataset used.",
            score=100.0,
        )
    ]


def test_parse_verifier_output_extracts_fenced_json() -> None:
    text = """```json
    {
      "decisions": [
        {
          "requirement_id": "req_num",
          "evidence_id": "ev_1",
          "support": "partial",
          "confidence": "medium",
          "answer_atoms": ["58%"],
          "risks": [],
          "reason": "The number appears, but the metric is unclear."
        }
      ]
    }
    ```"""

    decisions, error = parse_verifier_output(text)

    assert error is None
    assert len(decisions) == 1
    assert decisions[0].support == "partial"
    assert decisions[0].confidence == "medium"
    assert decisions[0].score == 45.0


def test_parse_verifier_output_reports_invalid_json() -> None:
    decisions, error = parse_verifier_output("not json")

    assert decisions == []
    assert error == "verifier did not return valid JSON"


def test_parse_verifier_output_drops_invalid_decisions() -> None:
    text = """
    {
      "decisions": [
        {
          "requirement_id": "req_1",
          "evidence_id": "ev_1",
          "support": "maybe",
          "confidence": "high",
          "answer_atoms": [],
          "risks": [],
          "reason": "invalid"
        },
        {
          "requirement_id": "req_1",
          "evidence_id": "ev_2",
          "support": "no",
          "confidence": "low",
          "answer_atoms": [],
          "risks": [],
          "reason": "related only"
        }
      ]
    }
    """

    decisions, error = parse_verifier_output(text)

    assert error is None
    assert [item.evidence_id for item in decisions] == ["ev_2"]
    assert decisions[0].score == 0.0


def _plan() -> QueryPlan:
    return QueryPlan(
        version="query_plan_v1",
        question="What dataset was used?",
        question_type="dataset_used",
        answer_shape="entity",
        intent_summary="Find the dataset used by the paper.",
        focus_terms=["dataset"],
        constraints=QueryConstraints(),
        evidence_requirements=[
            EvidenceRequirement("req_dataset", "dataset used by the paper", True),
            EvidenceRequirement("req_metric", "reported metric", True),
        ],
        queries=[
            PlannedQuery(
                "q_dataset",
                "focused_rewrite",
                "dataset used",
                ["req_dataset"],
                1,
            ),
            PlannedQuery(
                "q_metric",
                "focused_rewrite",
                "reported metric",
                ["req_metric"],
                2,
            ),
        ],
    )


def _list_plan() -> QueryPlan:
    return QueryPlan(
        version="query_plan_v1",
        question="Which methods did they compare with?",
        question_type="method_list",
        answer_shape="list",
        intent_summary="Find compared methods.",
        focus_terms=["methods"],
        constraints=QueryConstraints(),
        evidence_requirements=[
            EvidenceRequirement("req_methods", "methods compared by the paper", True),
        ],
        queries=[
            PlannedQuery(
                "q_methods",
                "focused_rewrite",
                "methods compared",
                ["req_methods"],
                1,
            ),
        ],
    )


def _item(
    item_id: str,
    text: str,
    score: float,
    targets: list[str],
) -> EvidenceItem:
    return EvidenceItem(
        id=item_id,
        paper_id="paper-1",
        chunk_id=item_id,
        chunk_text=text,
        best_score=score,
        matched_queries=[
            MatchedQuery(
                query_id=f"q_{item_id}",
                query="dataset used",
                role="focused_rewrite",
                rank=1,
                score=score,
                targets=targets,
            )
        ],
    )


def _pool(items: list[EvidenceItem]) -> EvidencePool:
    return EvidencePool(
        plan_id="plan-1",
        question="What dataset was used?",
        query_plan=_plan().to_dict(),
        items=items,
        summary_items=items[:2],
        missing_requirements=[],
        stats={
            "query_count": 2,
            "raw_result_count": len(items),
            "deduped_count": len(items),
        },
    )


def test_candidate_items_for_requirement_uses_targets_and_score_cap() -> None:
    items = [
        _item("ev_1", "low score target", 0.1, ["req_dataset"]),
        _item("ev_2", "high score target", 0.9, ["req_dataset"]),
        _item("ev_3", "other requirement", 1.0, ["req_metric"]),
    ]

    pool = _pool(items)
    pool.summary_items = []

    candidates = candidate_items_for_requirement(
        pool,
        "req_dataset",
        candidate_k=1,
    )

    assert [item.id for item in candidates] == ["ev_2"]


def test_candidate_items_include_summary_items_beyond_score_cap() -> None:
    items = [
        _item("ev_1", "generic baseline mention", 10.0, ["req_dataset"]),
        _item("ev_2", "generic baseline mention", 9.0, ["req_dataset"]),
        _item("ev_3", "generic baseline mention", 8.0, ["req_dataset"]),
        _item("ev_4", "specific baselines are QANet and BERT-Base", 1.0, ["req_dataset"]),
    ]
    pool = _pool(items)
    pool.summary_items = [items[0], items[3]]

    candidates = candidate_items_for_requirement(
        pool,
        "req_dataset",
        candidate_k=2,
    )

    assert [item.id for item in candidates] == ["ev_1", "ev_2", "ev_4"]


def test_build_verifier_prompt_contains_requirement_and_candidate_text() -> None:
    plan = _plan()
    requirement = plan.evidence_requirements[0]
    item = _item("ev_1", "The experiments use WikiHop.", 0.9, ["req_dataset"])

    prompt = build_verifier_prompt(
        plan=plan,
        requirement=requirement,
        candidates=[item],
    )

    assert "Return JSON only" in prompt
    assert "dataset used by the paper" in prompt
    assert "ev_1" in prompt
    assert "The experiments use WikiHop." in prompt
    assert "direct|partial|no" in prompt


def test_select_verified_summary_groups_by_requirement() -> None:
    plan = _plan()
    items = [
        _item("ev_1", "WikiHop is related work.", 0.99, ["req_dataset"]),
        _item("ev_2", "The experiments use WikiHop.", 0.50, ["req_dataset"]),
        _item("ev_3", "The result is 58%.", 0.80, ["req_metric"]),
    ]
    decisions, _ = parse_verifier_output("""
    {
      "decisions": [
        {
          "requirement_id": "req_dataset",
          "evidence_id": "ev_1",
          "support": "no",
          "confidence": "high",
          "answer_atoms": ["WikiHop"],
          "risks": ["related_work"],
          "reason": "Related work only."
        },
        {
          "requirement_id": "req_dataset",
          "evidence_id": "ev_2",
          "support": "direct",
          "confidence": "medium",
          "answer_atoms": ["WikiHop"],
          "risks": [],
          "reason": "Direct dataset relation."
        },
        {
          "requirement_id": "req_metric",
          "evidence_id": "ev_3",
          "support": "partial",
          "confidence": "high",
          "answer_atoms": ["58%"],
          "risks": ["metric_unclear"],
          "reason": "Value appears but metric is unclear."
        }
      ]
    }
    """)

    result = select_verified_summary(
        plan=plan,
        pool=_pool(items),
        decisions=decisions,
        summary_k=4,
    )

    assert result.verified_summary_items == ["ev_2", "ev_3"]
    assert result.stats == {
        "decision_count": 3,
        "direct_count": 1,
        "partial_count": 1,
        "no_count": 1,
    }


def test_select_verified_summary_marks_missing_requirement() -> None:
    plan = _plan()
    item = _item(
        "ev_1",
        "Only related work mentions WikiHop.",
        0.9,
        ["req_dataset"],
    )
    decisions, _ = parse_verifier_output("""
    {
      "decisions": [
        {
          "requirement_id": "req_dataset",
          "evidence_id": "ev_1",
          "support": "no",
          "confidence": "high",
          "answer_atoms": ["WikiHop"],
          "risks": ["related_work"],
          "reason": "Related only."
        }
      ]
    }
    """)

    result = select_verified_summary(
        plan=plan,
        pool=_pool([item]),
        decisions=decisions,
        summary_k=4,
    )

    assert result.verified_summary_items == []
    assert result.missing_verified_requirements == [
        {
            "requirement_id": "req_dataset",
            "description": "dataset used by the paper",
            "reason": "no_verified_direct_or_partial_evidence",
        },
        {
            "requirement_id": "req_metric",
            "description": "reported metric",
            "reason": "no_verifier_decisions",
        },
    ]


def test_select_verified_summary_records_conflicting_direct_atoms() -> None:
    plan = _plan()
    items = [
        _item("ev_1", "The paper uses HotpotQA.", 0.9, ["req_dataset"]),
        _item("ev_2", "The paper uses WikiHop.", 0.8, ["req_dataset"]),
    ]
    decisions, _ = parse_verifier_output("""
    {
      "decisions": [
        {
          "requirement_id": "req_dataset",
          "evidence_id": "ev_1",
          "support": "direct",
          "confidence": "high",
          "answer_atoms": ["HotpotQA"],
          "risks": [],
          "reason": "Direct."
        },
        {
          "requirement_id": "req_dataset",
          "evidence_id": "ev_2",
          "support": "direct",
          "confidence": "high",
          "answer_atoms": ["WikiHop"],
          "risks": [],
          "reason": "Direct."
        }
      ]
    }
    """)

    result = select_verified_summary(
        plan=plan,
        pool=_pool(items),
        decisions=decisions,
        summary_k=4,
    )

    assert result.verified_summary_items[:2] == ["ev_1", "ev_2"]
    assert result.conflicts == [
        {
            "requirement_id": "req_dataset",
            "evidence_ids": ["ev_1", "ev_2"],
            "answer_atoms": ["HotpotQA", "WikiHop"],
            "reason": "multiple_high_confidence_direct_answer_atoms",
        }
    ]


def test_select_verified_summary_does_not_conflict_on_single_list_evidence() -> None:
    plan = _plan()
    items = [
        _item(
            "ev_1",
            "The baselines are QANet and BERT-Base.",
            0.9,
            ["req_dataset"],
        ),
    ]
    decisions, _ = parse_verifier_output("""
    {
      "decisions": [
        {
          "requirement_id": "req_dataset",
          "evidence_id": "ev_1",
          "support": "direct",
          "confidence": "high",
          "answer_atoms": ["QANet", "BERT-Base"],
          "risks": [],
          "reason": "The chunk directly lists both baselines."
        }
      ]
    }
    """)

    result = select_verified_summary(
        plan=plan,
        pool=_pool(items),
        decisions=decisions,
        summary_k=4,
    )

    assert result.verified_summary_items == ["ev_1"]
    assert result.conflicts == []


def test_select_verified_summary_treats_multiple_list_directs_as_coverage() -> None:
    plan = _list_plan()
    items = [
        _item("ev_1", "Compared methods include K-means and AE.", 0.9, ["req_methods"]),
        _item("ev_2", "Other baselines include LSA and BOW.", 0.8, ["req_methods"]),
    ]
    decisions, _ = parse_verifier_output("""
    {
      "decisions": [
        {
          "requirement_id": "req_methods",
          "evidence_id": "ev_1",
          "support": "direct",
          "confidence": "high",
          "answer_atoms": ["K-means", "AE"],
          "risks": [],
          "reason": "The chunk directly lists compared methods."
        },
        {
          "requirement_id": "req_methods",
          "evidence_id": "ev_2",
          "support": "direct",
          "confidence": "high",
          "answer_atoms": ["LSA", "BOW"],
          "risks": [],
          "reason": "The chunk directly lists additional compared methods."
        }
      ]
    }
    """)

    result = select_verified_summary(
        plan=plan,
        pool=_pool(items),
        decisions=decisions,
        summary_k=4,
    )

    assert result.verified_summary_items == ["ev_1", "ev_2"]
    assert result.conflicts == []


class FakeVerifierClient:
    def __init__(
        self,
        responses: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responses = responses or []
        self.error = error
        self.calls: list[dict] = []

    def call(self, messages: list[dict], tools: list, *, system: str):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.responses.pop(0))


def test_run_evidence_verification_calls_once_per_required_requirement() -> None:
    plan = _plan()
    pool = _pool([
        _item("ev_1", "The experiments use WikiHop.", 0.9, ["req_dataset"]),
        _item("ev_2", "The reported accuracy is 58%.", 0.8, ["req_metric"]),
    ])
    client = FakeVerifierClient([
        """
        {"decisions": [{
          "requirement_id": "req_dataset",
          "evidence_id": "ev_1",
          "support": "direct",
          "confidence": "high",
          "answer_atoms": ["WikiHop"],
          "risks": [],
          "reason": "Direct."
        }]}
        """,
        """
        {"decisions": [{
          "requirement_id": "req_metric",
          "evidence_id": "ev_2",
          "support": "partial",
          "confidence": "medium",
          "answer_atoms": ["58%"],
          "risks": ["metric_unclear"],
          "reason": "Metric unclear."
        }]}
        """,
    ])

    result = run_evidence_verification(
        plan=plan,
        pool=pool,
        client=client,
        summary_k=4,
        verifier_candidate_k=6,
    )

    assert len(client.calls) == 2
    assert all(call["tools"] == [] for call in client.calls)
    assert result.verified_summary_items == ["ev_1", "ev_2"]
    assert result.verification_error is None


def test_run_evidence_verification_returns_error_result_on_client_failure() -> None:
    plan = _plan()
    pool = _pool([
        _item("ev_1", "The experiments use WikiHop.", 0.9, ["req_dataset"])
    ])
    client = FakeVerifierClient(error=RuntimeError("verifier unavailable"))

    result = run_evidence_verification(
        plan=plan,
        pool=pool,
        client=client,
        summary_k=4,
        verifier_candidate_k=6,
    )

    assert result.enabled is True
    assert result.verified_summary_items == []
    assert result.decisions == []
    assert result.verification_error == "RuntimeError: verifier unavailable"
