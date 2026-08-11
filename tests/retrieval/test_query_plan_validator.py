from paperpilot.retrieval.query_plan import minimal_fallback_plan
from paperpilot.retrieval.query_plan_validator import (
    parse_query_plan_json,
    validate_query_plan,
)


def valid_plan_dict(*, question: str = "question") -> dict:
    return {
        "version": "query_plan_v1",
        "question": question,
        "question_type": "other",
        "answer_shape": "freeform",
        "intent_summary": "Find direct evidence.",
        "focus_terms": [],
        "constraints": {
            "needs_numbers": False,
            "needs_comparison": False,
            "needs_table_or_figure": False,
            "polarity": "neutral",
        },
        "evidence_requirements": [
            {
                "id": "req_1",
                "description": "direct evidence",
                "required": True,
            }
        ],
        "queries": [
            {
                "id": "q_1",
                "role": "focused_rewrite",
                "query": "direct evidence",
                "targets": ["req_1"],
                "priority": 2,
            }
        ],
        "avoid": [],
        "expansion_hints": {
            "neighbor_window": 1,
            "prefer_tables": False,
            "prefer_captions": False,
        },
    }


def test_minimal_fallback_plan_has_literal_query_and_requirement() -> None:
    plan = minimal_fallback_plan("What dataset was used?")

    assert plan.version == "query_plan_v1"
    assert plan.question == "What dataset was used?"
    assert plan.question_type == "other"
    assert plan.answer_shape == "freeform"
    assert len(plan.evidence_requirements) == 1
    assert plan.evidence_requirements[0].id == "req_direct"
    assert plan.evidence_requirements[0].required is True
    assert len(plan.queries) == 1
    assert plan.queries[0].id == "q_lit"
    assert plan.queries[0].role == "literal"
    assert plan.queries[0].query == "What dataset was used?"
    assert plan.queries[0].targets == ["req_direct"]


def test_query_plan_to_dict_is_json_serializable() -> None:
    plan = minimal_fallback_plan("Which baseline is compared?")
    data = plan.to_dict()

    assert data["version"] == "query_plan_v1"
    assert data["constraints"]["polarity"] == "neutral"
    assert data["expansion_hints"]["neighbor_window"] == 1
    assert data["queries"][0]["priority"] == 1


def test_validate_query_plan_normalizes_valid_data() -> None:
    data = {
        "version": "query_plan_v1",
        "question": "What dataset was used?",
        "question_type": "dataset_used",
        "answer_shape": "single_entity",
        "intent_summary": "Find the dataset used in the experiments.",
        "focus_terms": ["dataset"],
        "constraints": {
            "needs_numbers": False,
            "needs_comparison": False,
            "needs_table_or_figure": False,
            "polarity": "neutral",
        },
        "evidence_requirements": [
            {
                "id": "req_dataset",
                "description": "dataset used by the paper",
                "required": True,
            }
        ],
        "queries": [
            {
                "id": "q_1",
                "role": "focused_rewrite",
                "query": "dataset used experiment corpus",
                "targets": ["req_dataset"],
                "priority": 2,
            }
        ],
        "avoid": ["related work datasets"],
        "expansion_hints": {
            "neighbor_window": 1,
            "prefer_tables": False,
            "prefer_captions": False,
        },
    }

    plan = validate_query_plan(data, question="What dataset was used?")

    assert plan.question_type == "dataset_used"
    assert plan.evidence_requirements[0].id == "req_dataset"
    assert plan.queries[0].id == "q_lit"
    assert plan.queries[0].role == "literal"
    assert plan.queries[1].query == "dataset used experiment corpus"
    assert plan.avoid == ["related work datasets"]


def test_validate_query_plan_caps_queries_and_removes_bad_targets() -> None:
    data = {
        "version": "query_plan_v1",
        "question": "Q?",
        "evidence_requirements": [
            {"id": "req_1", "description": "direct evidence", "required": True}
        ],
        "queries": [
            {
                "id": f"q_{i}",
                "role": "focused_rewrite",
                "query": f"query {i}",
                "targets": ["missing" if i == 2 else "req_1"],
                "priority": i,
            }
            for i in range(10)
        ],
    }

    plan = validate_query_plan(data, question="Q?")

    assert len(plan.queries) == 6
    assert all(query.targets == ["req_1"] for query in plan.queries)
    assert plan.queries[0].id == "q_lit"


def test_validate_query_plan_caps_evidence_requirements_at_six() -> None:
    data = valid_plan_dict()
    data["evidence_requirements"] = [
        {"id": f"req_{index}", "description": f"Requirement {index}"}
        for index in range(8)
    ]
    data["queries"] = [
        {"id": "q_1", "query": "evidence", "targets": ["req_1"]}
    ]

    plan = validate_query_plan(data, question="question")

    assert [item.id for item in plan.evidence_requirements] == [
        "req_0", "req_1", "req_2", "req_3", "req_4", "req_5"
    ]


def test_validate_query_plan_uses_first_retained_requirement_as_fallback_target() -> None:
    data = valid_plan_dict()
    data["evidence_requirements"] = [
        {"id": "req_0", "description": "First requirement"},
        {"id": "req_5", "description": "Second requirement"},
    ]
    data["queries"] = [
        {"id": "q_1", "query": "evidence", "targets": ["unknown"]}
    ]

    plan = validate_query_plan(data, question="question")

    assert plan.queries[1].targets == ["req_0"]


def test_parse_query_plan_json_falls_back_on_invalid_json() -> None:
    plan, meta = parse_query_plan_json("not json", question="What is used?")

    assert plan.queries[0].query == "What is used?"
    assert meta["fallback_used"] is True
    assert "invalid_json" in meta["fallback_reason"]
