from paperpilot.retrieval.query_plan import minimal_fallback_plan


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
