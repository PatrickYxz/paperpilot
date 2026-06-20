from paperpilot.eval.query_planner import (
    classify_question_type,
    infer_answer_shape,
    plan_queries,
)


def _queries(plan):
    return [item.query.lower() for item in plan.queries]


def test_plan_negative_exception_preserves_negative_intent() -> None:
    plan = plan_queries(
        "Which tasks does CamemBERT not improve on?",
        title="CamemBERT: a Tasty French Language Model",
    )

    assert plan.question_type == "negative_or_exception"
    assert plan.answer_shape == "list"
    assert plan.constraints.polarity == "negative"
    assert plan.constraints.needs_comparison is True
    assert "CamemBERT" in plan.focus_terms
    assert any("does not improve" in query or "except" in query for query in _queries(plan))
    assert any("general improvement" in item for item in plan.avoid)
    assert len(plan.queries) >= 4


def test_plan_dataset_question_targets_experiment_data() -> None:
    plan = plan_queries("What dataset was used in the experiment?")

    assert plan.question_type == "dataset_used"
    assert plan.answer_shape == "single_entity"
    assert plan.expansion_hints.neighbor_window == 2
    assert any("dataset used in experiment" in query for query in _queries(plan))
    assert any("related work datasets" in item for item in plan.avoid)


def test_plan_metric_question_prefers_tables_and_numbers() -> None:
    plan = plan_queries("How much better was CamemBERT than previous results on these tasks?")

    assert plan.question_type == "metric_or_result"
    assert plan.answer_shape == "number"
    assert plan.constraints.needs_numbers is True
    assert plan.constraints.needs_table is True
    assert plan.expansion_hints.prefer_tables is True
    assert any("table" in query for query in _queries(plan))
    assert any("score" in query or "accuracy" in query or "f1" in query for query in _queries(plan))


def test_plan_method_question_targets_baseline_lists() -> None:
    plan = plan_queries("Which popular clustering methods did they experiment with?")

    assert plan.question_type == "method_or_baseline_list"
    assert plan.answer_shape == "list"
    assert any("methods baselines approaches compared" in query for query in _queries(plan))
    assert any("complete list" in item for item in plan.must_find)


def test_plan_evaluation_protocol_question_targets_criteria() -> None:
    plan = plan_queries("How they perform manual evaluation, what is criteria?")

    assert plan.question_type == "evaluation_protocol"
    assert any("manual evaluation annotators scale criteria" in query for query in _queries(plan))
    assert any("who evaluated" in item for item in plan.must_find)


def test_plan_schema_is_json_serializable_shape() -> None:
    plan = plan_queries("What are the advantages of the proposed model?")
    data = plan.to_dict()

    assert data["question_type"] == "advantage_or_contribution"
    assert isinstance(data["constraints"], dict)
    assert isinstance(data["queries"], list)
    assert all(set(item) == {"role", "query"} for item in data["queries"])
    assert "oracle_spans" not in data
    assert "highlighted_evidence" not in data


def test_classification_and_shape_are_deterministic() -> None:
    question = "What are the state of the art approaches?"

    assert classify_question_type(question) == "method_or_baseline_list"
    assert infer_answer_shape(question) == "list"
    assert plan_queries(question).to_dict() == plan_queries(question).to_dict()
