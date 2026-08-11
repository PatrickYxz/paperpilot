import pytest

from paperpilot.retrieval.llm_query_planner import (
    RetrievalQueryPlanOutput,
    build_planner_prompt,
    build_retrieval_model,
    plan_with_llm,
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


class FakeStructuredRunnable:
    def __init__(self, result: dict | Exception) -> None:
        self.result = result
        self.calls: list[list] = []

    def invoke(self, messages: list) -> dict:
        self.calls.append(messages)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeStructuredModel:
    def __init__(self, result: dict | Exception) -> None:
        self.runnable = FakeStructuredRunnable(result)
        self.schemas: list[tuple[type, bool]] = []

    def with_structured_output(self, schema: type, *, include_raw: bool):
        self.schemas.append((schema, include_raw))
        return self.runnable


def test_build_planner_prompt_includes_question_and_metadata() -> None:
    prompt = build_planner_prompt(
        question="What dataset was used?",
        paper_title="A Dataset Paper",
        abstract="We evaluate on WikiHop.",
    )

    assert "What dataset was used?" in prompt
    assert "A Dataset Paper" in prompt
    assert "We evaluate on WikiHop." in prompt
    assert "query_plan_v1" in prompt
    assert "evidence_requirements" in prompt


def test_plan_with_llm_invokes_structured_model_once() -> None:
    model = FakeStructuredModel({
        "raw": object(),
        "parsed": RetrievalQueryPlanOutput.model_validate(valid_plan_dict()),
        "parsing_error": None,
    })

    plan, meta = plan_with_llm(
        question="question",
        model=model,
    )

    assert model.schemas == [(RetrievalQueryPlanOutput, True)]
    assert len(model.runnable.calls) == 1
    assert meta["fallback_used"] is False
    assert plan.queries[0].role == "literal"
    assert plan.queries[1].query == "direct evidence"


def test_plan_with_llm_falls_back_on_structured_parse_error() -> None:
    model = FakeStructuredModel({
        "raw": object(),
        "parsed": None,
        "parsing_error": ValueError("invalid output"),
    })

    plan, meta = plan_with_llm(question="Q?", model=model)

    assert meta["fallback_used"] is True
    assert meta["fallback_reason"] == "planner_parse_failed: ValueError"
    assert plan.queries[0].query == "Q?"


def test_plan_with_llm_falls_back_on_provider_failure_without_retry() -> None:
    model = FakeStructuredModel(RuntimeError("unavailable"))

    plan, meta = plan_with_llm(question="Q?", model=model)

    assert len(model.runnable.calls) == 1
    assert meta == {
        "fallback_used": True,
        "fallback_reason": "planner_call_failed: RuntimeError",
    }
    assert plan.queries[0].query == "Q?"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS",
            "not-an-int",
            "invalid literal for int()",
        ),
        (
            "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS",
            "0",
            "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS must be positive",
        ),
        (
            "PAPERPILOT_RESEARCH_MODEL_RETRIES",
            "-1",
            "PAPERPILOT_RESEARCH_MODEL_RETRIES must be nonnegative",
        ),
    ],
)
def test_build_retrieval_model_rejects_invalid_integer_configuration(
    monkeypatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        build_retrieval_model()
