from dataclasses import dataclass

from paperpilot.core.adapter import ParsedResponse
from paperpilot.retrieval.llm_query_planner import build_planner_prompt, plan_with_llm


@dataclass
class FakeClient:
    text: str

    def call(self, messages, tools, *, system):
        return ParsedResponse(
            text=self.text,
            tool_calls=[],
            usage={},
            raw=None,
        )


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


def test_plan_with_llm_returns_validated_plan() -> None:
    client = FakeClient(
        text=(
            '{"version":"query_plan_v1","question":"Q?",'
            '"evidence_requirements":[{"id":"req_1",'
            '"description":"direct evidence","required":true}],'
            '"queries":[{"id":"q_1","role":"focused_rewrite",'
            '"query":"direct evidence","targets":["req_1"],"priority":2}]}'
        )
    )

    plan, meta = plan_with_llm(
        question="Q?",
        paper_title="T",
        abstract="A",
        client=client,
    )

    assert meta["fallback_used"] is False
    assert plan.queries[0].role == "literal"
    assert plan.queries[1].query == "direct evidence"


def test_plan_with_llm_falls_back_on_invalid_response() -> None:
    client = FakeClient(text="not json")

    plan, meta = plan_with_llm(
        question="Q?",
        paper_title="T",
        abstract="A",
        client=client,
    )

    assert meta["fallback_used"] is True
    assert plan.queries[0].query == "Q?"
