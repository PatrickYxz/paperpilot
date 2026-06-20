"""LLM-backed JSON query planner."""
from __future__ import annotations

from typing import Any, Protocol

from paperpilot.core.adapter import LLMClient, Tool
from paperpilot.retrieval.query_plan import QueryPlan
from paperpilot.retrieval.query_plan_validator import parse_query_plan_json


class PlannerClient(Protocol):
    def call(self, messages: list[dict], tools: list[Tool], *, system: str) -> Any:
        ...


def build_planner_prompt(
    *,
    question: str,
    paper_title: str = "",
    abstract: str = "",
) -> str:
    return f"""You are PaperPilot's retrieval query planner.

Return only one JSON object. Do not wrap it in markdown.

Schema:
{{
  "version": "query_plan_v1",
  "question": "...",
  "question_type": "dataset_used | method_list | metric_result | comparison | yes_no | definition | evidence_location | other",
  "answer_shape": "single_entity | list | number | comparison | yes_no | freeform",
  "intent_summary": "...",
  "focus_terms": ["..."],
  "constraints": {{
    "needs_numbers": false,
    "needs_comparison": false,
    "needs_table_or_figure": false,
    "polarity": "neutral | negative | contrastive"
  }},
  "evidence_requirements": [
    {{"id": "req_1", "description": "what evidence must directly show", "required": true}}
  ],
  "queries": [
    {{"id": "q_1", "role": "focused_rewrite", "query": "search query", "targets": ["req_1"], "priority": 2}}
  ],
  "avoid": ["false positive traps"],
  "expansion_hints": {{
    "neighbor_window": 1,
    "prefer_tables": false,
    "prefer_captions": false
  }}
}}

Planning rules:
- Include evidence requirements that directly support the answer.
- Include 3 to 6 concise executable search queries.
- Preserve specific dataset, method, metric, task, and baseline names.
- Add avoid rules for likely false positives such as related-work mentions.
- Do not include gold answers, oracle spans, or external knowledge.

Paper title: {paper_title or "(unknown)"}

Abstract:
{abstract or "(not available)"}

Question:
{question}
"""


def plan_with_llm(
    *,
    question: str,
    paper_title: str = "",
    abstract: str = "",
    client: PlannerClient | None = None,
) -> tuple[QueryPlan, dict[str, Any]]:
    prompt = build_planner_prompt(
        question=question,
        paper_title=paper_title,
        abstract=abstract,
    )
    try:
        planner_client = client or LLMClient()
        response = planner_client.call(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            system="",
        )
        raw_text = response.text or ""
    except Exception as exc:  # noqa: BLE001
        plan, meta = parse_query_plan_json("", question=question)
        meta["fallback_reason"] = (
            f"planner_call_failed: {type(exc).__name__}: {exc}"
        )
        return plan, meta
    plan, meta = parse_query_plan_json(raw_text, question=question)
    meta["raw_response"] = raw_text
    return plan, meta
