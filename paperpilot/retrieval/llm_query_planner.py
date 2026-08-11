"""LLM-backed JSON query planner."""
from __future__ import annotations

import os
from typing import Any, Literal

from langchain.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, ConfigDict, Field

from paperpilot.retrieval.query_plan import QueryPlan, minimal_fallback_plan
from paperpilot.retrieval.query_plan_validator import validate_query_plan


class RetrievalConstraintsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_numbers: bool = False
    needs_comparison: bool = False
    needs_table_or_figure: bool = False
    polarity: Literal["neutral", "negative", "contrastive"] = "neutral"


class RetrievalRequirementOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    required: bool = True


class RetrievalQueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    role: str
    query: str
    targets: list[str]
    priority: int = 1


class RetrievalExpansionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    neighbor_window: int = 1
    prefer_tables: bool = False
    prefer_captions: bool = False


class RetrievalQueryPlanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["query_plan_v1"]
    question: str
    question_type: Literal[
        "dataset_used",
        "method_list",
        "metric_result",
        "comparison",
        "yes_no",
        "definition",
        "evidence_location",
        "other",
    ]
    answer_shape: Literal[
        "single_entity", "list", "number", "comparison", "yes_no", "freeform"
    ]
    intent_summary: str
    focus_terms: list[str] = Field(default_factory=list)
    constraints: RetrievalConstraintsOutput
    evidence_requirements: list[RetrievalRequirementOutput]
    queries: list[RetrievalQueryOutput]
    avoid: list[str] = Field(default_factory=list)
    expansion_hints: RetrievalExpansionOutput


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
    model: BaseChatModel | None = None,
) -> tuple[QueryPlan, dict[str, Any]]:
    planner_model = model or build_retrieval_model()
    runnable = planner_model.with_structured_output(
        RetrievalQueryPlanOutput,
        include_raw=True,
    )
    try:
        result = runnable.invoke(
            [HumanMessage(content=build_planner_prompt(
                question=question,
                paper_title=paper_title,
                abstract=abstract,
            ))]
        )
    except Exception as exc:  # noqa: BLE001
        return _fallback(question, f"planner_call_failed: {type(exc).__name__}")
    parsing_error = result.get("parsing_error")
    parsed = result.get("parsed")
    if parsing_error is not None or not isinstance(parsed, RetrievalQueryPlanOutput):
        error_name = type(parsing_error).__name__ if parsing_error else "missing_parsed"
        return _fallback(question, f"planner_parse_failed: {error_name}")
    plan = validate_query_plan(parsed.model_dump(mode="json"), question=question)
    return plan, {"fallback_used": False, "fallback_reason": None}


def _fallback(
    question: str,
    reason: str,
) -> tuple[QueryPlan, dict[str, Any]]:
    return minimal_fallback_plan(question), {
        "fallback_used": True,
        "fallback_reason": reason,
    }


def _int_env(name: str, *, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        relation = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be {relation}")
    return value


def build_retrieval_model() -> BaseChatModel:
    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=_int_env(
            "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS",
            default=4096,
            minimum=1,
        ),
        max_retries=_int_env(
            "PAPERPILOT_RESEARCH_MODEL_RETRIES",
            default=1,
            minimum=0,
        ),
    )
