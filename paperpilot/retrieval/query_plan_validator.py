"""Validation and parsing for LLM query plan JSON."""
from __future__ import annotations

import json
from typing import Any

from paperpilot.retrieval.query_plan import (
    PLAN_VERSION,
    EvidenceRequirement,
    ExpansionHints,
    PlannedQuery,
    QueryConstraints,
    QueryPlan,
    minimal_fallback_plan,
)

MAX_QUERIES = 6
MAX_QUERY_CHARS = 240


def parse_query_plan_json(
    text: str, *, question: str
) -> tuple[QueryPlan, dict[str, Any]]:
    try:
        data = json.loads(_extract_json_object(text))
    except Exception as exc:  # noqa: BLE001
        return minimal_fallback_plan(question), {
            "fallback_used": True,
            "fallback_reason": f"invalid_json: {type(exc).__name__}",
        }
    try:
        return validate_query_plan(data, question=question), {
            "fallback_used": False,
            "fallback_reason": None,
        }
    except Exception as exc:  # noqa: BLE001
        return minimal_fallback_plan(question), {
            "fallback_used": True,
            "fallback_reason": f"invalid_plan: {type(exc).__name__}: {exc}",
        }


def validate_query_plan(data: dict[str, Any], *, question: str) -> QueryPlan:
    if not isinstance(data, dict):
        raise ValueError("query plan must be a JSON object")
    if data.get("version") != PLAN_VERSION:
        raise ValueError(f"query plan version must be {PLAN_VERSION}")

    requirements = _requirements(data.get("evidence_requirements"))
    if not requirements:
        raise ValueError("query plan must include at least one evidence requirement")
    requirement_ids = {req.id for req in requirements}

    queries = _queries(data.get("queries"), requirement_ids=requirement_ids)
    literal = PlannedQuery(
        id="q_lit",
        role="literal",
        query=question,
        targets=[requirements[0].id],
        priority=1,
    )
    queries = [
        q
        for q in queries
        if q.role != "literal" and q.query.strip() != question.strip()
    ]
    queries = [literal, *queries][:MAX_QUERIES]
    if not queries:
        raise ValueError("query plan must include at least one valid query")

    constraints_data = (
        data.get("constraints") if isinstance(data.get("constraints"), dict) else {}
    )
    hints_data = (
        data.get("expansion_hints")
        if isinstance(data.get("expansion_hints"), dict)
        else {}
    )
    return QueryPlan(
        version=PLAN_VERSION,
        question=str(data.get("question") or question),
        question_type=str(data.get("question_type") or "other"),
        answer_shape=str(data.get("answer_shape") or "freeform"),
        intent_summary=str(
            data.get("intent_summary")
            or "Find direct evidence that answers the question."
        ),
        focus_terms=[
            str(item)
            for item in data.get("focus_terms") or []
            if str(item).strip()
        ],
        constraints=QueryConstraints(
            needs_numbers=bool(constraints_data.get("needs_numbers", False)),
            needs_comparison=bool(constraints_data.get("needs_comparison", False)),
            needs_table_or_figure=bool(constraints_data.get("needs_table_or_figure", False)),
            polarity=str(constraints_data.get("polarity") or "neutral"),
        ),
        evidence_requirements=requirements,
        queries=queries,
        avoid=[str(item) for item in data.get("avoid") or [] if str(item).strip()],
        expansion_hints=ExpansionHints(
            neighbor_window=int(hints_data.get("neighbor_window", 1) or 1),
            prefer_tables=bool(hints_data.get("prefer_tables", False)),
            prefer_captions=bool(hints_data.get("prefer_captions", False)),
        ),
    )


def _requirements(value: Any) -> list[EvidenceRequirement]:
    out: list[EvidenceRequirement] = []
    seen: set[str] = set()
    if not isinstance(value, list):
        return out
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        req_id = str(item.get("id") or f"req_{index}").strip()
        description = str(item.get("description") or "").strip()
        if not req_id or not description or req_id in seen:
            continue
        seen.add(req_id)
        out.append(
            EvidenceRequirement(
                id=req_id,
                description=description,
                required=bool(item.get("required", True)),
            )
        )
    return out


def _queries(value: Any, *, requirement_ids: set[str]) -> list[PlannedQuery]:
    out: list[PlannedQuery] = []
    seen: set[str] = set()
    if not isinstance(value, list):
        return out
    fallback_target = next(iter(requirement_ids))
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        query = " ".join(str(item.get("query") or "").split())
        if not query:
            continue
        query = query[:MAX_QUERY_CHARS]
        query_id = str(item.get("id") or f"q_{index}").strip()
        if not query_id or query_id in seen:
            query_id = f"q_{index}"
        seen.add(query_id)
        targets = [
            str(target)
            for target in item.get("targets") or []
            if str(target) in requirement_ids
        ]
        if not targets:
            targets = [fallback_target]
        out.append(
            PlannedQuery(
                id=query_id,
                role=str(item.get("role") or "focused_rewrite"),
                query=query,
                targets=targets,
                priority=int(item.get("priority", index) or index),
            )
        )
    return sorted(out, key=lambda query: query.priority)


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found")
    return stripped[start : end + 1]
