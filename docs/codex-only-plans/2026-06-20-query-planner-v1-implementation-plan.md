# Query Planner v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Query Planner v1 as a real PaperPilot deep-read retrieval pre-stage with LLM JSON planning, validation, forced planned query execution, and structured evidence pooling.

**Architecture:** Add a small `paperpilot.retrieval` planning layer that can be tested without starting MCP, then expose it through a new ColBERT MCP tool named `planned_retrieval`. The first product-path integration updates `deep-read-paper` so the agent normally calls `planned_retrieval` after `build_index`, while eval traces can record the structured plan and evidence pool.

**Tech Stack:** Python standard library, dataclasses, existing `LLMClient`, existing ColBERT `IndexManager`, existing MCP FastMCP server, pytest.

---

## Scope

This plan implements the approved v1 design in `docs/superpowers/specs/2026-06-20-query-planner-v1-design.md`.

In scope:

- LLM JSON planner module with deterministic fallback.
- Plan validation and normalization.
- Evidence pool exact dedupe and requirement-aware summary selection.
- New ColBERT MCP tool `planned_retrieval`.
- `deep-read-paper` skill update.
- Unit tests and focused MCP protocol tests.

Out of scope:

- reranker;
- semantic dedupe;
- table/caption special retrieval;
- chunking/index refactor;
- loop-level forced planner trigger;
- new dependencies.

## File Structure

- Create `paperpilot/retrieval/query_plan.py`
  - Dataclasses for `QueryPlan`, `EvidenceRequirement`, `PlannedQuery`, `QueryConstraints`, and `ExpansionHints`.
  - JSON serialization helpers.
  - `minimal_fallback_plan(question)`.

- Create `paperpilot/retrieval/query_plan_validator.py`
  - `validate_query_plan(data, question)` and `parse_query_plan_json(text, question)`.
  - Query cap, target validation, fallback behavior.

- Create `paperpilot/retrieval/llm_query_planner.py`
  - `build_planner_prompt(...)`.
  - `plan_with_llm(...)`.
  - JSON extraction and fallback.

- Create `paperpilot/retrieval/evidence_pool.py`
  - `EvidencePool`, `EvidenceItem`, `MatchedQuery`, `MissingRequirement`.
  - Exact dedupe by `paper_id + normalized_text_hash`.
  - Requirement-aware summary selection.
  - `format_evidence_summary(pool)`.

- Create `paperpilot/retrieval/planned_retrieval.py`
  - `run_planned_retrieval(...)` orchestration.
  - Accepts a search callable so tests can avoid real ColBERT.

- Modify `paperpilot/retrieval/__init__.py`
  - Export stable retrieval planning APIs.

- Modify `paperpilot/mcp_servers/colbert/server.py`
  - Add MCP tool `planned_retrieval`.
  - Route to the new orchestration layer.

- Modify `paperpilot/mcp_servers/colbert/index_manager.py`
  - Recommended small improvement: return `chunk_id` in `search` results using existing `paper_id::chunk_i` result id.

- Modify `paperpilot/skills/deep-read-paper.md`
  - Update workflow to call `mcp__colbert__planned_retrieval` after build_index.
  - Keep normal `mcp__colbert__search` as optional follow-up.

- Test `tests/retrieval/test_query_plan_validator.py`
  - Plan schema validation and fallback.

- Test `tests/retrieval/test_evidence_pool.py`
  - Exact dedupe, matched query provenance, requirement-aware summary, missing requirements.

- Test `tests/retrieval/test_planned_retrieval.py`
  - Forced multi-query execution with fake planner/search.

- Modify `tests/mcp_servers/test_index_manager.py`
  - Assert `chunk_id` is returned from `IndexManager.search`.

- Modify `tests/mcp_servers/test_colbert_via_client.py`
  - Add or update slow MCP integration coverage for the new `planned_retrieval` tool.

## Task 1: QueryPlan Dataclasses and Fallback

**Files:**

- Create: `paperpilot/retrieval/query_plan.py`
- Modify: `paperpilot/retrieval/__init__.py`
- Test: `tests/retrieval/test_query_plan_validator.py`

- [ ] **Step 1: Write failing tests for fallback shape**

Create `tests/retrieval/test_query_plan_validator.py` with:

```python
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/retrieval/test_query_plan_validator.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'paperpilot.retrieval.query_plan'`.

- [ ] **Step 3: Implement dataclasses and fallback**

Create `paperpilot/retrieval/query_plan.py`:

```python
"""Structured query planning schema for planned retrieval."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PLAN_VERSION = "query_plan_v1"


@dataclass(frozen=True)
class QueryConstraints:
    needs_numbers: bool = False
    needs_comparison: bool = False
    needs_table_or_figure: bool = False
    polarity: str = "neutral"


@dataclass(frozen=True)
class EvidenceRequirement:
    id: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class PlannedQuery:
    id: str
    role: str
    query: str
    targets: list[str]
    priority: int = 1


@dataclass(frozen=True)
class ExpansionHints:
    neighbor_window: int = 1
    prefer_tables: bool = False
    prefer_captions: bool = False


@dataclass(frozen=True)
class QueryPlan:
    version: str
    question: str
    question_type: str
    answer_shape: str
    intent_summary: str
    focus_terms: list[str]
    constraints: QueryConstraints
    evidence_requirements: list[EvidenceRequirement]
    queries: list[PlannedQuery]
    avoid: list[str] = field(default_factory=list)
    expansion_hints: ExpansionHints = field(default_factory=ExpansionHints)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def minimal_fallback_plan(question: str) -> QueryPlan:
    """Return a safe literal-query-only plan."""
    requirement = EvidenceRequirement(
        id="req_direct",
        description="direct evidence answering the question",
        required=True,
    )
    return QueryPlan(
        version=PLAN_VERSION,
        question=question,
        question_type="other",
        answer_shape="freeform",
        intent_summary="Find direct evidence that answers the user question.",
        focus_terms=[],
        constraints=QueryConstraints(),
        evidence_requirements=[requirement],
        queries=[
            PlannedQuery(
                id="q_lit",
                role="literal",
                query=question,
                targets=[requirement.id],
                priority=1,
            )
        ],
    )
```

Modify `paperpilot/retrieval/__init__.py`:

```python
"""Retrieval planning and evidence pooling helpers."""

from paperpilot.retrieval.query_plan import (
    EvidenceRequirement,
    ExpansionHints,
    PlannedQuery,
    QueryConstraints,
    QueryPlan,
    minimal_fallback_plan,
)

__all__ = [
    "EvidenceRequirement",
    "ExpansionHints",
    "PlannedQuery",
    "QueryConstraints",
    "QueryPlan",
    "minimal_fallback_plan",
]
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest tests/retrieval/test_query_plan_validator.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add paperpilot/retrieval/query_plan.py paperpilot/retrieval/__init__.py tests/retrieval/test_query_plan_validator.py
git commit -m "Add query plan schema"
```

Expected: commit succeeds.

## Task 2: Plan Validator and JSON Parser

**Files:**

- Modify: `paperpilot/retrieval/query_plan_validator.py`
- Modify: `tests/retrieval/test_query_plan_validator.py`

- [ ] **Step 1: Add failing validator tests**

Append to `tests/retrieval/test_query_plan_validator.py`:

```python
from paperpilot.retrieval.query_plan_validator import (
    parse_query_plan_json,
    validate_query_plan,
)


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


def test_parse_query_plan_json_falls_back_on_invalid_json() -> None:
    plan, meta = parse_query_plan_json("not json", question="What is used?")

    assert plan.queries[0].query == "What is used?"
    assert meta["fallback_used"] is True
    assert "invalid_json" in meta["fallback_reason"]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/retrieval/test_query_plan_validator.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'paperpilot.retrieval.query_plan_validator'`.

- [ ] **Step 3: Implement validator**

Create `paperpilot/retrieval/query_plan_validator.py`:

```python
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


def parse_query_plan_json(text: str, *, question: str) -> tuple[QueryPlan, dict[str, Any]]:
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
    queries = [q for q in queries if q.role != "literal" and q.query.strip() != question.strip()]
    queries = [literal, *queries][:MAX_QUERIES]
    if not queries:
        raise ValueError("query plan must include at least one valid query")

    constraints_data = data.get("constraints") if isinstance(data.get("constraints"), dict) else {}
    hints_data = data.get("expansion_hints") if isinstance(data.get("expansion_hints"), dict) else {}
    return QueryPlan(
        version=PLAN_VERSION,
        question=str(data.get("question") or question),
        question_type=str(data.get("question_type") or "other"),
        answer_shape=str(data.get("answer_shape") or "freeform"),
        intent_summary=str(data.get("intent_summary") or "Find direct evidence that answers the question."),
        focus_terms=[str(item) for item in data.get("focus_terms") or [] if str(item).strip()],
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
        out.append(EvidenceRequirement(
            id=req_id,
            description=description,
            required=bool(item.get("required", True)),
        ))
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
        out.append(PlannedQuery(
            id=query_id,
            role=str(item.get("role") or "focused_rewrite"),
            query=query,
            targets=targets,
            priority=int(item.get("priority", index) or index),
        ))
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
```

- [ ] **Step 4: Export validator APIs**

Modify `paperpilot/retrieval/__init__.py`:

```python
"""Retrieval planning and evidence pooling helpers."""

from paperpilot.retrieval.query_plan import (
    EvidenceRequirement,
    ExpansionHints,
    PlannedQuery,
    QueryConstraints,
    QueryPlan,
    minimal_fallback_plan,
)
from paperpilot.retrieval.query_plan_validator import (
    parse_query_plan_json,
    validate_query_plan,
)

__all__ = [
    "EvidenceRequirement",
    "ExpansionHints",
    "PlannedQuery",
    "QueryConstraints",
    "QueryPlan",
    "minimal_fallback_plan",
    "parse_query_plan_json",
    "validate_query_plan",
]
```

- [ ] **Step 5: Run tests and verify pass**

Run:

```bash
pytest tests/retrieval/test_query_plan_validator.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add paperpilot/retrieval/__init__.py paperpilot/retrieval/query_plan_validator.py tests/retrieval/test_query_plan_validator.py
git commit -m "Add query plan validation"
```

Expected: commit succeeds.

## Task 3: EvidencePool Deduplication and Summary Selection

**Files:**

- Create: `paperpilot/retrieval/evidence_pool.py`
- Create: `tests/retrieval/test_evidence_pool.py`
- Modify: `paperpilot/retrieval/__init__.py`

- [ ] **Step 1: Write failing EvidencePool tests**

Create `tests/retrieval/test_evidence_pool.py`:

```python
from paperpilot.retrieval.evidence_pool import (
    RawSearchHit,
    build_evidence_pool,
    format_evidence_summary,
)
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
            EvidenceRequirement("req_baseline", "baselines compared by the paper", True),
        ],
        queries=[
            PlannedQuery("q_dataset", "focused_rewrite", "dataset used experiment", ["req_dataset"], 1),
            PlannedQuery("q_baseline", "focused_rewrite", "baseline compared method", ["req_baseline"], 2),
        ],
    )


def test_exact_duplicate_chunks_merge_matched_queries() -> None:
    plan = _plan()
    hits = [
        RawSearchHit(
            paper_id="p1",
            chunk_id=None,
            chunk_text="The dataset is WikiHop.",
            score=10.0,
            query=plan.queries[0],
            rank=1,
        ),
        RawSearchHit(
            paper_id="p1",
            chunk_id=None,
            chunk_text="The dataset   is WikiHop.",
            score=8.0,
            query=plan.queries[1],
            rank=2,
        ),
    ]

    pool = build_evidence_pool("plan-1", plan, hits, summary_k=4)

    assert pool.stats["raw_result_count"] == 2
    assert pool.stats["deduped_count"] == 1
    assert len(pool.items) == 1
    assert pool.items[0].best_score == 10.0
    assert [match.query_id for match in pool.items[0].matched_queries] == [
        "q_dataset",
        "q_baseline",
    ]


def test_summary_selection_covers_required_requirements_before_score_fill() -> None:
    plan = _plan()
    hits = [
        RawSearchHit("p1", None, "Dataset evidence A.", 30.0, plan.queries[0], 1),
        RawSearchHit("p1", None, "Dataset evidence B.", 29.0, plan.queries[0], 2),
        RawSearchHit("p1", None, "Baseline evidence.", 20.0, plan.queries[1], 1),
    ]

    pool = build_evidence_pool("plan-1", plan, hits, summary_k=2)
    selected_texts = [item.chunk_text for item in pool.summary_items]

    assert selected_texts == ["Dataset evidence A.", "Baseline evidence."]
    assert pool.missing_requirements == []


def test_missing_requirement_records_no_candidates() -> None:
    plan = _plan()
    hits = [
        RawSearchHit("p1", None, "Dataset evidence A.", 30.0, plan.queries[0], 1),
    ]

    pool = build_evidence_pool("plan-1", plan, hits, summary_k=4)

    assert len(pool.missing_requirements) == 1
    assert pool.missing_requirements[0].requirement_id == "req_baseline"
    assert pool.missing_requirements[0].reason == "no_candidates"


def test_format_evidence_summary_includes_missing_requirements() -> None:
    plan = _plan()
    pool = build_evidence_pool("plan-1", plan, [], summary_k=4)

    text = format_evidence_summary(pool)

    assert "Planned retrieval completed." in text
    assert "Missing requirements:" in text
    assert "req_dataset" in text
    assert "req_baseline" in text
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/retrieval/test_evidence_pool.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'paperpilot.retrieval.evidence_pool'`.

- [ ] **Step 3: Implement EvidencePool**

Create `paperpilot/retrieval/evidence_pool.py`:

```python
"""Evidence pooling for planned retrieval results."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from paperpilot.retrieval.query_plan import PlannedQuery, QueryPlan


@dataclass(frozen=True)
class RawSearchHit:
    paper_id: str
    chunk_id: str | None
    chunk_text: str
    score: float
    query: PlannedQuery
    rank: int


@dataclass
class MatchedQuery:
    query_id: str
    query: str
    role: str
    rank: int
    score: float
    targets: list[str]


@dataclass
class EvidenceItem:
    id: str
    paper_id: str
    chunk_id: str | None
    chunk_text: str
    best_score: float
    matched_queries: list[MatchedQuery] = field(default_factory=list)

    def targets(self) -> set[str]:
        out: set[str] = set()
        for match in self.matched_queries:
            out.update(match.targets)
        return out


@dataclass(frozen=True)
class MissingRequirement:
    requirement_id: str
    description: str
    reason: str


@dataclass
class EvidencePool:
    plan_id: str
    question: str
    query_plan: dict[str, Any]
    items: list[EvidenceItem]
    summary_items: list[EvidenceItem]
    missing_requirements: list[MissingRequirement]
    stats: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "question": self.question,
            "query_plan": self.query_plan,
            "items": [asdict(item) for item in self.items],
            "summary_items": [item.id for item in self.summary_items],
            "missing_requirements": [asdict(item) for item in self.missing_requirements],
            "stats": dict(self.stats),
        }


def build_evidence_pool(
    plan_id: str,
    plan: QueryPlan,
    hits: list[RawSearchHit],
    *,
    summary_k: int = 8,
    overlap_threshold: float = 0.75,
) -> EvidencePool:
    deduped = _dedupe_hits(hits)
    summary, missing = _select_summary(
        plan,
        deduped,
        summary_k=summary_k,
        overlap_threshold=overlap_threshold,
    )
    return EvidencePool(
        plan_id=plan_id,
        question=plan.question,
        query_plan=plan.to_dict(),
        items=deduped,
        summary_items=summary,
        missing_requirements=missing,
        stats={
            "query_count": len(plan.queries),
            "raw_result_count": len(hits),
            "deduped_count": len(deduped),
        },
    )


def format_evidence_summary(pool: EvidencePool) -> str:
    missing = (
        "none"
        if not pool.missing_requirements
        else "\n".join(
            f"- {item.requirement_id}: {item.description} ({item.reason})"
            for item in pool.missing_requirements
        )
    )
    blocks = [
        "Planned retrieval completed.",
        f"Queries executed: {pool.stats['query_count']}",
        (
            "Evidence chunks: "
            f"{pool.stats['raw_result_count']} raw, "
            f"{pool.stats['deduped_count']} deduped"
        ),
        f"Missing requirements: {missing}",
        "",
        "Top evidence:",
    ]
    for item in pool.summary_items:
        query_ids = ", ".join(match.query_id for match in item.matched_queries)
        blocks.append(
            f"[{item.id}] Found by {query_ids}. Score {item.best_score:.2f}.\n"
            f"{item.chunk_text}"
        )
    if not pool.summary_items:
        blocks.append("No retrieved evidence.")
    return "\n\n".join(blocks)


def _dedupe_hits(hits: list[RawSearchHit]) -> list[EvidenceItem]:
    by_key: dict[str, EvidenceItem] = {}
    order: list[str] = []
    for hit in hits:
        key = _dedupe_key(hit)
        match = MatchedQuery(
            query_id=hit.query.id,
            query=hit.query.query,
            role=hit.query.role,
            rank=hit.rank,
            score=float(hit.score),
            targets=list(hit.query.targets),
        )
        existing = by_key.get(key)
        if existing is None:
            item = EvidenceItem(
                id=f"ev_{len(order) + 1}",
                paper_id=hit.paper_id,
                chunk_id=hit.chunk_id,
                chunk_text=hit.chunk_text,
                best_score=float(hit.score),
                matched_queries=[match],
            )
            by_key[key] = item
            order.append(key)
        else:
            existing.best_score = max(existing.best_score, float(hit.score))
            existing.matched_queries.append(match)
    return sorted((by_key[key] for key in order), key=lambda item: item.best_score, reverse=True)


def _select_summary(
    plan: QueryPlan,
    items: list[EvidenceItem],
    *,
    summary_k: int,
    overlap_threshold: float,
) -> tuple[list[EvidenceItem], list[MissingRequirement]]:
    selected: list[EvidenceItem] = []
    missing: list[MissingRequirement] = []
    query_targets = {target for query in plan.queries for target in query.targets}
    for requirement in plan.evidence_requirements:
        if not requirement.required:
            continue
        if requirement.id not in query_targets:
            missing.append(MissingRequirement(requirement.id, requirement.description, "no_targeted_query"))
            continue
        candidates = [item for item in items if requirement.id in item.targets()]
        candidate = _first_diverse(candidates, selected, overlap_threshold)
        if candidate is None:
            missing.append(MissingRequirement(requirement.id, requirement.description, "no_candidates"))
        else:
            selected.append(candidate)
        if len(selected) >= summary_k:
            return selected[:summary_k], missing
    for item in items:
        if len(selected) >= summary_k:
            break
        if item in selected:
            continue
        if _is_diverse(item, selected, overlap_threshold):
            selected.append(item)
    return selected, missing


def _first_diverse(
    candidates: list[EvidenceItem],
    selected: list[EvidenceItem],
    overlap_threshold: float,
) -> EvidenceItem | None:
    for candidate in candidates:
        if candidate not in selected and _is_diverse(candidate, selected, overlap_threshold):
            return candidate
    for candidate in candidates:
        if candidate not in selected:
            return candidate
    return None


def _is_diverse(candidate: EvidenceItem, selected: list[EvidenceItem], threshold: float) -> bool:
    return all(_token_overlap(candidate.chunk_text, item.chunk_text) <= threshold for item in selected)


def _token_overlap(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"[A-Za-z0-9_]+", a.lower()))
    tokens_b = set(re.findall(r"[A-Za-z0-9_]+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


def _dedupe_key(hit: RawSearchHit) -> str:
    if hit.chunk_id:
        return f"{hit.paper_id}::{hit.chunk_id}"
    normalized = " ".join(hit.chunk_text.split()).lower()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return f"{hit.paper_id}::{digest}"
```

- [ ] **Step 4: Export EvidencePool APIs**

Modify `paperpilot/retrieval/__init__.py`:

```python
"""Retrieval planning and evidence pooling helpers."""

from paperpilot.retrieval.evidence_pool import (
    EvidencePool,
    RawSearchHit,
    build_evidence_pool,
    format_evidence_summary,
)
from paperpilot.retrieval.query_plan import (
    EvidenceRequirement,
    ExpansionHints,
    PlannedQuery,
    QueryConstraints,
    QueryPlan,
    minimal_fallback_plan,
)
from paperpilot.retrieval.query_plan_validator import (
    parse_query_plan_json,
    validate_query_plan,
)

__all__ = [
    "EvidencePool",
    "EvidenceRequirement",
    "ExpansionHints",
    "PlannedQuery",
    "QueryConstraints",
    "QueryPlan",
    "RawSearchHit",
    "build_evidence_pool",
    "format_evidence_summary",
    "minimal_fallback_plan",
    "parse_query_plan_json",
    "validate_query_plan",
]
```

- [ ] **Step 5: Run tests and verify pass**

Run:

```bash
pytest tests/retrieval/test_evidence_pool.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add paperpilot/retrieval/__init__.py paperpilot/retrieval/evidence_pool.py tests/retrieval/test_evidence_pool.py
git commit -m "Add evidence pool selection"
```

Expected: commit succeeds.

## Task 4: LLM Query Planner

**Files:**

- Create: `paperpilot/retrieval/llm_query_planner.py`
- Create: `tests/retrieval/test_llm_query_planner.py`
- Modify: `paperpilot/retrieval/__init__.py`

- [ ] **Step 1: Write failing planner tests with fake client**

Create `tests/retrieval/test_llm_query_planner.py`:

```python
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
        text='{"version":"query_plan_v1","question":"Q?","evidence_requirements":[{"id":"req_1","description":"direct evidence","required":true}],"queries":[{"id":"q_1","role":"focused_rewrite","query":"direct evidence","targets":["req_1"],"priority":2}]}'
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/retrieval/test_llm_query_planner.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'paperpilot.retrieval.llm_query_planner'`.

- [ ] **Step 3: Implement LLM planner**

Create `paperpilot/retrieval/llm_query_planner.py`:

```python
"""LLM-backed JSON query planner."""
from __future__ import annotations

from typing import Any, Protocol

from paperpilot.core.adapter import LLMClient, Tool
from paperpilot.retrieval.query_plan import QueryPlan
from paperpilot.retrieval.query_plan_validator import parse_query_plan_json


class PlannerClient(Protocol):
    def call(self, messages: list[dict], tools: list[Tool], *, system: str) -> Any: ...


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
        meta["fallback_reason"] = f"planner_call_failed: {type(exc).__name__}: {exc}"
        return plan, meta
    plan, meta = parse_query_plan_json(raw_text, question=question)
    meta["raw_response"] = raw_text
    return plan, meta
```

- [ ] **Step 4: Export planner API**

Modify `paperpilot/retrieval/__init__.py` to add:

```python
from paperpilot.retrieval.llm_query_planner import (
    build_planner_prompt,
    plan_with_llm,
)
```

and add `"build_planner_prompt"` and `"plan_with_llm"` to `__all__`.

- [ ] **Step 5: Run tests and verify pass**

Run:

```bash
pytest tests/retrieval/test_llm_query_planner.py tests/retrieval/test_query_plan_validator.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add paperpilot/retrieval/__init__.py paperpilot/retrieval/llm_query_planner.py tests/retrieval/test_llm_query_planner.py
git commit -m "Add LLM query planner"
```

Expected: commit succeeds.

## Task 5: Planned Retrieval Orchestrator

**Files:**

- Create: `paperpilot/retrieval/planned_retrieval.py`
- Create: `tests/retrieval/test_planned_retrieval.py`
- Modify: `paperpilot/retrieval/__init__.py`

- [ ] **Step 1: Write failing orchestrator tests**

Create `tests/retrieval/test_planned_retrieval.py`:

```python
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
        question="What dataset was used?",
        question_type="dataset_used",
        answer_shape="single_entity",
        intent_summary="Find dataset.",
        focus_terms=["dataset"],
        constraints=QueryConstraints(),
        evidence_requirements=[
            EvidenceRequirement("req_dataset", "dataset used by the paper", True)
        ],
        queries=[
            PlannedQuery("q_lit", "literal", "What dataset was used?", ["req_dataset"], 1),
            PlannedQuery("q_1", "focused_rewrite", "dataset used experiment", ["req_dataset"], 2),
        ],
    )


def test_run_planned_retrieval_executes_all_plan_queries() -> None:
    calls = []

    def search(query: str, paper_id: str, top_k: int):
        calls.append((query, paper_id, top_k))
        return [
            {
                "paper_id": paper_id,
                "chunk_id": f"chunk_{len(calls)}",
                "chunk_text": f"Evidence for {query}",
                "score": 10.0,
            }
        ]

    result = run_planned_retrieval(
        plan_id="plan-1",
        plan=_plan(),
        paper_id="p1",
        search=search,
        top_k_each=3,
        summary_k=4,
    )

    assert [call[0] for call in calls] == [
        "What dataset was used?",
        "dataset used experiment",
    ]
    assert result.evidence_pool.stats["raw_result_count"] == 2
    assert "Planned retrieval completed." in result.summary_text


def test_run_planned_retrieval_records_query_errors_and_continues() -> None:
    def search(query: str, paper_id: str, top_k: int):
        if query == "What dataset was used?":
            raise RuntimeError("search unavailable")
        return [
            {
                "paper_id": paper_id,
                "chunk_text": "The dataset is WikiHop.",
                "score": 10.0,
            }
        ]

    result = run_planned_retrieval(
        plan_id="plan-1",
        plan=_plan(),
        paper_id="p1",
        search=search,
        top_k_each=3,
        summary_k=4,
    )

    assert result.query_errors == [
        {
            "query_id": "q_lit",
            "query": "What dataset was used?",
            "error": "RuntimeError: search unavailable",
        }
    ]
    assert result.evidence_pool.stats["raw_result_count"] == 1
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/retrieval/test_planned_retrieval.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'paperpilot.retrieval.planned_retrieval'`.

- [ ] **Step 3: Implement orchestrator**

Create `paperpilot/retrieval/planned_retrieval.py`:

```python
"""Forced execution for validated query plans."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from paperpilot.retrieval.evidence_pool import (
    EvidencePool,
    RawSearchHit,
    build_evidence_pool,
    format_evidence_summary,
)
from paperpilot.retrieval.query_plan import QueryPlan

SearchFn = Callable[[str, str, int], list[dict[str, Any]]]


@dataclass(frozen=True)
class PlannedRetrievalResult:
    evidence_pool: EvidencePool
    summary_text: str
    query_errors: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_text": self.summary_text,
            "evidence_pool": self.evidence_pool.to_dict(),
            "query_errors": list(self.query_errors),
        }


def run_planned_retrieval(
    *,
    plan_id: str,
    plan: QueryPlan,
    paper_id: str,
    search: SearchFn,
    top_k_each: int = 5,
    summary_k: int = 8,
) -> PlannedRetrievalResult:
    hits: list[RawSearchHit] = []
    errors: list[dict[str, str]] = []
    for planned_query in sorted(plan.queries, key=lambda query: query.priority):
        try:
            results = search(planned_query.query, paper_id, top_k_each)
        except Exception as exc:  # noqa: BLE001
            errors.append({
                "query_id": planned_query.id,
                "query": planned_query.query,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        for index, item in enumerate(results, start=1):
            chunk_text = str(item.get("chunk_text") or "")
            if not chunk_text.strip():
                continue
            hits.append(RawSearchHit(
                paper_id=str(item.get("paper_id") or paper_id),
                chunk_id=(
                    str(item["chunk_id"])
                    if item.get("chunk_id") is not None
                    else None
                ),
                chunk_text=chunk_text,
                score=float(item.get("score", 0.0)),
                query=planned_query,
                rank=index,
            ))
    pool = build_evidence_pool(plan_id, plan, hits, summary_k=summary_k)
    return PlannedRetrievalResult(
        evidence_pool=pool,
        summary_text=format_evidence_summary(pool),
        query_errors=errors,
    )
```

- [ ] **Step 4: Export orchestrator API**

Modify `paperpilot/retrieval/__init__.py` to add:

```python
from paperpilot.retrieval.planned_retrieval import (
    PlannedRetrievalResult,
    run_planned_retrieval,
)
```

and add `"PlannedRetrievalResult"` and `"run_planned_retrieval"` to `__all__`.

- [ ] **Step 5: Run tests and verify pass**

Run:

```bash
pytest tests/retrieval/test_planned_retrieval.py tests/retrieval/test_evidence_pool.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add paperpilot/retrieval/__init__.py paperpilot/retrieval/planned_retrieval.py tests/retrieval/test_planned_retrieval.py
git commit -m "Add planned retrieval execution"
```

Expected: commit succeeds.

## Task 6: ColBERT Search chunk_id and planned_retrieval MCP Tool

**Files:**

- Modify: `paperpilot/mcp_servers/colbert/index_manager.py`
- Modify: `paperpilot/mcp_servers/colbert/server.py`
- Modify: `tests/mcp_servers/test_index_manager.py`
- Modify: `tests/mcp_servers/test_colbert_via_client.py`

- [ ] **Step 1: Add failing index manager assertion for chunk_id**

In `tests/mcp_servers/test_index_manager.py`, modify `test_search_returns_only_matching_paper_chunks` so its final assertions are:

```python
    assert len(out) == 1
    assert out[0]["paper_id"] == "pA"
    assert out[0]["chunk_id"] == chunk_id_a.split("::", 1)[1]
    assert out[0]["chunk_text"] == state_a.chunk_texts[chunk_id_a]
```

Use the actual local variable names already present in that test: `out`, `chunk_id_a`, and `state_a`. The final test body should end as:

```python
    out = mgr.search("alpha", paper_id="pA", top_k=1)
    assert len(out) == 1
    assert out[0]["paper_id"] == "pA"
    assert out[0]["chunk_id"] == chunk_id_a.split("::", 1)[1]
    assert out[0]["chunk_text"] == state_a.chunk_texts[chunk_id_a]
```

- [ ] **Step 2: Run index manager test and verify failure**

Run:

```bash
pytest tests/mcp_servers/test_index_manager.py -q
```

Expected: FAIL because `chunk_id` is not returned.

- [ ] **Step 3: Return chunk_id from IndexManager.search**

Modify the return block in `paperpilot/mcp_servers/colbert/index_manager.py`:

```python
        out = []
        for result in scores[0]:
            full_id = result["id"]
            paper, chunk_id = full_id.split("::", 1)
            out.append({
                "paper_id": paper,
                "chunk_id": chunk_id,
                "chunk_text": state.chunk_texts[full_id],
                "score": float(result["score"]),
            })
        return out
```

- [ ] **Step 4: Run index manager tests and verify pass**

Run:

```bash
pytest tests/mcp_servers/test_index_manager.py -q
```

Expected: PASS.

- [ ] **Step 5: Add planned_retrieval MCP tool tests**

In `tests/mcp_servers/test_colbert_via_client.py`, add a slow integration test:

```python
@pytest.mark.slow
def test_planned_retrieval_tool_runs_multiple_queries(tmp_path):
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        tools = c.list_tools()
        build = next(t for t in tools if t.name == "mcp__colbert__build_index")
        planned = next(t for t in tools if t.name == "mcp__colbert__planned_retrieval")
        prefix = _paper_prefix(tmp_path)
        paper_id = f"{prefix}-planned"
        build.handler({
            "documents": [{
                "paper_id": paper_id,
                "text": (
                    "The experiments use the WikiHop dataset for multi-hop question answering. "
                    "The baseline system is compared against a standard neural reader. "
                ) * 8,
            }]
        })

        raw = planned.handler({
            "question": "What dataset was used?",
            "paper_id": paper_id,
            "paper_title": "A WikiHop Paper",
            "abstract": "We evaluate on WikiHop.",
            "top_k_each": 2,
            "summary_k": 3,
        })
        payload = json.loads(raw) if isinstance(raw, str) else raw

        assert "summary_text" in payload
        assert "evidence_pool" in payload
        assert payload["evidence_pool"]["stats"]["query_count"] >= 1
        assert payload["evidence_pool"]["stats"]["raw_result_count"] >= 1
        assert payload["evidence_pool"]["items"][0]["matched_queries"]
    finally:
        c.close()
```

- [ ] **Step 6: Run planned_retrieval MCP test and verify failure**

Run:

```bash
pytest -m slow tests/mcp_servers/test_colbert_via_client.py::test_planned_retrieval_tool_runs_multiple_queries -q
```

Expected: FAIL because `mcp__colbert__planned_retrieval` is not registered.

- [ ] **Step 7: Implement MCP tool**

Modify `paperpilot/mcp_servers/colbert/server.py` imports:

```python
import uuid

from paperpilot.retrieval.llm_query_planner import plan_with_llm
from paperpilot.retrieval.planned_retrieval import run_planned_retrieval
```

Add the MCP tool:

```python
@mcp.tool()
def planned_retrieval(
    question: str,
    paper_id: str,
    paper_title: str = "",
    abstract: str = "",
    top_k_each: int = 5,
    summary_k: int = 8,
) -> dict:
    """Plan and execute multiple ColBERT searches for one paper.

    Args:
        question: User question to answer from the paper.
        paper_id: The paper_id previously passed to build_index.
        paper_title: Optional paper title for planning.
        abstract: Optional abstract for planning.
        top_k_each: Number of chunks to retrieve per planned query.
        summary_k: Number of evidence items to show in the compact summary.

    Returns:
        dict with summary_text, evidence_pool, query_plan_meta, and query_errors.
    """
    return _planned_retrieval_impl(
        question=question,
        paper_id=paper_id,
        paper_title=paper_title,
        abstract=abstract,
        top_k_each=top_k_each,
        summary_k=summary_k,
    )


def _planned_retrieval_impl(
    *,
    question: str,
    paper_id: str,
    paper_title: str = "",
    abstract: str = "",
    top_k_each: int = 5,
    summary_k: int = 8,
) -> dict:
    assert _manager is not None, "IndexManager not initialized"
    plan, meta = plan_with_llm(
        question=question,
        paper_title=paper_title,
        abstract=abstract,
    )
    result = run_planned_retrieval(
        plan_id=f"plan-{uuid.uuid4().hex[:12]}",
        plan=plan,
        paper_id=paper_id,
        search=lambda query, pid, top_k: _manager.search(query, pid, top_k),
        top_k_each=top_k_each,
        summary_k=summary_k,
    )
    payload = result.to_dict()
    payload["query_plan_meta"] = meta
    return payload
```

Keep the return type as `dict`. Existing MCP handlers in this project may expose tool results to Python tests as either `dict` or JSON text, so the test parses `raw` only when it is a string.

- [ ] **Step 8: Run MCP and unit tests**

Run:

```bash
pytest tests/retrieval/test_planned_retrieval.py tests/retrieval/test_llm_query_planner.py tests/retrieval/test_evidence_pool.py -q
pytest tests/mcp_servers/test_index_manager.py -q
```

Expected: PASS.

Then run the slow MCP test if local ColBERT dependencies are available:

```bash
pytest -m slow tests/mcp_servers/test_colbert_via_client.py::test_planned_retrieval_tool_runs_multiple_queries -q
```

Expected: PASS or skip only for known local model/dependency constraints.

- [ ] **Step 9: Commit**

Run:

```bash
git add paperpilot/mcp_servers/colbert/index_manager.py paperpilot/mcp_servers/colbert/server.py tests/mcp_servers/test_index_manager.py tests/mcp_servers/test_colbert_via_client.py
git commit -m "Add planned retrieval MCP tool"
```

Expected: commit succeeds.

## Task 7: Deep Read Skill Integration

**Files:**

- Modify: `paperpilot/skills/deep-read-paper.md`
- Test: `tests/test_deep_read_skill.py`

- [ ] **Step 1: Add failing skill text test**

Create `tests/test_deep_read_skill.py`:

```python
from pathlib import Path


def test_deep_read_skill_requires_planned_retrieval_first() -> None:
    text = Path("paperpilot/skills/deep-read-paper.md").read_text(encoding="utf-8")

    assert "mcp__colbert__planned_retrieval" in text
    assert "planned_retrieval" in text
    assert "optional follow-up" in text.lower() or "补充" in text
    assert "mcp__colbert__search" in text
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_deep_read_skill.py -q
```

Expected: FAIL because the skill does not mention `planned_retrieval`.

- [ ] **Step 3: Update skill workflow**

Modify `paperpilot/skills/deep-read-paper.md` Workflow section to:

```markdown
1. 下载全文：调用 `mcp__arxiv__download_paper(arxiv_id="...")`。
2. 建索引：调用 `mcp__colbert__build_index(documents=[download_paper_result])`。
   - `documents` 必须是非空 list。
   - 每个元素必须包含 `paper_id` 和 `text`。
   - 通常直接把 `download_paper` 返回对象作为 list 里的唯一元素。
3. 计划检索：优先调用 `mcp__colbert__planned_retrieval(question="<用户问题>", paper_id="<indexed_paper_id>", paper_title="<title if known>", abstract="<abstract if known>", top_k_each=5, summary_k=8)`。
   - `paper_id` 必须与 build_index 时的 `paper_id` 一致。
   - planned retrieval 会生成结构化 QueryPlan，强制执行多条 planned queries，并返回 EvidencePool summary。
   - 回答应优先基于 planned retrieval 返回的 top evidence。
   - 如果返回 `missing_requirements`，不要猜测缺失部分；可以继续普通 search 补查。
4. 可选补充检索：只有当 planned retrieval 的证据不足、`missing_requirements` 未覆盖，或用户问题需要进一步澄清时，才调用 `mcp__colbert__search(query="...", paper_id="<indexed_paper_id>", top_k=3)`。
   - 补充 search 是为了填补缺口，不是替代 planned retrieval。
   - 换 query 时围绕 missing requirement、关键术语、同义词、表格/实验设置等线索。
5. 综合回答：只基于 planned retrieval 或 follow-up search 返回的具体段落回答。
```

- [ ] **Step 4: Run test and verify pass**

Run:

```bash
pytest tests/test_deep_read_skill.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add paperpilot/skills/deep-read-paper.md tests/test_deep_read_skill.py
git commit -m "Update deep read skill for planned retrieval"
```

Expected: commit succeeds.

## Task 8: Eval Baseline Uses Product Planned Retrieval

**Files:**

- Modify: `paperpilot/eval/baselines.py`
- Modify: `tests/eval/test_baselines.py`

- [ ] **Step 1: Add failing eval metadata test**

Append to `tests/eval/test_baselines.py`:

```python
def test_run_paperpilot_records_planned_retrieval_metadata(monkeypatch, tmp_path) -> None:
    events = []

    def fake_agent_run(prompt, max_iter, on_event):
        on_event("tool_result", {
            "name": "mcp__colbert__planned_retrieval",
            "content": '{"summary_text":"ok","evidence_pool":{"stats":{"raw_result_count":2,"deduped_count":1},"summary_items":["ev_1"],"missing_requirements":[]},"query_plan_meta":{"fallback_used":false},"query_errors":[]}',
        })
        events.append(prompt)
        return [{"role": "assistant", "content": "Short answer: WikiHop\n\nEvidence: WikiHop evidence."}]

    monkeypatch.setattr("paperpilot.main.run", fake_agent_run)
    result = baselines.run_paperpilot(CASE, trace_id="planned-meta-test")

    assert result["planned_retrieval_used"] is True
    assert result["planned_retrieval_stats"] == {"raw_result_count": 2, "deduped_count": 1}
    assert result["planned_retrieval_missing_requirements"] == []
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/eval/test_baselines.py::test_run_paperpilot_records_planned_retrieval_metadata -q
```

Expected: FAIL because result does not expose planned retrieval metadata.

- [ ] **Step 3: Implement trace parsing in baseline**

Modify `paperpilot/eval/baselines.py`:

```python
import json
```

Add helper:

```python
def _extract_planned_retrieval_metadata(trace_path: Path) -> dict[str, Any]:
    if not trace_path.exists():
        return {
            "planned_retrieval_used": False,
            "planned_retrieval_stats": None,
            "planned_retrieval_missing_requirements": None,
        }
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("kind") != "tool_result":
            continue
        payload = event.get("payload") or {}
        if payload.get("name") != "mcp__colbert__planned_retrieval":
            continue
        try:
            content = json.loads(payload.get("content") or "{}")
        except json.JSONDecodeError:
            continue
        pool = content.get("evidence_pool") or {}
        return {
            "planned_retrieval_used": True,
            "planned_retrieval_stats": pool.get("stats"),
            "planned_retrieval_missing_requirements": pool.get("missing_requirements"),
        }
    return {
        "planned_retrieval_used": False,
        "planned_retrieval_stats": None,
        "planned_retrieval_missing_requirements": None,
    }
```

In `run_paperpilot`, before the return dict:

```python
    planned_retrieval_meta = _extract_planned_retrieval_metadata(trace_path)
```

Add to returned dict:

```python
        **planned_retrieval_meta,
```

- [ ] **Step 4: Run test and verify pass**

Run:

```bash
pytest tests/eval/test_baselines.py::test_run_paperpilot_records_planned_retrieval_metadata -q
```

Expected: PASS.

- [ ] **Step 5: Run eval tests**

Run:

```bash
pytest tests/eval/test_baselines.py tests/eval/test_evidence_selection.py tests/eval/test_query_planner.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add paperpilot/eval/baselines.py tests/eval/test_baselines.py
git commit -m "Record planned retrieval eval metadata"
```

Expected: commit succeeds.

## Task 9: Final Verification

**Files:**

- Verify all files changed by prior tasks.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
pytest \
  tests/retrieval/test_query_plan_validator.py \
  tests/retrieval/test_evidence_pool.py \
  tests/retrieval/test_llm_query_planner.py \
  tests/retrieval/test_planned_retrieval.py \
  tests/mcp_servers/test_index_manager.py \
  tests/test_deep_read_skill.py \
  tests/eval/test_baselines.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run slow planned retrieval MCP smoke if dependencies are available**

Run:

```bash
pytest -m slow tests/mcp_servers/test_colbert_via_client.py::test_planned_retrieval_tool_runs_multiple_queries -q
```

Expected: PASS. If the local environment cannot load ColBERT model dependencies, record the exact failure in the final report.

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected: only intentional files changed or clean after commits. Existing unrelated Phase2 working-tree changes may remain and must not be reverted.

- [ ] **Step 4: Final implementation summary**

Report:

- files created and modified;
- tests run and results;
- whether slow MCP smoke passed;
- remaining risk: planner invocation is still skill/tool driven, not loop-forced;
- next possible step: QASPER 13-case recall diagnosis with planned retrieval.
