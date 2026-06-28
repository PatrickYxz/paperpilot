# Query Planner Phase 3 Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional requirement-level evidence verifier to planned retrieval so evidence shown to the agent is selected by direct support for each `EvidenceRequirement`, not only by ColBERT similarity.

**Architecture:** Implement the verifier first as pure retrieval code that consumes `QueryPlan` and `EvidencePool`, then wire it into `run_planned_retrieval` behind `verify_evidence=False`. MCP and eval get explicit opt-in flags so normal `deep-read-paper` behavior stays unchanged until verified by focused runs.

**Tech Stack:** Python dataclasses, existing `paperpilot.core.LLMClient`, pytest, existing ColBERT MCP server under `paperpilot/mcp_servers/colbert/server.py`, eval rerun script `scripts/day24_rerun_paperpilot_cases.py`.

---

## Context

Phase 3 implements the spec at:

- `docs/superpowers/specs/2026-06-28-query-planner-phase3-evidence-verifier-design.md`

Current retrieval path:

```text
deep-read-paper -> build_index -> mcp__colbert__planned_retrieval -> optional mcp__colbert__search -> answer
```

Current `planned_retrieval` behavior:

- `paperpilot/retrieval/planned_retrieval.py` executes planned queries.
- `paperpilot/retrieval/evidence_pool.py` dedupes raw hits and selects weak `summary_items`.
- `summary_items` are chosen by requirement target membership, ColBERT score, and diversity.
- There is no verification that a chunk directly supports the requirement it was targeted for.

Phase 3 desired behavior:

- Keep current behavior when `verify_evidence=False`.
- When `verify_evidence=True`, run an LLM verifier over candidate evidence grouped by requirement.
- Store verifier decisions under `evidence_pool.verification`.
- Store selected verified ids under `evidence_pool.verified_summary_items`.
- Prefer verified evidence in `summary_text` only when verification is enabled and successful enough to produce selected items.

## File Structure

Create:

- `paperpilot/retrieval/evidence_verifier.py`
  - Defines verifier dataclasses.
  - Builds strict JSON-only prompts.
  - Parses verifier JSON defensively.
  - Scores support/confidence decisions.
  - Selects verified summary evidence by requirement.
  - Runs optional verifier calls through an injectable client.

- `tests/retrieval/test_evidence_verifier.py`
  - Unit tests for parsing, scoring, candidate selection, verified summary selection, conflicts, errors, and prompt content.

Modify:

- `paperpilot/retrieval/evidence_pool.py`
  - Add optional verification fields to `EvidencePool`.
  - Add a formatter path for verified evidence.

- `paperpilot/retrieval/planned_retrieval.py`
  - Add `verify_evidence`, `verifier_candidate_k`, and optional `verifier_client`.
  - Call verifier after `build_evidence_pool`.
  - Preserve defaults and payload shape when verification is off.

- `paperpilot/retrieval/__init__.py`
  - Export new verifier types/helpers only after tests need imports.

- `paperpilot/mcp_servers/colbert/server.py`
  - Add MCP tool args `verify_evidence=False` and `verifier_candidate_k=6`.
  - Pass args through to `run_planned_retrieval`.

- `paperpilot/eval/baselines.py`
  - Add `verify_evidence=False` to `run_paperpilot`.
  - Add prompt guidance that tells the agent to set `verify_evidence=true`.
  - Extract verification metadata from planned retrieval trace payloads.

- `paperpilot/eval/evidence_selection.py`
  - Prefer `verified_summary_items` over `summary_items` when extracting planned retrieval chunks.

- `scripts/day24_rerun_paperpilot_cases.py`
  - Add `--verify-evidence`.
  - Include verification metadata in output JSONL.
  - Use a distinct trace id suffix when verification is enabled.

Modify tests:

- `tests/retrieval/test_planned_retrieval.py`
- `tests/mcp_servers/test_colbert_server_planned_retrieval.py`
- `tests/eval/test_baselines.py`
- `tests/eval/test_evidence_selection.py`
- `tests/eval/test_day24_rerun_paperpilot_cases.py`

Do not modify in Phase 3 unless user explicitly approves later:

- `paperpilot/skills/deep-read-paper.md`
- `paperpilot/mcp_servers.json`
- ColBERT index/model code
- answer-quality repair logic
- answer-level eval selector behavior, except for reading verified chunks

---

## Task 1: Add Verifier Schema and JSON Parser

**Files:**

- Create: `paperpilot/retrieval/evidence_verifier.py`
- Test: `tests/retrieval/test_evidence_verifier.py`

- [ ] **Step 1: Write parser tests**

Create `tests/retrieval/test_evidence_verifier.py` with these initial tests:

```python
from paperpilot.retrieval.evidence_verifier import (
    EvidenceVerificationDecision,
    parse_verifier_output,
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
```

- [ ] **Step 2: Run parser tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/retrieval/test_evidence_verifier.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'paperpilot.retrieval.evidence_verifier'
```

- [ ] **Step 3: Implement parser module**

Create `paperpilot/retrieval/evidence_verifier.py` with:

```python
"""Requirement-level evidence verification for planned retrieval."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

SUPPORT_VALUES = {"direct", "partial", "no"}
CONFIDENCE_VALUES = {"high", "medium", "low"}

SUPPORT_CONFIDENCE_SCORE = {
    ("direct", "high"): 100.0,
    ("direct", "medium"): 80.0,
    ("direct", "low"): 65.0,
    ("partial", "high"): 60.0,
    ("partial", "medium"): 45.0,
    ("partial", "low"): 30.0,
    ("no", "high"): 0.0,
    ("no", "medium"): 0.0,
    ("no", "low"): 0.0,
}


@dataclass(frozen=True)
class EvidenceVerificationDecision:
    requirement_id: str
    evidence_id: str
    support: str
    confidence: str
    answer_atoms: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    reason: str = ""
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceVerificationResult:
    enabled: bool
    method: str
    decisions: list[EvidenceVerificationDecision] = field(default_factory=list)
    verified_summary_items: list[str] = field(default_factory=list)
    missing_verified_requirements: list[dict[str, str]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    verification_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "enabled": self.enabled,
            "method": self.method,
            "decisions": [item.to_dict() for item in self.decisions],
            "verified_summary_items": list(self.verified_summary_items),
            "missing_verified_requirements": list(
                self.missing_verified_requirements
            ),
            "conflicts": list(self.conflicts),
            "stats": dict(self.stats),
        }
        if self.verification_error:
            data["verification_error"] = self.verification_error
        return data


def parse_verifier_output(
    text: str,
) -> tuple[list[EvidenceVerificationDecision], str | None]:
    extracted = _extract_json_object(text)
    if extracted is None:
        return [], "verifier did not return valid JSON"
    try:
        data = json.loads(extracted)
    except json.JSONDecodeError:
        return [], "verifier did not return valid JSON"
    if not isinstance(data, dict):
        return [], "verifier JSON was not an object"
    raw_decisions = data.get("decisions")
    if not isinstance(raw_decisions, list):
        return [], "verifier JSON missing decisions list"

    decisions: list[EvidenceVerificationDecision] = []
    for raw in raw_decisions:
        decision = _normalize_decision(raw)
        if decision is not None:
            decisions.append(decision)
    return decisions, None


def _normalize_decision(raw: Any) -> EvidenceVerificationDecision | None:
    if not isinstance(raw, dict):
        return None
    requirement_id = str(raw.get("requirement_id", "")).strip()
    evidence_id = str(raw.get("evidence_id", "")).strip()
    support = str(raw.get("support", "")).strip().lower()
    confidence = str(raw.get("confidence", "")).strip().lower()
    if (
        not requirement_id
        or not evidence_id
        or support not in SUPPORT_VALUES
        or confidence not in CONFIDENCE_VALUES
    ):
        return None
    answer_atoms = _string_list(raw.get("answer_atoms"))
    risks = _string_list(raw.get("risks"))
    reason = str(raw.get("reason", "")).strip()
    return EvidenceVerificationDecision(
        requirement_id=requirement_id,
        evidence_id=evidence_id,
        support=support,
        confidence=confidence,
        answer_atoms=answer_atoms,
        risks=risks,
        reason=reason,
        score=SUPPORT_CONFIDENCE_SCORE[(support, confidence)],
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _extract_json_object(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]
```

- [ ] **Step 4: Run parser tests and confirm pass**

Run:

```bash
.venv/bin/python -m pytest tests/retrieval/test_evidence_verifier.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add paperpilot/retrieval/evidence_verifier.py tests/retrieval/test_evidence_verifier.py
git commit -m "Add evidence verifier decision parsing"
```

---

## Task 2: Add Candidate Selection, Prompt Builder, and Verified Evidence Selection

**Files:**

- Modify: `paperpilot/retrieval/evidence_verifier.py`
- Test: `tests/retrieval/test_evidence_verifier.py`

- [ ] **Step 1: Add tests for candidates, prompt, scoring, missing, and conflicts**

Append to `tests/retrieval/test_evidence_verifier.py`:

```python
from paperpilot.retrieval.evidence_pool import EvidenceItem, EvidencePool, MatchedQuery
from paperpilot.retrieval.evidence_verifier import (
    build_verifier_prompt,
    candidate_items_for_requirement,
    select_verified_summary,
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
            PlannedQuery("q_dataset", "focused_rewrite", "dataset used", ["req_dataset"], 1),
            PlannedQuery("q_metric", "focused_rewrite", "reported metric", ["req_metric"], 2),
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
        stats={"query_count": 2, "raw_result_count": len(items), "deduped_count": len(items)},
    )


def test_candidate_items_for_requirement_uses_targets_and_score_cap() -> None:
    items = [
        _item("ev_1", "low score target", 0.1, ["req_dataset"]),
        _item("ev_2", "high score target", 0.9, ["req_dataset"]),
        _item("ev_3", "other requirement", 1.0, ["req_metric"]),
    ]

    candidates = candidate_items_for_requirement(
        _pool(items),
        "req_dataset",
        candidate_k=1,
    )

    assert [item.id for item in candidates] == ["ev_2"]


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
    item = _item("ev_1", "Only related work mentions WikiHop.", 0.9, ["req_dataset"])
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
```

- [ ] **Step 2: Run new tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/retrieval/test_evidence_verifier.py -q
```

Expected:

```text
ImportError: cannot import name 'build_verifier_prompt'
```

- [ ] **Step 3: Implement selection and prompt helpers**

Extend `paperpilot/retrieval/evidence_verifier.py` with:

```python
from paperpilot.retrieval.evidence_pool import EvidenceItem, EvidencePool
from paperpilot.retrieval.query_plan import EvidenceRequirement, QueryPlan

MAX_VERIFIER_CHUNK_CHARS = 1800


def candidate_items_for_requirement(
    pool: EvidencePool,
    requirement_id: str,
    *,
    candidate_k: int,
) -> list[EvidenceItem]:
    candidates = [item for item in pool.items if requirement_id in item.targets()]
    return sorted(candidates, key=lambda item: item.best_score, reverse=True)[:candidate_k]


def build_verifier_prompt(
    *,
    plan: QueryPlan,
    requirement: EvidenceRequirement,
    candidates: list[EvidenceItem],
) -> str:
    candidate_blocks = []
    for item in candidates:
        queries = ", ".join(match.query_id for match in item.matched_queries)
        text = item.chunk_text[:MAX_VERIFIER_CHUNK_CHARS]
        candidate_blocks.append(
            f"[{item.id}] score={item.best_score:.4f} queries={queries}\n{text}"
        )
    candidates_text = "\n\n".join(candidate_blocks) or "(no candidates)"
    constraints = plan.constraints
    return (
        "You are a requirement-level evidence verifier for a single-paper QA "
        "retrieval system.\n"
        "Return JSON only, with no markdown fences.\n\n"
        "Judge whether each candidate chunk supports the specific requirement, "
        "not whether it is generally related to the question.\n\n"
        "Allowed support values: direct|partial|no\n"
        "Allowed confidence values: high|medium|low\n\n"
        "Rules:\n"
        "- Use only the candidate chunk text and query plan metadata below.\n"
        "- Do not use prior knowledge.\n"
        "- Do not use benchmark gold answers or oracle spans.\n"
        "- Mark partial when relation, metric, comparator, or list coverage is incomplete.\n"
        "- Mark no for related work, background, nearby context, or a different entity role.\n"
        "- For numeric questions, metric, subset, comparator, and value must stay together.\n"
        "- For list questions, extract only items governed by the question phrase.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "decisions": [\n'
        "    {\n"
        '      "requirement_id": "",\n'
        '      "evidence_id": "",\n'
        '      "support": "direct|partial|no",\n'
        '      "confidence": "high|medium|low",\n'
        '      "answer_atoms": [],\n'
        '      "risks": [],\n'
        '      "reason": ""\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Question: {plan.question}\n"
        f"Question type: {plan.question_type}\n"
        f"Answer shape: {plan.answer_shape}\n"
        f"Intent: {plan.intent_summary}\n"
        f"Needs numbers: {constraints.needs_numbers}\n"
        f"Needs comparison: {constraints.needs_comparison}\n"
        f"Needs table or figure: {constraints.needs_table_or_figure}\n"
        f"Requirement id: {requirement.id}\n"
        f"Requirement description: {requirement.description}\n\n"
        f"Candidate chunks:\n{candidates_text}\n"
    )


def select_verified_summary(
    *,
    plan: QueryPlan,
    pool: EvidencePool,
    decisions: list[EvidenceVerificationDecision],
    summary_k: int,
) -> EvidenceVerificationResult:
    item_by_id = {item.id: item for item in pool.items}
    valid_decisions = [item for item in decisions if item.evidence_id in item_by_id]
    by_requirement: dict[str, list[EvidenceVerificationDecision]] = {}
    for decision in valid_decisions:
        by_requirement.setdefault(decision.requirement_id, []).append(decision)

    selected: list[str] = []
    missing: list[dict[str, str]] = []
    conflicts: list[dict[str, Any]] = []

    for requirement in plan.evidence_requirements:
        if not requirement.required:
            continue
        requirement_decisions = by_requirement.get(requirement.id, [])
        if not requirement_decisions:
            missing.append({
                "requirement_id": requirement.id,
                "description": requirement.description,
                "reason": "no_verifier_decisions",
            })
            continue

        ranked = sorted(
            requirement_decisions,
            key=lambda decision: (
                _verified_sort_score(decision, item_by_id),
                item_by_id[decision.evidence_id].best_score,
            ),
            reverse=True,
        )
        direct = [item for item in ranked if item.support == "direct"]
        partial = [item for item in ranked if item.support == "partial"]
        chosen = direct[:2] if direct else partial[:1]
        if chosen:
            for decision in chosen:
                if decision.evidence_id not in selected and len(selected) < summary_k:
                    selected.append(decision.evidence_id)
        else:
            missing.append({
                "requirement_id": requirement.id,
                "description": requirement.description,
                "reason": "no_verified_direct_or_partial_evidence",
            })
        conflict = _conflict_for_requirement(requirement.id, direct)
        if conflict is not None:
            conflicts.append(conflict)

    if len(selected) < summary_k:
        direct_remaining = sorted(
            [
                item
                for item in valid_decisions
                if item.support == "direct" and item.evidence_id not in selected
            ],
            key=lambda decision: (
                _verified_sort_score(decision, item_by_id),
                item_by_id[decision.evidence_id].best_score,
            ),
            reverse=True,
        )
        for decision in direct_remaining:
            if len(selected) >= summary_k:
                break
            selected.append(decision.evidence_id)

    return EvidenceVerificationResult(
        enabled=True,
        method="llm_requirement_verifier_v1",
        decisions=valid_decisions,
        verified_summary_items=selected,
        missing_verified_requirements=missing,
        conflicts=conflicts,
        stats=_decision_stats(valid_decisions),
    )


def _verified_sort_score(
    decision: EvidenceVerificationDecision,
    item_by_id: dict[str, EvidenceItem],
) -> float:
    item = item_by_id[decision.evidence_id]
    return decision.score + _normalized_colbert_score(item.best_score) * 10.0


def _normalized_colbert_score(score: float) -> float:
    if score <= 0:
        return 0.0
    return min(score / 10.0, 1.0)


def _decision_stats(
    decisions: list[EvidenceVerificationDecision],
) -> dict[str, int]:
    return {
        "decision_count": len(decisions),
        "direct_count": sum(1 for item in decisions if item.support == "direct"),
        "partial_count": sum(1 for item in decisions if item.support == "partial"),
        "no_count": sum(1 for item in decisions if item.support == "no"),
    }


def _conflict_for_requirement(
    requirement_id: str,
    direct: list[EvidenceVerificationDecision],
) -> dict[str, Any] | None:
    high_confidence = [item for item in direct if item.confidence == "high"]
    atoms: list[str] = []
    evidence_ids: list[str] = []
    for decision in high_confidence:
        for atom in decision.answer_atoms:
            if atom not in atoms:
                atoms.append(atom)
        if decision.evidence_id not in evidence_ids:
            evidence_ids.append(decision.evidence_id)
    if len(atoms) <= 1:
        return None
    return {
        "requirement_id": requirement_id,
        "evidence_ids": evidence_ids,
        "answer_atoms": atoms,
        "reason": "multiple_high_confidence_direct_answer_atoms",
    }
```

- [ ] **Step 4: Run verifier tests and confirm pass**

Run:

```bash
.venv/bin/python -m pytest tests/retrieval/test_evidence_verifier.py -q
```

Expected:

```text
9 passed
```

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add paperpilot/retrieval/evidence_verifier.py tests/retrieval/test_evidence_verifier.py
git commit -m "Add requirement-level verified evidence selection"
```

---

## Task 3: Add Verifier Runner With Injectable LLM Client

**Files:**

- Modify: `paperpilot/retrieval/evidence_verifier.py`
- Test: `tests/retrieval/test_evidence_verifier.py`

- [ ] **Step 1: Add fake-client runner tests**

Append:

```python
from types import SimpleNamespace

from paperpilot.retrieval.evidence_verifier import run_evidence_verification


class FakeVerifierClient:
    def __init__(self, responses: list[str] | None = None, error: Exception | None = None) -> None:
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
    pool = _pool([_item("ev_1", "The experiments use WikiHop.", 0.9, ["req_dataset"])])
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
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/retrieval/test_evidence_verifier.py -q
```

Expected:

```text
ImportError: cannot import name 'run_evidence_verification'
```

- [ ] **Step 3: Implement runner**

Extend `paperpilot/retrieval/evidence_verifier.py`:

```python
from paperpilot.core import LLMClient


def run_evidence_verification(
    *,
    plan: QueryPlan,
    pool: EvidencePool,
    client: Any | None = None,
    summary_k: int,
    verifier_candidate_k: int = 6,
) -> EvidenceVerificationResult:
    verifier_client = client or LLMClient()
    decisions: list[EvidenceVerificationDecision] = []
    parse_errors: list[str] = []

    try:
        for requirement in plan.evidence_requirements:
            if not requirement.required:
                continue
            candidates = candidate_items_for_requirement(
                pool,
                requirement.id,
                candidate_k=verifier_candidate_k,
            )
            if not candidates:
                continue
            prompt = build_verifier_prompt(
                plan=plan,
                requirement=requirement,
                candidates=candidates,
            )
            response = verifier_client.call(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                system="",
            )
            parsed, error = parse_verifier_output(response.text or "")
            if error is not None:
                parse_errors.append(f"{requirement.id}: {error}")
                continue
            decisions.extend(parsed)
    except Exception as exc:  # noqa: BLE001
        return EvidenceVerificationResult(
            enabled=True,
            method="llm_requirement_verifier_v1",
            verification_error=f"{type(exc).__name__}: {exc}",
            stats={
                "decision_count": 0,
                "direct_count": 0,
                "partial_count": 0,
                "no_count": 0,
            },
        )

    result = select_verified_summary(
        plan=plan,
        pool=pool,
        decisions=decisions,
        summary_k=summary_k,
    )
    if parse_errors:
        return EvidenceVerificationResult(
            enabled=result.enabled,
            method=result.method,
            decisions=result.decisions,
            verified_summary_items=result.verified_summary_items,
            missing_verified_requirements=result.missing_verified_requirements,
            conflicts=result.conflicts,
            stats=result.stats,
            verification_error="; ".join(parse_errors),
        )
    return result
```

- [ ] **Step 4: Run verifier tests and confirm pass**

Run:

```bash
.venv/bin/python -m pytest tests/retrieval/test_evidence_verifier.py -q
```

Expected:

```text
11 passed
```

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add paperpilot/retrieval/evidence_verifier.py tests/retrieval/test_evidence_verifier.py
git commit -m "Add LLM-backed evidence verifier runner"
```

---

## Task 4: Attach Verification Metadata to EvidencePool

**Files:**

- Modify: `paperpilot/retrieval/evidence_pool.py`
- Test: `tests/retrieval/test_evidence_pool.py`

- [ ] **Step 1: Add serialization and formatter tests**

Append to `tests/retrieval/test_evidence_pool.py`:

```python
from paperpilot.retrieval.evidence_verifier import EvidenceVerificationResult


def test_evidence_pool_serializes_verification_metadata() -> None:
    plan = _plan()
    pool = build_evidence_pool(
        "plan-1",
        plan,
        [
            RawSearchHit(
                paper_id="paper-1",
                chunk_id="chunk-1",
                chunk_text="The experiments use WikiHop.",
                score=9.0,
                query=plan.queries[0],
                rank=1,
            )
        ],
    )
    pool.verification = EvidenceVerificationResult(
        enabled=True,
        method="llm_requirement_verifier_v1",
        verified_summary_items=["ev_1"],
        stats={"decision_count": 1, "direct_count": 1, "partial_count": 0, "no_count": 0},
    )

    data = pool.to_dict()

    assert data["verified_summary_items"] == ["ev_1"]
    assert data["verification"]["enabled"] is True
    assert data["verification"]["stats"]["direct_count"] == 1


def test_format_evidence_summary_can_use_verified_items() -> None:
    plan = _plan()
    pool = build_evidence_pool(
        "plan-1",
        plan,
        [
            RawSearchHit(
                paper_id="paper-1",
                chunk_id="chunk-1",
                chunk_text="Noisy related work chunk.",
                score=10.0,
                query=plan.queries[0],
                rank=1,
            ),
            RawSearchHit(
                paper_id="paper-1",
                chunk_id="chunk-2",
                chunk_text="The experiments use WikiHop.",
                score=8.0,
                query=plan.queries[0],
                rank=2,
            ),
        ],
    )
    pool.verification = EvidenceVerificationResult(
        enabled=True,
        method="llm_requirement_verifier_v1",
        verified_summary_items=["ev_2"],
        stats={"decision_count": 2, "direct_count": 1, "partial_count": 0, "no_count": 1},
    )

    summary = format_evidence_summary(pool, use_verified=True)

    assert "Verified top evidence:" in summary
    assert "The experiments use WikiHop." in summary
    assert "Noisy related work chunk." not in summary
```

- [ ] **Step 2: Run targeted tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/retrieval/test_evidence_pool.py -q
```

Expected:

```text
TypeError: format_evidence_summary() got an unexpected keyword argument 'use_verified'
```

- [ ] **Step 3: Modify `EvidencePool` and formatter**

In `paperpilot/retrieval/evidence_pool.py`:

1. Add a forward-friendly field to `EvidencePool`:

```python
verification: Any | None = None
```

2. Update `to_dict()`:

```python
data = {
    "plan_id": self.plan_id,
    "question": self.question,
    "query_plan": self.query_plan,
    "items": [asdict(item) for item in self.items],
    "summary_items": [item.id for item in self.summary_items],
    "missing_requirements": [
        asdict(item) for item in self.missing_requirements
    ],
    "stats": dict(self.stats),
}
if self.verification is not None:
    verification = (
        self.verification.to_dict()
        if hasattr(self.verification, "to_dict")
        else dict(self.verification)
    )
    data["verification"] = verification
    data["verified_summary_items"] = list(
        verification.get("verified_summary_items") or []
    )
return data
```

3. Change formatter signature:

```python
def format_evidence_summary(pool: EvidencePool, *, use_verified: bool = False) -> str:
```

4. Choose displayed items:

```python
display_items = pool.summary_items
heading = "Top evidence:"
if use_verified and pool.verification is not None:
    verification = (
        pool.verification.to_dict()
        if hasattr(pool.verification, "to_dict")
        else dict(pool.verification)
    )
    verified_ids = [str(item) for item in verification.get("verified_summary_items") or []]
    by_id = {item.id: item for item in pool.items}
    verified_items = [by_id[item_id] for item_id in verified_ids if item_id in by_id]
    if verified_items:
        display_items = verified_items
        heading = "Verified top evidence:"
```

5. Replace the hard-coded `"Top evidence:"` block with `heading`, and iterate over `display_items`.

- [ ] **Step 4: Run evidence pool tests**

Run:

```bash
.venv/bin/python -m pytest tests/retrieval/test_evidence_pool.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add paperpilot/retrieval/evidence_pool.py tests/retrieval/test_evidence_pool.py
git commit -m "Add verification metadata to evidence pools"
```

---

## Task 5: Wire Verifier Into Planned Retrieval Behind a Default-Off Flag

**Files:**

- Modify: `paperpilot/retrieval/planned_retrieval.py`
- Modify: `paperpilot/retrieval/__init__.py`
- Test: `tests/retrieval/test_planned_retrieval.py`

- [ ] **Step 1: Add planned retrieval integration tests**

Append to `tests/retrieval/test_planned_retrieval.py`:

```python
from types import SimpleNamespace


class FakeVerifierClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def call(self, messages: list[dict], tools: list, *, system: str):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        return SimpleNamespace(text="""
        {"decisions": [{
          "requirement_id": "req_dataset",
          "evidence_id": "ev_1",
          "support": "direct",
          "confidence": "high",
          "answer_atoms": ["dataset"],
          "risks": [],
          "reason": "Direct evidence."
        }]}
        """)


def test_run_planned_retrieval_does_not_verify_by_default() -> None:
    verifier = FakeVerifierClient()

    def search(query: str, paper_id: str, top_k: int) -> list[dict]:
        return [
            {
                "paper_id": paper_id,
                "chunk_id": "chunk-1",
                "chunk_text": "Evidence for dataset used.",
                "score": 9.0,
            }
        ]

    result = run_planned_retrieval(
        plan_id="plan-1",
        plan=_plan(),
        paper_id="paper-1",
        search=search,
        verifier_client=verifier,
    )

    assert verifier.calls == []
    payload = result.to_dict()
    assert "verification" not in payload["evidence_pool"]
    assert "verified_summary_items" not in payload["evidence_pool"]


def test_run_planned_retrieval_can_verify_evidence() -> None:
    verifier = FakeVerifierClient()

    def search(query: str, paper_id: str, top_k: int) -> list[dict]:
        return [
            {
                "paper_id": paper_id,
                "chunk_id": "chunk-dataset",
                "chunk_text": "Evidence for dataset used.",
                "score": 9.0,
            }
        ]

    result = run_planned_retrieval(
        plan_id="plan-1",
        plan=_plan(),
        paper_id="paper-1",
        search=search,
        verify_evidence=True,
        verifier_client=verifier,
        verifier_candidate_k=3,
    )

    assert verifier.calls
    payload = result.to_dict()
    assert payload["evidence_pool"]["verification"]["enabled"] is True
    assert payload["evidence_pool"]["verified_summary_items"]
    assert "Verified top evidence:" in payload["summary_text"]
```

- [ ] **Step 2: Run planned retrieval tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/retrieval/test_planned_retrieval.py -q
```

Expected:

```text
TypeError: run_planned_retrieval() got an unexpected keyword argument 'verifier_client'
```

- [ ] **Step 3: Modify planned retrieval**

In `paperpilot/retrieval/planned_retrieval.py`:

1. Import verifier runner:

```python
from paperpilot.retrieval.evidence_verifier import run_evidence_verification
```

2. Extend `run_planned_retrieval` signature:

```python
def run_planned_retrieval(
    *,
    plan_id: str,
    plan: QueryPlan,
    paper_id: str,
    search: SearchFn,
    top_k_each: int = 5,
    summary_k: int = 8,
    verify_evidence: bool = False,
    verifier_candidate_k: int = 6,
    verifier_client: Any | None = None,
) -> PlannedRetrievalResult:
```

3. After `pool = build_evidence_pool(...)`, add:

```python
    if verify_evidence:
        pool.verification = run_evidence_verification(
            plan=plan,
            pool=pool,
            client=verifier_client,
            summary_k=summary_k,
            verifier_candidate_k=verifier_candidate_k,
        )
```

4. Change summary formatting:

```python
summary_text=format_evidence_summary(pool, use_verified=verify_evidence)
```

- [ ] **Step 4: Export verifier symbols**

In `paperpilot/retrieval/__init__.py`, export:

```python
from paperpilot.retrieval.evidence_verifier import (
    EvidenceVerificationDecision,
    EvidenceVerificationResult,
    run_evidence_verification,
)
```

And add these names to `__all__`.

- [ ] **Step 5: Run targeted retrieval tests**

Run:

```bash
.venv/bin/python -m pytest tests/retrieval/test_evidence_verifier.py tests/retrieval/test_evidence_pool.py tests/retrieval/test_planned_retrieval.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add paperpilot/retrieval/planned_retrieval.py paperpilot/retrieval/__init__.py tests/retrieval/test_planned_retrieval.py
git commit -m "Wire optional verifier into planned retrieval"
```

---

## Task 6: Add MCP Tool Parameters

**Files:**

- Modify: `paperpilot/mcp_servers/colbert/server.py`
- Test: `tests/mcp_servers/test_colbert_server_planned_retrieval.py`

- [ ] **Step 1: Add MCP implementation test**

Append to `tests/mcp_servers/test_colbert_server_planned_retrieval.py`:

```python
def test_planned_retrieval_impl_passes_verification_options(monkeypatch) -> None:
    manager = FakeManager()
    monkeypatch.setattr(server, "_manager", manager)

    def fake_plan_with_llm(**kwargs):
        return _plan(kwargs["question"]), {"fallback_used": False}

    captured: dict = {}

    def fake_run_planned_retrieval(**kwargs):
        captured.update(kwargs)

        class Result:
            def to_dict(self):
                return {
                    "summary_text": "Verified planned retrieval completed.",
                    "evidence_pool": {
                        "stats": {},
                        "summary_items": [],
                        "verified_summary_items": ["ev_1"],
                        "verification": {
                            "enabled": True,
                            "method": "llm_requirement_verifier_v1",
                            "decisions": [],
                            "missing_verified_requirements": [],
                            "conflicts": [],
                            "stats": {
                                "decision_count": 1,
                                "direct_count": 1,
                                "partial_count": 0,
                                "no_count": 0,
                            },
                        },
                    },
                    "query_errors": [],
                }

        return Result()

    monkeypatch.setattr(server, "plan_with_llm", fake_plan_with_llm)
    monkeypatch.setattr(server, "run_planned_retrieval", fake_run_planned_retrieval)

    payload = server._planned_retrieval_impl(
        question="What dataset was used?",
        paper_id="paper-1",
        paper_title="Title",
        abstract="Abstract",
        top_k_each=3,
        summary_k=2,
        verify_evidence=True,
        verifier_candidate_k=4,
    )

    assert captured["verify_evidence"] is True
    assert captured["verifier_candidate_k"] == 4
    assert payload["query_plan_meta"] == {"fallback_used": False}
    assert payload["evidence_pool"]["verification"]["enabled"] is True
```

- [ ] **Step 2: Run MCP server tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/mcp_servers/test_colbert_server_planned_retrieval.py -q
```

Expected:

```text
TypeError: _planned_retrieval_impl() got an unexpected keyword argument 'verify_evidence'
```

- [ ] **Step 3: Modify MCP tool and implementation signatures**

In `paperpilot/mcp_servers/colbert/server.py`, extend `planned_retrieval`:

```python
def planned_retrieval(
    question: str,
    paper_id: str,
    paper_title: str = "",
    abstract: str = "",
    top_k_each: int = 5,
    summary_k: int = 8,
    verify_evidence: bool = False,
    verifier_candidate_k: int = 6,
) -> dict:
```

Update docstring args for `verify_evidence` and `verifier_candidate_k`.

Pass both args to `_planned_retrieval_impl`.

Extend `_planned_retrieval_impl` signature:

```python
def _planned_retrieval_impl(
    question: str,
    paper_id: str,
    paper_title: str,
    abstract: str,
    top_k_each: int,
    summary_k: int,
    verify_evidence: bool = False,
    verifier_candidate_k: int = 6,
) -> dict:
```

Pass args to `run_planned_retrieval`:

```python
verify_evidence=verify_evidence,
verifier_candidate_k=verifier_candidate_k,
```

- [ ] **Step 4: Run MCP server tests**

Run:

```bash
.venv/bin/python -m pytest tests/mcp_servers/test_colbert_server_planned_retrieval.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 5: Commit Task 6**

Run:

```bash
git add paperpilot/mcp_servers/colbert/server.py tests/mcp_servers/test_colbert_server_planned_retrieval.py
git commit -m "Expose verifier options on planned retrieval MCP"
```

---

## Task 7: Add Eval Flag and Verification Metadata Extraction

**Files:**

- Modify: `paperpilot/eval/baselines.py`
- Modify: `scripts/day24_rerun_paperpilot_cases.py`
- Test: `tests/eval/test_baselines.py`
- Test: `tests/eval/test_day24_rerun_paperpilot_cases.py`

- [ ] **Step 1: Add baseline tests for prompt guidance and metadata**

Append to `tests/eval/test_baselines.py`:

```python
def test_run_paperpilot_can_request_verified_planned_retrieval(monkeypatch, tmp_path) -> None:
    raw = "Short answer: WikiHop.\n\nEvidence: Verified evidence."
    captured: dict[str, str] = {}

    monkeypatch.setattr(baselines, "_TRACE_DIR", tmp_path)
    _patch_agent_run_capture_prompt(monkeypatch, raw, captured)

    result = baselines.run_paperpilot(
        CASE,
        use_query_plan=True,
        verify_evidence=True,
    )

    assert result["evidence_verification_requested"] is True
    assert "set verify_evidence=true" in captured["prompt"]


def test_run_paperpilot_records_planned_retrieval_verification_metadata(
    monkeypatch, tmp_path
) -> None:
    raw = "Short answer: WikiHop.\n\nEvidence: Verified evidence."

    monkeypatch.setattr(baselines, "_TRACE_DIR", tmp_path)
    _patch_agent_run_with_verified_planned_retrieval(monkeypatch, raw)

    result = baselines.run_paperpilot(
        CASE,
        trace_id="verified-planned-retrieval-case",
        verify_evidence=True,
    )

    assert result["evidence_verification_requested"] is True
    assert result["evidence_verification_used"] is True
    assert result["verified_summary_count"] == 1
    assert result["direct_support_count"] == 1
    assert result["partial_support_count"] == 0
    assert result["unsupported_count"] == 0
    assert result["missing_verified_requirements"] == []
    assert result["verification_conflicts"] == []


def _patch_agent_run_with_verified_planned_retrieval(monkeypatch, final_text: str) -> None:
    import paperpilot.main

    def fake_run(query: str, *, max_iter: int, on_event):
        on_event("tool_call", {
            "name": "mcp__colbert__planned_retrieval",
            "arguments": {
                "question": "What dataset was used?",
                "paper_id": "1234.5678",
                "top_k_each": 5,
                "verify_evidence": True,
            },
        })
        on_event("tool_result", {
            "name": "mcp__colbert__planned_retrieval",
            "content": json.dumps({
                "summary_text": "Verified planned retrieval completed.",
                "evidence_pool": {
                    "stats": {
                        "raw_result_count": 2,
                        "deduped_count": 1,
                    },
                    "summary_items": ["ev_1"],
                    "verified_summary_items": ["ev_1"],
                    "missing_requirements": [],
                    "verification": {
                        "enabled": True,
                        "method": "llm_requirement_verifier_v1",
                        "decisions": [],
                        "missing_verified_requirements": [],
                        "conflicts": [],
                        "stats": {
                            "decision_count": 1,
                            "direct_count": 1,
                            "partial_count": 0,
                            "no_count": 0,
                        },
                    },
                },
                "query_plan_meta": {
                    "fallback_used": False,
                },
                "query_errors": [],
            }),
        })
        return [{"role": "assistant", "content": final_text}]

    monkeypatch.setattr(paperpilot.main, "run", fake_run)
```

- [ ] **Step 2: Run baseline tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/eval/test_baselines.py -q
```

Expected:

```text
TypeError: run_paperpilot() got an unexpected keyword argument 'verify_evidence'
```

- [ ] **Step 3: Modify `run_paperpilot` and prompt guidance**

In `paperpilot/eval/baselines.py`, change signature:

```python
def run_paperpilot(
    case: EvalCase,
    max_iter: int = 12,
    repair_client: LLMClient | None = None,
    evidence_client: LLMClient | None = None,
    use_query_plan: bool = False,
    verify_evidence: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
```

Pass to prompt builder:

```python
prompt = _build_paperpilot_prompt(
    case,
    query_plan=query_plan,
    verify_evidence=verify_evidence,
)
```

Return:

```python
"evidence_verification_requested": verify_evidence,
```

Change `_build_paperpilot_prompt` signature:

```python
def _build_paperpilot_prompt(
    case: EvalCase,
    *,
    query_plan: QueryPlan | None = None,
    verify_evidence: bool = False,
) -> str:
```

Add guidance:

```python
verification_guidance = (
    "Eval verification mode:\n"
    "- When calling mcp__colbert__planned_retrieval, set verify_evidence=true.\n"
    "- Prefer verified_summary_items when present, but still report insufficient evidence when requirements are missing.\n\n"
    if verify_evidence
    else ""
)
```

Insert `verification_guidance` before `Q:`.

- [ ] **Step 4: Extend metadata extraction defaults and return**

In `_extract_planned_retrieval_metadata`, add defaults:

```python
"evidence_verification_used": False,
"verified_summary_count": 0,
"direct_support_count": 0,
"partial_support_count": 0,
"unsupported_count": 0,
"missing_verified_requirements": None,
"verification_conflicts": None,
```

When parsing `pool`, add:

```python
verification = pool.get("verification")
if not isinstance(verification, dict):
    verification = {}
verification_stats = verification.get("stats")
if not isinstance(verification_stats, dict):
    verification_stats = {}
verified_summary_items = pool.get("verified_summary_items")
if not isinstance(verified_summary_items, list):
    verified_summary_items = []
```

Return these keys:

```python
"evidence_verification_used": bool(verification.get("enabled")),
"verified_summary_count": len(verified_summary_items),
"direct_support_count": int(verification_stats.get("direct_count") or 0),
"partial_support_count": int(verification_stats.get("partial_count") or 0),
"unsupported_count": int(verification_stats.get("no_count") or 0),
"missing_verified_requirements": verification.get("missing_verified_requirements"),
"verification_conflicts": verification.get("conflicts"),
```

- [ ] **Step 5: Add script CLI test**

In `tests/eval/test_day24_rerun_paperpilot_cases.py`, extend the existing record test or add:

```python
def test_record_includes_verification_metadata() -> None:
    case = type("Case", (), {
        "case_id": "case-1",
        "question": "What dataset was used?",
        "oracle_spans": ("WikiHop",),
    })()
    ans = {
        "predicted": "WikiHop",
        "elapsed_s": 1.0,
        "trace_path": "trace.jsonl",
        "tool_calls": ["mcp__colbert__planned_retrieval"],
        "error": None,
        "evidence_verification_requested": True,
        "evidence_verification_used": True,
        "verified_summary_count": 1,
        "direct_support_count": 1,
        "partial_support_count": 0,
        "unsupported_count": 0,
        "missing_verified_requirements": [],
        "verification_conflicts": [],
    }

    from scripts.day24_rerun_paperpilot_cases import _record

    record = _record(case, ans, use_query_plan=True, verify_evidence=True)

    assert record["baseline"] == "paperpilot_query_plan_v1_verified_rerun_cases"
    assert record["evidence_verification_requested"] is True
    assert record["evidence_verification_used"] is True
    assert record["verified_summary_count"] == 1
```

- [ ] **Step 6: Modify rerun script**

In `scripts/day24_rerun_paperpilot_cases.py`:

1. Add parser arg:

```python
parser.add_argument("--verify-evidence", action="store_true")
```

2. Pass to `run_paperpilot`:

```python
ans = baselines.run_paperpilot(
    case,
    use_query_plan=args.use_query_plan,
    verify_evidence=args.verify_evidence,
    trace_id=_trace_id(case.case_id, args.use_query_plan, args.verify_evidence),
)
```

3. Add helper:

```python
def _trace_id(case_id: str, use_query_plan: bool, verify_evidence: bool) -> str | None:
    if verify_evidence:
        return f"{case_id}__query_plan_v1_verified"
    if use_query_plan:
        return f"{case_id}__query_plan_v1"
    return None
```

4. Change `_record` signature:

```python
def _record(case: EvalCase, ans: dict, *, use_query_plan: bool, verify_evidence: bool) -> dict:
```

5. Choose baseline:

```python
if verify_evidence:
    baseline = "paperpilot_query_plan_v1_verified_rerun_cases"
elif use_query_plan:
    baseline = "paperpilot_query_plan_v1_rerun_cases"
else:
    baseline = "paperpilot_rerun_cases"
```

6. Include new keys in `extra_key` list:

```python
"evidence_verification_requested",
"evidence_verification_used",
"verified_summary_count",
"direct_support_count",
"partial_support_count",
"unsupported_count",
"missing_verified_requirements",
"verification_conflicts",
```

- [ ] **Step 7: Run eval tests**

Run:

```bash
.venv/bin/python -m pytest tests/eval/test_baselines.py tests/eval/test_day24_rerun_paperpilot_cases.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 8: Commit Task 7**

Run:

```bash
git add paperpilot/eval/baselines.py scripts/day24_rerun_paperpilot_cases.py tests/eval/test_baselines.py tests/eval/test_day24_rerun_paperpilot_cases.py
git commit -m "Record verified planned retrieval eval metadata"
```

---

## Task 8: Prefer Verified Summary Items in Eval Evidence Extraction

**Files:**

- Modify: `paperpilot/eval/evidence_selection.py`
- Test: `tests/eval/test_evidence_selection.py`

- [ ] **Step 1: Add extraction test**

Append to `tests/eval/test_evidence_selection.py`:

```python
def test_extract_retrieved_chunks_prefers_verified_summary_items(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join([
            json.dumps({
                "kind": "tool_result",
                "payload": {
                    "name": "mcp__colbert__planned_retrieval",
                    "content": json.dumps({
                        "evidence_pool": {
                            "items": [
                                {
                                    "id": "ev_1",
                                    "paper_id": "paper-1",
                                    "chunk_text": "Noisy related chunk.",
                                    "best_score": 9.0,
                                    "matched_queries": [{"query": "dataset"}],
                                },
                                {
                                    "id": "ev_2",
                                    "paper_id": "paper-1",
                                    "chunk_text": "The experiments use WikiHop.",
                                    "best_score": 8.0,
                                    "matched_queries": [{"query": "dataset"}],
                                },
                            ],
                            "summary_items": ["ev_1"],
                            "verified_summary_items": ["ev_2"],
                        }
                    }),
                },
            })
        ]),
        encoding="utf-8",
    )

    chunks = extract_retrieved_chunks(trace_path)

    assert [chunk["chunk_text"] for chunk in chunks] == [
        "The experiments use WikiHop."
    ]
```

- [ ] **Step 2: Run extraction test and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/eval/test_evidence_selection.py::test_extract_retrieved_chunks_prefers_verified_summary_items -q
```

Expected:

```text
AssertionError: ['Noisy related chunk.'] != ['The experiments use WikiHop.']
```

- [ ] **Step 3: Modify extraction priority**

In `paperpilot/eval/evidence_selection.py`, change:

```python
summary_ids = pool.get("summary_items")
```

to:

```python
summary_ids = pool.get("verified_summary_items")
if not isinstance(summary_ids, list) or not summary_ids:
    summary_ids = pool.get("summary_items")
```

- [ ] **Step 4: Run evidence selection tests**

Run:

```bash
.venv/bin/python -m pytest tests/eval/test_evidence_selection.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 5: Commit Task 8**

Run:

```bash
git add paperpilot/eval/evidence_selection.py tests/eval/test_evidence_selection.py
git commit -m "Prefer verified planned retrieval evidence in eval selector"
```

---

## Task 9: Focused Regression Suite

**Files:**

- No source changes expected.

- [ ] **Step 1: Run retrieval unit tests**

Run:

```bash
.venv/bin/python -m pytest tests/retrieval -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 2: Run MCP planned retrieval tests**

Run:

```bash
.venv/bin/python -m pytest tests/mcp_servers/test_colbert_server_planned_retrieval.py tests/mcp_servers/test_colbert_via_client.py::test_planned_retrieval_tool_runs -q
```

Expected:

```text
all selected tests passed
```

If `test_colbert_via_client.py::test_planned_retrieval_tool_runs` fails with MCP startup `Connection closed`, do not change verifier code first. Re-check ColBERT startup stderr and `HF_HUB_OFFLINE=1` propagation.

- [ ] **Step 3: Run eval unit tests**

Run:

```bash
.venv/bin/python -m pytest tests/eval/test_baselines.py tests/eval/test_evidence_selection.py tests/eval/test_day24_rerun_paperpilot_cases.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 4: Run full non-slow test suite**

Run:

```bash
.venv/bin/python -m pytest -m "not slow" -q
```

Expected:

```text
all non-slow tests passed
```

- [ ] **Step 5: Commit only if regression fixes were needed**

If Tasks 9.1-9.4 required no source changes, do not create an empty commit.

If fixes were needed:

```bash
git add <changed files>
git commit -m "Fix verifier regression test issues"
```

---

## Task 10: Run Three Phase3 Problem Cases

**Files:**

- No source changes expected.
- Output artifacts under `data/eval/` and `data/traces_rerun_*/` are eval artifacts; confirm with user before committing generated data unless the current branch has been explicitly collecting eval artifacts.

- [ ] **Step 1: Run verified rerun for three known cases**

Run:

```bash
.venv/bin/python scripts/day24_rerun_paperpilot_cases.py \
  --case-id qasper-1910.04601-q1 \
  --case-id qasper-1701.00185-q1 \
  --case-id qasper-1910.07181-q0 \
  --use-query-plan \
  --verify-evidence \
  --out-path data/eval/paperpilot_phase3_verified_cases_20260628.jsonl \
  --trace-dir data/traces_phase3_verified_20260628
```

Expected:

```text
The script writes a JSONL result file and a trace directory.
Each record includes evidence_verification_used and verifier support counts.
```

- [ ] **Step 2: Summarize evidence verification outcomes**

Run:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("data/eval/paperpilot_phase3_verified_cases_20260628.jsonl")
for line in path.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    print(row["case_id"])
    print("  passed:", row.get("passed"))
    print("  verification:", row.get("evidence_verification_used"))
    print("  verified_summary_count:", row.get("verified_summary_count"))
    print("  direct/partial/no:", row.get("direct_support_count"), row.get("partial_support_count"), row.get("unsupported_count"))
    print("  missing:", row.get("missing_verified_requirements"))
    print("  conflicts:", row.get("verification_conflicts"))
    print("  predicted:", str(row.get("predicted", ""))[:300].replace("\\n", " "))
PY
```

Expected:

```text
Human-readable comparison of the three cases.
```

- [ ] **Step 3: Decide whether to commit eval artifacts**

Ask user before committing generated `data/eval/...jsonl` and trace files.

If user approves:

```bash
git add data/eval/paperpilot_phase3_verified_cases_20260628.jsonl data/traces_phase3_verified_20260628
git commit -m "Capture phase 3 verified retrieval eval artifacts"
```

If user declines:

```bash
git status --short
```

Then report generated files as local artifacts only.

---

## Plan Self-Review

- [x] Spec coverage:
  - Optional verifier schema and parser: Tasks 1-3.
  - Requirement-level selection and conflict/missing tracking: Task 2.
  - Default-off planned retrieval integration: Task 5.
  - MCP flags: Task 6.
  - Eval flag and metadata: Task 7.
  - Verified summary extraction for answer selector: Task 8.
  - Problem-case rerun: Task 10.

- [x] Backward compatibility:
  - `verify_evidence=False` remains default in `run_planned_retrieval`.
  - MCP defaults keep old call shape valid.
  - `EvidencePool.to_dict()` only emits `verification` and `verified_summary_items` when verification exists.
  - Existing `summary_items` remains available.

- [x] Error handling:
  - Verifier client failure returns `verification_error` and no verified items.
  - Retrieval still returns original `summary_items`.
  - Eval metadata defaults are stable when no verification payload exists.

- [x] Risk areas:
  - `LLMClient()` requires `DEEPSEEK_API_KEY`; unit tests must always inject fake clients.
  - MCP smoke may fail from ColBERT startup/network cache issues; diagnose separately from verifier behavior.
  - `verified_summary_items` are ids, not item dicts; all extractors must resolve ids through `items`.

## Recommended Execution Order

Use subagent-driven development if available, one task per subagent:

1. Tasks 1-3: pure verifier module.
2. Tasks 4-5: retrieval integration.
3. Task 6: MCP surface.
4. Tasks 7-8: eval integration.
5. Tasks 9-10: verification and problem-case rerun.

If executing inline, complete and commit each task before starting the next one. The first meaningful review checkpoint should be after Task 5, because at that point the verifier exists and planned retrieval can run with verification on, but MCP/eval are still untouched.
