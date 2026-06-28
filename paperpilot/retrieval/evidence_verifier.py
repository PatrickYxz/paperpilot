"""Requirement-level evidence verification for planned retrieval."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from paperpilot.core import LLMClient
from paperpilot.retrieval.evidence_pool import EvidenceItem, EvidencePool
from paperpilot.retrieval.query_plan import EvidenceRequirement, QueryPlan

SUPPORT_VALUES = {"direct", "partial", "no"}
CONFIDENCE_VALUES = {"high", "medium", "low"}
MAX_VERIFIER_CHUNK_CHARS = 1800

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


def candidate_items_for_requirement(
    pool: EvidencePool,
    requirement_id: str,
    *,
    candidate_k: int,
) -> list[EvidenceItem]:
    candidates = [item for item in pool.items if requirement_id in item.targets()]
    return sorted(candidates, key=lambda item: item.best_score, reverse=True)[
        :candidate_k
    ]


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
