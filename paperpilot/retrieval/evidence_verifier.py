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
