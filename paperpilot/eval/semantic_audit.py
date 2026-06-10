"""Semantic audit schema for QASPER eval results.

This module defines data structures only. It does not call an LLM judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SemanticLabel = Literal[
    "correct",
    "partial",
    "incorrect",
    "contradictory",
    "unverifiable",
    "judge_uncertain",
]

ALLOWED_SEMANTIC_LABELS: set[str] = {
    "correct",
    "partial",
    "incorrect",
    "contradictory",
    "unverifiable",
    "judge_uncertain",
}


@dataclass(frozen=True)
class SemanticAuditRecord:
    case_id: str
    baseline: str
    strict_pass: bool
    semantic_label: SemanticLabel
    confidence: str
    reason: str
    used_gold_evidence: bool
    audit_model: str | None
    audit_version: str

    def __post_init__(self) -> None:
        if self.semantic_label not in ALLOWED_SEMANTIC_LABELS:
            raise ValueError(f"invalid semantic_label: {self.semantic_label!r}")
        if not self.case_id.strip():
            raise ValueError("case_id is required")
        if not self.baseline.strip():
            raise ValueError("baseline is required")
        if not self.reason.strip():
            raise ValueError("reason is required")
        if not self.audit_version.strip():
            raise ValueError("audit_version is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "baseline": self.baseline,
            "strict_pass": self.strict_pass,
            "semantic_label": self.semantic_label,
            "confidence": self.confidence,
            "reason": self.reason,
            "used_gold_evidence": self.used_gold_evidence,
            "audit_model": self.audit_model,
            "audit_version": self.audit_version,
        }


def semantic_audit_record_from_dict(row: dict[str, Any]) -> SemanticAuditRecord:
    strict_pass = row.get("strict_pass", row.get("passed"))
    if not isinstance(strict_pass, bool):
        raise ValueError("strict_pass or passed must be a boolean")
    return SemanticAuditRecord(
        case_id=str(row["case_id"]),
        baseline=str(row["baseline"]),
        strict_pass=strict_pass,
        semantic_label=str(row["semantic_label"]),  # type: ignore[arg-type]
        confidence=str(row.get("confidence") or "unknown"),
        reason=str(row["reason"]),
        used_gold_evidence=bool(row.get("used_gold_evidence", False)),
        audit_model=(
            None if row.get("audit_model") is None else str(row.get("audit_model"))
        ),
        audit_version=str(row.get("audit_version") or "v1"),
    )
