"""Tests for semantic audit schema helpers."""
import pytest

from paperpilot.eval.semantic_audit import (
    ALLOWED_SEMANTIC_LABELS,
    SemanticAuditRecord,
    semantic_audit_record_from_dict,
)


def test_allowed_semantic_labels_are_fixed() -> None:
    assert ALLOWED_SEMANTIC_LABELS == {
        "correct",
        "partial",
        "incorrect",
        "contradictory",
        "unverifiable",
        "judge_uncertain",
    }


def test_semantic_audit_record_to_dict() -> None:
    record = SemanticAuditRecord(
        case_id="qasper-1-q0",
        baseline="paperpilot",
        strict_pass=True,
        semantic_label="partial",
        confidence="medium",
        reason="The answer includes one required span but misses another.",
        used_gold_evidence=True,
        audit_model=None,
        audit_version="v1",
    )

    assert record.to_dict() == {
        "case_id": "qasper-1-q0",
        "baseline": "paperpilot",
        "strict_pass": True,
        "semantic_label": "partial",
        "confidence": "medium",
        "reason": "The answer includes one required span but misses another.",
        "used_gold_evidence": True,
        "audit_model": None,
        "audit_version": "v1",
    }


def test_semantic_audit_record_rejects_bad_label() -> None:
    with pytest.raises(ValueError, match="invalid semantic_label"):
        SemanticAuditRecord(
            case_id="qasper-1-q0",
            baseline="paperpilot",
            strict_pass=True,
            semantic_label="almost_right",
            confidence="medium",
            reason="Bad label.",
            used_gold_evidence=True,
            audit_model=None,
            audit_version="v1",
        )


def test_semantic_audit_record_from_dict_uses_legacy_passed() -> None:
    record = semantic_audit_record_from_dict({
        "case_id": "qasper-1-q0",
        "baseline": "paperpilot",
        "passed": False,
        "semantic_label": "incorrect",
        "confidence": "high",
        "reason": "The answer does not answer the question.",
        "used_gold_evidence": True,
        "audit_model": "manual",
        "audit_version": "v1",
    })

    assert record.strict_pass is False
    assert record.audit_model == "manual"
