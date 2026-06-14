"""Eval summary helper tests."""
import json
from pathlib import Path

from paperpilot.web.eval_summary import (
    build_eval_snapshot,
    list_calibration_candidates,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_eval_snapshot_computes_semantic_and_calibrated_scores(tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    calibration_path = tmp_path / "calibration.jsonl"
    _write_jsonl(
        audit_path,
        [
            {
                "case_id": "case-1",
                "strict_pass": True,
                "semantic_label": "correct",
            },
            {
                "case_id": "case-2",
                "strict_pass": False,
                "semantic_label": "partial",
            },
            {
                "case_id": "case-3",
                "strict_pass": True,
                "semantic_label": "incorrect",
            },
        ],
    )
    _write_jsonl(
        calibration_path,
        [
            {
                "case_id": "case-2",
                "review_decision": "accept_as_correct",
            },
            {
                "case_id": "case-3",
                "review_decision": "downgrade_to_incorrect",
            },
        ],
    )

    snapshot = build_eval_snapshot(
        audit_path=audit_path,
        calibration_path=calibration_path,
    )

    assert snapshot["available"] is True
    assert snapshot["total_cases"] == 3
    assert snapshot["strict"]["pass_count"] == 2
    assert snapshot["strict"]["rate"] == 0.6667
    assert snapshot["semantic"]["correct_count"] == 1
    assert snapshot["semantic"]["weighted_count"] == 1.5
    assert snapshot["calibrated"]["correct_count"] == 2
    assert snapshot["calibrated"]["weighted_count"] == 2.0
    assert snapshot["calibrated"]["decision_counts"] == {
        "accept_as_correct": 1,
        "downgrade_to_incorrect": 1,
    }


def test_build_eval_snapshot_returns_unavailable_for_missing_audit(tmp_path: Path):
    snapshot = build_eval_snapshot(
        audit_path=tmp_path / "missing.jsonl",
        calibration_path=tmp_path / "calibration.jsonl",
    )

    assert snapshot["available"] is False
    assert "not found" in snapshot["message"]


def test_list_calibration_candidates_filters_and_truncates_rows(tmp_path: Path):
    calibration_path = tmp_path / "calibration.jsonl"
    _write_jsonl(
        calibration_path,
        [
            {
                "case_id": "case-1",
                "category": "partial_high",
                "review_decision": "accept_as_correct",
                "strict_pass": True,
                "semantic_label": "partial",
                "confidence": "high",
                "question": "Q1?",
                "oracle_spans": ["A"],
                "predicted": "x" * 2000,
                "judge_reason": "Reason 1",
                "review_notes": "Note 1",
            },
            {
                "case_id": "case-2",
                "category": "partial_medium",
                "review_decision": "keep_partial",
                "strict_pass": False,
                "semantic_label": "partial",
                "confidence": "medium",
                "question": "Q2?",
                "oracle_spans": ["B"],
                "predicted": "short",
                "judge_reason": "Reason 2",
                "review_notes": "Note 2",
            },
        ],
    )

    payload = list_calibration_candidates(
        calibration_path=calibration_path,
        category="partial_high",
        review_decision="accept_as_correct",
    )

    assert payload["available"] is True
    assert payload["total_candidates"] == 2
    assert payload["filtered_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["case_id"] == "case-1"
    assert candidate["review_notes"] == "Note 1"
    assert "[truncated]" in candidate["predicted_excerpt"]


def test_list_calibration_candidates_returns_unavailable_for_missing_file(tmp_path: Path):
    payload = list_calibration_candidates(
        calibration_path=tmp_path / "missing.jsonl",
    )

    assert payload["available"] is False
    assert payload["candidates"] == []
